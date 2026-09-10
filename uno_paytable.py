"""
uno_paytable.py
===============

Builds paytables that hit a target RTP against a simulated score distribution.

RTP is linear in the payouts -- RTP = sum over scores of P(score) * pay(score) --
so this is solved rather than searched. What makes it interesting is that the
system is badly underdetermined: about twenty paying tiers against a single RTP
equation. Any number of tables hit 92%. So the script does not look for "the"
answer; it takes a SHAPE and solves for the one free scale inside it.


THE CONSTRAINTS
---------------
    * a score of 5 always pays exactly 1.5x
    * every step up pays at least 0.25x more than the step below
    * the 0.0x tiers at the bottom are exempt, and must be one contiguous block
    * every payout is a whole multiple of 0.25x

Those first two together set a hard floor on RTP. Anchored at 1.5x on score 5,
the cheapest legal table climbs by exactly 0.25x per step, which still pays
0.25 * (score + 1) at every score above 5. Push the win line as high as it can
go -- score 5, since 5 must pay -- and that table's RTP is the minimum this rule
set can produce. Ask for less and ``balance`` will tell you it is impossible
rather than quietly returning something close.

There is no ceiling: the tail can grow without limit.


SHAPES
------
Each shape is a payout curve with one free parameter, solved by bisection.
All of them respect the constraints above; they differ in where the money sits.

    minimal       every step the least legal 0.25x, no free parameter at all.
                  Only useful for reading off the floor.
    linear        a constant step everywhere. Flattest sensible table.
    progressive   steps that grow with score, so the curve bends upward.
    geometric     a constant multiplier per step. Bends harder.
    tail          minimal steps through the middle, everything loaded into the
                  top few scores. This is the shape of a jackpot game, and the
                  shape the current hand-built table is closest to.

Lower shapes on that list put more of the return into common scores; higher ones
put it into rare ones. Same RTP, very different game.


A WARNING ABOUT THE DISTRIBUTION
--------------------------------
A table is only balanced as well as the distribution it was balanced against,
and the scores that matter most are the rarest. A 250x jackpot on a 0.03% score
carries a real share of the RTP off maybe 300 observations in a million hands.
So ``balance`` reports the sampling error it inherited, and ``verify`` re-runs
the finished table on seeds that were never used to build it. Treat the verify
number as the real one.


USAGE
-----
    python3 uno_paytable.py

    from uno_paytable import score_distribution, balance, verify
    dist = score_distribution(hands=1_000_000, seed=1)
    table = balance(dist, target_rtp=0.92, shape="tail")
    print(table.as_python())
    print(verify(table, hands=1_000_000, seed=999))
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from uno_sim import JACKPOT_SCORE, play_hand
from uno_strategy import DEFAULT_STRATEGY

__all__ = [
    "STEP",
    "ANCHOR_SCORE",
    "ANCHOR_PAY",
    "SHAPES",
    "Paytable",
    "score_distribution",
    "rtp_of",
    "payout_stdev",
    "minimum_rtp",
    "low_tiers_for",
    "feasible_range",
    "validate",
    "balance",
    "verify",
]


# --------------------------------------------------------------------------
# The rules the tables must obey
# --------------------------------------------------------------------------

#: Payout granularity. Every payout is a whole multiple of this.
STEP = 0.25

#: The pinned tier: this score always pays exactly this much.
ANCHOR_SCORE = 5
ANCHOR_PAY = 1.5

#: Smallest legal increase between consecutive paying tiers, in STEP units.
MIN_STEP_UNITS = 1

assert abs(ANCHOR_PAY / STEP - round(ANCHOR_PAY / STEP)) < 1e-9, \
    "the anchor payout must be a whole multiple of STEP"
ANCHOR_UNITS = round(ANCHOR_PAY / STEP)


def _units(pay: float) -> int:
    """Convert a payout to whole STEP units."""
    return round(pay / STEP)


# --------------------------------------------------------------------------
# Score distribution
# --------------------------------------------------------------------------

def score_distribution(hands: int = 1_000_000,
                       hand_size: int = 5,
                       strategy: str = DEFAULT_STRATEGY,
                       seed: Optional[int] = None,
                       jackpot_score: int = JACKPOT_SCORE) -> Counter:
    """Simulate ``hands`` hands and return how often each score came up.

    Accumulates only the histogram, so memory stays flat no matter the batch
    size -- which matters, because balancing a jackpot tier needs millions of
    hands before its frequency settles.
    """
    if hands < 1:
        raise ValueError("hands must be at least 1")
    rng = random.Random(seed)
    dist: Counter = Counter()
    for _ in range(hands):
        dist[play_hand(hand_size=hand_size, strategy=strategy, rng=rng,
                       jackpot_score=jackpot_score).points] += 1
    return dist


def _probabilities(dist: Counter) -> Dict[int, float]:
    total = sum(dist.values())
    if not total:
        raise ValueError("the distribution is empty")
    return {s: c / total for s, c in dist.items()}


def rtp_of(paytable: Dict[int, float], dist: Counter) -> float:
    """Return to player: the average payout per unit staked."""
    p = _probabilities(dist)
    return sum(prob * paytable.get(s, 0.0) for s, prob in p.items())


def payout_stdev(paytable: Dict[int, float], dist: Counter) -> float:
    """Standard deviation of the payout on a single hand.

    Drives how many hands any RTP estimate needs. A long tail makes this large.
    """
    p = _probabilities(dist)
    mean = sum(prob * paytable.get(s, 0.0) for s, prob in p.items())
    var = sum(prob * (paytable.get(s, 0.0) - mean) ** 2 for s, prob in p.items())
    return math.sqrt(max(0.0, var))


# --------------------------------------------------------------------------
# Shapes: a payout curve with one free parameter
# --------------------------------------------------------------------------

#: A shape maps (score, lam, max_score) -> desired payout, before the minimum
#: step rule is enforced. Must be non-decreasing in ``lam`` so the solver can
#: bisect on it.
Shape = Callable[[int, float, int], float]


def _shape_minimal(score: int, lam: float, hi: int) -> float:
    return ANCHOR_PAY + STEP * (score - ANCHOR_SCORE)


def _shape_linear(score: int, lam: float, hi: int) -> float:
    return ANCHOR_PAY + (STEP + lam) * (score - ANCHOR_SCORE)


def _shape_progressive(score: int, lam: float, hi: int) -> float:
    d = score - ANCHOR_SCORE
    return ANCHOR_PAY + STEP * d + lam * d * d


def _shape_geometric(score: int, lam: float, hi: int) -> float:
    d = score - ANCHOR_SCORE
    # (1 + lam) per step, with the minimal ladder as a floor for small lam.
    return max(ANCHOR_PAY + STEP * d, ANCHOR_PAY * (1.0 + lam) ** d)


def _make_tail_shape(tail_start: int) -> Shape:
    """Minimal steps below ``tail_start``, everything loaded above it."""
    def shape(score: int, lam: float, hi: int) -> float:
        base = ANCHOR_PAY + STEP * (score - ANCHOR_SCORE)
        over = max(0, score - tail_start + 1)
        return base + lam * over * over
    return shape


#: Shapes by name. ``tail`` defaults to loading the top five scores.
SHAPES: Dict[str, Shape] = {
    "minimal": _shape_minimal,
    "linear": _shape_linear,
    "progressive": _shape_progressive,
    "geometric": _shape_geometric,
    "tail": _make_tail_shape(JACKPOT_SCORE - 4),
}


# --------------------------------------------------------------------------
# Building and checking a table
# --------------------------------------------------------------------------

@dataclass(slots=True)
class Paytable:
    """A finished paytable, plus how it was built and how well it landed."""

    payouts: Dict[int, float]
    target_rtp: float
    achieved_rtp: float
    shape: str
    lam: float
    win_line: int
    balanced_on: int
    #: Sampling error the table inherited from the distribution it was fitted to.
    fit_stderr: float = 0.0
    hands_fitted: int = 0

    @property
    def error(self) -> float:
        """Signed miss against the target."""
        return self.achieved_rtp - self.target_rtp

    @property
    def house_edge(self) -> float:
        return 1.0 - self.achieved_rtp

    def contributions(self, dist: Counter) -> List[Tuple[int, float, float, float]]:
        """Per score: (score, frequency, payout, share of RTP). Paying tiers only."""
        p = _probabilities(dist)
        rows = []
        for s in sorted(self.payouts):
            pay = self.payouts[s]
            if pay <= 0:
                continue
            freq = p.get(s, 0.0)
            rows.append((s, freq, pay, freq * pay))
        return rows

    def as_python(self) -> str:
        """The table as a paste-ready PAYTABLE literal for uno_sim."""
        lines = ["PAYTABLE: Dict[int, float] = {"]
        for s in sorted(self.payouts):
            pay = self.payouts[s]
            note = ""
            if s == ANCHOR_SCORE:
                note = "   # anchored"
            elif s == self.balanced_on:
                note = "   # balancing tier"
            lines.append(f"    {s:>2}: {pay:g},{note}")
        lines.append("}")
        return "\n".join(lines)

    def describe(self, dist: Counter) -> str:
        rows = self.contributions(dist)
        out = [
            f"shape={self.shape}  target={self.target_rtp:.3%}  "
            f"achieved={self.achieved_rtp:.4%}  miss={self.error:+.4%}",
            f"house edge {self.house_edge:.3%}   win line at score {self.win_line}"
            f"   balanced on score {self.balanced_on}",
        ]
        if self.hands_fitted:
            out.append(f"fitted on {self.hands_fitted:,} hands, so the fit itself "
                       f"carries +/- {self.fit_stderr:.3%}")
        out += ["", "  score      freq      pays   contributes   share"]
        for s, freq, pay, contrib in rows:
            share = contrib / self.achieved_rtp if self.achieved_rtp else 0.0
            out.append(f"  {s:>5}  {freq:8.4%}  {pay:7.2f}x  {contrib:11.4f}"
                       f"   {share:6.1%}")
        hit = sum(f for _, f, _, _ in rows)
        real = sum(f for _, f, pay, _ in rows if pay >= 1.0)
        out += [
            "  " + "-" * 50,
            f"  hit rate {hit:.3%}, of which {real:.3%} actually returns the stake "
            f"or better",
        ]
        return "\n".join(out)


def validate(payouts: Dict[int, float]) -> List[str]:
    """Check a table against the rules. Returns a list of problems, empty if fine."""
    problems: List[str] = []
    scores = sorted(payouts)
    if not scores:
        return ["the table is empty"]

    for s in scores:
        u = payouts[s] / STEP
        if abs(u - round(u)) > 1e-9:
            problems.append(f"score {s} pays {payouts[s]:g}, not a multiple of {STEP}")
        if payouts[s] < 0:
            problems.append(f"score {s} pays a negative amount")

    if ANCHOR_SCORE in payouts and _units(payouts[ANCHOR_SCORE]) != ANCHOR_UNITS:
        problems.append(f"score {ANCHOR_SCORE} pays {payouts[ANCHOR_SCORE]:g}, "
                        f"must be {ANCHOR_PAY:g}")

    # zeros must be one contiguous block at the bottom
    paying = [s for s in scores if payouts[s] > 0]
    if paying:
        first = paying[0]
        for s in scores:
            if s > first and payouts[s] <= 0:
                problems.append(f"score {s} pays nothing but a lower score pays; "
                                f"the 0.0x tiers must be one block at the bottom")

    # every step up gains at least MIN_STEP_UNITS, among paying tiers
    for a, b in zip(paying, paying[1:]):
        gain = _units(payouts[b]) - _units(payouts[a])
        if gain < MIN_STEP_UNITS:
            problems.append(
                f"score {b} pays {payouts[b]:g} against {payouts[a]:g} at score {a}; "
                f"needs at least +{STEP * MIN_STEP_UNITS:g}")
    return problems


def _build(shape: Shape, lam: float, win_line: int, max_score: int,
           low_tiers: Dict[int, float]) -> Dict[int, int]:
    """Lay out a table in STEP units: quantise the curve, then repair the ladder."""
    units: Dict[int, int] = {}

    for s in range(0, win_line):
        units[s] = 0

    # Below the anchor the tiers are given outright -- there are only a few, and
    # pinning them is clearer than fitting them.
    for s in range(win_line, ANCHOR_SCORE):
        units[s] = _units(low_tiers[s])
    units[ANCHOR_SCORE] = ANCHOR_UNITS

    # Above the anchor, follow the shape but never break the minimum step.
    for s in range(ANCHOR_SCORE + 1, max_score + 1):
        want = _units(shape(s, lam, max_score))
        units[s] = max(want, units[s - 1] + MIN_STEP_UNITS)
    return units


def _rtp_units(units: Dict[int, int], probs: Dict[int, float]) -> float:
    return sum(prob * STEP * units.get(s, 0) for s, prob in probs.items())


def low_tiers_for(win_line: int, mode: str = "min") -> Dict[int, float]:
    """The tiers between the win line and the anchor.

    These are few but they dominate RTP, because low scores are far and away the
    most common. Score 4 alone turns up on roughly one hand in six, and the rules
    allow it anything from 0.25x up to 1.25x -- a swing of about 15 points of RTP
    from that single tier. So this is the coarse control, and the tail is the fine
    one, which is the opposite of what it looks like.

        "min"  the cheapest legal ladder: 0.25x at the win line, +0.25x per step
        "max"  the dearest legal ladder, running right up to the anchor
    """
    span = range(win_line, ANCHOR_SCORE)
    if mode == "min":
        return {s: STEP * (1 + s - win_line) for s in span}
    if mode == "max":
        return {s: ANCHOR_PAY - STEP * (ANCHOR_SCORE - s) for s in span}
    raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")


def feasible_range(dist: Counter,
                   shape: str = "tail",
                   win_line: int = 4,
                   low_tiers: Optional[Dict[int, float]] = None,
                   max_score: int = JACKPOT_SCORE) -> float:
    """Lowest RTP reachable with this shape, win line, and low tiers.

    There is no upper bound -- the tail can grow without limit -- so only the
    floor is worth reporting.
    """
    probs = _probabilities(dist)
    if low_tiers is None:
        low_tiers = low_tiers_for(win_line)
    return _rtp_units(_build(SHAPES[shape], 0.0, win_line, max_score, low_tiers),
                      probs)


def minimum_rtp(dist: Counter, max_score: int = JACKPOT_SCORE) -> float:
    """The lowest RTP the constraints allow.

    Win line as high as it can go -- the anchor score, since that tier must pay --
    and every step above it the minimum 0.25x.
    """
    probs = _probabilities(dist)
    units = _build(_shape_minimal, 0.0, ANCHOR_SCORE, max_score, {})
    return _rtp_units(units, probs)


def balance(dist: Counter,
            target_rtp: float,
            shape: str = "tail",
            win_line: int = 4,
            low_tiers: Optional[Dict[int, float]] = None,
            max_score: int = JACKPOT_SCORE,
            balance_on: Optional[int] = None,
            hands_fitted: int = 0) -> Paytable:
    """Build a table of the given shape that pays ``target_rtp``.

    Solved in two passes. Bisection on the shape's free parameter gets close;
    quantising to 0.25x makes RTP a staircase in that parameter, so it cannot
    land exactly. The remainder is then closed on a single tier -- by default the
    top score, whose frequency is so low that one 0.25x step moves RTP by a tiny
    amount, which is exactly the fine adjustment needed.

    That is also how a real jackpot game is balanced: the base game is a
    deliberate shape, and the top prize absorbs the rounding.
    """
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {tuple(SHAPES)}, got {shape!r}")
    if not 0 < target_rtp < 100:
        raise ValueError("target_rtp is a fraction, e.g. 0.92 for 92%")
    if win_line > ANCHOR_SCORE:
        raise ValueError(f"win_line cannot exceed {ANCHOR_SCORE}, which must pay")

    probs = _probabilities(dist)
    fn = SHAPES[shape]
    if balance_on is None:
        balance_on = max_score

    if low_tiers is None:
        # Cheapest legal ladder, which keeps the floor as low as possible and so
        # leaves the most room to hit a target. Pass low_tiers explicitly for a
        # more generous shoulder -- see low_tiers_for().
        low_tiers = low_tiers_for(win_line)

    floor = _rtp_units(_build(fn, 0.0, win_line, max_score, low_tiers), probs)
    if target_rtp < floor:
        hard = minimum_rtp(dist, max_score)
        levers = []
        if win_line < ANCHOR_SCORE:
            cheap = low_tiers_for(win_line)
            if any(low_tiers[s] > cheap[s] for s in cheap):
                levers.append("cheapen the low tiers (low_tiers_for(win_line, 'min'))")
            levers.append(f"raise win_line towards {ANCHOR_SCORE}")
        raise ValueError(
            f"target {target_rtp:.3%} is below what the constraints allow here. "
            f"This shape, win line {win_line} and these low tiers floor out at "
            f"{floor:.3%}. The lowest any legal table reaches is {hard:.3%}, with "
            f"nothing paid below score {ANCHOR_SCORE} and every step the minimum "
            f"{STEP:g}x -- a score of 5 paying {ANCHOR_PAY:g}x is what sets that. "
            + (f"To come down: {', '.join(levers)}." if levers else "")
        )

    # Bracket, then bisect. RTP rises with lam for every shape here.
    lo, hi = 0.0, 1.0
    for _ in range(200):
        if _rtp_units(_build(fn, hi, win_line, max_score, low_tiers), probs) >= target_rtp:
            break
        hi *= 2.0
    else:
        raise RuntimeError("could not bracket the target; check the shape")

    for _ in range(80):
        mid = (lo + hi) / 2
        if _rtp_units(_build(fn, mid, win_line, max_score, low_tiers), probs) < target_rtp:
            lo = mid
        else:
            hi = mid
    lam = lo

    units = _build(fn, lam, win_line, max_score, low_tiers)

    # Close the remainder on the balancing tier, respecting its neighbours.
    prob_b = probs.get(balance_on, 0.0)
    if prob_b > 0:
        gap = target_rtp - _rtp_units(units, probs)
        delta = round(gap / (STEP * prob_b))
        lower = units.get(balance_on - 1, 0) + MIN_STEP_UNITS
        upper = (units[balance_on + 1] - MIN_STEP_UNITS
                 if balance_on + 1 in units else None)
        want = units[balance_on] + delta
        want = max(want, lower)
        if upper is not None:
            want = min(want, upper)
        units[balance_on] = want

    payouts = {s: round(u * STEP, 10) for s, u in units.items()}
    problems = validate(payouts)
    if problems:
        raise RuntimeError("built an illegal table: " + "; ".join(problems))

    achieved = _rtp_units(units, probs)
    sd = payout_stdev(payouts, dist)
    return Paytable(
        payouts=payouts,
        target_rtp=target_rtp,
        achieved_rtp=achieved,
        shape=shape,
        lam=lam,
        win_line=win_line,
        balanced_on=balance_on,
        fit_stderr=sd / math.sqrt(sum(dist.values())),
        hands_fitted=hands_fitted or sum(dist.values()),
    )


def verify(table: Paytable,
           hands: int = 1_000_000,
           hand_size: int = 5,
           strategy: str = DEFAULT_STRATEGY,
           seed: Optional[int] = None,
           jackpot_score: int = JACKPOT_SCORE) -> Tuple[float, float]:
    """Re-measure a finished table on fresh hands. Returns (rtp, standard error).

    Use a seed that was not used to fit the table. Balancing against a
    distribution and then reporting the RTP off that same distribution just
    reads back the number that was fitted; it says nothing about whether the
    tail frequencies were right.
    """
    dist = score_distribution(hands=hands, hand_size=hand_size, strategy=strategy,
                              seed=seed, jackpot_score=jackpot_score)
    rtp = rtp_of(table.payouts, dist)
    se = payout_stdev(table.payouts, dist) / math.sqrt(hands)
    return rtp, se


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

if __name__ == "__main__":
    TARGET = 0.95
    FIT_HANDS = 400_000
    CHECK_HANDS = 400_000
    WIN_LINE = 4
    LOW_TIERS = {4: 0.5}            # matches the hand-built table's shoulder
    SHAPES_TO_TRY = ("linear", "progressive", "geometric", "tail")

    print(f"simulating {FIT_HANDS:,} hands to fit against...")
    fit_dist = score_distribution(hands=FIT_HANDS, seed=20260801)

    print(f"\nfloor imposed by the constraints: {minimum_rtp(fit_dist):.3%}")
    print(f"  (score {ANCHOR_SCORE} must pay {ANCHOR_PAY:g}x and every step must gain "
          f"{STEP:g}x, so no legal table goes below this)")

    print("\nthe low tiers are the coarse lever, not the tail -- floor by win line:")
    for w in (5, 4, 3):
        lo = feasible_range(fit_dist, win_line=w)
        hi = feasible_range(fit_dist, win_line=w,
                            low_tiers=low_tiers_for(w, "max")) if w < ANCHOR_SCORE else lo
        print(f"  win line {w}: {lo:.2%} on the cheapest shoulder, "
              f"{hi:.2%} on the dearest")

    print(f"\ntargeting {TARGET:.1%} with win line {WIN_LINE}, "
          f"score 4 paying {LOW_TIERS[4]:g}x\n")

    tables = {}
    for name in SHAPES_TO_TRY:
        t = balance(fit_dist, target_rtp=TARGET, shape=name,
                    win_line=WIN_LINE, low_tiers=LOW_TIERS)
        tables[name] = t
        top = max(t.payouts)
        print(f"  {name:12} achieved {t.achieved_rtp:.4%}  miss {t.error:+.4%}  "
              f"top prize {t.payouts[top]:>7g}x  "
              f"volatility {payout_stdev(t.payouts, fit_dist):6.2f}")

    # One fresh sample, shared across all the tables -- simulating per table would
    # cost four runs for no extra information.
    print(f"\nverifying on {CHECK_HANDS:,} fresh hands, seed never used to fit:")
    check_dist = score_distribution(hands=CHECK_HANDS, seed=99991)
    for name, t in tables.items():
        rtp = rtp_of(t.payouts, check_dist)
        se = payout_stdev(t.payouts, check_dist) / math.sqrt(CHECK_HANDS)
        print(f"  {name:12} fitted {t.achieved_rtp:.3%} -> measured {rtp:.3%} "
              f"+/- {se:.3%}   drift {rtp - t.achieved_rtp:+.3%}")
    print("  (all four drift together -- that is the fit distribution's tail error,"
          "\n   shared by every table built from it, not noise in the tables)")

    pick = "tail"
    print(f"\n{'=' * 64}\nfull breakdown, {pick} shape\n{'=' * 64}")
    print(tables[pick].describe(fit_dist))
    print()
    print(tables[pick].as_python())