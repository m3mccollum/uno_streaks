"""
uno_sim.py
==========

Hand simulator for the Uno-based casino game. Builds on ``uno_deck``.

One hand, as currently defined:

    1. Shuffle one deck and deal the player a hand.
    2. Deal one card to the dealer. If it is not a number card (0-9), discard
       it and deal again, so the player always matches against a number.
    3. If the player holds a card that plays on the dealer's card -- matching
       color, matching number, or any wild -- play it: remove it from hand and
       score. A Skip scores 2 points; every other card scores 1.
    3b. If that card was a Reverse, the player immediately plays a second card
       alongside it, scoring for both. The rider must match the Reverse: same
       color, any wild, or any Reverse of any color. If the rider is itself a
       Reverse it grants another rider, so a hand holding several Reverses can
       chain through all of them in a single turn.
    4. If the card played was a Draw Two, the player draws 2 cards. If it was a
       Wild Draw Four, the player draws 4. Plain Wilds and blanks draw nothing.
       When the draw pile runs out, the discard pile is reshuffled into it.
    5. If the score has reached the jackpot threshold (21 by default), the round
       ends immediately as a jackpot -- the maximum payout.
    6. Deal the dealer a fresh card and repeat from step 3.
    7. The hand ends when the player runs out of cards, or when the player
       holds nothing that matches the dealer's card.

Because +2 and +4 put cards back into the hand, a round can run well past the
number of cards it was dealt. A Skip scores 2, but never past the jackpot
threshold: a Skip played at 20 scores 1 and finishes the round on exactly 21.

Run directly to simulate a batch of hands. Set ``hands`` and ``size`` at the
bottom of this file; ``hands = 1`` prints that hand turn by turn instead.

    $ python3 uno_sim.py

Use from code:

    from uno_sim import play_hand, simulate

    result = play_hand(hand_size=7, seed=42)     # one hand, full detail
    print(result.points, result.ended_reason)
    for turn in result.turns:
        print(turn.dealer_card, turn.played)

    batch = simulate(hands=100, seed=42)         # many hands, aggregated
    print(batch.average, batch.scores)


A NOTE ON CARD CHOICE
---------------------
When more than one card in hand matches, the rules do not say which to play,
but the choice changes the result. Since the dealer only ever shows a number
card, each kind of card in hand has a fixed chance of matching one:

    action card   25.0%   can only ever match by color
    number 0      28.9%   one 0 per color, so few rank-matches
    number 1-9    32.9%   two of each per color
    wild         100.0%   always matches

So spend the least flexible match and keep the most flexible cards back.
Available via ``strategy=``:

    FIRST           play the first match in deal order (naive baseline)
    SAVE_WILDS      play any non-wild match before spending a wild
    LEAST_FLEXIBLE  action, then 0, then 1-9, then wild (default)

Ties inside a tier break on deal order for now. The better tiebreak is to spend
from the player's largest color group -- same-color cards are redundant, and
color coverage is what survives turns -- but that is not implemented yet.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from uno_deck import (
    Card,
    Deck,
    Rank,
    DECK_SIZE,
    playable_cards,
    sort_hand,
)

__all__ = [
    "Turn",
    "HandResult",
    "SimResult",
    "play_hand",
    "simulate",
    "JACKPOT_SCORE",
    "PAYTABLE",
    "payout",
    "CARD_POINTS",
    "card_points",
    "matches_reverse",
    "FIRST",
    "SAVE_WILDS",
    "LEAST_FLEXIBLE",
    "STRATEGIES",
]


# --------------------------------------------------------------------------
# How the player picks among several matching cards
# --------------------------------------------------------------------------

FIRST = "first"
SAVE_WILDS = "save_wilds"
LEAST_FLEXIBLE = "least_flexible"


#: Flexibility tiers, lowest first. Lower is less useful to keep, so spend it
#: sooner. The numbers are the chance each kind of card matches a dealer number
#: card drawn from a fresh deck (the dealer only ever shows a number):
#:
#:     action card   25.0%   can only ever match by color
#:     number 0      28.9%   one 0 per color, so few rank-matches
#:     number 1-9    32.9%   two of each per color
#:     wild         100.0%   always matches
#:
#: 0s sit in their own tier because there is only one per color, giving them
#: strictly fewer rank-matches than a 1-9.
TIER_ACTION = 0
TIER_ZERO = 1
TIER_NUMBER = 2
TIER_WILD = 3


def _flexibility(card: Card) -> int:
    """Which flexibility tier a card belongs to. Lower means spend it sooner."""
    if card.is_wild:
        return TIER_WILD
    if card.is_number:
        return TIER_ZERO if card.rank is Rank.ZERO else TIER_NUMBER
    return TIER_ACTION


def _choose_first(playable: List[Card]) -> Card:
    """Naive baseline: whatever matched first in deal order."""
    return playable[0]


def _choose_save_wilds(playable: List[Card]) -> Card:
    """Spend any non-wild match first, keeping wilds as insurance.

    A wild matches anything, so spending one while an ordinary match is
    available throws away a guaranteed future point.
    """
    non_wild = [c for c in playable if not c.is_wild]
    return non_wild[0] if non_wild else playable[0]


def _choose_least_flexible(playable: List[Card]) -> Card:
    """Spend the least flexible match: action, then 0, then 1-9, then wild.

    Ties within a tier currently break on deal order, which is arbitrary. The
    intended tiebreak is to spend from the player's largest color group, since
    same-color cards are redundant and diversity is what survives turns; that
    needs the whole hand, not just the playable subset, so it is not wired in
    yet.
    """
    return min(playable, key=_flexibility)


STRATEGIES: Dict[str, Callable[[List[Card]], Card]] = {
    FIRST: _choose_first,
    SAVE_WILDS: _choose_save_wilds,
    LEAST_FLEXIBLE: _choose_least_flexible,
}


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

#: Points scored for playing each card. Anything absent scores DEFAULT_CARD_POINTS.
#: A Skip pays double, which makes it the only card whose value differs from the
#: number of cards it represents -- so score and cards-played are separate
#: quantities and must not be used interchangeably. A card's value is also capped
#: by the room left below the jackpot threshold, so a Skip does not always pay 2.
CARD_POINTS: Dict[Rank, int] = {
    Rank.SKIP: 2,
}
DEFAULT_CARD_POINTS = 1


def card_points(card: Card) -> int:
    """Points the player scores for playing ``card``."""
    return CARD_POINTS.get(card.rank, DEFAULT_CARD_POINTS)


def matches_reverse(card: Card, reverse: Card) -> bool:
    """Can ``card`` ride along with a played Reverse?

    Three ways to qualify: same color as the Reverse, any wild, or any Reverse
    regardless of color. That last clause is what makes long chains possible --
    every one of the deck's Reverses can follow every other, so a chain is
    limited only by how many Reverses the hand holds.
    """
    return (card.is_wild
            or card.rank is Rank.REVERSE
            or card.color is reverse.color)


#: How many cards the player draws after playing each card.
DRAW_COUNTS: Dict[Rank, int] = {
    Rank.DRAW_TWO: 2,
    Rank.WILD_DRAW_FOUR: 4,
}


def draw_count(card: Card) -> int:
    """Cards the player draws after playing ``card``. 0 for most cards."""
    return DRAW_COUNTS.get(card.rank, 0)


#: Score that ends the round as a jackpot, paying the maximum.
JACKPOT_SCORE = 25

#: Score -> payout, as a multiple of the bet. Any score not listed pays nothing,
#: so this table alone decides both the paytable and where the win line sits.
#: Edit these numbers to balance the game; the RTP printed on every run is
#: recalculated straight from this table and the simulated distribution.
#:
#: Keep it monotonic (a higher score should never pay less). Note that 21 is a
#: more common result than 19 or 20, because capping the score makes the top
#: bucket absorb every round that would otherwise have scored higher -- so the
#: jackpot multiplier does more damage to the RTP than its position suggests.
PAYTABLE: Dict[int, float] = {
     0: 0,
     1: 0,
     2: 0,
     3: 0,
     4: 0.5,
     5: 2,   # anchored
     6: 2.25,
     7: 2.5,
     8: 2.75,
     9: 3,
    10: 3.25,
    11: 3.5,
    12: 3.75,
    13: 4,
    14: 4.25,
    15: 4.5,
    16: 4.75,
    17: 5,
    18: 5.25,
    19: 5.5,
    20: 5.75,
    21: 6.0,
    22: 7.0,
    23: 8.0,
    24: 10.0,
    25: 12.0,   # balancing tier
}


def payout(score: int) -> float:
    """Payout for a score, as a multiple of the bet. Unlisted scores pay 0."""
    return PAYTABLE.get(score, 0.0)

#: Why a hand stopped.
OUT_OF_CARDS = "out_of_cards"      # played every card
NO_MATCH = "no_match"              # nothing in hand matched the dealer
JACKPOT = "jackpot"                # reached the jackpot threshold
DECK_EXHAUSTED = "deck_exhausted"  # no cards left to draw, even after reshuffle
TURN_LIMIT = "turn_limit"          # safety cap tripped; should never happen
END_REASONS = (OUT_OF_CARDS, NO_MATCH, JACKPOT, DECK_EXHAUSTED, TURN_LIMIT)


@dataclass(slots=True)
class Turn:
    """One dealer card and the player's response to it."""

    number: int
    dealer_card: Card
    hand_before: List[Card]
    playable: List[Card]
    #: The card matched against the dealer's card.
    played: Optional[Card]
    #: The hand once the plays and any draws have resolved. Stored rather than
    #: derived, because +2 and +4 add cards that subtraction would not show.
    hand_after: List[Card] = field(default_factory=list)
    #: Points for the matched card alone.
    main_points: int = 0
    #: Extra cards played on the back of a Reverse, in the order played. A
    #: Reverse rider chains into another, so this can hold several cards.
    riders: List[Card] = field(default_factory=list)
    #: Points for each rider, parallel to ``riders``.
    rider_points: List[int] = field(default_factory=list)
    #: Cards drawn as a result of playing a +2 or +4 (either card counts).
    drawn: List[Card] = field(default_factory=list)
    dealer_burns: List[Card] = field(default_factory=list)
    cards_remaining: int = 0

    @property
    def scored(self) -> bool:
        return self.played is not None

    @property
    def points_scored(self) -> int:
        """Everything this turn was worth, matched card plus any Reverse rider."""
        return self.main_points + sum(self.rider_points)

    @property
    def cards_played(self) -> List[Card]:
        """One card normally, two when a Reverse brought a rider along."""
        return ([] if self.played is None else [self.played]) + list(self.riders)

    @property
    def is_bonus_turn(self) -> bool:
        return bool(self.riders)

    @property
    def chain_length(self) -> int:
        """How many riders this turn pulled. 0 on an ordinary turn."""
        return len(self.riders)

    @property
    def draw_count(self) -> int:
        return len(self.drawn)

    @property
    def points_capped(self) -> bool:
        """True if the jackpot threshold cut either card's value short.

        Only reachable on a Skip played with just one point of room left.
        """
        if self.played is not None and self.main_points < card_points(self.played):
            return True
        return any(pts < card_points(c)
                   for c, pts in zip(self.riders, self.rider_points))

    @property
    def match_count(self) -> int:
        return len(self.playable)

    @property
    def burn_count(self) -> int:
        return len(self.dealer_burns)


@dataclass(slots=True)
class HandResult:
    """The full record of one hand, for later analysis."""

    starting_hand: List[Card]
    turns: List[Turn]
    points: int
    final_hand: List[Card]
    ended_reason: str
    strategy: str
    cards_remaining: int

    @property
    def cards_played(self) -> List[Card]:
        """Every card the player put down, riders included."""
        return [c for t in self.turns for c in t.cards_played]

    @property
    def cards_played_count(self) -> int:
        """Number of cards played. No longer equal to ``points``, since a Skip
        scores 2 for one card."""
        return len(self.cards_played)

    @property
    def skips_played(self) -> int:
        """Skip cards played. Not all of them necessarily paid 2 -- see
        ``points_capped``."""
        return sum(1 for c in self.cards_played if c.rank is Rank.SKIP)

    @property
    def reverses_played(self) -> int:
        return sum(1 for c in self.cards_played if c.rank is Rank.REVERSE)

    @property
    def bonus_plays(self) -> int:
        """Turns where a Reverse brought at least one extra card along."""
        return sum(1 for t in self.turns if t.is_bonus_turn)

    @property
    def total_riders(self) -> int:
        """Extra cards played on the back of Reverses this round."""
        return sum(t.chain_length for t in self.turns)

    @property
    def longest_chain(self) -> int:
        """Most riders pulled by any single turn this round."""
        return max((t.chain_length for t in self.turns), default=0)

    @property
    def points_capped(self) -> bool:
        """Did the jackpot threshold cut a Skip's value short this round?"""
        return any(t.points_capped for t in self.turns)

    @property
    def cleared_hand(self) -> bool:
        """Did the player play every card?"""
        return not self.final_hand

    @property
    def is_jackpot(self) -> bool:
        return self.ended_reason == JACKPOT

    @property
    def payout(self) -> float:
        """What this round returns, as a multiple of the bet."""
        return payout(self.points)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def total_burns(self) -> int:
        return sum(t.burn_count for t in self.turns)

    @property
    def total_drawn(self) -> int:
        """Cards added to the hand mid-play by +2 and +4."""
        return sum(t.draw_count for t in self.turns)

    @property
    def max_hand_size(self) -> int:
        """Largest the hand ever got, useful for spotting runaway hands."""
        sizes = [len(self.starting_hand)] + [len(t.hand_after) for t in self.turns]
        return max(sizes)

    def describe(self) -> str:
        start = " ".join(str(c) for c in sort_hand(self.starting_hand))
        lines = [
            f"Starting hand:  {start}   ({len(self.starting_hand)} cards)",
            f"Strategy:       {self.strategy}",
            "",
        ]
        for t in self.turns:
            burns = f" (burned {' '.join(str(c) for c in t.dealer_burns)})" if t.dealer_burns else ""
            plays = " ".join(str(c) for c in sort_hand(t.playable)) or "none"
            if t.scored:
                action = f"play {t.played}"
                if t.riders:
                    action += " + rider" + ("s" if t.chain_length > 1 else "")
                    action += " " + " ".join(str(c) for c in t.riders)
                if t.points_scored != DEFAULT_CARD_POINTS or t.points_capped:
                    action += f" [{t.points_scored} pts"
                    action += ", capped by the jackpot]" if t.points_capped else "]"
                if t.drawn:
                    action += f", draw {t.draw_count} ({' '.join(str(c) for c in t.drawn)})"
            else:
                action = "no match, hand ends"
            after = t.hand_after
            remaining = " ".join(str(c) for c in sort_hand(after)) or "none"
            lines.append(
                f"  Turn {t.number}: dealer {str(t.dealer_card):<6}{burns}\n"
                f"          playable: {plays}\n"
                f"          -> {action}\n"
                f"          hand now: {remaining}   ({len(after)} left)"
            )
        left = " ".join(str(c) for c in sort_hand(self.final_hand)) or "none"
        lines += [
            "",
            f"Points:         {self.points}"
            f"   ({self.cards_played_count} cards played, {self.skips_played} skips,"
            f" {self.total_riders} riders on {self.bonus_plays} reverses)",
            f"Cards drawn:    {self.total_drawn}   (hand peaked at {self.max_hand_size})",
            f"Ended because:  {self.ended_reason}",
            f"Cards in hand:  {len(self.final_hand)}   ({left})",
            f"Deck:           {self.cards_remaining} of {DECK_SIZE} cards remaining",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Dealer card
# --------------------------------------------------------------------------

def _draw_dealer_card(deck: Deck, numbers_only: bool) -> Tuple[Card, List[Card]]:
    """Draw the dealer's card, discarding non-numbers if ``numbers_only``.

    Returns ``(dealer_card, burned)``. Burned cards go to the discard pile, so
    total cards are conserved.

    Termination is guaranteed: only non-number cards are ever burned, so a
    number card confirmed to exist up front is still there to be found.
    """
    if not numbers_only:
        card = deck.draw()
        deck.discard(card)
        return card, []

    # Reshuffling means the discard pile is still reachable, so check both piles
    # before looping. Without this the loop below could spin forever recycling a
    # pile that contains no numbers at all.
    reachable = list(deck.draw_pile) + list(deck.discard_pile)
    if not any(c.is_number for c in reachable):
        raise RuntimeError("no number cards left for the dealer to show")

    burned: List[Card] = []
    while True:
        card = deck.draw()
        deck.discard(card)
        if card.is_number:
            return card, burned
        burned.append(card)


# --------------------------------------------------------------------------
# The hand
# --------------------------------------------------------------------------

def play_hand(hand_size: int = 7,
              strategy: str = LEAST_FLEXIBLE,
              deck: Optional[Deck] = None,
              seed: Optional[int] = None,
              rng: Optional[random.Random] = None,
              dealer_numbers_only: bool = True,
              jackpot_score: int = JACKPOT_SCORE,
              max_turns: int = 1000) -> HandResult:
    """Play one hand to completion and return the full record.

    Every card comes off a single deck: the player's hand first, then each
    dealer card, any burns, and any cards drawn from a +2 or +4. Cards the
    player plays move to the discard pile, so hand + draw pile + discard pile
    always totals the deck size. When the draw pile empties the discard pile is
    reshuffled back into it.

    Reaching ``jackpot_score`` ends the round immediately as a jackpot. A card
    never scores more than the room left below the threshold, so the score lands
    on it exactly and a jackpot always records ``jackpot_score``. Every turn still
    adds at least one point, so a round cannot exceed ``jackpot_score + 1`` turns.

    ``max_turns`` is a redundant backstop given that bound, kept only so a future
    rule change cannot reintroduce an unbounded loop unnoticed.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {tuple(STRATEGIES)}, got {strategy!r}")
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if jackpot_score < 1:
        raise ValueError("jackpot_score must be at least 1")
    choose = STRATEGIES[strategy]

    if deck is None:
        deck = Deck(seed=seed, rng=rng)

    # RULE 1: deal the hand. `hand` is mutated in place for the whole round, so
    # keep an independent snapshot of what was dealt.
    hand = deck.deal(hand_size)
    starting_hand = list(hand)

    turns: List[Turn] = []
    points = 0
    # If the while loop exits on its own the hand emptied, so that is the
    # default; every other ending overwrites this before breaking.
    ended_reason = OUT_OF_CARDS
    turn_number = 0

    while hand:
        # Unreachable while the jackpot caps the round at jackpot_score + 1
        # turns. Kept so a future rule cannot reintroduce a runaway loop
        # unnoticed.
        if turn_number >= max_turns:
            ended_reason = TURN_LIMIT
            break
        turn_number += 1

        # RULE 2: the dealer must show a number, so non-numbers get burned until
        # one turns up.
        try:
            dealer_card, burned = _draw_dealer_card(deck, dealer_numbers_only)
        except RuntimeError:
            # A hand grown large by +2s and +4s can hold every number card,
            # leaving the dealer nothing legal to show.
            ended_reason = DECK_EXHAUSTED
            break

        # RULE 3: what can the player legally play -- matching color, matching
        # number, or any wild.
        matches = playable_cards(hand, dealer_card)
        hand_before = list(hand)   # snapshot before this turn changes anything

        # A dead end ends the hand. Still recorded as a turn, so the log shows
        # which dealer card stopped the player and what they were left holding.
        if not matches:
            turns.append(Turn(
                number=turn_number,
                dealer_card=dealer_card,
                hand_before=hand_before,
                playable=[],
                played=None,
                hand_after=list(hand),
                dealer_burns=burned,
                cards_remaining=deck.remaining,
            ))
            ended_reason = NO_MATCH
            break

        # The strategy decides which of several matches to spend. This choice is
        # not fixed by the rules and measurably changes the outcome.
        chosen = choose(matches)
        hand.remove(chosen)          # removes one copy; duplicates compare equal
        deck.discard(chosen)         # discards can be reshuffled back in later
        # A card scores its face value, but never more than the room left below
        # the jackpot threshold: a Skip played at 20 pays 1, not 2, so the score
        # lands on 21 exactly rather than overshooting to 22. Room is always at
        # least 1 here, because the loop breaks as soon as the threshold is met,
        # so every turn still adds at least one point.
        main_points = min(card_points(chosen), jackpot_score - points)
        points += main_points

        # RULE 3b: a Reverse brings a rider with it -- one extra card matching
        # the Reverse by color, or any wild -- scoring for both. Taken whenever
        # one is available, since a free play is never worse than holding the
        # card. A rider that is itself a Reverse chains into another rider, each
        # matched against the card just played rather than the original.
        #
        # The chain terminates on three separate grounds: it stops on the first
        # non-Reverse rider, it removes a card from the hand every step, and it
        # stops the moment the jackpot threshold is reached. Since any Reverse
        # follows any other, the practical limit is how many Reverses are in
        # hand -- at most the 8 the deck holds.
        riders: List[Card] = []
        rider_points: List[int] = []
        top = chosen
        while top.rank is Rank.REVERSE and points < jackpot_score:
            candidates = [c for c in hand if matches_reverse(c, top)]
            if not candidates:
                break
            rider = choose(candidates)
            hand.remove(rider)
            deck.discard(rider)
            scored = min(card_points(rider), jackpot_score - points)
            points += scored
            riders.append(rider)
            rider_points.append(scored)
            top = rider          # the next rider matches the card just played

        # RULE 5: reaching the threshold ends the round at once. Cards played for
        # the winning points do not draw -- there is no next turn for those cards
        # to matter in.
        jackpot_hit = points >= jackpot_score

        # RULE 4: a +2 or +4 puts cards back into the hand, which is why a round
        # can outlast the cards it was dealt. Either card this turn can trigger
        # it. Deck.draw() reshuffles the discard pile in when the draw pile
        # empties.
        drawn: List[Card] = []
        exhausted = False
        if not jackpot_hit:
            to_draw = draw_count(chosen) + sum(draw_count(c) for c in riders)
            for _ in range(to_draw):
                try:
                    card = deck.draw()
                except RuntimeError:
                    # Nothing left anywhere to draw; take the points and stop.
                    exhausted = True
                    break
                drawn.append(card)
                hand.append(card)

        turns.append(Turn(
            number=turn_number,
            dealer_card=dealer_card,
            hand_before=hand_before,
            playable=matches,
            played=chosen,
            hand_after=list(hand),
            main_points=main_points,
            riders=riders,
            rider_points=rider_points,
            drawn=drawn,
            dealer_burns=burned,
            cards_remaining=deck.remaining,
        ))

        if jackpot_hit:
            ended_reason = JACKPOT
            break
        if exhausted:
            ended_reason = DECK_EXHAUSTED
            break

    return HandResult(
        starting_hand=starting_hand,
        turns=turns,
        points=points,
        final_hand=hand,
        ended_reason=ended_reason,
        strategy=strategy,
        cards_remaining=deck.remaining,
    )


# --------------------------------------------------------------------------
# Batches of hands
# --------------------------------------------------------------------------

@dataclass(slots=True)
class SimResult:
    """Aggregate over a batch of hands. Keeps every HandResult for drill-down."""

    results: List[HandResult]
    strategy: str
    hand_size: int

    @property
    def hands(self) -> int:
        return len(self.results)

    @property
    def scores(self) -> List[int]:
        return [r.points for r in self.results]

    @property
    def total_points(self) -> int:
        return sum(self.scores)

    @property
    def average(self) -> float:
        return statistics.fmean(self.scores) if self.results else 0.0

    @property
    def stdev(self) -> float:
        """Sample standard deviation of per-hand score."""
        return statistics.stdev(self.scores) if len(self.results) > 1 else 0.0

    @property
    def standard_error(self) -> float:
        """Uncertainty on the average. Shrinks with sqrt(hands).

        Reported because a small batch is noisy: 100 hands pins the average only
        to about +/- 0.4, which is wider than the gap between strategies.
        """
        return self.stdev / (self.hands ** 0.5) if self.hands else 0.0

    @property
    def distribution(self) -> Counter:
        """How many hands finished on each score."""
        return Counter(self.scores)

    @property
    def cleared(self) -> int:
        """Hands where the player played every card."""
        return sum(1 for r in self.results if r.cleared_hand)

    @property
    def clear_rate(self) -> float:
        return self.cleared / self.hands if self.hands else 0.0

    @property
    def payouts(self) -> List[float]:
        return [payout(s) for s in self.scores]

    @property
    def total_payout(self) -> float:
        return sum(self.payouts)

    @property
    def rtp(self) -> float:
        """Return to player: average payout per hand, with a bet of 1 per hand."""
        return self.total_payout / self.hands if self.hands else 0.0

    @property
    def house_edge(self) -> float:
        return 1.0 - self.rtp

    @property
    def rtp_standard_error(self) -> float:
        """Uncertainty on the RTP estimate.

        Large prizes at low frequency make payout far more volatile than score,
        so this is wide unless the batch is big.
        """
        p = self.payouts
        if len(p) < 2:
            return 0.0
        return statistics.stdev(p) / (self.hands ** 0.5)

    @property
    def hit_rate(self) -> float:
        """Fraction of hands that pay anything at all."""
        return sum(1 for x in self.payouts if x > 0) / self.hands if self.hands else 0.0

    @property
    def capped(self) -> int:
        """Rounds where a closing Skip was cut short by the jackpot threshold."""
        return sum(1 for r in self.results if r.points_capped)

    @property
    def jackpots(self) -> int:
        """Hands that hit the jackpot threshold.

        Rare enough (well under 1% at a 7-card deal) that a batch needs to be in
        the tens of thousands before this rate means anything.
        """
        return sum(1 for r in self.results if r.is_jackpot)

    @property
    def jackpot_rate(self) -> float:
        return self.jackpots / self.hands if self.hands else 0.0

    def describe(self, histogram: bool = True) -> str:
        lines = [
            f"{self.hands} hands | {self.hand_size} cards | strategy={self.strategy}",
            "",
            "Scores per hand:",
        ]
        row = ""
        for i, s in enumerate(self.scores):
            row += f"{s:>3}"
            if (i + 1) % 25 == 0:
                lines.append(f"  {row}")
                row = ""
        if row:
            lines.append(f"  {row}")

        lines += [
            "",
            f"Average score:   {self.average:.2f} points per hand"
            f"   (+/- {self.standard_error:.2f} standard error)",
            f"Total points:    {self.total_points}",
            f"Score range:     {min(self.scores)} to {max(self.scores)}",
            f"Cleared hand:    {self.cleared} of {self.hands}  ({self.clear_rate:.1%})",
            f"Jackpots:        {self.jackpots} of {self.hands}  ({self.jackpot_rate:.2%})"
            + (f", {self.capped} closed on a capped Skip" if self.capped else ""),
        ]

        reasons = Counter(r.ended_reason for r in self.results)
        lines.append("Ended because:   " + ", ".join(
            f"{k} {v}" for k, v in reasons.most_common()))
        drawn = sum(r.total_drawn for r in self.results)
        lines.append(f"Cards drawn:     {drawn} total"
                     f"   ({drawn / self.hands:.2f} per hand)")

        if histogram:
            dist = self.distribution
            widest = max(dist.values())
            lines += ["", "Score distribution:"]
            # range must follow the observed scores: +2 and +4 draws mean a hand
            # can score well above the number of cards it started with
            for s in range(max(self.scores) + 1):
                count = dist.get(s, 0)
                bar = "#" * round(count / widest * 40) if widest else ""
                pay = payout(s)
                tag = f"{pay:>6.1f}x" if pay else "      -"
                lines.append(f"  {s:>2} |{tag} | {bar:<32} {count:>5}  ({count / self.hands:5.1%})")

        lines += self._payout_lines()
        return "\n".join(lines)

    def _payout_lines(self) -> List[str]:
        """The RTP breakdown: which scores actually drive the return."""
        dist = self.distribution
        lines = ["", "Payout analysis (bet = 1 unit per hand):",
                 "  score      freq     pays   contributes"]
        # Only paying scores matter to the RTP; the rest contribute exactly zero.
        for s in sorted(k for k in PAYTABLE if dist.get(k)):
            freq = dist[s] / self.hands
            lines.append(f"  {s:>5}  {freq:8.4%}  {PAYTABLE[s]:6.1f}x"
                         f"   {freq * PAYTABLE[s]:8.4f}")
        unpaid = sorted(k for k in PAYTABLE if not dist.get(k))
        if unpaid:
            lines.append(f"  (no hands scored {', '.join(map(str, unpaid))}"
                         f" -- those prizes went unpaid in this batch)")
        lines += [
            "  " + "-" * 38,
            f"  hit rate    {self.hit_rate:8.3%}",
            f"  RTP         {self.rtp:8.3%}   (+/- {self.rtp_standard_error:.3%})",
            f"  house edge  {self.house_edge:8.3%}",
        ]
        return lines


def simulate(hands: int = 100,
             hand_size: int = 7,
             strategy: str = LEAST_FLEXIBLE,
             seed: Optional[int] = None,
             rng: Optional[random.Random] = None,
             dealer_numbers_only: bool = True,
             jackpot_score: int = JACKPOT_SCORE) -> SimResult:
    """Play ``hands`` independent hands and collect the results.

    Each hand gets its own freshly shuffled full deck, but all hands share one
    random stream, so a single ``seed`` reproduces the whole batch.
    """
    if hands < 1:
        raise ValueError("hands must be at least 1")
    # One shared generator across all hands, so a single seed reproduces the
    # whole batch rather than just one hand.
    if rng is None:
        rng = random.Random(seed)

    results = [
        play_hand(hand_size=hand_size, strategy=strategy, rng=rng,
                  dealer_numbers_only=dealer_numbers_only,
                  jackpot_score=jackpot_score)
        for _ in range(hands)
    ]
    return SimResult(results=results, strategy=strategy, hand_size=hand_size)


if __name__ == "__main__":
    hands = 100000
    size = 5

    if hands == 1:
        # a single hand is worth seeing turn by turn
        print(play_hand(hand_size=size).describe())
    else:
        print(simulate(hands=hands, hand_size=size).describe())