"""
uno_deck.py
===========

A 112-card Uno deck model built for fast hand simulation.

Deck composition lives in one place: the ``DECK_COMPOSITION`` table. Edit that
table to change the deck; ``FULL_DECK``, ``DECK_SIZE``, ``Deck``, ``sample_hand``
and ``verify_deck`` all follow automatically. Defaults to the Mattel 112-card
deck (standard 108 + 4 blanks).

Run directly to deal a hand:

    $ python3 uno_deck.py         # a random 7-card hand
    $ python3 uno_deck.py 10      # a random 10-card hand

Quick start
-----------
    from uno_deck import Deck, sample_hand, color_counts

    hand = sample_hand(7)               # fast: one independent random hand
    print(hand)                         # [R5, G+2, +4, ...]
    print(color_counts(hand))

    deck = Deck(seed=42)                # stateful: full game simulation
    hands = deck.deal_hands(4, 7)
    top = deck.flip_start_card()

Trying a different deck without editing the module:

    from uno_deck import DECK_COMPOSITION, Rank, build_deck

    comp = {**DECK_COMPOSITION, Rank.BLANK: 0, Rank.ZERO: 0}
    variant = build_deck(comp)          # 104 cards
    hand = sample_hand(7, deck=variant)
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Color",
    "Rank",
    "Card",
    "Deck",
    "DECK_COMPOSITION",
    "DECK_SIZE",
    "FULL_DECK",
    "build_deck",
    "verify_deck",
    "sample_hand",
    "sample_hands",
    "by_color",
    "color_counts",
    "rank_counts",
    "is_playable",
    "playable_cards",
    "sort_hand",
    "parse",
]


# --------------------------------------------------------------------------
# Colors
# --------------------------------------------------------------------------

class Color(Enum):
    RED = "Red"
    YELLOW = "Yellow"
    GREEN = "Green"
    BLUE = "Blue"
    WILD = "Wild"  # sentinel for colorless cards; not a real playable color

    @property
    def short(self) -> str:
        return {"Red": "R", "Yellow": "Y", "Green": "G", "Blue": "B", "Wild": "W"}[self.value]

    def __repr__(self) -> str:  # keeps debug output tight
        return self.short


#: The four real colors, in canonical order. Excludes the WILD sentinel.
COLORS: Tuple[Color, ...] = (Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE)


# --------------------------------------------------------------------------
# Ranks
# --------------------------------------------------------------------------

class Rank(Enum):
    ZERO = "0"
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    SKIP = "Skip"
    REVERSE = "Reverse"
    DRAW_TWO = "Draw Two"
    WILD = "Wild"
    WILD_DRAW_FOUR = "Wild Draw Four"
    BLANK = "Wild Customizable"

    # -- classification ----------------------------------------------------

    @property
    def is_number(self) -> bool:
        return self in NUMBER_RANKS

    @property
    def is_action(self) -> bool:
        """Colored action card: Skip, Reverse, Draw Two."""
        return self in ACTION_RANKS

    @property
    def is_wild(self) -> bool:
        """Colorless card: Wild, Wild Draw Four, or blank."""
        return self in WILD_RANKS

    @property
    def number(self) -> Optional[int]:
        """Face value 0-9, or None for non-number cards."""
        return int(self.value) if self.is_number else None

    @property
    def short(self) -> str:
        return RANK_SHORT[self]

    def __repr__(self) -> str:
        return self.short


NUMBER_RANKS: Tuple[Rank, ...] = (
    Rank.ZERO, Rank.ONE, Rank.TWO, Rank.THREE, Rank.FOUR,
    Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE,
)
ACTION_RANKS: Tuple[Rank, ...] = (Rank.SKIP, Rank.REVERSE, Rank.DRAW_TWO)
WILD_RANKS: Tuple[Rank, ...] = (Rank.WILD, Rank.WILD_DRAW_FOUR, Rank.BLANK)

RANK_SHORT: Dict[Rank, str] = {
    **{r: r.value for r in NUMBER_RANKS},
    Rank.SKIP: "S",
    Rank.REVERSE: "REV",
    Rank.DRAW_TWO: "+2",
    Rank.WILD: "WILD",
    Rank.WILD_DRAW_FOUR: "+4",
    Rank.BLANK: "BLANK",
}

# Sort ordering: color groups first (in canonical order), wilds last.
_COLOR_ORDER: Dict[Color, int] = {c: i for i, c in enumerate(COLORS)}
_COLOR_ORDER[Color.WILD] = len(COLORS)
_RANK_ORDER: Dict[Rank, int] = {r: i for i, r in enumerate(Rank)}


# --------------------------------------------------------------------------
# Card
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True, order=False)
class Card:
    """An immutable, hashable Uno card. Safe to use as a dict key or in a set.

    Note that duplicate physical cards are equal (there are two Red 7s and they
    compare equal), so use lists/Counters rather than sets when multiplicity
    matters.
    """

    color: Color
    rank: Rank

    def __post_init__(self) -> None:
        # Rejecting impossible cards here means a malformed card can never reach
        # a deck, a hand, or the matching logic.
        if self.rank.is_wild and self.color is not Color.WILD:
            raise ValueError(f"{self.rank.value} must be colorless")
        if not self.rank.is_wild and self.color is Color.WILD:
            raise ValueError(f"{self.rank.value} requires a real color")

    # -- convenience passthroughs -----------------------------------------

    @property
    def is_wild(self) -> bool:
        return self.rank.is_wild

    @property
    def is_number(self) -> bool:
        return self.rank.is_number

    @property
    def is_action(self) -> bool:
        return self.rank.is_action

    @property
    def number(self) -> Optional[int]:
        return self.rank.number

    @property
    def code(self) -> str:
        """Compact string: 'R5', 'G+2', 'BREV', 'WILD', '+4', 'BLANK'."""
        if self.is_wild:
            return self.rank.short
        return f"{self.color.short}{self.rank.short}"

    @property
    def sort_key(self) -> Tuple[int, int]:
        return (_COLOR_ORDER[self.color], _RANK_ORDER[self.rank])

    def __str__(self) -> str:
        return self.code

    def __repr__(self) -> str:
        return self.code


#: How many copies of each rank the deck holds. THIS IS THE ONLY PLACE DECK
#: COMPOSITION IS DEFINED -- edit here and everything downstream follows.
#:
#: For colored ranks the number is copies *per color* (so 2 means eight cards,
#: two in each of the four colors). For wild ranks it is the total number of
#: copies, since they are colorless.
#:
#: Set a rank to 0 (or delete its line) to remove it from the deck entirely.
#: The Rank enum member still exists, so parse("BLANK") and .is_wild keep
#: working -- the card just never gets dealt.
#:
#:     Rank.BLANK: 0,   -> drops the 4 blank wilds      (108-card deck)
#:     Rank.ZERO:  0,   -> drops the four 0s            (108-card deck)
#:                         both together                (104-card deck)
DECK_COMPOSITION: Dict[Rank, int] = {
    Rank.ZERO: 1,
    **{r: 2 for r in NUMBER_RANKS[1:]},
    **{r: 2 for r in ACTION_RANKS},
    Rank.WILD: 4,
    Rank.WILD_DRAW_FOUR: 4,
    Rank.BLANK: 0,
}


def _expected_size(composition: Dict[Rank, int]) -> int:
    return sum(n if r.is_wild else n * len(COLORS) for r, n in composition.items())


def build_deck(composition: Optional[Dict[Rank, int]] = None) -> Tuple[Card, ...]:
    """Build an unshuffled deck from a composition table.

    Defaults to ``DECK_COMPOSITION``. Pass your own table to try a variant
    without editing the module -- handy for comparing deck configurations
    side by side.
    """
    comp = DECK_COMPOSITION if composition is None else composition
    cards: List[Card] = []
    for rank, count in comp.items():
        if count < 0:
            raise ValueError(f"negative count for {rank.value}")
        if rank.is_wild:
            # Colorless, so the count is the total number of copies.
            cards += [Card(Color.WILD, rank)] * count
        else:
            # Colored, so the count is per color and multiplies by four.
            for color in COLORS:
                cards += [Card(color, rank)] * count
    return tuple(cards)


#: Canonical, unshuffled deck built from DECK_COMPOSITION. Treat as read-only.
FULL_DECK: Tuple[Card, ...] = build_deck()

#: Number of cards in FULL_DECK (112 with the default composition).
DECK_SIZE: int = len(FULL_DECK)

assert DECK_SIZE == _expected_size(DECK_COMPOSITION), "deck build disagrees with table"

_FULL_DECK_LIST: List[Card] = list(FULL_DECK)  # random.sample needs a sequence


# --------------------------------------------------------------------------
# Deck: stateful, for simulating a whole game
# --------------------------------------------------------------------------

class Deck:
    """A shuffled, stateful deck with a draw pile and a discard pile.

    Use this when the sequence of draws matters (dealing to several players,
    drawing mid-hand, reshuffling). If you only need independent random hands,
    ``sample_hand`` is faster.
    """

    def __init__(self, seed: Optional[int] = None, rng: Optional[random.Random] = None,
                 shuffle: bool = True, cards: Optional[Sequence[Card]] = None) -> None:
        self.rng = rng if rng is not None else random.Random(seed)
        self._source: Tuple[Card, ...] = FULL_DECK if cards is None else tuple(cards)
        self.draw_pile: List[Card] = list(self._source)
        self.discard_pile: List[Card] = []
        if shuffle:
            self.shuffle()

    # -- state -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.draw_pile)

    @property
    def remaining(self) -> int:
        return len(self.draw_pile)

    @property
    def top_card(self) -> Optional[Card]:
        return self.discard_pile[-1] if self.discard_pile else None

    def shuffle(self) -> "Deck":
        self.rng.shuffle(self.draw_pile)
        return self

    def reset(self, shuffle: bool = True) -> "Deck":
        """Return every card to the draw pile."""
        self.draw_pile = list(self._source)
        self.discard_pile = []
        if shuffle:
            self.shuffle()
        return self

    # -- dealing -----------------------------------------------------------

    def deal(self, n: int, sort: bool = False) -> List[Card]:
        """Deal ``n`` cards off the top as a hand."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n > len(self.draw_pile):
            raise ValueError(f"only {len(self.draw_pile)} cards left, asked for {n}")
        hand = [self.draw_pile.pop() for _ in range(n)]
        return sort_hand(hand) if sort else hand

    def deal_hands(self, num_hands: int, cards_each: int,
                   sort: bool = False) -> List[List[Card]]:
        """Deal several hands round-robin, the way a dealer would."""
        total = num_hands * cards_each
        if total > len(self.draw_pile):
            raise ValueError(f"only {len(self.draw_pile)} cards left, asked for {total}")
        hands: List[List[Card]] = [[] for _ in range(num_hands)]
        for _ in range(cards_each):
            for hand in hands:
                hand.append(self.draw_pile.pop())
        return [sort_hand(h) for h in hands] if sort else hands

    def draw(self) -> Card:
        """Draw a single card, reshuffling the discards if the pile is empty."""
        if not self.draw_pile:
            self._recycle_discards()
        return self.draw_pile.pop()

    def flip_start_card(self, avoid_wild: bool = True) -> Card:
        """Turn the starting card face up onto the discard pile.

        With ``avoid_wild``, wilds are buried and the next card is tried, which
        mirrors the standard rule.
        """
        while True:
            card = self.draw()
            if avoid_wild and card.is_wild:
                self.draw_pile.insert(0, card)
                continue
            self.discard_pile.append(card)
            return card

    def discard(self, card: Card) -> None:
        self.discard_pile.append(card)

    def _recycle_discards(self) -> None:
        # The face-up top card stays put; everything beneath it is reshuffled
        # into a new draw pile.
        if len(self.discard_pile) <= 1:
            raise RuntimeError("no cards left to draw and nothing to recycle")
        top = self.discard_pile.pop()
        self.draw_pile = self.discard_pile
        self.discard_pile = [top]
        self.shuffle()

    def __repr__(self) -> str:
        return f"Deck(remaining={self.remaining}, top={self.top_card})"


# --------------------------------------------------------------------------
# Fast independent sampling, for Monte Carlo runs
# --------------------------------------------------------------------------

def sample_hand(n: int = 7, rng: Optional[random.Random] = None,
                deck: Optional[Sequence[Card]] = None) -> List[Card]:
    """Draw ``n`` cards from a fresh full deck without shuffling the whole thing.

    Much faster than building a Deck per trial, so use this for large Monte
    Carlo loops over a single hand. Pass ``deck`` to sample from a variant built
    by ``build_deck``.
    """
    # Samples from an untouched template rather than consuming a pile, which is
    # why this is fast but has no notion of cards remaining.
    pool = _FULL_DECK_LIST if deck is None else list(deck)
    if not 0 <= n <= len(pool):
        raise ValueError(f"n must be between 0 and {len(pool)}")
    picker = rng.sample if rng is not None else random.sample
    return picker(pool, n)


def sample_hands(num_hands: int, cards_each: int = 7,
                 rng: Optional[random.Random] = None,
                 deck: Optional[Sequence[Card]] = None) -> List[List[Card]]:
    """Deal several mutually exclusive hands from one fresh deck."""
    total = num_hands * cards_each
    cards = sample_hand(total, rng, deck)
    return [cards[i * cards_each:(i + 1) * cards_each] for i in range(num_hands)]


# --------------------------------------------------------------------------
# Hand analysis helpers
# --------------------------------------------------------------------------

def sort_hand(cards: Iterable[Card]) -> List[Card]:
    """Sort into color groups, ascending rank, wilds last."""
    return sorted(cards, key=lambda c: c.sort_key)


def by_color(cards: Iterable[Card]) -> Dict[Color, List[Card]]:
    """Group a hand by color. Wilds land under ``Color.WILD``."""
    groups: Dict[Color, List[Card]] = {c: [] for c in COLORS}
    groups[Color.WILD] = []
    for card in cards:
        groups[card.color].append(card)
    return groups


def color_counts(cards: Iterable[Card], count_wilds: bool = False) -> Counter:
    """Count cards per color. Wilds excluded unless ``count_wilds``."""
    return Counter(c.color for c in cards if count_wilds or not c.is_wild)


def rank_counts(cards: Iterable[Card]) -> Counter:
    return Counter(c.rank for c in cards)


def is_playable(card: Card, top: Card, active_color: Optional[Color] = None) -> bool:
    """Standard match test: same color, same rank, or a wild.

    ``active_color`` is the color declared by whoever played a wild. Rank
    matching covers symbols too, so Skip plays on Skip across colors. Wild Draw
    Four's "only if you can't match" restriction is not enforced here, since
    that is a table rule rather than a property of the card.
    """
    if card.is_wild:
        return True
    effective = active_color if active_color is not None else top.color
    if effective is Color.WILD:
        return True  # wild on top with no color declared yet
    # Rank comparison covers symbols as well as digits, so Skip plays on Skip.
    return card.color is effective or card.rank is top.rank


def playable_cards(cards: Iterable[Card], top: Card,
                   active_color: Optional[Color] = None) -> List[Card]:
    return [c for c in cards if is_playable(c, top, active_color)]


def parse(code: str) -> Card:
    """Build a Card from a compact code: 'R5', 'g+2', 'BREV', 'wild', '+4'."""
    s = code.strip().upper()
    for rank in WILD_RANKS:
        if s in (rank.short, rank.value.upper()):
            return Card(Color.WILD, rank)
    if not s:
        raise ValueError("empty card code")
    color_map = {c.short: c for c in COLORS}
    if s[0] not in color_map:
        raise ValueError(f"unknown color in {code!r}")
    color, rest = color_map[s[0]], s[1:]
    for rank in NUMBER_RANKS + ACTION_RANKS:
        if rest in (rank.short.upper(), rank.value.upper()):
            return Card(color, rank)
    raise ValueError(f"unknown rank in {code!r}")


# --------------------------------------------------------------------------
# Self-check (call manually; not run on import or on __main__)
# --------------------------------------------------------------------------

def verify_deck(deck: Optional[Sequence[Card]] = None,
                composition: Optional[Dict[Rank, int]] = None) -> None:
    """Assert a deck matches its composition table.

    Checks every rank appears the expected number of times, that the total is
    right, and that no unexpected cards snuck in. Works for any composition,
    so it keeps passing after you edit DECK_COMPOSITION.
    """
    cards = FULL_DECK if deck is None else deck
    comp = DECK_COMPOSITION if composition is None else composition
    counts = Counter(cards)

    for rank, n in comp.items():
        if rank.is_wild:
            got = counts[Card(Color.WILD, rank)]
            assert got == n, f"{rank.value}: expected {n}, found {got}"
        else:
            for color in COLORS:
                got = counts[Card(color, rank)]
                assert got == n, f"{color.value} {rank.value}: expected {n}, found {got}"

    assert len(cards) == _expected_size(comp), "total card count disagrees with table"
    stray = {c.rank for c in cards} - {r for r, n in comp.items() if n > 0}
    assert not stray, f"unexpected ranks in deck: {stray}"


# --------------------------------------------------------------------------
# Entry point: deal one hand
# --------------------------------------------------------------------------

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    deck = Deck()                 # stateful, so it tracks what is left
    hand = deck.deal(n)
    print(hand)
    print(f"{deck.remaining} of {DECK_SIZE} cards remaining")