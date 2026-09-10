/*
 * engine.js -- the rules of UNO Streaks, as a reducer.
 *
 * NO DOM IN THIS FILE. The UI drives a round with `nextState(state, action)`
 * and a headless driver can run one to completion the same way, so the rules
 * exist once and the version players use is the version that gets tested.
 *
 * Every number this file needs -- deck contents, card points, draw counts,
 * jackpot, paytable -- arrives in `data`, which is generated out of the Python
 * model by export_web_data.py. None of the maths is written down here.
 *
 *     const engine = UnoEngine.create(UNO_DATA);
 *     let state = engine.newRound(seed);          // dealt, dealer card up
 *     state = engine.nextState(state, {type: 'play', index: 2});
 *     // ... until state.phase === 'roundOver'
 */
(function (root, factory) {
  const api = factory();
  root.UnoEngine = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  // -- phases ---------------------------------------------------------------
  const AWAITING_PLAY = 'awaitingPlay';   // pick a card matching the dealer
  const AWAITING_RIDER = 'awaitingRider'; // pick a card to ride a Reverse
  const ROUND_OVER = 'roundOver';

  // -- end reasons, mirroring uno_sim.py ------------------------------------
  const OUT_OF_CARDS = 'out_of_cards';
  const NO_MATCH = 'no_match';
  const JACKPOT = 'jackpot';
  const DECK_EXHAUSTED = 'deck_exhausted';

  const END_TEXT = {
    out_of_cards: 'Hand empty -- every card played.',
    no_match: 'Nothing in hand matched the dealer. Round over.',
    jackpot: 'JACKPOT! The score reached the top of the table.',
    deck_exhausted: 'The deck ran dry. Round over.',
  };

  // Thrown internally when no card can be produced; never escapes nextState.
  function Exhausted() {}

  // -- seeded RNG -----------------------------------------------------------
  // Lives inside the state, so a round replays exactly from its seed alone.
  function nextRandom(state) {
    let t = (state.rng = (state.rng + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  function shuffle(cards, state) {
    for (let i = cards.length - 1; i > 0; i--) {
      const j = Math.floor(nextRandom(state) * (i + 1));
      const tmp = cards[i];
      cards[i] = cards[j];
      cards[j] = tmp;
    }
  }

  function create(data) {
    const JACKPOT_SCORE = data.jackpotScore;
    const RANKS = data.ranks;

    // -- card facts, every one looked up from the generated table -----------
    const isWild = (card) => RANKS[card.r].kind === 'wild';
    const isNumber = (card) => RANKS[card.r].kind === 'number';
    const points = (card) => RANKS[card.r].points;
    const draws = (card) => RANKS[card.r].draws;
    const label = (card) => (isWild(card) ? card.r : card.c + card.r);

    /* The dealer always shows a number, so an action card in hand can only
     * ever match by colour -- never by rank. */
    function playsOnDealer(card, dealer) {
      return isWild(card) || card.c === dealer.c || card.r === dealer.r;
    }

    /* Riders use their own, looser rule: same colour as the card just played,
     * any wild, or any Reverse of any colour. */
    function ridesOn(card, top) {
      return isWild(card) || card.r === 'REV' || card.c === top.c;
    }

    // -- deck ---------------------------------------------------------------
    function drawCard(s) {
      if (s.drawPile.length === 0) {
        // The face-up card stays out; everything under it is reshuffled in.
        if (s.discardPile.length <= 1) throw new Exhausted();
        const top = s.discardPile.pop();
        s.drawPile = s.discardPile;
        s.discardPile = [top];
        shuffle(s.drawPile, s);
      }
      return s.drawPile.pop();
    }

    /* The dealer draws until a number appears. Non-numbers are burned to the
     * discard pile, not returned to the draw pile. */
    function drawDealerCard(s) {
      const reachable = s.drawPile.concat(s.discardPile);
      if (!reachable.some(isNumber)) throw new Exhausted();
      const burned = [];
      for (;;) {
        const card = drawCard(s);
        s.discardPile.push(card);
        if (isNumber(card)) return { card: card, burned: burned };
        burned.push(card);
      }
    }

    // -- round flow ---------------------------------------------------------
    function endRound(s, reason) {
      s.phase = ROUND_OVER;
      s.endReason = reason;
      s.legal = [];
      s.riderTop = null;
      s.payout = payoutFor(s.score);
      s.message = END_TEXT[reason];
      return s;
    }

    /* Deal the next dealer card and work out what the player may play. */
    function beginTurn(s) {
      s.playedThisTurn = [];
      s.pendingDraw = 0;
      s.riderTop = null;
      s.drawnLastTurn = s.drawnLastTurn || [];
      if (s.hand.length === 0) return endRound(s, OUT_OF_CARDS);

      let dealt;
      try {
        dealt = drawDealerCard(s);
      } catch (e) {
        if (e instanceof Exhausted) return endRound(s, DECK_EXHAUSTED);
        throw e;
      }
      s.turn += 1;
      s.dealerCard = dealt.card;
      s.burned = dealt.burned;

      s.legal = indicesWhere(s.hand, (c) => playsOnDealer(c, dealt.card));
      if (s.legal.length === 0) return endRound(s, NO_MATCH);

      s.phase = AWAITING_PLAY;
      s.message = 'Dealer shows ' + label(dealt.card) + '. Play a match.';
      return s;
    }

    /* The rider chain is over: take the turn's draws, then deal again. */
    function finishTurn(s) {
      let exhausted = false;
      const drawn = [];
      for (let i = 0; i < s.pendingDraw; i++) {
        try {
          const card = drawCard(s);
          s.hand.push(card);
          drawn.push(card);
        } catch (e) {
          if (!(e instanceof Exhausted)) throw e;
          exhausted = true;
          break;
        }
      }
      s.pendingDraw = 0;
      if (exhausted) return endRound(s, DECK_EXHAUSTED);
      const next = beginTurn(s);
      next.drawnLastTurn = drawn;
      return next;
    }

    /* Score one card and route to whatever comes next. */
    function playCard(s, index) {
      const card = s.hand.splice(index, 1)[0];
      s.discardPile.push(card);

      // A card never scores past the jackpot: a Skip one below the threshold
      // pays 1, not 2, so the score lands on it exactly.
      const scored = Math.min(points(card), JACKPOT_SCORE - s.score);
      s.score += scored;
      s.playedThisTurn.push({ card: card, points: scored });
      s.log.push('T' + s.turn + ': ' + label(card) + ' +' + scored +
                 ' (score ' + s.score + ')');

      // Draws are collected and taken only once the whole turn is done, so a
      // freshly drawn card can never be used as a rider.
      s.pendingDraw += draws(card);

      // The card that reaches the jackpot does not draw -- there is no next
      // turn for those cards to matter in.
      if (s.score >= JACKPOT_SCORE) {
        s.pendingDraw = 0;
        return endRound(s, JACKPOT);
      }

      // A Reverse brings a rider with it, and a rider that is itself a Reverse
      // grants another -- each matched against the card just played.
      if (card.r === 'REV') {
        const riders = indicesWhere(s.hand, (c) => ridesOn(c, card));
        if (riders.length > 0) {
          s.phase = AWAITING_RIDER;
          s.riderTop = card;
          s.legal = riders;
          s.message = 'Reverse! Ride it with a card matching ' + label(card) + '.';
          return s;
        }
      }
      return finishTurn(s);
    }

    // -- the reducer --------------------------------------------------------
    function nextState(state, action) {
      const s = clone(state);
      switch (action.type) {
        case 'play': {
          if (s.phase !== AWAITING_PLAY && s.phase !== AWAITING_RIDER) {
            throw new Error('cannot play in phase ' + s.phase);
          }
          if (s.legal.indexOf(action.index) === -1) {
            throw new Error('card ' + action.index + ' is not a legal play');
          }
          return playCard(s, action.index);
        }
        default:
          throw new Error('unknown action ' + action.type);
      }
    }

    // -- a fresh round ------------------------------------------------------
    function newRound(seed, handSize) {
      const size = handSize === undefined ? data.handSize : handSize;
      const s = {
        rng: (seed === undefined ? Math.random() * 4294967296 : seed) >>> 0,
        seed: seed,
        drawPile: data.deck.map((pair) => ({ c: pair[0], r: pair[1] })),
        discardPile: [],
        hand: [],
        startingHand: [],
        dealerCard: null,
        burned: [],
        legal: [],
        riderTop: null,
        pendingDraw: 0,
        playedThisTurn: [],
        drawnLastTurn: [],
        score: 0,
        turn: 0,
        phase: AWAITING_PLAY,
        endReason: null,
        payout: null,
        message: '',
        log: [],
      };
      shuffle(s.drawPile, s);
      for (let i = 0; i < size; i++) s.hand.push(s.drawPile.pop());
      s.startingHand = s.hand.slice();
      return beginTurn(s);
    }

    // -- the CHART strategy, ported for the hint button ---------------------
    // Play the highest available: +2 -> Reverse -> +4 -> Skip -> 0 -> 1-9 ->
    // Wild. Ties go to the card from the largest colour group. One exception:
    // a Skip jumps ahead of a +4 when the hand holds 3 or more of the Skip's
    // colour (the Skip itself counts towards those three).
    const CHART_ORDER = { '+2': 0, REV: 1, '+4': 2, S: 3, '0': 4, WILD: 6 };

    function colourCount(hand, colour) {
      let n = 0;
      for (const card of hand) if (card.c === colour) n++;
      return n;
    }

    function chartRank(card, hand) {
      if (card.r === 'S') return colourCount(hand, card.c) >= 3 ? 1.5 : 3;
      const fixed = CHART_ORDER[card.r];
      return fixed === undefined ? 5 : fixed; // 1-9 all share one tier
    }

    /* Which of `state.legal` the chart would play, as a hand index. */
    function chartPick(state) {
      let best = null;
      for (const index of state.legal) {
        const card = state.hand[index];
        const rank = chartRank(card, state.hand);
        const group = -colourCount(state.hand, card.c);
        if (best === null || rank < best.rank ||
            (rank === best.rank && group < best.group)) {
          best = { index: index, rank: rank, group: group };
        }
      }
      return best === null ? null : best.index;
    }

    // -- payouts ------------------------------------------------------------
    function payoutFor(score) {
      const pay = data.paytable[String(score)];
      return pay === undefined ? 0 : pay;
    }

    return {
      newRound: newRound,
      nextState: nextState,
      chartPick: chartPick,
      payoutFor: payoutFor,
      label: label,
      isWild: isWild,
      cardPoints: points,
      cardDraws: draws,
      jackpotScore: JACKPOT_SCORE,
      phases: { AWAITING_PLAY, AWAITING_RIDER, ROUND_OVER },
      endReasons: { OUT_OF_CARDS, NO_MATCH, JACKPOT, DECK_EXHAUSTED },
    };
  }

  // -- small helpers --------------------------------------------------------
  function indicesWhere(list, test) {
    const out = [];
    for (let i = 0; i < list.length; i++) if (test(list[i])) out.push(i);
    return out;
  }

  function clone(state) {
    return Object.assign({}, state, {
      drawPile: state.drawPile.slice(),
      discardPile: state.discardPile.slice(),
      hand: state.hand.slice(),
      legal: state.legal.slice(),
      burned: state.burned.slice(),
      playedThisTurn: state.playedThisTurn.slice(),
      drawnLastTurn: state.drawnLastTurn.slice(),
      log: state.log.slice(),
    });
  }

  return { create: create };
});
