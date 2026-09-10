/*
 * ui.js -- clicks in, engine actions out. All the DOM lives here.
 *
 * This file owns no rules. It reads state the engine hands back and turns the
 * player's click into `{type: 'play', index}`; anything it needed to decide for
 * itself would be a second implementation of the rules.
 */
'use strict';

(function () {
  const data = globalThis.UNO_DATA;
  const engine = UnoEngine.create(data);
  const $ = (id) => document.getElementById(id);

  const COLOR_CLASS = { R: 'red', Y: 'yellow', G: 'green', B: 'blue', W: 'wild' };

  let state = null;      // current engine state, or null between rounds
  let credits = 100;
  let bet = 5;

  // index.html?seed=42 deals reproducible rounds (42, 43, 44 ...), so an odd
  // hand can be handed to someone else exactly as it happened.
  const seedParam = new URLSearchParams(location.search).get('seed');
  let seed = seedParam === null ? null : Number(seedParam) >>> 0;

  // -- card rendering -------------------------------------------------------
  function cardEl(card, extra) {
    const el = document.createElement('div');
    el.className = 'card ' + COLOR_CLASS[card.c] + (extra ? ' ' + extra : '');
    const face = document.createElement('span');
    face.className = 'face' + (card.r.length > 2 ? ' small' : '');
    face.textContent = card.r;
    el.appendChild(face);
    const corner = document.createElement('span');
    corner.className = 'corner';
    corner.textContent = card.r;
    el.appendChild(corner);
    return el;
  }

  // -- static furniture -----------------------------------------------------
  function buildPaytable() {
    const table = $('paytable');
    table.innerHTML = '';
    const scores = Object.keys(data.paytable).map(Number).sort((a, b) => b - a);
    for (const score of scores) {
      const pays = data.paytable[String(score)];
      const row = table.insertRow();
      row.id = 'pay-' + score;
      row.className = pays > 0 ? '' : 'dead';
      if (score === data.jackpotScore) row.classList.add('jackpot-row');
      row.insertCell().textContent = score;
      const cell = row.insertCell();
      cell.textContent = pays.toFixed(2) + 'x';
      cell.className = 'pays';
    }
  }

  function buildTrack() {
    const track = $('track');
    track.innerHTML = '';
    for (let i = 1; i <= data.jackpotScore; i++) {
      const pip = document.createElement('div');
      pip.className = 'pip' + (data.paytable[String(i)] > 0 ? '' : ' nopay');
      pip.id = 'pip-' + i;
      pip.title = i + ' pays ' + (data.paytable[String(i)] || 0) + 'x';
      track.appendChild(pip);
    }
  }

  // -- the whole view, redrawn from state -----------------------------------
  function render() {
    $('credits').textContent = credits;
    $('score').textContent = state ? state.score : 0;
    $('deal').textContent = state ? 'Deal (round in play)' : 'Deal';
    $('deal').disabled = !!state || credits < bet;

    renderDealer();
    renderHand();
    renderTrack();
    renderPlayed();
    renderLog();

    $('message').textContent = state ? state.message
      : (credits < bet ? 'Not enough credits for that bet.' : 'Press Deal to start.');
    $('deck-count').textContent = state
      ? state.drawPile.length + ' in the draw pile, ' + state.discardPile.length + ' discarded'
      : '';

    // Highlight the paying tier the player currently sits on.
    for (const row of $('paytable').rows) row.classList.remove('here');
    if (state) {
      const row = $('pay-' + state.score);
      if (row) row.classList.add('here');
    }
  }

  function renderDealer() {
    const slot = $('dealer-card');
    slot.innerHTML = '';
    if (state && state.dealerCard) {
      slot.appendChild(cardEl(state.dealerCard, 'big'));
    } else {
      const back = document.createElement('div');
      back.className = 'card back big';
      slot.appendChild(back);
    }

    const burned = $('burned');
    burned.innerHTML = '';
    if (state && state.burned.length) {
      const note = document.createElement('span');
      note.className = 'burn-note';
      note.textContent = 'burned:';
      burned.appendChild(note);
      for (const card of state.burned) burned.appendChild(cardEl(card, 'tiny'));
    }
  }

  function renderHand() {
    const hand = $('hand');
    hand.innerHTML = '';
    if (!state) return;

    const hintOn = $('hint').checked;
    const pick = hintOn && state.phase !== engine.phases.ROUND_OVER
      ? engine.chartPick(state) : null;
    const legal = new Set(state.legal);

    state.hand.forEach((card, index) => {
      const playable = legal.has(index);
      const el = cardEl(card, playable ? 'playable' : 'dimmed');
      if (index === pick) el.classList.add('hinted');
      if (state.drawnLastTurn.indexOf(card) !== -1) el.classList.add('fresh');
      if (playable) {
        el.tabIndex = 0;
        el.addEventListener('click', () => play(index));
        el.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); play(index); }
        });
      }
      hand.appendChild(el);
    });

    if (state.hand.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'empty-hand';
      empty.textContent = 'Hand empty.';
      hand.appendChild(empty);
    }
  }

  function renderTrack() {
    const score = state ? state.score : 0;
    for (let i = 1; i <= data.jackpotScore; i++) {
      const pip = $('pip-' + i);
      pip.classList.toggle('lit', i <= score);
      pip.classList.toggle('tip', i === score);
    }
  }

  function renderPlayed() {
    const box = $('played');
    box.innerHTML = '';
    if (!state || !state.playedThisTurn.length) return;
    const note = document.createElement('span');
    note.className = 'burn-note';
    note.textContent = 'this turn:';
    box.appendChild(note);
    for (const entry of state.playedThisTurn) {
      const wrap = document.createElement('div');
      wrap.className = 'played-card';
      wrap.appendChild(cardEl(entry.card, 'tiny'));
      const pts = document.createElement('span');
      pts.className = 'pts';
      pts.textContent = '+' + entry.points;
      wrap.appendChild(pts);
      box.appendChild(wrap);
    }
  }

  function renderLog() {
    const log = $('log');
    log.innerHTML = '';
    if (!state) return;
    for (const line of state.log) {
      const li = document.createElement('li');
      li.textContent = line;
      log.appendChild(li);
    }
    log.scrollTop = log.scrollHeight;
  }

  // -- actions --------------------------------------------------------------
  function play(index) {
    state = engine.nextState(state, { type: 'play', index: index });
    if (state.phase === engine.phases.ROUND_OVER) return settle();
    render();
  }

  function settle() {
    const won = bet * state.payout;
    credits += won;
    $('lastwin').textContent = state.payout > 0
      ? won + ' (' + state.payout.toFixed(2) + 'x)' : '--';
    const finished = state;
    render();
    $('message').textContent = finished.message + ' Scored ' + finished.score +
      ' for ' + finished.payout.toFixed(2) + 'x -- ' +
      (won > 0 ? 'won ' + won + '.' : 'no win.');
    if (finished.endReason === engine.endReasons.JACKPOT) {
      document.body.classList.add('jackpot-flash');
      setTimeout(() => document.body.classList.remove('jackpot-flash'), 2000);
    }
    // Keep the finished board on screen; the next Deal clears it.
    state = null;
    $('deal').disabled = credits < bet;
    $('deal').textContent = 'Deal';
    renderHandFrozen(finished);
  }

  /* After a round the hand is shown as it finished: nothing clickable. */
  function renderHandFrozen(finished) {
    const hand = $('hand');
    hand.innerHTML = '';
    for (const card of finished.hand) hand.appendChild(cardEl(card, 'dimmed'));
    if (!finished.hand.length) {
      const empty = document.createElement('div');
      empty.className = 'empty-hand';
      empty.textContent = 'Hand empty -- every card played.';
      hand.appendChild(empty);
    }
    $('score').textContent = finished.score;
    for (let i = 1; i <= data.jackpotScore; i++) {
      $('pip-' + i).classList.toggle('lit', i <= finished.score);
    }
    const row = $('pay-' + finished.score);
    if (row) row.classList.add('here');
  }

  function deal() {
    if (credits < bet) return;
    credits -= bet;
    state = engine.newRound(seed === null ? undefined : seed++);
    if (state.phase === engine.phases.ROUND_OVER) return settle();
    render();
  }

  // -- wiring ---------------------------------------------------------------
  $('deal').addEventListener('click', deal);
  $('hint').addEventListener('change', () => { if (state) renderHand(); });
  $('bet').addEventListener('change', (e) => {
    bet = Number(e.target.value);
    if (!state) render();
  });

  $('jackpot-score').textContent = data.jackpotScore;
  buildPaytable();
  buildTrack();
  render();

  // Exposed so a console session (or a screenshot script) can drive the same
  // reducer the buttons drive.
  globalThis.UNO_UI = { engine: engine, get state() { return state; } };
})();
