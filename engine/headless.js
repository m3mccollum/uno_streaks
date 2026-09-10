/*
 * headless.js -- run the browser engine with no browser.
 *
 * The point of the reducer is that the rules can be driven by something other
 * than clicks. This plays batches with the CHART strategy and prints the score
 * distribution and RTP, so the JS can be checked against uno_sim.py.
 *
 *     node engine/headless.js 200000
 *
 * Remember the sample-size rule from CLAUDE.md: 10k hands is +-4% on RTP and
 * you need ~170k for +-1%. A jackpot frequency needs millions.
 */
'use strict';

const path = require('path');
const data = require(path.join(__dirname, '..', 'data', 'game-data.js'));
const UnoEngine = require(path.join(__dirname, 'engine.js'));

const engine = UnoEngine.create(data);

/** Play one round through, always taking the chart's pick. */
function playRound(seed) {
  let state = engine.newRound(seed);
  while (state.phase !== engine.phases.ROUND_OVER) {
    const index = engine.chartPick(state);
    state = engine.nextState(state, { type: 'play', index: index });
  }
  return state;
}

function main() {
  const rounds = Number(process.argv[2] || 100000);
  const dist = new Map();
  const reasons = new Map();
  let paid = 0;

  for (let i = 0; i < rounds; i++) {
    const state = playRound(i + 1);
    dist.set(state.score, (dist.get(state.score) || 0) + 1);
    reasons.set(state.endReason, (reasons.get(state.endReason) || 0) + 1);
    paid += state.payout;
  }

  const scores = [...dist.keys()].sort((a, b) => a - b);
  console.log('score   freq       pays    contribution');
  for (const score of scores) {
    const freq = dist.get(score) / rounds;
    const pays = engine.payoutFor(score);
    console.log(
      String(score).padStart(5),
      (freq * 100).toFixed(4).padStart(8) + '%',
      pays.toFixed(2).padStart(8) + 'x',
      (freq * pays).toFixed(4).padStart(12)
    );
  }
  console.log('');
  for (const [reason, n] of reasons) {
    console.log(reason.padEnd(16), ((n / rounds) * 100).toFixed(3) + '%');
  }
  // Standard error of the mean payout, so the RTP comes with its uncertainty.
  console.log('');
  console.log('rounds  ' + rounds);
  console.log('RTP     ' + ((paid / rounds) * 100).toFixed(3) + '%');
}

main();
