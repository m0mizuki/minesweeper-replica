import test from "node:test";
import assert from "node:assert/strict";
import {
  cellKey,
  createExampleBoard,
  createFactorGraph,
  createBPState,
  runCycle,
} from "../js/bp-engine.js";

const approximately = (actual, expected, epsilon = 1e-10) => {
  assert.ok(Math.abs(actual - expected) < epsilon, `${actual} ≈ ${expected}`);
};

test("制約のない変数の周辺確率は事前分布と一致する", () => {
  for (const rho of [0, 0.2, 0.5, 1]) {
    const graph = createFactorGraph({ size: 2, rho, revealed: new Set(), mines: new Set() });
    const state = createBPState(graph);
    for (const values of Object.values(state.marginals)) {
      approximately(values[0], 1 - rho);
      approximately(values[1], rho);
    }
  }
});

test("変数→因子の初期値は事前分布、因子→変数は中立とする", () => {
  const graph = createFactorGraph(createExampleBoard());
  const state = createBPState(graph);
  for (const edge of graph.edges) {
    assert.deepEqual(state.variableToFactor[edge.key], [1 - graph.rho, graph.rho]);
    assert.deepEqual(state.factorToVariable[edge.key], [0.5, 0.5]);
  }
});

test("周辺確率のトレースは事前因子を1回掛けた値を正規化する", () => {
  const graph = createFactorGraph({
    size: 2, rho: 0.25,
    revealed: new Set([cellKey(0, 0)]),
    mines: new Set([cellKey(1, 1)]),
  });
  const { state, trace } = runCycle(graph, createBPState(graph));
  assert.equal(trace.rho, 0.25);
  assert.equal(trace.marginalCalculations.length, 3);
  for (const calculation of trace.marginalCalculations) {
    const [zero, one] = calculation.rows;
    approximately(zero.prior, 0.75);
    approximately(one.prior, 0.25);
    assert.equal(zero.terms.length, 1);
    approximately(zero.terms[0].value, 0.4);
    approximately(one.terms[0].value, 0.6);
    approximately(zero.raw, 0.3);
    approximately(one.raw, 0.15);
    approximately(calculation.normalization, 0.45);
    approximately(calculation.normalized[1], 1 / 3);
    assert.deepEqual(calculation.normalized, state.marginals[calculation.variableId]);
  }
});

test("資料例の盤面から4因子・5変数・12辺を構成する", () => {
  const graph = createFactorGraph(createExampleBoard());
  assert.equal(graph.variables.length, 5);
  assert.equal(graph.factors.length, 4);
  assert.equal(graph.edges.length, 12);
  assert.deepEqual(graph.factors.map((factor) => factor.clue), [1, 1, 2, 2]);
});

test("2×2盤面から因子グラフを構成してBPを更新できる", () => {
  const board = {
    size: 2,
    rho: 0.25,
    revealed: new Set([cellKey(0, 0)]),
    mines: new Set([cellKey(1, 1)]),
  };
  const graph = createFactorGraph(board);
  assert.equal(graph.variables.length, 3);
  assert.equal(graph.factors.length, 1);
  assert.equal(graph.edges.length, 3);
  assert.equal(graph.factors[0].clue, 1);

  const { state, trace } = runCycle(graph, createBPState(graph));
  assert.equal(state.cycle, 1);
  assert.equal(trace.variableCalculations.length, 3);
  assert.equal(trace.factorCalculations.length, 3);
});

test("1サイクル目の a1→x1 は事前分布を反映した [1/2, 1/2]", () => {
  const graph = createFactorGraph(createExampleBoard());
  const initial = createBPState(graph);
  const { state, trace } = runCycle(graph, initial);
  const message = state.factorToVariable["x1|a1"];
  approximately(message[0], 1 / 2);
  approximately(message[1], 1 / 2);
  const calculation = trace.factorCalculations.find(({ name }) => name === "m_a1→x1");
  assert.deepEqual(calculation.rows.map((row) => row.assignments.length), [2, 1]);
  assert.deepEqual(
    calculation.rows[0].assignments[0].terms.map(({ from, to, argument, probability }) => ({
      from,
      to,
      argument,
      probability,
    })),
    [
      { from: "x2", to: "a1", argument: 1, probability: 1 / 3 },
      { from: "x3", to: "a1", argument: 0, probability: 1 - 1 / 3 },
    ],
  );
});

test("2サイクル目の x3→a1 は事前因子を含む [1/9, 8/9]", () => {
  const graph = createFactorGraph(createExampleBoard());
  const first = runCycle(graph, createBPState(graph));
  const second = runCycle(graph, first.state);
  const message = second.state.variableToFactor["x3|a1"];
  approximately(message[0], 1 / 9);
  approximately(message[1], 8 / 9);
});

test("各メッセージは正規化される", () => {
  const graph = createFactorGraph(createExampleBoard());
  const { state } = runCycle(graph, createBPState(graph));
  for (const message of [
    ...Object.values(state.variableToFactor),
    ...Object.values(state.factorToVariable),
    ...Object.values(state.marginals),
  ]) {
    approximately(message[0] + message[1], 1);
  }
});
