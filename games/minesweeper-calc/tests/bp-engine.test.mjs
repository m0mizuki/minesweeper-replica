import test from "node:test";
import assert from "node:assert/strict";
import {
  createExampleBoard,
  createFactorGraph,
  createBPState,
  runCycle,
} from "../js/bp-engine.mjs";

const approximately = (actual, expected, epsilon = 1e-10) => {
  assert.ok(Math.abs(actual - expected) < epsilon, `${actual} ≈ ${expected}`);
};

test("資料例の盤面から4因子・5変数・12辺を構成する", () => {
  const graph = createFactorGraph(createExampleBoard());
  assert.equal(graph.variables.length, 5);
  assert.equal(graph.factors.length, 4);
  assert.equal(graph.edges.length, 12);
  assert.deepEqual(graph.factors.map((factor) => factor.clue), [1, 1, 2, 2]);
});

test("1サイクル目の a1→x1 は [2/3, 1/3]", () => {
  const graph = createFactorGraph(createExampleBoard());
  const initial = createBPState(graph);
  const { state, trace } = runCycle(graph, initial);
  const message = state.factorToVariable["x1|a1"];
  approximately(message[0], 2 / 3);
  approximately(message[1], 1 / 3);
  const calculation = trace.factorCalculations.find(({ name }) => name === "m_a1→x1");
  assert.deepEqual(calculation.rows.map((row) => row.assignments.length), [2, 1]);
});

test("2サイクル目の x3→a1 は [1/3, 2/3]", () => {
  const graph = createFactorGraph(createExampleBoard());
  const first = runCycle(graph, createBPState(graph));
  const second = runCycle(graph, first.state);
  const message = second.state.variableToFactor["x3|a1"];
  approximately(message[0], 1 / 3);
  approximately(message[1], 2 / 3);
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
