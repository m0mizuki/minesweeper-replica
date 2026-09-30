import test from "node:test";
import assert from "node:assert/strict";
import {
  createGibbsBlocks,
  createGibbsState,
  runGibbsStep,
} from "../js/gibbs-engine.mjs";
import {
  cellKey,
  createExampleBoard,
  createFactorGraph,
} from "../js/bp-engine.mjs";

const satisfiesAllClues = (graph, assignment) => graph.factors.every((factor) => (
  factor.variableIds.reduce((sum, id) => sum + assignment[id], 0) === factor.clue
));

test("数字制約で連結した変数を1つのブロックにまとめる", () => {
  const graph = createFactorGraph(createExampleBoard());
  const blocks = createGibbsBlocks(graph);
  assert.equal(blocks.length, 1);
  assert.deepEqual(blocks[0].sourceFactorIds, ["a1", "a2", "a3", "a4"]);
  assert.deepEqual(blocks[0].variableIds, ["x1", "x2", "x3", "x4", "x5"]);
});

test("初期状態はすべての数字制約を満たす", () => {
  const graph = createFactorGraph(createExampleBoard());
  const state = createGibbsState(graph, 1 / 3);
  assert.equal(state.step, 0);
  assert.equal(state.sampleCount, 0);
  assert.ok(satisfiesAllClues(graph, state.assignment));
});

test("1ステップで候補を列挙・正規化し乱数区間から選択する", () => {
  const graph = createFactorGraph(createExampleBoard());
  const initial = createGibbsState(graph, 1 / 3);
  const { state, trace } = runGibbsStep(graph, initial, () => 0.8);
  assert.equal(state.step, 1);
  assert.equal(state.sampleCount, 1);
  assert.equal(trace.candidates.length, 32);
  assert.ok(satisfiesAllClues(graph, state.assignment));
  assert.ok(trace.candidates[trace.selectedIndex].valid);
  assert.notDeepEqual(state.assignment, initial.assignment);
  const probabilitySum = trace.candidates.reduce((sum, candidate) => sum + candidate.probability, 0);
  assert.ok(Math.abs(probabilitySum - 1) < 1e-12);
  const validProbabilities = trace.candidates
    .filter(({ valid }) => valid)
    .map(({ probability }) => probability)
    .sort((left, right) => left - right);
  assert.ok(Math.abs(validProbabilities[0] - 1 / 3) < 1e-12);
  assert.ok(Math.abs(validProbabilities[1] - 2 / 3) < 1e-12);
  const selected = trace.candidates[trace.selectedIndex];
  assert.ok(trace.randomValue >= selected.interval[0]);
  assert.ok(trace.randomValue < selected.interval[1]);
});

test("ステップごとに複数ブロックを巡回し経験周辺確率を更新する", () => {
  const graph = createFactorGraph({
    size: 2,
    rho: 1 / 3,
    revealed: new Set(),
    mines: new Set(),
  });
  let state = createGibbsState(graph, 1 / 3);
  assert.equal(state.blocks.length, 4);
  const first = runGibbsStep(graph, state, () => 0);
  state = first.state;
  const second = runGibbsStep(graph, state, () => 0);
  state = second.state;
  assert.equal(first.trace.block.id, "B1");
  assert.equal(second.trace.block.id, "B2");
  assert.equal(state.sampleCount, 2);
  for (const values of Object.values(state.marginals)) {
    assert.ok(Math.abs(values[0] + values[1] - 1) < 1e-12);
  }
});

test("因子に接続しない変数も単独ブロックとして更新できる", () => {
  const graph = createFactorGraph({
    size: 2,
    rho: 0.25,
    revealed: new Set([cellKey(0, 0)]),
    mines: new Set(),
  });
  const isolated = graph.variables.find((variable) => variable.factorIds.length === 0);
  assert.equal(isolated, undefined);

  const noFactors = createFactorGraph({
    size: 2,
    rho: 0.25,
    revealed: new Set(),
    mines: new Set(),
  });
  const blocks = createGibbsBlocks(noFactors);
  assert.equal(blocks.length, 4);
  assert.ok(blocks.every((block) => block.variableIds.length === 1));
});
