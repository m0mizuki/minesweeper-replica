import test from "node:test";
import assert from "node:assert/strict";
import {
  createGibbsBlocks,
  createGibbsState,
  runGibbsStep,
} from "../js/gibbs-engine.mjs";
import {
  createExampleBoard,
  createFactorGraph,
} from "../js/bp-engine.mjs";

const satisfiesAllClues = (graph, assignment) => graph.factors.every((factor) => (
  factor.variableIds.reduce((sum, id) => sum + assignment[id], 0) === factor.clue
));

test("指定したサイズの変数部分集合をブロックとして列挙する", () => {
  const graph = createFactorGraph(createExampleBoard());
  const pairBlocks = createGibbsBlocks(graph, 2);
  assert.equal(pairBlocks.length, 10);
  assert.ok(pairBlocks.every((block) => block.variableIds.length === 2));
  assert.deepEqual(pairBlocks[0].variableIds, ["x1", "x2"]);
  assert.deepEqual(pairBlocks.at(-1).variableIds, ["x4", "x5"]);

  const fullBlock = createGibbsBlocks(graph, graph.variables.length);
  assert.equal(fullBlock.length, 1);
  assert.deepEqual(fullBlock[0].sourceFactorIds, ["a1", "a2", "a3", "a4"]);
  assert.deepEqual(fullBlock[0].variableIds, ["x1", "x2", "x3", "x4", "x5"]);
});

test("初期状態はすべての数字制約を満たす", () => {
  const graph = createFactorGraph(createExampleBoard());
  const state = createGibbsState(graph);
  assert.equal(state.step, 0);
  assert.equal(state.sampleCount, 0);
  assert.ok(satisfiesAllClues(graph, state.assignment));
});

test("1ステップで候補を列挙・正規化し乱数区間から選択する", () => {
  const graph = createFactorGraph(createExampleBoard());
  const initial = createGibbsState(graph);
  const { state, trace } = runGibbsStep(graph, initial, () => 0.8);
  assert.equal(state.step, 1);
  assert.equal(state.sampleCount, 1);
  assert.equal(trace.candidates.length, 32);
  assert.ok(satisfiesAllClues(graph, state.assignment));
  assert.ok(trace.candidates[trace.selectedIndex].valid);
  assert.notDeepEqual(state.assignment, initial.assignment);
  trace.candidates.forEach((candidate) => {
    const deltaProduct = candidate.factorChecks.every(({ satisfied }) => satisfied) ? 1 : 0;
    assert.equal(candidate.rawWeight, deltaProduct);
  });
  assert.equal(trace.normalization, 2);
  const probabilitySum = trace.candidates.reduce((sum, candidate) => sum + candidate.probability, 0);
  assert.ok(Math.abs(probabilitySum - 1) < 1e-12);
  const validProbabilities = trace.candidates
    .filter(({ valid }) => valid)
    .map(({ probability }) => probability)
    .sort((left, right) => left - right);
  assert.ok(Math.abs(validProbabilities[0] - 1 / 2) < 1e-12);
  assert.ok(Math.abs(validProbabilities[1] - 1 / 2) < 1e-12);
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
  const noShuffle = () => 1 - Number.EPSILON;
  let state = createGibbsState(graph, 2, noShuffle);
  assert.equal(state.blockSize, 2);
  assert.equal(state.blocks.length, 6);
  const first = runGibbsStep(graph, state, () => 0);
  assert.equal(first.trace.block.variableIds.length, 2);
  assert.equal(first.trace.candidates.length, 4);
  state = first.state;
  const second = runGibbsStep(graph, state, () => 0);
  state = second.state;
  assert.equal(first.trace.block.id, "B1");
  assert.equal(second.trace.block.id, "B2");
  assert.equal(state.lastBlockIndex, second.trace.blockIndex);
  assert.equal(state.sampleCount, 2);
  for (const values of Object.values(state.marginals)) {
    assert.ok(Math.abs(values[0] + values[1] - 1) < 1e-12);
  }
});

test("全ブロックをスイープごとにランダムな順番で1回ずつ更新する", () => {
  const graph = createFactorGraph({
    size: 2,
    rho: 0.25,
    revealed: new Set(),
    mines: new Set(),
  });
  let state = createGibbsState(graph, 2, () => 0);
  const identityOrder = state.blocks.map((_, index) => index);
  const firstSweepOrder = [...state.blockOrder];
  assert.notDeepEqual(firstSweepOrder, identityOrder);
  assert.deepEqual([...firstSweepOrder].sort((a, b) => a - b), identityOrder);

  const visited = [];
  for (let index = 0; index < state.blocks.length; index += 1) {
    const result = runGibbsStep(graph, state, () => 0);
    visited.push(result.trace.blockIndex);
    state = result.state;
  }
  assert.deepEqual(visited, firstSweepOrder);
  assert.equal(new Set(visited).size, state.blocks.length);
  assert.equal(state.blockCursor, 0);
  assert.deepEqual([...state.blockOrder].sort((a, b) => a - b), identityOrder);
});

test("因子に接続しない変数も指定サイズのブロックで更新できる", () => {
  const noFactors = createFactorGraph({
    size: 2,
    rho: 0.25,
    revealed: new Set(),
    mines: new Set(),
  });
  const blocks = createGibbsBlocks(noFactors, 2);
  assert.equal(blocks.length, 6);
  assert.ok(blocks.every((block) => block.variableIds.length === 2));
  assert.ok(blocks.every((block) => block.sourceFactorIds.length === 0));
});

test("ブロックサイズは2から全変数までに制限する", () => {
  const graph = createFactorGraph(createExampleBoard());
  assert.throws(() => createGibbsBlocks(graph, 1), RangeError);
  assert.throws(() => createGibbsBlocks(graph, graph.variables.length + 1), RangeError);
});
