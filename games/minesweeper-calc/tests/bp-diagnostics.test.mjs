import test from "node:test";
import assert from "node:assert/strict";
import {
  createExampleBoard, createFactorGraph, createBPState, runCycle,
} from "../js/bp-engine.js";
import { referenceInitial, referenceCycle, maxMessageDifference } from "../../../scripts/diagnose_calc_bp.mjs";

test("dampingの既定値と明示的なオフは従来の更新と一致する", () => {
  const graph = createFactorGraph(createExampleBoard());
  const initial = createBPState(graph);
  const omitted = runCycle(graph, initial);
  const off = runCycle(graph, initial, { damping: 0 });
  assert.deepEqual(omitted, off);
  assert.equal(omitted.trace.damping, 0);
  for (const calculation of omitted.trace.factorCalculations) {
    assert.deepEqual(calculation.normalized, calculation.undamped);
  }
});

test("dampingは正規化したBP値と旧メッセージを指定割合で混合する", () => {
  const graph = createFactorGraph(createExampleBoard());
  const { state, trace } = runCycle(graph, createBPState(graph), { damping: 0.2 });
  const calculation = trace.factorCalculations.find(({ name }) => name === "m_a3→x5");
  assert.equal(calculation.damping, 0.2);
  assert.deepEqual(calculation.previous, [0.5, 0.5]);
  assert.ok(Math.abs(calculation.undamped[0] - 0.2) < 1e-12);
  assert.ok(Math.abs(state.factorToVariable["x5|a3"][0] - 0.26) < 1e-12);
  assert.ok(Math.abs(state.factorToVariable["x5|a3"][1] - 0.74) < 1e-12);
});

test("dampingのオン・係数変更・オフは独立した参照計算と一致する", () => {
  const graph = createFactorGraph(createExampleBoard());
  let reference = referenceInitial(graph);
  let state = createBPState(graph);
  for (const damping of [...Array(10).fill(0.2), ...Array(10).fill(0.5), ...Array(3).fill(0)]) {
    const next = runCycle(graph, state, { damping });
    state = next.state;
    reference = referenceCycle(graph, reference, damping);
    assert.equal(next.trace.damping, damping);
    assert.equal(next.trace.hasContradiction, false);
    for (const direction of ["variableToFactor", "factorToVariable"]) {
      for (const [key, values] of Object.entries(state[direction])) {
        assert.ok(Math.abs(values[0] + values[1] - 1) < 1e-12);
        for (const x of [0, 1]) assert.ok(Math.abs(values[x] - Math.exp(reference[direction][key][x])) < 1e-12);
      }
    }
    for (const [id, marginal] of Object.entries(state.marginals)) {
      assert.ok(Math.abs(marginal[1] - Math.exp(reference.marginalLogs[id][1])) < 1e-12);
    }
  }
});

test("dampingで資料例を100サイクル更新しても正規化失敗しない", () => {
  const graph = createFactorGraph(createExampleBoard());
  let state = createBPState(graph);
  for (let i = 0; i < 100; i += 1) {
    const next = runCycle(graph, state, { damping: 0.2 });
    assert.equal(next.trace.hasContradiction, false);
    state = next.state;
  }
});

test("不正なdamping係数を拒否し、ゼロ質量の更新を旧メッセージで隠さない", () => {
  const graph = createFactorGraph(createExampleBoard());
  const initial = createBPState(graph);
  for (const damping of [-0.1, 1, NaN, Infinity, "0.2", true]) {
    assert.throws(() => runCycle(graph, initial, { damping }), RangeError);
  }
  const impossible = createFactorGraph(createExampleBoard());
  impossible.factors[0].clue = 9;
  const { trace } = runCycle(impossible, createBPState(impossible), { damping: 0.2 });
  assert.equal(trace.hasContradiction, true);
  const failure = trace.factorCalculations.find(({ impossible }) => impossible);
  assert.deepEqual(failure.normalized, [0, 0]);
});

test("独立した対数空間BPとブラウザBPの更新が丸め誤差内で一致する", () => {
  for (const rho of [0.1, 0.3, 1 / 3, 0.5, 0.8]) {
    const graph = createFactorGraph({ ...createExampleBoard(), rho });
    let reference = referenceInitial(graph);
    let state = createBPState(graph);
    for (let cycle = 0; cycle < 10; cycle += 1) {
      state = runCycle(graph, state).state;
      reference = referenceCycle(graph, reference);
      const converted = Object.fromEntries(["variableToFactor", "factorToVariable"].map((direction) => [
        direction, Object.fromEntries(Object.entries(reference[direction]).map(([key, logs]) => [key, logs.map(Math.exp)])),
      ]));
      assert.ok(maxMessageDifference(state, converted) < 1e-12);
    }
  }
});

test("木上では地雷数の異なる2解の厳密なBernoulli事後確率と一致する", () => {
  // x1+x2=1, x2+x3=1; the two states are 010 and 101.
  for (const rho of [0.1, 0.3, 0.5, 0.8]) {
    const graph = {
      rho,
      variables: [
        { id: "x1", factorIds: ["a1"] },
        { id: "x2", factorIds: ["a1", "a2"] },
        { id: "x3", factorIds: ["a2"] },
      ],
      factors: [
        { id: "a1", clue: 1, variableIds: ["x1", "x2"] },
        { id: "a2", clue: 1, variableIds: ["x2", "x3"] },
      ],
      edges: [
        { key: "x1|a1", variableId: "x1", factorId: "a1" },
        { key: "x2|a1", variableId: "x2", factorId: "a1" },
        { key: "x2|a2", variableId: "x2", factorId: "a2" },
        { key: "x3|a2", variableId: "x3", factorId: "a2" },
      ],
    };
    let state = createBPState(graph);
    for (let cycle = 0; cycle < 5; cycle += 1) state = runCycle(graph, state).state;
    for (const [index, expected] of [rho, 1 - rho, rho].entries()) {
      assert.ok(Math.abs(state.marginals[`x${index + 1}`][1] - expected) < 1e-12);
    }
  }
});

test("資料例のBP振動は対数空間でも残り、数値アンダーフローだけが原因ではない", () => {
  const graph = createFactorGraph(createExampleBoard());
  let state = referenceInitial(graph);
  const center = [];
  for (let cycle = 0; cycle < 30; cycle += 1) {
    state = referenceCycle(graph, state);
    center.push(Math.exp(state.marginalLogs.x3[1]));
    assert.ok(Object.values(state.variableToFactor).flat().every(Number.isFinite));
    assert.ok(Object.values(state.factorToVariable).flat().every(Number.isFinite));
  }
  assert.ok(center[0] > 0.88 && center[1] < 0.34);
  assert.ok(center[28] > 0.99 && center[29] < 0.01);
});

test("dampingによる数値的収束は厳密周辺確率との一致を保証しない", () => {
  const graph = createFactorGraph(createExampleBoard());
  let state = referenceInitial(graph);
  for (let cycle = 0; cycle < 100; cycle += 1) state = referenceCycle(graph, state, 0.2);
  // The exact center probability is 2/3; this loopy BP fixed point selects one solution.
  assert.ok(Math.exp(state.marginalLogs.x3[1]) > 0.99);
  assert.ok(Math.abs(Math.exp(state.marginalLogs.x3[1]) - 2 / 3) > 0.3);
});
