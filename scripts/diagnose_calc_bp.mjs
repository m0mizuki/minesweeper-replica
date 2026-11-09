// Reproduce browser BP against an independently implemented log-space update.
import { pathToFileURL } from "node:url";
import {
  createExampleBoard, createFactorGraph, createBPState, runCycle,
} from "../games/minesweeper-calc/js/bp-engine.js";

const logAdd = (a, b) => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const max = Math.max(a, b);
  return max + Math.log1p(Math.exp(Math.min(a, b) - max));
};
const normalizeLogs = (logs) => {
  const max = Math.max(...logs);
  if (!Number.isFinite(max)) throw new Error("reference zero mass");
  const shifted = logs.map((value) => value - max);
  const z = Math.log(Math.exp(shifted[0]) + Math.exp(shifted[1]));
  return shifted.map((value) => value - z);
};
const key = (v, a) => `${v}|${a}`;
const probabilities = (map) => Object.fromEntries(Object.entries(map)
  .map(([id, logs]) => [id, logs.map(Math.exp)]));

export function referenceInitial(graph) {
  return {
    variableToFactor: Object.fromEntries(graph.edges.map(({ key: id }) => [
      id, [Math.log1p(-graph.rho), Math.log(graph.rho)],
    ])),
    factorToVariable: Object.fromEntries(graph.edges.map(({ key: id }) => [
      id, [-Math.LN2, -Math.LN2],
    ])),
  };
}

export function referenceCycle(graph, state, damping = 0) {
  const prior = [Math.log1p(-graph.rho), Math.log(graph.rho)];
  const mix = (logs, old) => damping === 0 ? logs : normalizeLogs(logs.map((v, x) => (
    logAdd(Math.log1p(-damping) + v, Math.log(damping) + old[x])
  )));
  const variableToFactor = {};
  for (const variable of graph.variables) {
    for (const target of variable.factorIds) {
      const logs = prior.map((p, x) => variable.factorIds
        .filter((id) => id !== target)
        .reduce((sum, id) => sum + state.factorToVariable[key(variable.id, id)][x], p));
      const id = key(variable.id, target);
      variableToFactor[id] = mix(normalizeLogs(logs), state.variableToFactor[id]);
    }
  }
  const factorToVariable = {};
  for (const factor of graph.factors) {
    for (const target of factor.variableIds) {
      // Sum over assignments by dynamic programming on the mine count.
      let counts = [0];
      for (const id of factor.variableIds.filter((id) => id !== target)) {
        const next = Array(counts.length + 1).fill(-Infinity);
        counts.forEach((value, count) => {
          for (const x of [0, 1]) {
            next[count + x] = logAdd(next[count + x], value + variableToFactor[key(id, factor.id)][x]);
          }
        });
        counts = next;
      }
      const logs = [counts[factor.clue] ?? -Infinity, counts[factor.clue - 1] ?? -Infinity];
      const id = key(target, factor.id);
      factorToVariable[id] = mix(normalizeLogs(logs), state.factorToVariable[id]);
    }
  }
  const marginalLogs = Object.fromEntries(graph.variables.map((variable) => [
    variable.id, normalizeLogs(prior.map((p, x) => variable.factorIds
      .reduce((sum, id) => sum + factorToVariable[key(variable.id, id)][x], p))),
  ]));
  return { variableToFactor, factorToVariable, marginalLogs };
}

export function maxMessageDifference(left, right) {
  let maximum = 0;
  for (const direction of ["variableToFactor", "factorToVariable"]) {
    for (const id of Object.keys(left[direction])) {
      for (const x of [0, 1]) maximum = Math.max(maximum, Math.abs(left[direction][id][x] - right[direction][id][x]));
    }
  }
  return maximum;
}

const asProbabilities = (state) => ({
  variableToFactor: probabilities(state.variableToFactor),
  factorToVariable: probabilities(state.factorToVariable),
});

function exact(graph) {
  const states = [];
  let z = 0;
  for (let mask = 0; mask < 2 ** graph.variables.length; mask += 1) {
    const bits = graph.variables.map((_, index) => (mask >> index) & 1);
    const assignment = Object.fromEntries(graph.variables.map(({ id }, i) => [id, bits[i]]));
    if (!graph.factors.every((f) => f.variableIds.reduce((n, id) => n + assignment[id], 0) === f.clue)) continue;
    const weight = bits.reduce((w, bit) => w * (bit ? graph.rho : 1 - graph.rho), 1);
    states.push({ bits, weight });
    z += weight;
  }
  return { z, states: states.map(({ bits, weight }) => ({ bits, probability: weight / z })) };
}

function example(rho, damping = 0) {
  const graph = createFactorGraph({ ...createExampleBoard(), rho });
  let browser = createBPState(graph);
  let reference = referenceInitial(graph);
  let firstZero = null;
  let maxReferenceErrorFirstTenCycles = 0;
  const checkpoints = [];
  for (let n = 1; n <= 100; n += 1) {
    const nextReference = referenceCycle(graph, reference, damping);
    const refDelta = maxMessageDifference(asProbabilities(reference), asProbabilities(nextReference));
    reference = nextReference;
    if (firstZero === null && damping === 0) {
      const next = runCycle(graph, browser);
      if (next.trace.hasContradiction) firstZero = n;
      else if (n <= 10) maxReferenceErrorFirstTenCycles = Math.max(
        maxReferenceErrorFirstTenCycles, maxMessageDifference(next.state, asProbabilities(reference)),
      );
      browser = next.state;
    }
    if ([1, 2, 3, 4, 18, 19, 30, 50, 100].includes(n)) checkpoints.push({
      cycle: n, residual: refDelta,
      centerProbability: Math.exp(reference.marginalLogs.x3[1]),
      centerLogOdds: reference.marginalLogs.x3[1] - reference.marginalLogs.x3[0],
      allLogMessagesFinite: Object.values(reference.variableToFactor).flat().every(Number.isFinite)
        && Object.values(reference.factorToVariable).flat().every(Number.isFinite),
    });
  }
  return { rho, damping, firstZero, maxReferenceErrorFirstTenCycles, exact: exact(graph), checkpoints };
}

function scan(generationRho, inferenceRho) {
  const base = createExampleBoard();
  const cells = createFactorGraph(base).variables.map(({ key: id }) => id);
  const counts = { converged: 0, zero_mass: 0, iteration_limit: 0 };
  let probabilityOfZeroMass = 0;
  const failures = [];
  for (let mask = 0; mask < 2 ** cells.length; mask += 1) {
    const mines = new Set(cells.filter((_, i) => (mask >> i) & 1));
    const graph = createFactorGraph({ ...base, rho: inferenceRho, mines });
    let state = createBPState(graph);
    let status = "iteration_limit";
    for (let n = 0; n < 100; n += 1) {
      const next = runCycle(graph, state);
      const delta = maxMessageDifference(state, next.state);
      state = next.state;
      if (next.trace.hasContradiction) { status = "zero_mass"; break; }
      if (delta < 1e-10) { status = "converged"; break; }
    }
    counts[status] += 1;
    if (status === "zero_mass") {
      probabilityOfZeroMass += generationRho ** mines.size * (1 - generationRho) ** (cells.length - mines.size);
      failures.push(mask);
    }
  }
  return { generationRho, inferenceRho, cycles: 100, tolerance: 1e-10, counts, probabilityOfZeroMass, failures };
}

function dampedDensityPoint(rho) {
  const graph = createFactorGraph(createExampleBoard({ rho }));
  let state = createBPState(graph);
  let status = "iteration_limit";
  let cycle = 0;
  let residual = 0;
  for (cycle = 1; cycle <= 1000; cycle += 1) {
    const next = runCycle(graph, state, { damping: 0.2 });
    residual = maxMessageDifference(state, next.state);
    state = next.state;
    if (next.trace.hasContradiction) { status = "zero_mass"; break; }
    if (residual < 1e-10) { status = "converged"; break; }
  }
  return {
    rho, damping: 0.2, status, cycles: Math.min(cycle, 1000), residual,
    bpMarginals: graph.variables.map(({ id }) => state.marginals[id][1]),
    exactMarginals: [0, rho, 1 - rho, rho, 1],
    exact: exact(graph),
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  console.log(JSON.stringify({
    examples: [example(1 / 3), example(0.5), example(1 / 3, 0.2)],
    scans: [0.1, 0.3, 1 / 3, 0.5, 0.8].map((rho) => ({
      oldUniform: scan(rho, 0.5),
      newBernoulli: scan(rho, rho),
    })),
    dampedDensitySweep: [0.1, 0.2, 1 / 3, 0.4, 0.49, 0.5, 0.51, 0.6, 0.8, 0.9].map(dampedDensityPoint),
  }, null, 2));
}
