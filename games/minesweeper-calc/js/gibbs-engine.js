// ES module semantics are shared by browsers and this directory's package.json.
function enumerateBits(length) {
  const assignments = [];
  for (let mask = 0; mask < 2 ** length; mask += 1) {
    assignments.push(Array.from({ length }, (_, index) => (mask >> index) & 1));
  }
  return assignments;
}

function cloneAssignment(assignment) {
  return { ...assignment };
}

function factorValue(factor, assignment) {
  return factor.variableIds.reduce((sum, variableId) => sum + assignment[variableId], 0);
}

function isValidAssignment(graph, assignment) {
  return graph.factors.every((factor) => factorValue(factor, assignment) === factor.clue);
}

function logPriorWeight(bits, rho) {
  const count = bits.reduce((sum, bit) => sum + bit, 0);
  if (rho === 0) return count === 0 ? 0 : -Infinity;
  if (rho === 1) return count === bits.length ? 0 : -Infinity;
  return count * Math.log(rho) + (bits.length - count) * Math.log1p(-rho);
}

function createInitialAssignment(graph) {
  for (const bits of enumerateBits(graph.variables.length)) {
    const assignment = Object.fromEntries(
      graph.variables.map((variable, index) => [variable.id, bits[index]]),
    );
    if (isValidAssignment(graph, assignment) && Number.isFinite(logPriorWeight(bits, graph.rho))) {
      return assignment;
    }
  }
  throw new Error("盤面の数字を満たし、事前確率が正の地雷配置がありません。");
}

function combinations(values, size, start = 0, prefix = [], result = []) {
  if (prefix.length === size) {
    result.push([...prefix]);
    return result;
  }
  const remaining = size - prefix.length;
  for (let index = start; index <= values.length - remaining; index += 1) {
    prefix.push(values[index]);
    combinations(values, size, index + 1, prefix, result);
    prefix.pop();
  }
  return result;
}

function randomUnitInterval(random) {
  return Math.min(Math.max(Number(random()), 0), 1 - Number.EPSILON);
}

function shuffledBlockOrder(length, random) {
  const order = Array.from({ length }, (_, index) => index);
  for (let index = order.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(randomUnitInterval(random) * (index + 1));
    [order[index], order[swapIndex]] = [order[swapIndex], order[index]];
  }
  return order;
}

export function createGibbsBlocks(graph, blockSize = graph.variables.length) {
  const variableIds = graph.variables.map(({ id }) => id);
  if (variableIds.length === 0) return [];
  if (!Number.isInteger(blockSize) || blockSize < 2 || blockSize > variableIds.length) {
    throw new RangeError(`blockSize は 2 以上 ${variableIds.length} 以下で指定してください。`);
  }

  return combinations(variableIds, blockSize).map((blockVariableIds, index) => ({
    id: `B${index + 1}`,
    variableIds: blockVariableIds,
    sourceFactorIds: graph.factors
      .filter((factor) => factor.variableIds.some((id) => blockVariableIds.includes(id)))
      .map(({ id }) => id),
  }));
}

function empiricalMarginals(graph, assignment, mineCounts, sampleCount) {
  return Object.fromEntries(graph.variables.map((variable) => {
    const probability = sampleCount > 0
      ? mineCounts[variable.id] / sampleCount
      : assignment[variable.id];
    return [variable.id, [1 - probability, probability]];
  }));
}

export function createGibbsState(graph, blockSize = graph.variables.length, random = Math.random) {
  const blocks = createGibbsBlocks(graph, blockSize);
  const assignment = createInitialAssignment(graph);
  const mineCounts = Object.fromEntries(graph.variables.map(({ id }) => [id, 0]));
  return {
    step: 0,
    blockSize,
    blocks,
    blockOrder: shuffledBlockOrder(blocks.length, random),
    blockCursor: 0,
    lastBlockIndex: null,
    assignment,
    sampleCount: 0,
    mineCounts,
    marginals: empiricalMarginals(graph, assignment, mineCounts, 0),
  };
}

export function runGibbsStep(graph, state, random = Math.random) {
  if (state.blocks.length === 0) throw new Error("更新できるブロックがありません。");
  const blockIndex = state.blockOrder[state.blockCursor];
  const block = state.blocks[blockIndex];
  const affectedFactors = graph.factors.filter((factor) => (
    factor.variableIds.some((variableId) => block.variableIds.includes(variableId))
  ));

  const candidates = enumerateBits(block.variableIds.length).map((bits) => {
    const assignment = cloneAssignment(state.assignment);
    block.variableIds.forEach((variableId, index) => {
      assignment[variableId] = bits[index];
    });
    const factorChecks = affectedFactors.map((factor) => {
      const sum = factorValue(factor, assignment);
      return { factorId: factor.id, sum, clue: factor.clue, satisfied: sum === factor.clue };
    });
    const valid = factorChecks.every(({ satisfied }) => satisfied);
    const priorLogWeight = logPriorWeight(bits, graph.rho);
    return {
      bits,
      values: Object.fromEntries(block.variableIds.map((id, index) => [id, bits[index]])),
      factorChecks,
      valid,
      mineCount: bits.reduce((sum, bit) => sum + bit, 0),
      priorWeight: Math.exp(priorLogWeight),
      constraintWeight: valid ? 1 : 0,
      logWeight: valid ? priorLogWeight : -Infinity,
      rawWeight: valid ? Math.exp(priorLogWeight) : 0,
      probability: 0,
      interval: [0, 0],
    };
  });

  const normalization = candidates.reduce((sum, candidate) => sum + candidate.rawWeight, 0);
  const maximum = Math.max(...candidates.map(({ logWeight }) => logWeight));
  if (!Number.isFinite(maximum)) {
    throw new Error(`${block.id} の条件付き分布に有効な割当がありません。`);
  }
  const scaledTotal = candidates.reduce((sum, candidate) => (
    sum + Math.exp(candidate.logWeight - maximum)
  ), 0);

  let cumulative = 0;
  candidates.forEach((candidate) => {
    const lower = cumulative;
    candidate.probability = Math.exp(candidate.logWeight - maximum) / scaledTotal;
    cumulative += candidate.probability;
    candidate.interval = [lower, cumulative];
  });

  const randomValue = randomUnitInterval(random);
  let selectedIndex = candidates.findIndex((candidate) => (
    candidate.probability > 0 && randomValue < candidate.interval[1]
  ));
  if (selectedIndex < 0) selectedIndex = candidates.findLastIndex(({ probability }) => probability > 0);
  const selected = candidates[selectedIndex];
  const assignment = cloneAssignment(state.assignment);
  block.variableIds.forEach((variableId) => {
    assignment[variableId] = selected.values[variableId];
  });

  const mineCounts = { ...state.mineCounts };
  graph.variables.forEach((variable) => {
    mineCounts[variable.id] += assignment[variable.id];
  });
  const sampleCount = state.sampleCount + 1;
  let blockCursor = state.blockCursor + 1;
  let blockOrder = [...state.blockOrder];
  if (blockCursor >= state.blocks.length) {
    blockOrder = shuffledBlockOrder(state.blocks.length, random);
    blockCursor = 0;
  }
  const nextState = {
    ...state,
    step: state.step + 1,
    assignment,
    sampleCount,
    mineCounts,
    blockOrder,
    blockCursor,
    lastBlockIndex: blockIndex,
    marginals: empiricalMarginals(graph, assignment, mineCounts, sampleCount),
  };

  return {
    state: nextState,
    trace: {
      step: nextState.step,
      rho: graph.rho,
      blockIndex,
      blockPosition: state.blockCursor + 1,
      block: { ...block, variableIds: [...block.variableIds], sourceFactorIds: [...block.sourceFactorIds] },
      beforeAssignment: cloneAssignment(state.assignment),
      afterAssignment: cloneAssignment(assignment),
      affectedFactorIds: affectedFactors.map(({ id }) => id),
      candidates,
      normalization,
      logNormalization: maximum + Math.log(scaledTotal),
      randomValue,
      selectedIndex,
      sampleCount,
      marginals: Object.fromEntries(
        Object.entries(nextState.marginals).map(([id, values]) => [id, [...values]]),
      ),
    },
  };
}
