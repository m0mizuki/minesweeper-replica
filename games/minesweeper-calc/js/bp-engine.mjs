const OFFSETS = [
  [-1, -1], [-1, 0], [-1, 1],
  [0, -1],             [0, 1],
  [1, -1],  [1, 0],   [1, 1],
];

export const cellKey = (row, col) => `${row},${col}`;

export function adjacentCells(row, col, size = 3) {
  return OFFSETS
    .map(([dr, dc]) => ({ row: row + dr, col: col + dc }))
    .filter(({ row: nextRow, col: nextCol }) => (
      nextRow >= 0 && nextRow < size && nextCol >= 0 && nextCol < size
    ));
}

function normalize(values) {
  const total = values[0] + values[1];
  if (!Number.isFinite(total) || total <= 0) return [0, 0];
  return [values[0] / total, values[1] / total];
}

function cloneMessageMap(messages) {
  return Object.fromEntries(
    Object.entries(messages).map(([key, value]) => [key, [...value]]),
  );
}

function enumerateBits(length) {
  const assignments = [];
  for (let mask = 0; mask < 2 ** length; mask += 1) {
    assignments.push(Array.from({ length }, (_, index) => (mask >> index) & 1));
  }
  return assignments;
}

export function createBoard({ size = 3, rho = 0.3, revealedKeys = [], random = Math.random }) {
  const revealed = new Set(revealedKeys);
  const mines = new Set();
  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col < size; col += 1) {
      const key = cellKey(row, col);
      if (!revealed.has(key) && random() < rho) mines.add(key);
    }
  }
  return { size, rho, revealed, mines };
}

export function createExampleBoard() {
  return {
    size: 3,
    rho: 1 / 3,
    revealed: new Set([cellKey(0, 1), cellKey(1, 0), cellKey(1, 2), cellKey(2, 1)]),
    mines: new Set([cellKey(0, 2), cellKey(2, 0), cellKey(2, 2)]),
  };
}

export function createFactorGraph(board) {
  const cells = [];
  const variables = [];
  const factors = [];
  let variableIndex = 1;
  let factorIndex = 1;

  for (let row = 0; row < board.size; row += 1) {
    for (let col = 0; col < board.size; col += 1) {
      const key = cellKey(row, col);
      const cell = { row, col, key };
      if (board.revealed.has(key)) {
        const clue = adjacentCells(row, col, board.size)
          .filter((neighbor) => board.mines.has(cellKey(neighbor.row, neighbor.col))).length;
        const factor = { ...cell, id: `a${factorIndex}`, index: factorIndex, clue, variableIds: [] };
        factors.push(factor);
        cells.push({ ...factor, kind: "factor" });
        factorIndex += 1;
      } else {
        const variable = { ...cell, id: `x${variableIndex}`, index: variableIndex, factorIds: [] };
        variables.push(variable);
        cells.push({ ...variable, kind: "variable" });
        variableIndex += 1;
      }
    }
  }

  const variableByKey = Object.fromEntries(variables.map((variable) => [variable.key, variable]));
  const edges = [];
  for (const factor of factors) {
    for (const neighbor of adjacentCells(factor.row, factor.col, board.size)) {
      const variable = variableByKey[cellKey(neighbor.row, neighbor.col)];
      if (!variable) continue;
      factor.variableIds.push(variable.id);
      variable.factorIds.push(factor.id);
      edges.push({
        key: `${variable.id}|${factor.id}`,
        variableId: variable.id,
        factorId: factor.id,
      });
    }
  }

  return { size: board.size, cells, variables, factors, edges };
}

export function createBPState(graph) {
  const variableToFactor = {};
  const factorToVariable = {};
  for (const edge of graph.edges) {
    variableToFactor[edge.key] = [0.5, 0.5];
    factorToVariable[edge.key] = [0.5, 0.5];
  }
  return {
    cycle: 0,
    variableToFactor,
    factorToVariable,
    marginals: computeMarginals(graph, factorToVariable),
  };
}

function edgeKey(variableId, factorId) {
  return `${variableId}|${factorId}`;
}

function product(values) {
  return values.reduce((result, value) => result * value, 1);
}

export function computeMarginals(graph, factorToVariable) {
  const marginals = {};
  for (const variable of graph.variables) {
    const raw = [0, 1].map((value) => product(
      variable.factorIds.map((factorId) => factorToVariable[edgeKey(variable.id, factorId)][value]),
    ));
    marginals[variable.id] = normalize(raw);
  }
  return marginals;
}

function updateVariableMessages(graph, previousFactorToVariable) {
  const messages = {};
  const calculations = [];

  for (const edge of graph.edges) {
    const variable = graph.variables.find(({ id }) => id === edge.variableId);
    const sourceFactorIds = variable.factorIds.filter((id) => id !== edge.factorId);
    const rows = [0, 1].map((value) => {
      const terms = sourceFactorIds.map((factorId) => ({
        message: `m_${factorId}→${variable.id}(${value})`,
        value: previousFactorToVariable[edgeKey(variable.id, factorId)][value],
      }));
      return { value, terms, raw: product(terms.map((term) => term.value)) };
    });
    const raw = rows.map((row) => row.raw);
    const normalized = normalize(raw);
    messages[edge.key] = normalized;
    calculations.push({
      direction: "variable-to-factor",
      from: variable.id,
      to: edge.factorId,
      name: `m_${variable.id}→${edge.factorId}`,
      excluded: edge.factorId,
      rows,
      raw,
      normalized,
      normalization: raw[0] + raw[1],
      impossible: normalized[0] === 0 && normalized[1] === 0,
    });
  }
  return { messages, calculations };
}

function updateFactorMessages(graph, variableToFactor) {
  const messages = {};
  const calculations = [];

  for (const edge of graph.edges) {
    const factor = graph.factors.find(({ id }) => id === edge.factorId);
    const otherVariableIds = factor.variableIds.filter((id) => id !== edge.variableId);
    const rows = [0, 1].map((targetValue) => {
      const assignments = enumerateBits(otherVariableIds.length)
        .filter((bits) => targetValue + bits.reduce((sum, bit) => sum + bit, 0) === factor.clue)
        .map((bits) => {
          const terms = bits.map((bit, index) => {
            const variableId = otherVariableIds[index];
            return {
              variableId,
              value: bit,
              message: `m_${variableId}→${factor.id}(${bit})`,
              probability: variableToFactor[edgeKey(variableId, factor.id)][bit],
            };
          });
          return {
            values: Object.fromEntries(otherVariableIds.map((id, index) => [id, bits[index]])),
            terms,
            product: product(terms.map((term) => term.probability)),
          };
        });
      return {
        value: targetValue,
        assignments,
        raw: assignments.reduce((sum, assignment) => sum + assignment.product, 0),
      };
    });
    const raw = rows.map((row) => row.raw);
    const normalized = normalize(raw);
    messages[edge.key] = normalized;
    calculations.push({
      direction: "factor-to-variable",
      from: factor.id,
      to: edge.variableId,
      name: `m_${factor.id}→${edge.variableId}`,
      clue: factor.clue,
      otherVariableIds,
      rows,
      raw,
      normalized,
      normalization: raw[0] + raw[1],
      impossible: normalized[0] === 0 && normalized[1] === 0,
    });
  }
  return { messages, calculations };
}

export function runCycle(graph, state) {
  const variablePhase = updateVariableMessages(graph, state.factorToVariable);
  const factorPhase = updateFactorMessages(graph, variablePhase.messages);
  const nextState = {
    cycle: state.cycle + 1,
    variableToFactor: cloneMessageMap(variablePhase.messages),
    factorToVariable: cloneMessageMap(factorPhase.messages),
    marginals: computeMarginals(graph, factorPhase.messages),
  };
  return {
    state: nextState,
    trace: {
      cycle: nextState.cycle,
      variableCalculations: variablePhase.calculations,
      factorCalculations: factorPhase.calculations,
      marginals: cloneMessageMap(nextState.marginals),
      hasContradiction: [
        ...variablePhase.calculations,
        ...factorPhase.calculations,
      ].some((calculation) => calculation.impossible),
    },
  };
}
