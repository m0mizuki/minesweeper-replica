import {
  cellKey,
  createBoard,
  createExampleBoard,
  createFactorGraph,
  createBPState,
  runCycle,
} from "./bp-engine.mjs";

const elements = {
  board: document.querySelector("#board"),
  rho: document.querySelector("#rho"),
  rhoOutput: document.querySelector("#rho-output"),
  setupPanel: document.querySelector("#setup-panel"),
  runPanel: document.querySelector("#run-panel"),
  startButton: document.querySelector("#start-button"),
  exampleButton: document.querySelector("#example-button"),
  nextButton: document.querySelector("#next-cycle-button"),
  editButton: document.querySelector("#edit-button"),
  cycleNumber: document.querySelector("#cycle-number"),
  modeStatus: document.querySelector("#mode-status"),
  variableCount: document.querySelector("#variable-count"),
  factorCount: document.querySelector("#factor-count"),
  edgeCount: document.querySelector("#edge-count"),
  emptyState: document.querySelector("#empty-state"),
  trace: document.querySelector("#trace"),
  expandButton: document.querySelector("#expand-button"),
  template: document.querySelector("#message-card-template"),
};

let mode = "setup";
let selectedRevealed = new Set([cellKey(0, 1), cellKey(1, 0), cellKey(1, 2), cellKey(2, 1)]);
let boardModel = null;
let graph = null;
let bpState = null;
let history = [];

const formatNumber = (value) => {
  if (value === 0 || value === 1) return String(value);
  return Number(value.toFixed(6)).toString();
};

const vectorText = (vector) => `[${formatNumber(vector[0])}, ${formatNumber(vector[1])}]`;

function subscriptLabel(id) {
  return `${id[0]}<sub>${id.slice(1)}</sub>`;
}

function setMode(nextMode) {
  mode = nextMode;
  const isSetup = mode === "setup";
  elements.setupPanel.hidden = !isSetup;
  elements.runPanel.hidden = isSetup;
  elements.modeStatus.textContent = isSetup ? "設定中" : "計算中";
  elements.modeStatus.classList.toggle("is-running", !isSetup);
  renderBoard();
}

function renderBoard() {
  elements.board.replaceChildren();
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      const key = cellKey(row, col);
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "board-cell";
      cell.setAttribute("role", "gridcell");

      if (mode === "setup") {
        const isRevealed = selectedRevealed.has(key);
        cell.classList.toggle("setup-revealed", isRevealed);
        cell.innerHTML = isRevealed
          ? "<span class=\"setup-mark\">OPEN</span><span class=\"setup-label\">開示</span>"
          : "<span class=\"setup-mark\">?</span><span class=\"setup-label\">未開示</span>";
        cell.setAttribute("aria-pressed", String(isRevealed));
        cell.setAttribute("aria-label", `${row + 1}行${col + 1}列を${isRevealed ? "未開示" : "開示"}にする`);
        cell.addEventListener("click", () => {
          if (selectedRevealed.has(key)) selectedRevealed.delete(key);
          else selectedRevealed.add(key);
          renderBoard();
        });
      } else {
        const modelCell = graph.cells.find((candidate) => candidate.key === key);
        cell.disabled = true;
        if (modelCell.kind === "factor") {
          cell.classList.add("factor-cell");
          cell.innerHTML = `<strong>${modelCell.clue}</strong><span>${subscriptLabel(modelCell.id)}</span>`;
          cell.setAttribute("aria-label", `開示マス ${modelCell.id}、数字 ${modelCell.clue}`);
        } else {
          const marginal = bpState.marginals[modelCell.id] ?? [0.5, 0.5];
          const mineProbability = marginal[1];
          cell.classList.add("variable-cell");
          cell.style.setProperty("--probability", `${mineProbability * 100}%`);
          cell.innerHTML = `<strong>${subscriptLabel(modelCell.id)}</strong><span>P(1) ${formatNumber(mineProbability)}</span>`;
          cell.setAttribute("aria-label", `未開示マス ${modelCell.id}、地雷確率 ${formatNumber(mineProbability)}`);
        }
      }
      elements.board.append(cell);
    }
  }
}

function initialize(board) {
  boardModel = board;
  graph = createFactorGraph(boardModel);
  bpState = createBPState(graph);
  history = [];
  elements.cycleNumber.textContent = "0";
  elements.variableCount.textContent = graph.variables.length;
  elements.factorCount.textContent = graph.factors.length;
  elements.edgeCount.textContent = graph.edges.length;
  elements.trace.replaceChildren();
  elements.emptyState.hidden = false;
  elements.emptyState.querySelector("h3").textContent = graph.edges.length
    ? "初期メッセージを設定しました"
    : "更新できるメッセージがありません";
  elements.emptyState.querySelector("p").textContent = graph.edges.length
    ? "すべての有向メッセージは [0.5, 0.5] です。「1サイクル進める」で更新式を展開します。"
    : "開示マスと隣接する未開示マスがないため、因子グラフに辺がありません。";
  elements.nextButton.disabled = graph.edges.length === 0;
  elements.expandButton.hidden = true;
  setMode("running");
}

function renderVariableCalculation(calculation) {
  const card = elements.template.content.firstElementChild.cloneNode(true);
  card.querySelector(".message-name").textContent = calculation.name;
  card.querySelector(".message-vector").textContent = vectorText(calculation.normalized);
  const body = card.querySelector(".calculation-lines");

  calculation.rows.forEach((row) => {
    const line = document.createElement("p");
    const expression = row.terms.length
      ? row.terms.map((term) => `${term.message} = ${formatNumber(term.value)}`).join(" × ")
      : "1（空積）";
    line.innerHTML = `<span class="formula-key">${calculation.name}(${row.value})</span><span> ∝ ${expression}</span><b>= ${formatNumber(row.raw)}</b>`;
    body.append(line);
  });
  appendNormalization(body, calculation);
  return card;
}

function renderFactorCalculation(calculation) {
  const card = elements.template.content.firstElementChild.cloneNode(true);
  card.querySelector(".message-name").textContent = calculation.name;
  card.querySelector(".message-vector").textContent = vectorText(calculation.normalized);
  const body = card.querySelector(".calculation-lines");

  const constraint = document.createElement("p");
  constraint.className = "constraint-line";
  const variables = [calculation.to, ...calculation.otherVariableIds].join(" + ");
  constraint.textContent = `制約: ${variables} = ${calculation.clue}`;
  body.append(constraint);

  calculation.rows.forEach((row) => {
    const group = document.createElement("div");
    group.className = "assignment-group";
    const heading = document.createElement("p");
    heading.innerHTML = `<span class="formula-key">${calculation.name}(${row.value})</span><span> ∝ 有効な割当の和</span>`;
    group.append(heading);

    if (row.assignments.length === 0) {
      const empty = document.createElement("p");
      empty.className = "assignment-row no-assignment";
      empty.textContent = "有効な割当なし → 0";
      group.append(empty);
    } else {
      row.assignments.forEach((assignment) => {
        const assignmentLine = document.createElement("p");
        assignmentLine.className = "assignment-row";
        const values = calculation.otherVariableIds.length
          ? `(${calculation.otherVariableIds.map((id) => `${id}=${assignment.values[id]}`).join(", ")})`
          : "(他変数なし)";
        const productExpression = assignment.terms.length
          ? assignment.terms.map((term) => formatNumber(term.probability)).join(" × ")
          : "1";
        assignmentLine.innerHTML = `<span>${values}</span><span>${productExpression}</span><b>${formatNumber(assignment.product)}</b>`;
        group.append(assignmentLine);
      });
      const sumLine = document.createElement("p");
      sumLine.className = "sum-line";
      sumLine.textContent = `合計 = ${formatNumber(row.raw)}`;
      group.append(sumLine);
    }
    body.append(group);
  });
  appendNormalization(body, calculation);
  return card;
}

function appendNormalization(body, calculation) {
  const result = document.createElement("p");
  result.className = calculation.impossible ? "normalization is-error" : "normalization";
  result.innerHTML = calculation.impossible
    ? "Z = 0 — この局所制約は矛盾しています"
    : `Z = ${formatNumber(calculation.normalization)}　→　<strong>${vectorText(calculation.normalized)}</strong>`;
  body.append(result);
}

function createPhase(title, formula, calculations, kind) {
  const section = document.createElement("section");
  section.className = `phase phase-${kind}`;
  const heading = document.createElement("div");
  heading.className = "phase-heading";
  heading.innerHTML = `<div><span class="phase-chip">${kind === "variable" ? "STEP 1" : "STEP 2"}</span><h4>${title}</h4></div><code>${formula}</code>`;
  section.append(heading);
  const grid = document.createElement("div");
  grid.className = "message-grid";
  calculations.forEach((calculation) => {
    grid.append(kind === "variable"
      ? renderVariableCalculation(calculation)
      : renderFactorCalculation(calculation));
  });
  section.append(grid);
  return section;
}

function renderTrace(trace) {
  const details = document.createElement("details");
  details.className = "cycle-trace";
  details.open = true;
  const summary = document.createElement("summary");
  summary.innerHTML = `<span>Cycle ${trace.cycle}</span><span>${trace.variableCalculations.length + trace.factorCalculations.length} messages${trace.hasContradiction ? " · 矛盾あり" : ""}</span>`;
  details.append(summary);

  const content = document.createElement("div");
  content.className = "cycle-content";
  content.append(createPhase(
    "変数 → 因子",
    "mᵢ→ₐ(xᵢ) ∝ ∏ᵦ∈∂ᵢ∖ₐ mᵦ→ᵢ(xᵢ)",
    trace.variableCalculations,
    "variable",
  ));
  content.append(createPhase(
    "因子 → 変数",
    "mₐ→ᵢ(xᵢ) ∝ Σₓ∂ₐ∖ᵢ δ(xᵢ + Σⱼ∈∂ₐ∖ᵢ xⱼ, cₐ) ∏ⱼ∈∂ₐ∖ᵢ mⱼ→ₐ(xⱼ)",
    trace.factorCalculations,
    "factor",
  ));

  const marginal = document.createElement("section");
  marginal.className = "marginal-strip";
  marginal.innerHTML = `<div><span class="phase-chip">RESULT</span><h4>Cycle ${trace.cycle} 後の周辺確率</h4></div>`;
  const values = document.createElement("div");
  values.className = "marginal-values";
  graph.variables.forEach((variable) => {
    const item = document.createElement("span");
    item.innerHTML = `${subscriptLabel(variable.id)} <b>${vectorText(trace.marginals[variable.id])}</b>`;
    values.append(item);
  });
  marginal.append(values);
  content.append(marginal);
  details.append(content);
  return details;
}

function advanceCycle() {
  const result = runCycle(graph, bpState);
  bpState = result.state;
  history.push(result.trace);
  elements.cycleNumber.textContent = String(bpState.cycle);
  elements.emptyState.hidden = true;
  elements.expandButton.hidden = false;

  elements.trace.querySelectorAll("details").forEach((item) => { item.open = false; });
  elements.trace.prepend(renderTrace(result.trace));
  renderBoard();
  elements.trace.querySelector("summary")?.focus();
}

function beginWithCurrentSetup() {
  if (selectedRevealed.size === 0) {
    elements.modeStatus.textContent = "開示マスを選択してください";
    elements.modeStatus.classList.add("is-warning");
    return;
  }
  elements.modeStatus.classList.remove("is-warning");
  initialize(createBoard({
    size: 3,
    rho: Number(elements.rho.value),
    revealedKeys: selectedRevealed,
  }));
}

elements.rho.addEventListener("input", () => {
  elements.rhoOutput.textContent = Number(elements.rho.value).toFixed(2);
});
elements.startButton.addEventListener("click", beginWithCurrentSetup);
elements.exampleButton.addEventListener("click", () => {
  const example = createExampleBoard();
  selectedRevealed = new Set(example.revealed);
  elements.rho.value = example.rho.toFixed(2);
  elements.rhoOutput.textContent = example.rho.toFixed(2);
  initialize(example);
});
elements.nextButton.addEventListener("click", advanceCycle);
elements.editButton.addEventListener("click", () => {
  history = [];
  elements.trace.replaceChildren();
  elements.emptyState.hidden = false;
  elements.emptyState.querySelector("h3").textContent = "計算の準備をします";
  elements.emptyState.querySelector("p").textContent = "左の盤面で ρ と開示マスを設定すると、因子グラフと初期メッセージが作られます。";
  elements.expandButton.hidden = true;
  elements.cycleNumber.textContent = "0";
  setMode("setup");
});
elements.expandButton.addEventListener("click", () => {
  const details = [...elements.trace.querySelectorAll("details")];
  const shouldOpen = details.some((item) => !item.open);
  details.forEach((item) => { item.open = shouldOpen; });
  elements.expandButton.textContent = shouldOpen ? "すべて折りたたむ" : "すべて展開";
});

renderBoard();
