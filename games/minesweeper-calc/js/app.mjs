import {
  cellKey,
  createBoard,
  createExampleBoard,
  createFactorGraph,
  createBPState,
  runCycle,
} from "./bp-engine.mjs";
import {
  idToLatex,
  joinProductLatex,
  messageToLatex,
  renderLatex,
  renderStaticLatex,
  vectorToLatex,
  waitForMathRenderer,
} from "./math-renderer.mjs";

await waitForMathRenderer();

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
const vectorLatex = (vector) => vectorToLatex(vector, formatNumber);

function mathSpan(latex, className = "") {
  const element = document.createElement("span");
  if (className) element.className = className;
  return renderLatex(element, latex);
}

function setEmptyState(title, message, latex = null) {
  elements.emptyState.querySelector("h3").textContent = title;
  const paragraph = elements.emptyState.querySelector("p");
  paragraph.replaceChildren(document.createTextNode(message));
  if (latex) {
    paragraph.append(" ", mathSpan(latex, "math-inline"), " です。『1サイクル進める』で更新式を展開します。");
  }
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
          const clue = document.createElement("strong");
          clue.textContent = modelCell.clue;
          cell.append(clue, mathSpan(idToLatex(modelCell.id)));
          cell.setAttribute("aria-label", `開示マス ${modelCell.id}、数字 ${modelCell.clue}`);
        } else {
          const marginal = bpState.marginals[modelCell.id] ?? [0.5, 0.5];
          const mineProbability = marginal[1];
          cell.classList.add("variable-cell");
          cell.style.setProperty("--probability", `${mineProbability * 100}%`);
          const label = document.createElement("strong");
          renderLatex(label, idToLatex(modelCell.id));
          const probability = mathSpan(`P(${idToLatex(modelCell.id)}=1)=${formatNumber(mineProbability)}`);
          cell.append(label, probability);
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
  if (graph.edges.length) {
    setEmptyState("初期メッセージを設定しました", "すべての有向メッセージは", "[0.5,\\,0.5]");
  } else {
    setEmptyState("更新できるメッセージがありません", "開示マスと隣接する未開示マスがないため、因子グラフに辺がありません。");
  }
  elements.nextButton.disabled = graph.edges.length === 0;
  elements.expandButton.hidden = true;
  setMode("running");
}

function renderVariableCalculation(calculation) {
  const card = elements.template.content.firstElementChild.cloneNode(true);
  renderLatex(card.querySelector(".message-name"), messageToLatex(calculation.from, calculation.to));
  renderLatex(card.querySelector(".message-vector"), vectorLatex(calculation.normalized));
  const body = card.querySelector(".calculation-lines");

  calculation.rows.forEach((row) => {
    const line = document.createElement("p");
    const expression = row.terms.length
      ? joinProductLatex(row.terms.map((term) => `${messageToLatex(term.from, term.to, term.argument)}=${formatNumber(term.value)}`))
      : "1\\;\\text{（空積）}";
    line.append(
      mathSpan(messageToLatex(calculation.from, calculation.to, row.value), "formula-key"),
      mathSpan(`\\propto ${expression}`),
      mathSpan(`=${formatNumber(row.raw)}`, "result-value"),
    );
    body.append(line);
  });
  appendNormalization(body, calculation);
  return card;
}

function renderFactorCalculation(calculation) {
  const card = elements.template.content.firstElementChild.cloneNode(true);
  renderLatex(card.querySelector(".message-name"), messageToLatex(calculation.from, calculation.to));
  renderLatex(card.querySelector(".message-vector"), vectorLatex(calculation.normalized));
  const body = card.querySelector(".calculation-lines");

  const constraint = document.createElement("p");
  constraint.className = "constraint-line";
  const variables = [calculation.to, ...calculation.otherVariableIds].map(idToLatex).join("+");
  renderLatex(constraint, `\\text{制約: }${variables}=${calculation.clue}`);
  body.append(constraint);

  calculation.rows.forEach((row) => {
    const group = document.createElement("div");
    group.className = "assignment-group";
    const heading = document.createElement("p");
    heading.append(
      mathSpan(messageToLatex(calculation.from, calculation.to, row.value), "formula-key"),
      mathSpan("\\propto\\;\\text{有効な割当の和}"),
    );
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
          ? `\\left(${calculation.otherVariableIds.map((id) => `${idToLatex(id)}=${assignment.values[id]}`).join(",\\;")}\\right)`
          : "\\text{（他変数なし）}";
        const productExpression = assignment.terms.length
          ? joinProductLatex(assignment.terms.map((term) => formatNumber(term.probability)))
          : "1";
        assignmentLine.append(
          mathSpan(values),
          mathSpan(productExpression),
          mathSpan(`=${formatNumber(assignment.product)}`, "result-value"),
        );
        group.append(assignmentLine);
      });
      const sumLine = document.createElement("p");
      sumLine.className = "sum-line";
      renderLatex(sumLine, `\\text{合計}=${formatNumber(row.raw)}`);
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
  renderLatex(
    result,
    calculation.impossible
      ? "Z=0\\quad\\text{— この局所制約は矛盾しています}"
      : `Z=${formatNumber(calculation.normalization)}\\quad\\Longrightarrow\\quad ${vectorLatex(calculation.normalized)}`,
  );
  body.append(result);
}

function createPhase(title, formula, calculations, kind) {
  const section = document.createElement("section");
  section.className = `phase phase-${kind}`;
  const heading = document.createElement("div");
  heading.className = "phase-heading";
  const headingLabel = document.createElement("div");
  const chip = document.createElement("span");
  chip.className = "phase-chip";
  chip.textContent = kind === "variable" ? "STEP 1" : "STEP 2";
  const headingTitle = document.createElement("h4");
  headingTitle.textContent = title;
  const formulaElement = document.createElement("code");
  renderLatex(formulaElement, formula);
  headingLabel.append(chip, headingTitle);
  heading.append(headingLabel, formulaElement);
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
    "m_{i\\to a}(x_i)\\propto\\prod_{b\\in\\partial i\\setminus a}m_{b\\to i}(x_i)",
    trace.variableCalculations,
    "variable",
  ));
  content.append(createPhase(
    "因子 → 変数",
    "m_{a\\to i}(x_i)\\propto\\sum_{\\mathbf{x}_{\\partial a\\setminus i}}\\delta\\!\\left(x_i+\\sum_{j\\in\\partial a\\setminus i}x_j,c_a\\right)\\prod_{j\\in\\partial a\\setminus i}m_{j\\to a}(x_j)",
    trace.factorCalculations,
    "factor",
  ));

  const marginal = document.createElement("section");
  marginal.className = "marginal-strip";
  const marginalHeading = document.createElement("div");
  const marginalChip = document.createElement("span");
  marginalChip.className = "phase-chip";
  marginalChip.textContent = "RESULT";
  const marginalTitle = document.createElement("h4");
  marginalTitle.textContent = `Cycle ${trace.cycle} 後の周辺確率`;
  marginalHeading.append(marginalChip, marginalTitle);
  marginal.append(marginalHeading);
  const values = document.createElement("div");
  values.className = "marginal-values";
  graph.variables.forEach((variable) => {
    const item = document.createElement("span");
    renderLatex(item, `${idToLatex(variable.id)}:\\;${vectorLatex(trace.marginals[variable.id])}`);
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
  setEmptyState("計算の準備をします", "左の盤面で ρ と開示マスを設定すると、因子グラフと初期メッセージが作られます。");
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

renderStaticLatex();
renderBoard();
