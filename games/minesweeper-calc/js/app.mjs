import {
  cellKey,
  createBoard,
  createExampleBoard,
  createFactorGraph,
  createBPState,
  runCycle,
} from "./bp-engine.mjs";
import {
  createGibbsState,
  runGibbsStep,
} from "./gibbs-engine.mjs";
import {
  idToLatex,
  joinProductLatex,
  messageToLatex,
  messageValueToLatex,
  renderLatex,
  renderLatexSource,
  renderStaticLatex,
  vectorToLatex,
  waitForMathRenderer,
} from "./math-renderer.mjs";

await waitForMathRenderer();

const elements = {
  board: document.querySelector("#board"),
  boardTitle: document.querySelector("#board-title"),
  methodInputs: [...document.querySelectorAll('input[name="inference-method"]')],
  sizeInputs: [...document.querySelectorAll('input[name="board-size"]')],
  rho: document.querySelector("#rho"),
  rhoOutput: document.querySelector("#rho-output"),
  setupPanel: document.querySelector("#setup-panel"),
  runPanel: document.querySelector("#run-panel"),
  startButton: document.querySelector("#start-button"),
  exampleButton: document.querySelector("#example-button"),
  nextButton: document.querySelector("#next-cycle-button"),
  nextButtonLabel: document.querySelector("#next-button-label"),
  nextButtonDetail: document.querySelector("#next-button-detail"),
  editButton: document.querySelector("#edit-button"),
  cycleNumber: document.querySelector("#cycle-number"),
  counterLabel: document.querySelector("#counter-label"),
  modeStatus: document.querySelector("#mode-status"),
  variableCount: document.querySelector("#variable-count"),
  factorCount: document.querySelector("#factor-count"),
  edgeCount: document.querySelector("#edge-count"),
  thirdMetricLabel: document.querySelector("#third-metric-label"),
  modelNoteTitle: document.querySelector("#model-note-title"),
  modelNoteCopy: document.querySelector("#model-note-copy"),
  traceIndex: document.querySelector("#trace-index"),
  messagesTitle: document.querySelector("#messages-title"),
  emptyState: document.querySelector("#empty-state"),
  trace: document.querySelector("#trace"),
  expandButton: document.querySelector("#expand-button"),
  katexDetails: document.querySelector("#katex-details"),
  mathModeLabel: document.querySelector("#math-mode-label"),
  template: document.querySelector("#message-card-template"),
};

let mode = "setup";
let inferenceMethod = "bp";
let boardSize = 3;
const revealedBySize = new Map([
  [2, new Set([cellKey(0, 0)])],
  [3, new Set([cellKey(0, 1), cellKey(1, 0), cellKey(1, 2), cellKey(2, 1)])],
]);
let selectedRevealed = revealedBySize.get(boardSize);
let boardModel = null;
let graph = null;
let bpState = null;
let gibbsState = null;
let history = [];
let useKatexForDetails = true;
const traceByElement = new WeakMap();

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

function renderDetailLatex(element, latex) {
  return useKatexForDetails
    ? renderLatex(element, latex)
    : renderLatexSource(element, latex);
}

function detailMathSpan(latex, className = "") {
  const element = document.createElement("span");
  if (className) element.className = className;
  return renderDetailLatex(element, latex);
}

function setEmptyState(title, message, latex = null) {
  elements.emptyState.querySelector("h3").textContent = title;
  const paragraph = elements.emptyState.querySelector("p");
  paragraph.replaceChildren(document.createTextNode(message));
  if (latex) {
    const action = inferenceMethod === "bp" ? "1サイクル進める" : "1ステップ進める";
    paragraph.append(" ", mathSpan(latex, "math-inline"), ` です。『${action}』で計算を展開します。`);
  }
}

function updateMethodCopy() {
  const isBP = inferenceMethod === "bp";
  elements.methodInputs.forEach((input) => {
    input.checked = input.value === inferenceMethod;
  });
  elements.counterLabel.textContent = isBP ? "cycle" : "step";
  elements.nextButtonLabel.textContent = isBP ? "1サイクル進める" : "1ステップ進める";
  elements.nextButtonDetail.replaceChildren();
  if (isBP) {
    elements.nextButtonDetail.append(
      mathSpan("m_{i\\to a}", "math-inline"),
      " → ",
      mathSpan("m_{a\\to i}", "math-inline"),
    );
  } else {
    elements.nextButtonDetail.append(mathSpan(
      "\\mathbf{x}_{B_t}\\sim P(\\mathbf{x}_{B_t}\\mid\\mathbf{x}_{\\backslash B_t})",
      "math-inline",
    ));
  }
  elements.traceIndex.textContent = isBP ? "02 / MESSAGE TRACE" : "02 / SAMPLING TRACE";
  elements.messagesTitle.textContent = isBP ? "メッセージ計算" : "Blocked Gibbs 計算";
  elements.thirdMetricLabel.textContent = isBP ? "edges" : "blocks";
  elements.modelNoteTitle.textContent = isBP ? "地雷密度の扱い" : "条件付き分布の扱い";
  elements.modelNoteCopy.replaceChildren();
  if (isBP) {
    elements.modelNoteCopy.append(
      mathSpan("\\rho", "math-inline"),
      " は盤面生成のみに使用します。BP の初期メッセージはすべて ",
      mathSpan("[0.5,\\,0.5]", "math-inline"),
      " です。",
    );
  } else {
    elements.modelNoteCopy.append(
      mathSpan("\\rho", "math-inline"),
      " は盤面生成のみに使用します。目的分布は ",
      mathSpan(
        "P(\\mathbf{x}\\mid\\mathbf{c})=\\frac{1}{Z}\\prod_a\\delta\\!\\left(\\sum_{i\\in\\partial a}x_i,c_a\\right)",
        "math-inline",
      ),
      "、ブロック条件付き確率は制約因子の積だけで決まります。",
    );
  }
}

function setMode(nextMode) {
  mode = nextMode;
  const isSetup = mode === "setup";
  elements.setupPanel.hidden = !isSetup;
  elements.runPanel.hidden = isSetup;
  elements.modeStatus.textContent = isSetup
    ? "設定中"
    : inferenceMethod === "bp" ? "BP 計算中" : "Gibbs 計算中";
  elements.modeStatus.classList.toggle("is-running", !isSetup);
  renderBoard();
}

function syncBoardSizeUI() {
  const label = `${boardSize}×${boardSize} 盤面`;
  elements.boardTitle.textContent = label;
  elements.board.setAttribute("aria-label", `${boardSize} × ${boardSize} マインスイーパー盤面`);
  elements.board.setAttribute("aria-rowcount", String(boardSize));
  elements.board.setAttribute("aria-colcount", String(boardSize));
  elements.board.style.setProperty("--board-size", String(boardSize));
  elements.sizeInputs.forEach((input) => {
    input.checked = Number(input.value) === boardSize;
  });
}

function selectBoardSize(nextSize) {
  boardSize = nextSize;
  selectedRevealed = revealedBySize.get(boardSize);
  elements.modeStatus.classList.remove("is-warning");
  elements.modeStatus.textContent = "設定中";
  syncBoardSizeUI();
  renderBoard();
}

function renderBoard() {
  syncBoardSizeUI();
  elements.board.replaceChildren();
  for (let row = 0; row < boardSize; row += 1) {
    for (let col = 0; col < boardSize; col += 1) {
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
          const activeState = inferenceMethod === "bp" ? bpState : gibbsState;
          const marginal = activeState.marginals[modelCell.id] ?? [0.5, 0.5];
          const mineProbability = marginal[1];
          cell.classList.add("variable-cell");
          const currentValue = inferenceMethod === "gibbs"
            ? gibbsState.assignment[modelCell.id]
            : null;
          if (currentValue === 1) cell.classList.add("is-current-mine");
          cell.style.setProperty("--probability", `${mineProbability * 100}%`);
          const label = document.createElement("strong");
          renderLatex(label, idToLatex(modelCell.id));
          const probability = inferenceMethod === "bp"
            ? mathSpan(`P(${idToLatex(modelCell.id)}=1)=${formatNumber(mineProbability)}`)
            : mathSpan(
              gibbsState.sampleCount > 0
                ? `${idToLatex(modelCell.id)}=${currentValue},\\;\\hat P(1)=${formatNumber(mineProbability)}`
                : `${idToLatex(modelCell.id)}=${currentValue}\\;\\text{（初期値）}`,
            );
          cell.append(label, probability);
          cell.setAttribute(
            "aria-label",
            inferenceMethod === "bp"
              ? `未開示マス ${modelCell.id}、地雷確率 ${formatNumber(mineProbability)}`
              : `未開示マス ${modelCell.id}、現在値 ${currentValue}、経験地雷確率 ${formatNumber(mineProbability)}`,
          );
        }
      }
      elements.board.append(cell);
    }
  }
}

function initialize(board) {
  boardModel = board;
  boardSize = board.size;
  selectedRevealed = new Set(board.revealed);
  revealedBySize.set(boardSize, selectedRevealed);
  syncBoardSizeUI();
  graph = createFactorGraph(boardModel);
  bpState = inferenceMethod === "bp" ? createBPState(graph) : null;
  gibbsState = inferenceMethod === "gibbs" ? createGibbsState(graph) : null;
  history = [];
  elements.cycleNumber.textContent = "0";
  elements.variableCount.textContent = graph.variables.length;
  elements.factorCount.textContent = graph.factors.length;
  elements.edgeCount.textContent = inferenceMethod === "bp"
    ? graph.edges.length
    : gibbsState.blocks.length;
  elements.trace.replaceChildren();
  elements.emptyState.hidden = false;
  if (inferenceMethod === "bp" && graph.edges.length) {
    setEmptyState("初期メッセージを設定しました", "すべての有向メッセージは", "[0.5,\\,0.5]");
  } else if (inferenceMethod === "gibbs" && gibbsState.blocks.length) {
    setEmptyState(
      "制約を満たす初期配置を作りました",
      `${gibbsState.blocks.length} 個のブロックを順番に更新します。最初の更新対象は`,
      gibbsState.blocks[0].id,
    );
  } else {
    setEmptyState(
      "更新できる対象がありません",
      inferenceMethod === "bp"
        ? "開示マスと隣接する未開示マスがないため、因子グラフに辺がありません。"
        : "未開示変数がないため、更新するブロックがありません。",
    );
  }
  elements.nextButton.disabled = inferenceMethod === "bp"
    ? graph.edges.length === 0
    : gibbsState.blocks.length === 0;
  elements.expandButton.hidden = true;
  setMode("running");
}

function renderVariableCalculation(calculation) {
  const card = elements.template.content.firstElementChild.cloneNode(true);
  renderDetailLatex(card.querySelector(".message-name"), messageToLatex(calculation.from, calculation.to));
  renderDetailLatex(card.querySelector(".message-vector"), vectorLatex(calculation.normalized));
  const body = card.querySelector(".calculation-lines");

  calculation.rows.forEach((row) => {
    const line = document.createElement("p");
    const expression = row.terms.length
      ? joinProductLatex(row.terms.map((term) => messageValueToLatex(term, term.value, formatNumber)))
      : "1\\;\\text{（空積）}";
    line.append(
      detailMathSpan(messageToLatex(calculation.from, calculation.to, row.value), "formula-key"),
      detailMathSpan(`\\propto ${expression}`),
      detailMathSpan(`=${formatNumber(row.raw)}`, "result-value"),
    );
    body.append(line);
  });
  appendNormalization(body, calculation);
  return card;
}

function renderFactorCalculation(calculation) {
  const card = elements.template.content.firstElementChild.cloneNode(true);
  renderDetailLatex(card.querySelector(".message-name"), messageToLatex(calculation.from, calculation.to));
  renderDetailLatex(card.querySelector(".message-vector"), vectorLatex(calculation.normalized));
  const body = card.querySelector(".calculation-lines");

  const constraint = document.createElement("p");
  constraint.className = "constraint-line";
  const variables = [calculation.to, ...calculation.otherVariableIds].map(idToLatex).join("+");
  renderDetailLatex(constraint, `\\text{制約: }${variables}=${calculation.clue}`);
  body.append(constraint);

  calculation.rows.forEach((row) => {
    const group = document.createElement("div");
    group.className = "assignment-group";
    const heading = document.createElement("p");
    heading.append(
      detailMathSpan(messageToLatex(calculation.from, calculation.to, row.value), "formula-key"),
      detailMathSpan("\\propto\\;\\text{有効な割当の和}"),
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
          ? joinProductLatex(assignment.terms.map((term) => (
            messageValueToLatex(term, term.probability, formatNumber)
          )))
          : "1";
        assignmentLine.append(
          detailMathSpan(values),
          detailMathSpan(productExpression),
          detailMathSpan(`=${formatNumber(assignment.product)}`, "result-value"),
        );
        group.append(assignmentLine);
      });
      const sumLine = document.createElement("p");
      sumLine.className = "sum-line";
      renderDetailLatex(sumLine, `\\text{合計}=${formatNumber(row.raw)}`);
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
  renderDetailLatex(
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

function createCycleContent(trace) {
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
    renderDetailLatex(item, `${idToLatex(variable.id)}:\\;${vectorLatex(trace.marginals[variable.id])}`);
    values.append(item);
  });
  marginal.append(values);
  content.append(marginal);
  return content;
}

function assignmentLatex(variableIds, values) {
  return variableIds.map((id) => `${idToLatex(id)}=${values[id]}`).join(",\\;");
}

function createGibbsStepContent(trace) {
  const content = document.createElement("div");
  content.className = "cycle-content gibbs-content";

  const conditioning = document.createElement("section");
  conditioning.className = "phase gibbs-conditioning";
  const conditioningHeading = document.createElement("div");
  conditioningHeading.className = "phase-heading";
  const conditioningLabel = document.createElement("div");
  const conditioningChip = document.createElement("span");
  conditioningChip.className = "phase-chip";
  conditioningChip.textContent = "STEP 1";
  const conditioningTitle = document.createElement("h4");
  conditioningTitle.textContent = `${trace.block.id} を更新対象にする`;
  conditioningLabel.append(conditioningChip, conditioningTitle);
  const conditioningFormula = document.createElement("code");
  renderDetailLatex(
    conditioningFormula,
    `${idToLatex(trace.block.id)}=\\{${trace.block.variableIds.map(idToLatex).join(",\\;")}\\}`,
  );
  conditioningHeading.append(conditioningLabel, conditioningFormula);
  conditioning.append(conditioningHeading);

  const context = document.createElement("div");
  context.className = "gibbs-context";
  const before = document.createElement("p");
  before.append(
    document.createTextNode("更新前の標本 "),
    detailMathSpan(assignmentLatex(graph.variables.map(({ id }) => id), trace.beforeAssignment)),
  );
  const factors = document.createElement("p");
  factors.append(
    document.createTextNode("再評価する制約 "),
    detailMathSpan(
      trace.affectedFactorIds.length
        ? trace.affectedFactorIds.map(idToLatex).join(",\\;")
        : "\\text{なし}",
    ),
  );
  context.append(before, factors);
  conditioning.append(context);
  content.append(conditioning);

  const enumeration = document.createElement("section");
  enumeration.className = "phase gibbs-enumeration";
  const enumerationHeading = document.createElement("div");
  enumerationHeading.className = "phase-heading";
  const enumerationLabel = document.createElement("div");
  const enumerationChip = document.createElement("span");
  enumerationChip.className = "phase-chip";
  enumerationChip.textContent = "STEP 2";
  const enumerationTitle = document.createElement("h4");
  enumerationTitle.textContent = `${trace.candidates.length} 通りを列挙して重みを計算`;
  enumerationLabel.append(enumerationChip, enumerationTitle);
  const enumerationFormula = document.createElement("code");
  renderDetailLatex(
    enumerationFormula,
    "P(\\mathbf{x}_B\\mid\\mathbf{x}_{\\backslash B})\\propto\\prod_{a:\\,\\partial a\\cap B\\ne\\emptyset}\\delta\\!\\left(\\sum_{i\\in\\partial a}x_i,c_a\\right)",
  );
  enumerationHeading.append(enumerationLabel, enumerationFormula);
  enumeration.append(enumerationHeading);

  const candidateGrid = document.createElement("div");
  candidateGrid.className = "candidate-grid";
  trace.candidates.forEach((candidate, index) => {
    const card = document.createElement("article");
    card.className = "candidate-card";
    if (!candidate.valid) card.classList.add("is-invalid");
    if (index === trace.selectedIndex) card.classList.add("is-selected");

    const header = document.createElement("header");
    const assignment = document.createElement("code");
    renderDetailLatex(assignment, assignmentLatex(trace.block.variableIds, candidate.values));
    const status = document.createElement("span");
    status.className = "candidate-status";
    status.textContent = index === trace.selectedIndex
      ? "採択"
      : candidate.valid ? "有効" : "制約違反";
    header.append(assignment, status);

    const body = document.createElement("div");
    body.className = "candidate-body";
    const checks = document.createElement("p");
    checks.className = "factor-checks";
    if (candidate.factorChecks.length) {
      candidate.factorChecks.forEach((check, checkIndex) => {
        if (checkIndex) checks.append(" · ");
        checks.append(
          detailMathSpan(`${idToLatex(check.factorId)}:\\;${check.sum}${check.satisfied ? "=" : "\\ne"}${check.clue}`),
        );
      });
    } else {
      checks.textContent = "このブロックに接続する数字制約はありません";
    }
    const weight = document.createElement("p");
    weight.append(
      detailMathSpan(
        `w=\\prod_{a:\\,\\partial a\\cap B\\ne\\emptyset}\\delta_a=${formatNumber(candidate.rawWeight)}`,
      ),
    );
    const probability = document.createElement("p");
    probability.className = "candidate-probability";
    probability.append(
      detailMathSpan(`P=${formatNumber(candidate.probability)}`),
      detailMathSpan(
        `I=[${formatNumber(candidate.interval[0])},\\;${formatNumber(candidate.interval[1])})`,
      ),
    );
    body.append(checks, weight, probability);
    card.append(header, body);
    candidateGrid.append(card);
  });
  enumeration.append(candidateGrid);
  content.append(enumeration);

  const draw = document.createElement("section");
  draw.className = "phase gibbs-draw";
  const drawHeading = document.createElement("div");
  drawHeading.className = "phase-heading";
  const drawLabel = document.createElement("div");
  const drawChip = document.createElement("span");
  drawChip.className = "phase-chip";
  drawChip.textContent = "STEP 3";
  const drawTitle = document.createElement("h4");
  drawTitle.textContent = "正規化し、乱数区間から採択";
  drawLabel.append(drawChip, drawTitle);
  const drawFormula = document.createElement("code");
  renderDetailLatex(drawFormula, `Z_B=${formatNumber(trace.normalization)},\\quad u=${formatNumber(trace.randomValue)}`);
  drawHeading.append(drawLabel, drawFormula);
  draw.append(drawHeading);
  const selected = trace.candidates[trace.selectedIndex];
  const decision = document.createElement("div");
  decision.className = "gibbs-decision";
  decision.append(
    detailMathSpan(
      `${formatNumber(selected.interval[0])}\\le u<${formatNumber(selected.interval[1])}`,
    ),
    document.createTextNode(" より "),
    detailMathSpan(assignmentLatex(trace.block.variableIds, selected.values)),
    document.createTextNode(" を採択"),
  );
  draw.append(decision);
  content.append(draw);

  const result = document.createElement("section");
  result.className = "marginal-strip gibbs-result";
  const resultHeading = document.createElement("div");
  const resultChip = document.createElement("span");
  resultChip.className = "phase-chip";
  resultChip.textContent = "RESULT";
  const resultTitle = document.createElement("h4");
  resultTitle.textContent = `Step ${trace.step} 後の標本と経験確率（n=${trace.sampleCount}）`;
  resultHeading.append(resultChip, resultTitle);
  result.append(resultHeading);
  const values = document.createElement("div");
  values.className = "marginal-values";
  graph.variables.forEach((variable) => {
    const item = document.createElement("span");
    renderDetailLatex(
      item,
      `${idToLatex(variable.id)}=${trace.afterAssignment[variable.id]},\\;\\hat P(1)=${formatNumber(trace.marginals[variable.id][1])}`,
    );
    values.append(item);
  });
  result.append(values);
  content.append(result);
  return content;
}

function mountTraceContent(details) {
  if (details.querySelector(":scope > .cycle-content")) return;
  const entry = traceByElement.get(details);
  if (!entry) return;
  details.append(entry.method === "bp"
    ? createCycleContent(entry.trace)
    : createGibbsStepContent(entry.trace));
}

function unmountTraceContent(details) {
  details.querySelector(":scope > .cycle-content")?.remove();
}

function rerenderExpandedCycles() {
  elements.trace.querySelectorAll("details[open]").forEach((details) => {
    unmountTraceContent(details);
    mountTraceContent(details);
  });
}

function updateExpandButtonLabel() {
  const details = [...elements.trace.querySelectorAll("details")];
  if (!details.length) return;
  elements.expandButton.textContent = details.every((item) => item.open)
    ? "すべて折りたたむ"
    : "すべて展開";
}

function setCycleExpanded(details, expanded) {
  details.open = expanded;
  if (expanded) mountTraceContent(details);
  else unmountTraceContent(details);
}

function renderBPTrace(trace) {
  const details = document.createElement("details");
  details.className = "cycle-trace";
  traceByElement.set(details, { method: "bp", trace });

  const summary = document.createElement("summary");
  summary.innerHTML = `<span>Cycle ${trace.cycle}</span><span>${trace.variableCalculations.length + trace.factorCalculations.length} messages${trace.hasContradiction ? " · 矛盾あり" : ""}</span>`;
  details.append(summary);

  details.addEventListener("toggle", () => {
    if (details.open) mountTraceContent(details);
    else unmountTraceContent(details);
    updateExpandButtonLabel();
  });
  setCycleExpanded(details, true);
  return details;
}

function renderGibbsTrace(trace) {
  const details = document.createElement("details");
  details.className = "cycle-trace gibbs-trace";
  traceByElement.set(details, { method: "gibbs", trace });
  const validCount = trace.candidates.filter(({ valid }) => valid).length;
  const summary = document.createElement("summary");
  summary.innerHTML = `<span>Step ${trace.step} · ${trace.block.id}</span><span>${validCount} / ${trace.candidates.length} valid · u=${formatNumber(trace.randomValue)}</span>`;
  details.append(summary);
  details.addEventListener("toggle", () => {
    if (details.open) mountTraceContent(details);
    else unmountTraceContent(details);
    updateExpandButtonLabel();
  });
  setCycleExpanded(details, true);
  return details;
}

function advance() {
  const result = inferenceMethod === "bp"
    ? runCycle(graph, bpState)
    : runGibbsStep(graph, gibbsState);
  if (inferenceMethod === "bp") bpState = result.state;
  else gibbsState = result.state;
  history.push(result.trace);
  elements.cycleNumber.textContent = String(
    inferenceMethod === "bp" ? bpState.cycle : gibbsState.step,
  );
  elements.emptyState.hidden = true;
  elements.expandButton.hidden = false;

  elements.trace.querySelectorAll("details").forEach((item) => setCycleExpanded(item, false));
  elements.trace.prepend(
    inferenceMethod === "bp" ? renderBPTrace(result.trace) : renderGibbsTrace(result.trace),
  );
  updateExpandButtonLabel();
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
    size: boardSize,
    rho: Number(elements.rho.value),
    revealedKeys: selectedRevealed,
  }));
}

elements.methodInputs.forEach((input) => {
  input.addEventListener("change", () => {
    if (!input.checked) return;
    inferenceMethod = input.value;
    updateMethodCopy();
    elements.modeStatus.textContent = "設定中";
    elements.modeStatus.classList.remove("is-warning");
  });
});
elements.sizeInputs.forEach((input) => {
  input.addEventListener("change", () => {
    if (input.checked) selectBoardSize(Number(input.value));
  });
});
elements.rho.addEventListener("input", () => {
  elements.rhoOutput.textContent = Number(elements.rho.value).toFixed(2);
});
elements.katexDetails.addEventListener("change", () => {
  useKatexForDetails = elements.katexDetails.checked;
  elements.mathModeLabel.textContent = useKatexForDetails ? "KaTeX" : "軽量";
  rerenderExpandedCycles();
});
elements.startButton.addEventListener("click", beginWithCurrentSetup);
elements.exampleButton.addEventListener("click", () => {
  const example = createExampleBoard();
  boardSize = example.size;
  selectedRevealed = new Set(example.revealed);
  revealedBySize.set(boardSize, selectedRevealed);
  elements.rho.value = example.rho.toFixed(2);
  elements.rhoOutput.textContent = example.rho.toFixed(2);
  initialize(example);
});
elements.nextButton.addEventListener("click", advance);
elements.editButton.addEventListener("click", () => {
  history = [];
  elements.trace.replaceChildren();
  elements.emptyState.hidden = false;
  setEmptyState(
    "計算の準備をします",
    "左の盤面で推論手法、ρ、開示マスを設定すると計算状態が作られます。",
  );
  elements.expandButton.hidden = true;
  elements.cycleNumber.textContent = "0";
  setMode("setup");
});
elements.expandButton.addEventListener("click", () => {
  const details = [...elements.trace.querySelectorAll("details")];
  const shouldOpen = details.some((item) => !item.open);
  details.forEach((item) => setCycleExpanded(item, shouldOpen));
  updateExpandButtonLabel();
});

updateMethodCopy();
renderStaticLatex();
renderBoard();
