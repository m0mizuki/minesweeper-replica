const boardElement = document.querySelector("#board");
const flagCounter = document.querySelector("#flag-counter");
const timerElement = document.querySelector("#timer");
const faceButton = document.querySelector("#new-game");
const solveButton = document.querySelector("#solve-bp");
const rhoInput = document.querySelector("#rho");
const rhoValue = document.querySelector("#rho-value");
const sizeInput = document.querySelector("#size");
const bpState = document.querySelector("#bp-state");
const message = document.querySelector("#message");
const diagStatus = document.querySelector("#diag-status");
const diagIterations = document.querySelector("#diag-iterations");
const diagResidual = document.querySelector("#diag-residual");
const diagFactors = document.querySelector("#diag-factors");

let game = null;
let timerId = null;
let localStart = null;
let isSolving = false;

function formatCounter(value) {
  return String(Math.max(0, Math.min(999, value))).padStart(3, "0");
}

function updateTimer() {
  if (!game || !game.started || ["won", "lost"].includes(game.status)) {
    timerElement.textContent = formatCounter(game?.elapsed ?? 0);
    return;
  }
  timerElement.textContent = formatCounter(Math.floor((Date.now() - localStart) / 1000));
}

async function postJson(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch (_) {
      // Keep the HTTP status when the response is not JSON.
    }
    throw new Error(detail);
  }
  return response.json();
}

function setGame(nextGame) {
  const wasStarted = game?.started;
  game = nextGame;
  if (game.started && !wasStarted) {
    localStart = Date.now() - game.elapsed * 1000;
    timerId = window.setInterval(updateTimer, 250);
  }
  if (!game.started || ["won", "lost"].includes(game.status)) {
    window.clearInterval(timerId);
    timerId = null;
  }
  render();
}

function clearDiagnostics() {
  diagStatus.textContent = "—";
  diagIterations.textContent = "—";
  diagResidual.textContent = "—";
  diagFactors.textContent = "—";
  bpState.textContent = "READY";
  bpState.className = "state";
}

async function newGame() {
  window.clearInterval(timerId);
  timerId = null;
  localStart = null;
  message.textContent = "";
  clearDiagnostics();
  try {
    setGame(await postJson("/api/new", {
      size: Number(sizeInput.value),
      rho: Number(rhoInput.value),
    }));
  } catch (error) {
    message.textContent = error.message;
  }
}

async function reveal(row, col) {
  if (!game || ["won", "lost"].includes(game.status)) return;
  faceButton.classList.remove("pressed");
  try {
    setGame(await postJson(`/api/${game.game_id}/reveal`, { row, col }));
  } catch (error) {
    message.textContent = error.message;
  }
}

async function toggleFlag(row, col) {
  if (!game || ["won", "lost"].includes(game.status)) return;
  try {
    setGame(await postJson(`/api/${game.game_id}/flag`, { row, col }));
  } catch (error) {
    message.textContent = error.message;
  }
}

async function solveWithBP() {
  if (!game || isSolving || ["won", "lost"].includes(game.status)) return;
  isSolving = true;
  message.textContent = "メッセージを更新しています…";
  bpState.textContent = "RUNNING";
  bpState.className = "state running";
  render();
  try {
    setGame(await postJson(`/api/${game.game_id}/solve`, {
      max_iterations: 500,
      tolerance: 1e-9,
      damping: 0.2,
    }));
    message.textContent = "各セルに周辺地雷確率を表示しました。";
  } catch (error) {
    message.textContent = error.message;
    bpState.textContent = "ERROR";
    bpState.className = "state error";
  } finally {
    isSolving = false;
    render();
  }
}

function probabilityColor(probability) {
  const hue = 146 - probability * 142;
  const lightness = 92 - probability * 42;
  return `hsl(${hue} 58% ${lightness}%)`;
}

function renderDiagnostics() {
  if (!game.bp) {
    if (bpState.textContent !== "RUNNING" && bpState.textContent !== "ERROR") clearDiagnostics();
    return;
  }
  diagStatus.textContent = game.bp.status.toUpperCase();
  diagIterations.textContent = game.bp.iterations;
  diagResidual.textContent = game.bp.final_max_message_delta.toExponential(2);
  diagFactors.textContent = game.bp.constraints;
  bpState.textContent = game.bp.status === "converged" ? "CONVERGED" : game.bp.status.toUpperCase();
  bpState.className = `state ${game.bp.status === "converged" ? "ok" : "warning"}`;
}

function render() {
  if (!game) return;
  flagCounter.textContent = formatCounter(game.flags);
  updateTimer();
  renderDiagnostics();

  faceButton.classList.toggle("won", game.status === "won");
  faceButton.classList.toggle("lost", game.status === "lost");
  solveButton.disabled = isSolving || ["won", "lost"].includes(game.status);
  boardElement.style.setProperty("--size", game.size);
  boardElement.setAttribute("aria-label", `${game.size} × ${game.size} マインスイーパー盤面`);
  boardElement.innerHTML = "";

  for (const row of game.cells) {
    for (const cell of row) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "cell";
      button.setAttribute("role", "gridcell");
      button.setAttribute("aria-label", `${cell.row + 1}行 ${cell.col + 1}列`);

      if (cell.revealed) {
        button.classList.add("revealed");
        if (cell.value > 0) {
          button.dataset.value = cell.value;
          button.textContent = cell.value;
        }
      } else if (cell.flagged) {
        button.classList.add("flagged");
      }

      if (["mine", "hit-mine"].includes(cell.value)) button.classList.add(cell.value);
      if (cell.value === "wrong-flag") {
        button.classList.add("flagged", "wrong-flag", "game-over");
        const cross = document.createElement("span");
        cross.className = "cross";
        button.append(cross);
      }

      if (!cell.revealed && typeof cell.mine_probability === "number") {
        button.style.backgroundColor = probabilityColor(cell.mine_probability);
        const probability = document.createElement("span");
        probability.className = "bp-probability";
        probability.textContent = `${Math.round(cell.mine_probability * 100)}%`;
        button.append(probability);
        button.setAttribute("aria-label", `${button.getAttribute("aria-label")}、地雷確率 ${Math.round(cell.mine_probability * 100)}%`);
      }
      if (["won", "lost"].includes(game.status)) button.classList.add("game-over");

      button.addEventListener("mousedown", (event) => {
        if (event.button === 0 && !cell.revealed && !cell.flagged) {
          button.classList.add("down");
          faceButton.classList.add("pressed");
        }
      });
      button.addEventListener("mouseup", () => button.classList.remove("down"));
      button.addEventListener("mouseleave", () => button.classList.remove("down"));
      button.addEventListener("click", () => reveal(cell.row, cell.col));
      button.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        toggleFlag(cell.row, cell.col);
      });
      boardElement.append(button);
    }
  }

  if (game.status === "won") message.textContent = `クリア。実現した地雷数は ${game.realized_mines} 個でした。`;
  if (game.status === "lost") message.textContent = `地雷でした。実現した地雷数は ${game.realized_mines} 個です。`;
}

rhoInput.addEventListener("input", () => { rhoValue.textContent = Number(rhoInput.value).toFixed(3); });
rhoInput.addEventListener("change", newGame);
sizeInput.addEventListener("change", newGame);
faceButton.addEventListener("click", newGame);
solveButton.addEventListener("click", solveWithBP);
document.addEventListener("mouseup", () => faceButton.classList.remove("pressed"));
newGame();
