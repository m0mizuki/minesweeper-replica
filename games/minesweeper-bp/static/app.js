const boardElement = document.querySelector("#board");
const flagCounter = document.querySelector("#flag-counter");
const timerElement = document.querySelector("#timer");
const faceButton = document.querySelector("#new-game");
const solveButton = document.querySelector("#solve-bp");
const bpStatus = document.querySelector("#bp-status");
const rhoInput = document.querySelector("#rho");
const rhoValue = document.querySelector("#rho-value");

let game = null;
let timerId = null;
let localStart = null;
let isSolving = false;

function renderEmptyBoard() {
  const fragment = document.createDocumentFragment();
  for (let index = 0; index < 81; index += 1) {
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = "cell";
    cell.disabled = true;
    cell.tabIndex = -1;
    cell.setAttribute("aria-hidden", "true");
    fragment.append(cell);
  }
  boardElement.replaceChildren(fragment);
}

function formatCounter(value) {
  return String(Math.max(0, Math.min(999, value))).padStart(3, "0");
}

function updateTimer() {
  if (!game || !game.started || game.status === "won" || game.status === "lost") {
    timerElement.textContent = formatCounter(game?.elapsed ?? 0);
    return;
  }
  const elapsed = Math.min(999, Math.floor((Date.now() - localStart) / 1000));
  timerElement.textContent = formatCounter(elapsed);
}

function setGame(nextGame) {
  const wasStarted = game?.started;
  game = nextGame;

  if (game.started && !wasStarted) {
    localStart = Date.now() - game.elapsed * 1000;
    timerId = window.setInterval(updateTimer, 250);
  }
  if (!game.started || game.status === "won" || game.status === "lost") {
    window.clearInterval(timerId);
    timerId = null;
  }
  render();
}

async function postJson(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = `Request failed: ${response.status}`;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch (_) {
      // Use the HTTP status when the response has no JSON body.
    }
    throw new Error(detail);
  }
  return response.json();
}

async function newGame() {
  window.clearInterval(timerId);
  timerId = null;
  localStart = null;
  bpStatus.textContent = "";
  try {
    setGame(await postJson("/api/new", { rho: Number(rhoInput.value) }));
  } catch (error) {
    bpStatus.textContent = "error";
    console.error(error);
  }
}

async function reveal(row, col) {
  if (!game || game.status === "won" || game.status === "lost") return;
  faceButton.classList.remove("pressed");
  try {
    setGame(await postJson(`/api/${game.game_id}/reveal`, { row, col }));
  } catch (error) {
    bpStatus.textContent = "error";
    console.error(error);
  }
}

async function toggleFlag(row, col) {
  if (!game || game.status === "won" || game.status === "lost") return;
  try {
    setGame(await postJson(`/api/${game.game_id}/flag`, { row, col }));
  } catch (error) {
    bpStatus.textContent = "error";
    console.error(error);
  }
}

async function solveWithBP() {
  if (!game || isSolving || game.status === "won" || game.status === "lost") return;
  isSolving = true;
  bpStatus.textContent = "BP...";
  render();
  try {
    setGame(await postJson(`/api/${game.game_id}/solve`, {}));
  } catch (error) {
    bpStatus.textContent = "error";
    console.error(error);
  } finally {
    isSolving = false;
    render();
  }
}

function render() {
  if (!game) return;
  flagCounter.textContent = formatCounter(game.flags);
  updateTimer();

  faceButton.classList.toggle("won", game.status === "won");
  faceButton.classList.toggle("lost", game.status === "lost");
  solveButton.disabled = isSolving || game.status === "won" || game.status === "lost";
  if (!isSolving && game.bp) {
    bpStatus.textContent = `${game.bp.status} · ${game.bp.iterations} iter`;
  }

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

      if (cell.value === "mine" || cell.value === "hit-mine") {
        button.classList.add(cell.value);
      }
      if (cell.value === "wrong-flag") {
        button.classList.add("flagged", "wrong-flag", "game-over");
        const cross = document.createElement("span");
        cross.className = "cross";
        button.append(cross);
      }
      if (
        !cell.revealed
        && game.status !== "won"
        && game.status !== "lost"
        && typeof cell.mine_probability === "number"
      ) {
        const probability = document.createElement("span");
        probability.className = "bp-probability";
        probability.textContent = `${Math.round(cell.mine_probability * 100)}%`;
        button.append(probability);
      }
      if (game.status === "won" || game.status === "lost") {
        button.classList.add("game-over");
      }

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
}

faceButton.addEventListener("click", newGame);
solveButton.addEventListener("click", solveWithBP);
rhoInput.addEventListener("input", () => {
  rhoValue.textContent = Number(rhoInput.value).toFixed(3);
});
rhoInput.addEventListener("change", newGame);
document.addEventListener("mouseup", () => faceButton.classList.remove("pressed"));
renderEmptyBoard();
newGame();
