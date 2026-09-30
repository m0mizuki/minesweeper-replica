const READY_TIMEOUT_MS = 3500;

export async function waitForMathRenderer() {
  if (window.katex?.render) return true;

  const loaded = new Promise((resolve) => {
    window.addEventListener("katex-ready", () => resolve(true), { once: true });
  });
  const timedOut = new Promise((resolve) => {
    window.setTimeout(() => resolve(false), READY_TIMEOUT_MS);
  });
  return Promise.race([loaded, timedOut]);
}

export function renderLatex(element, latex, options = {}) {
  element.dataset.latex = latex;
  element.setAttribute("aria-label", latex);
  if (window.katex?.render) {
    window.katex.render(latex, element, {
      displayMode: false,
      throwOnError: false,
      strict: "warn",
      trust: false,
      ...options,
    });
  } else {
    element.textContent = latex;
    element.classList.add("latex-fallback");
  }
  return element;
}

export function renderStaticLatex(root = document) {
  root.querySelectorAll("[data-latex]").forEach((element) => {
    renderLatex(element, element.dataset.latex);
  });
}

export function idToLatex(id) {
  return `${id[0]}_{${id.slice(1)}}`;
}

export function messageToLatex(from, to, argument = null) {
  const suffix = argument === null ? "" : `(${argument})`;
  return `m_{${idToLatex(from)}\\to ${idToLatex(to)}}${suffix}`;
}

export function vectorToLatex(vector, formatter) {
  return `\\left[${formatter(vector[0])},\\;${formatter(vector[1])}\\right]`;
}

export function joinProductLatex(terms) {
  return terms.join("\\,\\times\\,");
}
