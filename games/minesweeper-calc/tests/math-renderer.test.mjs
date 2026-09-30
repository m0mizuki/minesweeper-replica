import test from "node:test";
import assert from "node:assert/strict";
import {
  idToLatex,
  joinProductLatex,
  messageToLatex,
  vectorToLatex,
} from "../js/math-renderer.mjs";

test("変数・因子ラベルをLaTeXの添字へ変換する", () => {
  assert.equal(idToLatex("x12"), "x_{12}");
  assert.equal(idToLatex("a3"), "a_{3}");
});

test("有向メッセージをLaTeX形式へ変換する", () => {
  assert.equal(messageToLatex("x3", "a1", 0), "m_{x_{3}\\to a_{1}}(0)");
  assert.equal(messageToLatex("a2", "x4"), "m_{a_{2}\\to x_{4}}");
});

test("確率ベクトルをLaTeX形式へ変換する", () => {
  assert.equal(
    vectorToLatex([2 / 3, 1 / 3], (value) => value.toFixed(3)),
    "\\left[0.667,\\;0.333\\right]",
  );
});

test("積記号と次の数式をLaTeXコマンドとして分離する", () => {
  const product = joinProductLatex([
    "m_{a_{2}\\to x_{3}}(0)=0.5",
    "m_{a_{3}\\to x_{3}}(0)=0.5",
  ]);
  assert.equal(
    product,
    "m_{a_{2}\\to x_{3}}(0)=0.5\\,\\times\\,m_{a_{3}\\to x_{3}}(0)=0.5",
  );
  assert.doesNotMatch(product, /\\\\timesm/);
});
