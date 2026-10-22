import assert from "node:assert/strict";
import test from "node:test";
import { speculate } from "./speculate.ts";

const gold = Array.from({ length: 80 }, (_, index) => `t${index}`);

test("speculative text is greedy text", () => {
  for (const k of [1, 2, 4, 8]) {
    for (const agree of [0, 0.35, 0.7, 1]) {
      const result = speculate(gold, 6, 18, k, agree, 7);
      assert.deepEqual(result.text, result.greedy);
      assert.equal(result.identical, true);
      if (agree === 0) assert.equal(result.qwenForwards, 18);
      if (agree === 1) assert.ok(result.qwenForwards <= Math.ceil(18 / (k + 1)));
    }
  }
});
