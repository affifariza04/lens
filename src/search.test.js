import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bars, dot, normalise, rank } from './search.js';

test('dot of normalised vectors is cosine', () => {
  const a = normalise([3, 4]), b = normalise([4, 3]);
  assert.ok(Math.abs(dot(a, b) - 24 / 25) < 1e-12);
  assert.ok(Math.abs(dot(a, a) - 1) < 1e-12);
});

test('rank orders by similarity and softmax sums to 1', () => {
  const q = normalise([1, 0, 0]);
  const items = [{ id: 'a', e: normalise([0, 1, 0]) }, { id: 'b', e: normalise([1, .1, 0]) }, { id: 'c', e: normalise([1, 1, 0]) }];
  const r = rank(q, items);
  assert.deepEqual(r.map(x => x.id), ['b', 'c', 'a']);
  assert.ok(Math.abs(r.reduce((s, x) => s + x.p, 0) - 1) < 1e-9);
  assert.ok(r[0].p > r[1].p && r[1].p > r[2].p);
});

test('bars map the best to 1 and the worst to 0', () => {
  const r = [{ score: .31 }, { score: .25 }, { score: .19 }];
  assert.deepEqual(bars(r).map(x => +x.toFixed(2)), [1, .5, 0]);
  assert.deepEqual(bars([{ score: .2 }, { score: .2 }]), [1, 1]);
});

test('rank does not mutate the input', () => {
  const items = [{ id: 'x', e: [1, 0] }];
  rank([1, 0], items);
  assert.equal(items[0].score, undefined);
});
