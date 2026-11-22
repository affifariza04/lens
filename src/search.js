// Pure ranking helpers (no DOM, no model) so they can be unit-tested in Node.

/** Dot product of two equal-length vectors. Embeddings are L2-normalised, so this is cosine similarity. */
export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function normalise(v) {
  const n = Math.hypot(...v) || 1;
  return Array.from(v, x => x / n);
}

/**
 * Rank gallery items against a query embedding.
 * Returns items sorted by similarity with `score` (raw cosine) and `p` (softmax over the gallery at CLIP's
 * logit scale, so it reads like "how sure is the model that this is the best match").
 */
export function rank(query, items, logitScale = 100) {
  const scored = items.map(it => ({ ...it, score: dot(query, it.e) }));
  const max = Math.max(...scored.map(s => s.score));
  const exps = scored.map(s => Math.exp((s.score - max) * logitScale));
  const total = exps.reduce((a, b) => a + b, 0);
  scored.forEach((s, i) => { s.p = exps[i] / total; });
  return scored.sort((a, b) => b.score - a.score);
}

/** Map raw cosine scores onto 0..1 for a readable bar (relative to this result set). */
export function bars(ranked) {
  const hi = ranked[0]?.score ?? 0, lo = ranked[ranked.length - 1]?.score ?? 0;
  return ranked.map(r => (hi === lo ? 1 : (r.score - lo) / (hi - lo)));
}
