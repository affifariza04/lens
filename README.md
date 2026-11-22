# lens

**Search photos by meaning, entirely in the browser.** Type "yellow taxis in the city" or "coffee on a wooden
table" and a vision-language model (CLIP) running inside the tab ranks every photo by how well it matches.
No server, no API key, nothing uploaded.

### ▶ [Live demo](https://affifariza04.github.io/lens/)

[![test-and-deploy](https://github.com/affifariza04/lens/actions/workflows/pages.yml/badge.svg)](https://github.com/affifariza04/lens/actions/workflows/pages.yml)
![stack](https://img.shields.io/badge/CLIP%20ViT--B%2F32-Transformers.js%203-7CF0C8) ![license](https://img.shields.io/badge/license-MIT-green)

## How it works

```
build time (Node)                          in the browser
─────────────────                          ──────────────
photo ─► CLIP image encoder (int8) ─┐      query ─► tokenizer ─► CLIP text encoder (int8, cached)
                                    │                                 │
                   gallery/index.json  ◄── 72 × 512-d unit vectors     ▼ 512-d unit vector
                                    └────────────► cosine similarity ─► softmax @ logit scale 100 ─► FLIP re-order
```

- **Split encoders.** Image embeddings are computed once at build time (`tools/embed.mjs`) with the same model and
  quantisation the browser uses for text, so both live in the same embedding space. The page ships 72 vectors in a
  small JSON and only the ~64 MB text encoder is downloaded, once, then cached by the browser.
- **Fast enough to search as you type.** A query is tokenised and encoded on-device in tens of milliseconds;
  ranking 72 photos is a few dot products. Requests are sequenced so a slow early query never overwrites a newer one.
- **Readable scores.** Raw cosine similarity for CLIP sits around 0.2–0.35, which means little to people. The UI shows a
  bar relative to the result set and a softmax "confidence" at CLIP's logit scale for the top match.
- **Search by example.** "≈ similar" uses a photo's own embedding as the query (image → image retrieval).
- **Smooth re-ordering.** Cards move to their new rank with a FLIP animation instead of jumping.

## Run it

```bash
npx serve .                 # any static server
npm test                    # ranking maths (node:test)

# re-embed the gallery after adding photos to gallery/ and tools/gallery.json
cd tools && npm i && node embed.mjs
```

## Files

| | |
|---|---|
| `src/search.js` | cosine ranking, softmax confidence, relative bars (pure, tested) |
| `src/app.js` | model loading with progress, on-device text encoding, FLIP re-ordering |
| `tools/embed.mjs` | build-time image embeddings |
| `gallery/` | 72 photos (Unsplash via picsum.photos) + `index.json` |

---

Built by [Affif Ariza Adiwijaya](https://affifariza04.github.io/portfolio/). MIT licensed. Photos: Unsplash license.
