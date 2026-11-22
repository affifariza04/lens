import { AutoTokenizer, CLIPTextModelWithProjection, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1';
import { bars, normalise, rank } from './search.js';

env.allowLocalModels = false;
const MODEL = 'Xenova/clip-vit-base-patch32';
const $ = s => document.querySelector(s);
const grid = $('#grid'), input = $('#q'), status = $('#status'), meter = $('#meter i');

// ---------------------------------------------------------------- gallery (image embeddings precomputed at build time)
const index = await fetch('gallery/index.json').then(r => r.json());
const items = index.items;
const cards = new Map();
for (const it of items) {
  const el = document.createElement('figure');
  el.className = 'card';
  el.innerHTML = `<img src="gallery/${it.id}.jpg" alt="" loading="lazy" decoding="async">
    <figcaption><span class="bar"><i></i></span><b class="pct"></b><button class="sim" title="Find photos like this">≈ similar</button></figcaption>`;
  el.querySelector('.sim').addEventListener('click', () => { input.value = ''; search(it.e, `photos like #${it.id}`); });
  grid.append(el); cards.set(it.id, el);
}

// ---------------------------------------------------------------- text encoder, on-device
let tokenizer, textModel;
const progress = {};
function onProgress(p) {
  if (p.status === 'progress' && p.total) {
    progress[p.file] = [p.loaded, p.total];
    const [l, t] = Object.values(progress).reduce(([a, b], [c, d]) => [a + c, b + d], [0, 0]);
    meter.style.width = `${(100 * l / t).toFixed(1)}%`;
    status.textContent = `Downloading the model to your browser… ${(l / 1e6).toFixed(1)} / ${(t / 1e6).toFixed(1)} MB (cached after the first visit)`;
  }
}
const ready = (async () => {
  status.textContent = 'Loading the CLIP text encoder…';
  tokenizer = await AutoTokenizer.from_pretrained(MODEL);
  textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL, { dtype: 'q8', progress_callback: onProgress });
  document.body.classList.add('ready'); meter.style.width = '100%';
  status.textContent = 'Model ready. Everything runs on this device: your query never leaves the browser.';
})().catch(err => { status.textContent = `Could not load the model: ${err.message}`; throw err; });

async function embedText(text) {
  const inputs = tokenizer([text], { padding: true, truncation: true });
  const { text_embeds } = await textModel(inputs);
  return normalise(text_embeds.data);
}

// ---------------------------------------------------------------- search + animated re-order (FLIP)
let seq = 0;
async function search(vec, label) {
  const mine = ++seq;
  const t0 = performance.now();
  if (typeof vec === 'string') { await ready; vec = await embedText(vec); }
  if (mine !== seq) return;
  const ranked = rank(vec, items), b = bars(ranked), ms = performance.now() - t0;
  const first = new Map([...cards].map(([id, el]) => [id, el.getBoundingClientRect()]));
  ranked.forEach((r, i) => {
    const el = cards.get(r.id);
    grid.append(el);
    el.style.setProperty('--b', b[i]);
    el.querySelector('.pct').textContent = `${(r.score * 100).toFixed(1)}`;
    el.classList.toggle('top', i < 3);
    el.classList.toggle('dim', b[i] < .35);
  });
  for (const [id, el] of cards) {
    const a = first.get(id), z = el.getBoundingClientRect();
    const dx = a.left - z.left, dy = a.top - z.top;
    if (dx || dy) el.animate([{ transform: `translate(${dx}px,${dy}px)` }, { transform: 'none' }], { duration: 650, easing: 'cubic-bezier(.2,.7,.2,1)' });
  }
  status.textContent = `“${label}” · ranked ${items.length} photos in ${ms.toFixed(0)} ms on this device · best match ${(ranked[0].p * 100).toFixed(0)}% confident`;
  grid.classList.add('ranked');
}

let timer;
input.addEventListener('input', () => {
  clearTimeout(timer);
  const q = input.value.trim();
  if (q.length > 1) timer = setTimeout(() => search(q, q), 280);
});
$('#form').addEventListener('submit', e => { e.preventDefault(); const q = input.value.trim(); if (q) search(q, q); });
document.querySelectorAll('[data-q]').forEach(b => b.addEventListener('click', () => { input.value = b.dataset.q; search(b.dataset.q, b.dataset.q); }));
