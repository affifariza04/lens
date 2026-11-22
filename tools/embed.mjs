// Precompute CLIP image embeddings for the gallery (same model + dtype the browser uses for text).
import { AutoProcessor, CLIPVisionModelWithProjection, RawImage, env } from '@huggingface/transformers';
env.localModelPath = './models/'; env.allowRemoteModels = false;
import { readFileSync, writeFileSync } from 'node:fs';
const MODEL = 'Xenova/clip-vit-base-patch32';
const meta = JSON.parse(readFileSync('gallery.json', 'utf8'));
const processor = await AutoProcessor.from_pretrained(MODEL);
const vision = await CLIPVisionModelWithProjection.from_pretrained(MODEL, { dtype: 'q8' });
const out = [];
for (const m of meta) {
  const image = await RawImage.read(`../gallery/${m.id}.jpg`);
  const { image_embeds } = await vision(await processor(image));
  const v = Array.from(image_embeds.data); const n = Math.hypot(...v);
  out.push({ ...m, e: v.map(x => +(x / n).toFixed(4)) });
  process.stdout.write('.');
}
writeFileSync('../gallery/index.json', JSON.stringify({ model: MODEL, dtype: 'q8', dims: out[0].e.length, items: out }));
console.log('\nwrote', out.length);
