# One-shot generator: embeds ORT runtime + trained ONNX into model.js
import base64
import os


def b64_chunks(path, chunk=24000):
    raw = open(path, "rb").read()
    enc = base64.b64encode(raw).decode()
    return [enc[i : i + chunk] for i in range(0, len(enc), chunk)], len(raw)


def block(name, path):
    chunks, rawlen = b64_chunks(path)
    lines = [f"const {name} = ["]
    for c in chunks:
        lines.append(f'"{c}",')
    lines.append('].join("");')
    print(f"  {name}: {path} -> {len(chunks)} chunks ({rawlen} raw bytes)")
    return "\n".join(lines)


print("Embedding payloads...")
payloads = "\n\n".join(
    [
        block("ORT_JS", "vendor/ort.min.js"),
        block("ORT_MJS", "vendor/ort-wasm-simd-threaded.jsep.mjs"),
        block("ORT_WASM", "vendor/ort-wasm-simd-threaded.jsep.wasm"),
        block(
            "MODEL_ONNX",
            "sih-sonar-debris/sih-sonar-debris/model/sonar_model_edge.onnx",
        ),
    ]
)

CORE = r"""/* ============================================================
   SonarModel — self-contained in-browser inference module.
   Bundles ONNX Runtime Web 1.22 (MIT) and the trained U-Net
   (sonar_model_edge.onnx) as base64 payloads: no CDN, no cloud,
   no extra fetches. API:
     SonarModel.load()               -> Promise<boolean>
     SonarModel.infer(canvas, gate)  -> { contacts, vetoed, mask }
   Pipeline per capture mirrors the Python edge pipeline:
     CLAHE (preprocess.py) -> /255 -> U-Net sigmoid mask ->
     threshold 0.5 -> morphological close/open (predict.py) ->
     connected components -> IoU merge -> filter.py-style veto.
   ============================================================ */
"use strict";

const SonarModel = (() => {

  // ---------- base64 payload decoding ----------
  function b64ToBytes(arr) {
    const bin = atob(arr);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }

  function loadScriptBlob(bytes) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(
        new Blob([bytes], { type: "text/javascript" })
      );
      const s = document.createElement("script");
      s.src = url;
      s.onload = () => { URL.revokeObjectURL(url); resolve(); };
      s.onerror = () => { URL.revokeObjectURL(url); reject(new Error("ort script load failed")); };
      document.head.appendChild(s);
    });
  }

  // ---------- pipeline constants (mirror predict.py / filter.py) ----------
  const MS = 256;          // model input resolution
  const THRESHOLD = 0.5;   // mask binarisation
  const MIN_AREA = 40;     // min contour px area
  const MAX_DET = 5;       // max contacts per sweep
  const IOU_MERGE = 0.05;  // box merge threshold

  // ---------- tiny image ops (no OpenCV in the browser) ----------

  function toGray(u8, w, h) {
    const g = new Float32Array(w * h);
    for (let i = 0, p = 0; i < g.length; i++, p += 4) {
      g[i] = (0.299 * u8[p] + 0.587 * u8[p + 1] + 0.114 * u8[p + 2]) / 255;
    }
    return g;
  }

  // CLAHE equivalent: clip-limited tile equalisation with bilinear
  // interpolation between tile maps (cv2.createCLAHE clip=2.0, 8x8 grid).
  function clahe(gray, w, h, clipLimit = 2.0, tile = 8) {
    const tx = Math.ceil(w / tile), ty = Math.ceil(h / tile);
    const maps = new Array(tx * ty);
    for (let tyy = 0; tyy < ty; tyy++) {
      for (let txx = 0; txx < tx; txx++) {
        const hist = new Float64Array(256);
        let n = 0;
        const y1 = Math.min((tyy + 1) * tile, h), x1 = Math.min((txx + 1) * tile, w);
        for (let y = tyy * tile; y < y1; y++) {
          for (let x = txx * tile; x < x1; x++) {
            hist[Math.min(255, (gray[y * w + x] * 255) | 0)]++;
            n++;
          }
        }
        if (!n) { maps[tyy * tx + txx] = new Float64Array(256).fill(0.5); continue; }
        const limit = Math.max(1, ((clipLimit * n) / 256) | 0);
        let excess = 0;
        for (let i = 0; i < 256; i++) {
          if (hist[i] > limit) { excess += hist[i] - limit; hist[i] = limit; }
        }
        const share = excess / 256;
        const map = new Float64Array(256);
        let acc = 0;
        for (let i = 0; i < 256; i++) { acc += hist[i] + share; map[i] = acc / Math.max(1, n + excess); }
        maps[tyy * tx + txx] = map;
      }
    }
    const out = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const gy = Math.min(ty - 1, (y / tile) | 0), y1 = Math.min(ty - 1, gy + 1);
      const fy = y / tile - gy;
      for (let x = 0; x < w; x++) {
        const gx = Math.min(tx - 1, (x / tile) | 0), x1 = Math.min(tx - 1, gx + 1);
        const fx = x / tile - gx;
        const v = Math.min(255, (gray[y * w + x] * 255) | 0);
        const a = maps[gy * tx + gx][v], b = maps[gy * tx + x1][v];
        const c = maps[y1 * tx + gx][v], d = maps[y1 * tx + x1][v];
        out[y * w + x] = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
      }
    }
    return out;
  }

  function resizeGray(src, sw, sh, dw, dh) {
    const out = new Float32Array(dw * dh);
    const rx = sw / dw, ry = sh / dh;
    for (let y = 0; y < dh; y++) {
      const sy0 = Math.min(sh - 1, (y * ry) | 0);
      const sy1 = Math.min(sh - 1, sy0 + 1);
      const fy = y * ry - sy0;
      for (let x = 0; x < dw; x++) {
        const sx0 = Math.min(sw - 1, (x * rx) | 0);
        const sx1 = Math.min(sw - 1, sx0 + 1);
        const fx = x * rx - sx0;
        const a = src[sy0 * sw + sx0], b = src[sy0 * sw + sx1];
        const c = src[sy1 * sw + sx0], d = src[sy1 * sw + sx1];
        out[y * dw + x] = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
      }
    }
    return out;
  }

  // ---------- mask post-processing (exact predict.py order) ----------

  function morph(mask, w, h, dilatePass) {
    // one 5x5 morphology pass; dilatePass=true -> max filter, else min filter
    const out = new Uint8Array(mask.length);
    const r = 2;
    for (let y = 0; y < h; y++) {
      const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
      for (let x = 0; x < w; x++) {
        const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r);
        let hit = dilatePass ? 0 : 1;
        for (let yy = y0; yy <= y1; yy++) {
          const row = yy * w;
          for (let xx = x0; xx <= x1; xx++) {
            const v = mask[row + xx];
            if (dilatePass ? v : !v) { hit = dilatePass ? 1 : 0; yy = y1 + 1; break; }
          }
        }
        out[y * w + x] = hit;
      }
    }
    return out;
  }

  function morphology(binary, w, h) {
    // close(5x5 x2): dilate, dilate, erode, erode
    let m = morph(binary, w, h, true);
    m = morph(m, w, h, true);
    m = morph(m, w, h, false);
    m = morph(m, w, h, false);
    // open(5x5 x1): erode, dilate
    m = morph(m, w, h, false);
    m = morph(m, w, h, true);
    return m;
  }

  function components(mask, w, h) {
    const label = new Int32Array(mask.length);
    const comps = [];
    const stack = [];
    for (let i = 0; i < mask.length; i++) {
      if (!mask[i] || label[i]) continue;
      const id = comps.length + 1;
      let minX = w, minY = h, maxX = 0, maxY = 0, area = 0;
      stack.length = 0;
      stack.push(i);
      label[i] = id;
      while (stack.length) {
        const p = stack.pop();
        const px = p % w, py = (p / w) | 0;
        area++;
        if (px < minX) minX = px;
        if (py < minY) minY = py;
        if (px > maxX) maxX = px;
        if (py > maxY) maxY = py;
        if (px > 0 && mask[p - 1] && !label[p - 1]) { label[p - 1] = id; stack.push(p - 1); }
        if (px < w - 1 && mask[p + 1] && !label[p + 1]) { label[p + 1] = id; stack.push(p + 1); }
        if (py > 0 && mask[p - w] && !label[p - w]) { label[p - w] = id; stack.push(p - w); }
        if (py < h - 1 && mask[p + w] && !label[p + w]) { label[p + w] = id; stack.push(p + w); }
      }
      comps.push({ minX, minY, maxX, maxY, area });
    }
    return comps;
  }

  // ---------- box merge (predict.py) ----------

  function iou(a, b) {
    const ix1 = Math.max(a.xmin, b.xmin), iy1 = Math.max(a.ymin, b.ymin);
    const ix2 = Math.min(a.xmax, b.xmax), iy2 = Math.min(a.ymax, b.ymax);
    const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
    const inter = iw * ih;
    const aa = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    const bb = (b.xmax - b.xmin) * (b.ymax - b.ymin);
    return inter / Math.max(aa + bb - inter, 1e-6);
  }

  function mergeOverlapping(boxes, thresh) {
    boxes = boxes.slice().sort((x, y) => y.confidence - x.confidence);
    const used = new Array(boxes.length).fill(false);
    const out = [];
    for (let i = 0; i < boxes.length; i++) {
      if (used[i]) continue;
      const group = [boxes[i]];
      used[i] = true;
      for (let j = i + 1; j < boxes.length; j++) {
        if (!used[j] && iou(boxes[i], boxes[j]) > thresh) {
          group.push(boxes[j]);
          used[j] = true;
        }
      }
      out.push({
        xmin: Math.min(...group.map((g) => g.xmin)),
        ymin: Math.min(...group.map((g) => g.ymin)),
        xmax: Math.max(...group.map((g) => g.xmax)),
        ymax: Math.max(...group.map((g) => g.ymax)),
        confidence: Math.max(...group.map((g) => g.confidence)),
      });
    }
    return out;
  }

  // ---------- confidence veto (confidence_scoring/filter.py) ----------
  // compact blobs score high, elongated ones (ridges/rocks) low,
  // specks and terrain-scale patches are flagged as noise.

  function vetoConfidence(regionConf, w, h) {
    const area = w * h;
    if (area < 10) return 15;                    // speckle noise
    if (area > 0.75 * MS * MS) return 15;        // terrain-scale segmentation
    const ar = Math.max(w, h) / (Math.min(w, h) + 1e-5);
    if (ar < 2.5) return Math.min(99, Math.round(regionConf * 100));
    return Math.round(regionConf * 45);          // elongated ridge profile
  }

  // ---------- module state ----------

  let session = null;
  let loadPromise = null;

  const SonarModel = {
    status: "idle",
    error: null,
    inputName: "sonar_input",
    outputName: "debris_mask_output",

    load() {
      if (this.status === "ready") return Promise.resolve(true);
      if (this.status === "error") return Promise.resolve(false);
      if (loadPromise) return loadPromise;

      loadPromise = (async () => {
        try {
          if (typeof ort === "undefined") {
            await loadScriptBlob(b64ToBytes(ORT_JS));
          }
          if (typeof ort === "undefined") {
            throw new Error("ort global missing after bootstrap");
          }
          ort.env.wasm.numThreads = 1;          // single-threaded: no COOP/COEP needed
          const isHttp =
            typeof location !== "undefined" && /^https?:$/.test(location.protocol);
          if (isHttp) {
            // served over http(s): resolve runtime artifacts against the page URL
            // (absolute — ORT's script runs from a blob context where relative
            // specifiers cannot resolve)
            const base = location.href;
            // glue + binary must be the matching jsep pair
            ort.env.wasm.wasmPaths = {
              mjs: new URL("./vendor/ort-wasm-simd-threaded.jsep.mjs", base).href,
              wasm: new URL("./vendor/ort-wasm-simd-threaded.jsep.wasm", base).href,
            };
          } else {
            // file:// or sandboxed: feed ORT the embedded bytes, glue via blob URL
            ort.env.wasm.wasmBinary = b64ToBytes(ORT_WASM);
            const mjsUrl = URL.createObjectURL(
              new Blob([b64ToBytes(ORT_MJS)], { type: "text/javascript" })
            );
            this.__mjsUrl = mjsUrl;
            ort.env.wasm.wasmPaths = { mjs: mjsUrl };
          }

          session = await ort.InferenceSession.create(
            b64ToBytes(MODEL_ONNX),
            { executionProviders: ["wasm"], graphOptimizationLevel: "all" }
          );
          if (this.__mjsUrl) URL.revokeObjectURL(this.__mjsUrl);
          this.status = "ready";
          return true;
        } catch (e) {
          this.status = "error";
          this.error = String((e && e.message) || e).slice(0, 140);
          return false;
        }
      })();

      return loadPromise;
    },

    // canvas: the display viewport (boxes map back onto it);
    // native: optional un-stretched source (native CLAHE, like preprocess.py)
    async infer(canvas, gate, native) {
      if (this.status !== "ready" || !session) throw new Error("runtime not ready");
      const src = native && native.width ? native : canvas;
      const w = src.width, h = src.height;
      const u8 = src.getContext("2d", { willReadFrequently: true })
        .getImageData(0, 0, w, h).data;

      let gray = toGray(u8, w, h);
      gray = clahe(gray, w, h);
      const input = resizeGray(gray, w, h, MS, MS);

      const feeds = new ort.Tensor("float32", input, [1, 1, MS, MS]);
      const results = await session.run({ [this.inputName]: feeds });
      const mask = results[this.outputName].data; // sigmoid probs, 256x256

      const binary = new Uint8Array(mask.length);
      for (let i = 0; i < mask.length; i++) binary[i] = mask[i] > THRESHOLD ? 1 : 0;
      const clean = morphology(binary, MS, MS);

      const comps = components(clean, MS, MS);
      const raw = [];
      for (const c of comps) {
        if (c.area < MIN_AREA) continue;
        let sum = 0, cnt = 0;
        for (let y = c.minY; y <= c.maxY; y++) {
          for (let x = c.minX; x <= c.maxX; x++) { sum += mask[y * MS + x]; cnt++; }
        }
        raw.push({
          xmin: c.minX, ymin: c.minY,
          xmax: c.maxX + 1, ymax: c.maxY + 1,
          confidence: sum / cnt,
        });
      }

      const merged = mergeOverlapping(raw, IOU_MERGE)
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, MAX_DET);

      const contacts = [];
      let vetoed = 0;
      for (const b of merged) {
        const bw = b.xmax - b.xmin, bh = b.ymax - b.ymin;
        const conf = vetoConfidence(b.confidence, bw, bh);
        if (conf < gate) { vetoed++; continue; }
        // boxes are in 256-model-space; the overlay draws from 0-1 normals
        contacts.push({
          id: "C" + String(contacts.length + 1).padStart(2, "0"),
          cls: "debris",
          conf,
          u: ((b.xmin + b.xmax) / 2) / MS,
          v: ((b.ymin + b.ymax) / 2) / MS,
          w: bw / MS,
          h: bh / MS,
          len: +(bw * 0.35).toFixed(1),
          wid: +(bh * 0.35).toFixed(1),
          box: { xmin: b.xmin, ymin: b.ymin, xmax: b.xmax, ymax: b.ymax },
        });
      }
      return { contacts, vetoed, mask };
    },
  };

  return SonarModel;
})();

window.SonarModel = SonarModel;
"""

out = CORE.replace("/*__PAYLOADS__*/", payloads)
# place payloads right after the IIFE opening (before helper functions)
marker = 'const SonarModel = (() => {\n'
assert marker in out, "marker missing"
out = out.replace(marker, marker + "\n  // ---------- embedded payloads (base64) ----------\n\n" + payloads + "\n", 1)

with open("model.js", "w", encoding="utf-8", newline="\n") as f:
    f.write(out)
print("model.js written:", os.path.getsize("model.js"), "bytes")
