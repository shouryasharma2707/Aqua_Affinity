# AQUA AFFINITY

**Ghost gear detection for side-scan sonar.** A self-contained demo website that turns
side-scan sonar imagery into confidence-scored, geotagged marine-debris contacts —
with the full detection loop (ingest → condition → detect → score → report) running
**entirely in your browser**. No cloud. No telemetry.

The site pairs a polished landing page with a working analysis console backed by the
team's [U-Net ONNX model](sih-sonar-debris/)

---

## ✨ What's on the page

| Section | Description |
|---|---|
| **Hero** | Project pitch and headline figures (≈38 ms p95/tile, <2.1% FPR, GeoJSON export) |
| **Console** | The live prototype: load a capture, run real U-Net inference, inspect contacts, export reports |
| **Approach** | How the detector separates wrecks from rocks — despeckling, shadow↔object geometry, confidence veto |
| **Pipeline** | The five-stage edge pipeline (ingest, condition, detect, score & veto, report) |
| **Benchmarks** | Stated evaluation numbers on hand-verified sonar tiles |

## 🖥️ The console (the fun part)

The console in `index.html` is **not a mockup** — it runs the same ONNX model the
Python pipeline exports, via [ONNX Runtime Web](https://onnxruntime.ai/) (WASM execution provider):

- **Capture sources** — four dataset presets from `sih-sonar-debris/data/raw/`, local
  file upload, or drag-and-drop any image onto the viewport.
- **Tuning** — confidence gate slider (30–95%) and overlay toggle, with instant re-inference.
- **Detection** — grayscale → CLAHE enhancement → 256×256 U-Net inference → threshold →
  connected-component blobs → mean-probability + rule-based confidence scoring.
  Blob geometry drives indicative labels (drum / net / pipe / wreck).
- **Event log** — timestamped console output mirroring the edge pipeline's stages.
- **Export** — download contacts as **GeoJSON** (`aquaaffinity_contacts.json`) or
  **CSV** (`aquaffinity_contacts.csv`) with lat/lon, confidence, bounding box and
  estimated dimensions.
- **Deterministic geotag** — ping headers aren't in the dataset, so coordinates are
  derived from contact position on a fixed survey-line origin (Gulf of Mannar block).
  Same input frame → same coordinates.

`model.js` mirrors the Python pipeline 1:1 so browser and offline results stay
comparable (`preprocessing/preprocess.py` ↔ `clahe()`, `model/predict.py` ↔ threshold +
blobs, `confidence_scoring/filter.py` ↔ rule scoring).

---

## 🚀 Run it locally

It's a fully static site — no build step, no package install. Any static file server works:

```bash
# from the repo root — pick one:
python -m http.server 8000
npx serve .
```

Then open <http://localhost:8000>.

> **Use an HTTP server, not `file://`.** The WASM runtime, its ES-module glue, and the
> ONNX model are all fetched at runtime, which browsers block on `file://`.


## 🧠 The model

- **Architecture:** lightweight single-class U-Net ("debris"), 1×1×256×256 input
- **Export:** `sih-sonar-debris/model/sonar_model_edge.onnx` (+ external weights `.onnx.data`)
- **Preprocessing:** grayscale (BT.601 weights) → CLAHE (clip 2.0, 8×8 grid) → resize to 256² → normalize
- **Post-processing:** sigmoid threshold 0.5 → connected components → min-area 15 filter
- **Confidence:** mean mask probability blended with an aspect/area rule score (mirrors `filter.py`)
- **Size:** ~74 KB — small enough to load in seconds and run CPU-only on edge hardware

The U-Net is single-class; the human-readable labels (drum / net / pipe / wreck) are
derived from blob geometry and are **indicative, not identified** — the UI footnotes
this everywhere.
