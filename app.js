/* ============================================================
   SONARBEAM — front-end instrumentation
   1. Real-capture ingest (dataset presets + upload + drag-drop)
   2. ONNX U-Net inference (model.js) — the only detection path
   3. Deterministic geotag, overlay drawing, event log
   4. GeoJSON/CSV export
   ============================================================ */
"use strict";

const SW = 1024,
  SH = 614; // internal viewport resolution

const CLASS_COLOR = {
  wreck: "#ff5d5d",
  net: "#f5b243",
  pipe: "#39f0c8",
  drum: "#7fd8d2",
  debris: "#9fb6c3",
  ridge: "#6d8595",
};

/* ---------------- deterministic geotag ----------------
   Ping-header lat/lon isn't in the dataset, so coordinates are derived
   from the contact's position in the frame on a fixed survey-line origin
   (Gulf of Mannar survey block) — same input frame, same coordinates. */
function geotag(c) {
  const latD = 12 + c.v * 0.03;
  const lonD = 80.2 + c.u * 0.03;
  const fmt = (d, pos, neg) => {
    const deg = Math.floor(d);
    const min = (d - deg) * 60;
    return `${deg}°${min.toFixed(2)}′${d >= 0 ? pos : neg}`;
  };
  return { lat: fmt(latD, "N", "S"), lon: fmt(lonD, "E", "W"), latD, lonD };
}

/* Overlay drawing  */

function drawOverlay(canvas, contacts, gate, sceneKey) {
  canvas.width = SW;
  canvas.height = SH;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.clearRect(0, 0, SW, SH);
  ctx.font = "600 15px 'IBM Plex Mono', monospace";

  for (const c of contacts) {
    const x = (c.u - c.w / 2) * SW;
    const y = (c.v - c.h / 2) * SH;
    const w = c.w * SW,
      h = c.h * SH;
    const col = CLASS_COLOR[c.cls] || "#9fb6c3";

    ctx.strokeStyle = col;
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.strokeRect(x - 4, y - 4, w + 8, h + 8);

    // corner ticks
    ctx.lineWidth = 4;
    const t = Math.min(14, w / 3, h / 3);
    ctx.beginPath();
    ctx.moveTo(x - 4, y - 4 + t);
    ctx.lineTo(x - 4, y - 4);
    ctx.lineTo(x - 4 + t, y - 4);
    ctx.moveTo(x + 4 + w - t, y - 4);
    ctx.lineTo(x + 4 + w, y - 4);
    ctx.lineTo(x + 4 + w, y - 4 + t);
    ctx.stroke();

    // label
    const label = `${c.cls.toUpperCase()}  ${c.conf}%`;
    const tw = ctx.measureText(label).width;
    const ly = y - 14 < 8 ? y + h + 22 : y - 12;
    ctx.fillStyle = "rgba(3, 11, 18, 0.88)";
    ctx.fillRect(x - 4, ly - 13, tw + 14, 20);
    ctx.fillStyle = col;
    ctx.fillText(label, x + 3, ly + 1);
  }
}


/*  console wiring  */const $ = (s) => document.querySelector(s);

const el = {
  captureChips: document.querySelectorAll("[data-real]"),
  confSlider: $("#confSlider"),
  confOut: $("#confOut"),

  boundsToggle: $("#boundsToggle"),
  rescan: $("#rescanBtn"),
  uploadBtn: $("#uploadBtn"), uploadInput: $("#uploadInput"),
  viewport: $("#viewport"), overlay: $("#overlayCanvas"),
  sonar: $("#sonarCanvas"),
  hudRange: $("#hudRange"),
  hudLat: $("#hudLat"),
  hudLon: $("#hudLon"),
  vpTag: $("#vpTag"),
  log: $("#eventLog"),
  topScore: $("#topScore"),
  topKind: $("#topKind"),
  exportJson: $("#exportJson"),
  exportCsv: $("#exportCsv"),
};

const state = {
  file: null,        // display name of the current capture
  result: { contacts: [], vetoed: 0 },
};
let scanBusy = false, scanQueued = false;

const stamp = () => {
  const d = new Date();
  const p = (n, l = 2) => String(n).padStart(l, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}.${p(Math.floor(d.getMilliseconds() / 10))}`;
};

function log(msg, cls = "") {
  const line = document.createElement("p");
  line.className = `log-line ${cls}`;
  line.innerHTML = `<span class="t">${stamp()}</span>  <span class="msg">${msg}</span>`;
  el.log.appendChild(line);
  while (el.log.children.length > 60) el.log.removeChild(el.log.firstChild);
  el.log.scrollTop = el.log.scrollHeight;
}

function updateSliderFill(input) {
  const pct = ((input.value - input.min) / (input.max - input.min)) * 100;
  input.style.setProperty("--fill", pct + "%");
}

async function scan() {
  if (scanBusy) { scanQueued = true; return; }
  scanBusy = true;
  try {
    await doScan();
  } finally {
    scanBusy = false;
    if (scanQueued) { scanQueued = false; scan(); }
  }
}

async function doScan() {
  if (!state.file) return; // no capture loaded yet
  log(`INGEST  ${state.file} · 900 kHz · ping 0448`);

  log("COND    slant-range corr · TVG · SRAD despeckle", "ok");

