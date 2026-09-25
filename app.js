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
