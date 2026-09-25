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
