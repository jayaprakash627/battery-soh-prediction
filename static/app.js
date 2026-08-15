/* =============================================================================
 *  Battery State of Health — front end
 * =============================================================================
 *  No framework, no build step, no chart library. Three canvases drawn by hand,
 *  device-pixel-ratio aware, re-reading the CSS custom properties on theme
 *  toggle so they recolour in place.
 *
 *  The page is a sequence, not a document. Someone arriving knowing nothing
 *  should understand each step before the next one needs it: the problem, what
 *  a charge looks like, what ageing does to it, how the model reads it, whether
 *  it actually works, where it stops. The progress rail exists so that order is
 *  visible rather than implied.
 *
 *  Every interaction drives the real model over HTTP. Nothing here is a mock-up,
 *  and the model is deliberately NOT reimplemented in JavaScript to make the
 *  sliders feel snappier — that would be two models, and they would drift.
 * ========================================================================== */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const fmt = (n, d = 2) => Number(n).toFixed(d);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

const state = {
  card: null,          // /api/model
  demo: null,          // real charge curves
  life: null,          // leave-one-battery-out predictions
  lifeCell: null,
  ideaCurve: null,     // the curve shown in step 1
  hover: null,         // sample index while the mouse is over the idea chart
  ageCell: null,
  ageRows: [],         // that battery's curves, newest first
  ageIndex: 0,
  agePrediction: null,
  ageNewest: null,     // the ghost line: the same battery when new
  whyBase: null,       // the real measurements behind the current answer
  whyNow: null,        // those measurements as the sliders currently have them
  whyPrediction: null,
  predCache: new Map(),
};

/* ---- theme --------------------------------------------------------------- */
const root = document.documentElement;
// Paper is this report's primary theme, so an unset preference opens light.
// The fleet console next door does the opposite, because it is a different
// kind of thing.
root.dataset.theme = localStorage.getItem("soh-theme")
  || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

$("#theme-toggle").addEventListener("click", () => {
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem("soh-theme", root.dataset.theme);
  redrawAll();
});

const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();

/* ---- network ------------------------------------------------------------- */
const getJSON = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
};

const postJSON = async (url, body) => {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) throw Object.assign(new Error("request failed"), { detail: data.detail });
  return data;
};

/** Predictions cached by battery and charge number. Dragging the age slider
 *  back and forth would otherwise re-ask the server for an answer it has
 *  already given, and the number would flicker on every pass. */
async function predictCurve(row) {
  const key = `${row.cell_id}:${row.cycle_number}`;
  if (state.predCache.has(key)) return state.predCache.get(key);
  const result = await postJSON("/api/predict", { ...row.curve, confidence: 0.9 });
  state.predCache.set(key, result);
  return result;
}

/* ---- canvas -------------------------------------------------------------- */
function prepare(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  // From data-h, never the height attribute: assigning canvas.height *writes*
  // that attribute, so reading it back returns the device-pixel value and the
  // chart doubles on every redraw.
  const height = Number(canvas.dataset.h) || 300;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.height = height + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, height);
  return { ctx, w: rect.width, h: height };
}

/** The three channels of one charge, each scaled to its own range. Volts, amps
 *  and degrees share no scale, so the shape carries the information — which is
 *  why there are no numbers on the vertical axis. */
function drawCharge(ctx, box, curve, { alpha = 1, width = 1.9 } = {}) {
  const t = curve.time_s;
  const tMin = t[0], tMax = t[t.length - 1];
  const X = (v) => box.x0 + ((v - tMin) / (tMax - tMin || 1)) * (box.x1 - box.x0);
  const series = [
    { data: curve.voltage_v, color: css("--teal") },
    { data: curve.current_a.map(Math.abs), color: css("--blue") },
    { data: curve.temperature_c, color: css("--violet") },
  ];
  ctx.save();
  ctx.globalAlpha = alpha;
  for (const s of series) {
    const lo = Math.min(...s.data), hi = Math.max(...s.data);
    const Y = (v) => box.y1 - ((v - lo) / (hi - lo || 1)) * (box.y1 - box.y0) * 0.9;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.beginPath();
    s.data.forEach((v, i) => (i ? ctx.lineTo(X(t[i]), Y(v)) : ctx.moveTo(X(t[i]), Y(v))));
    ctx.stroke();
  }
  ctx.restore();
  return { X };
}

/** The shaded strip: the only part of the charge the model reads. */
function drawWindow(ctx, box, curve, X, label) {
  const iLow = curve.voltage_v.findIndex((v) => v >= 3.9);
  const iHigh = curve.voltage_v.findIndex((v) => v >= 4.15);
  if (iLow < 0 || iHigh <= iLow) return;
  const t = curve.time_s;
  ctx.fillStyle = css("--teal") + "1f";
  ctx.fillRect(X(t[iLow]), box.y0, X(t[iHigh]) - X(t[iLow]), box.y1 - box.y0);
  ctx.strokeStyle = css("--teal") + "55";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(X(t[iLow]) + 0.5, box.y0); ctx.lineTo(X(t[iLow]) + 0.5, box.y1);
  ctx.moveTo(X(t[iHigh]) + 0.5, box.y0); ctx.lineTo(X(t[iHigh]) + 0.5, box.y1);
  ctx.stroke();
  if (label) {
    ctx.fillStyle = css("--text-faint");
    ctx.font = "10px " + css("--font-mono");
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(label, X(t[iLow]) + 5, box.y0 + 4);
  }
}

function drawAxisFoot(ctx, box, leftLabel, rightLabel, centreLabel) {
  ctx.fillStyle = css("--text-faint");
  ctx.font = "10px " + css("--font-mono");
  ctx.textBaseline = "top";
  ctx.textAlign = "left"; ctx.fillText(leftLabel, box.x0, box.y1 + 8);
  ctx.textAlign = "right"; ctx.fillText(rightLabel, box.x1, box.y1 + 8);
  ctx.textAlign = "center"; ctx.fillText(centreLabel, (box.x0 + box.x1) / 2, box.y1 + 8);
}

/* ---- chart 1: what a charge looks like, with a hover readout -------------- */
function drawIdea() {
  const canvas = $("#idea-chart");
  if (!canvas || !state.ideaCurve) return;
  const { ctx, w, h } = prepare(canvas);
  const box = { x0: 14, x1: w - 14, y0: 16, y1: h - 26 };
  const curve = state.ideaCurve.curve;

  const { X } = drawCharge(ctx, box, curve);
  drawWindow(ctx, box, curve, X, "the part the model reads");
  drawAxisFoot(ctx, box, "start", `${Math.round(curve.time_s.at(-1) / 60)} min`,
               "time through the charge");

  if (state.hover !== null) {
    const x = X(curve.time_s[state.hover]);
    ctx.strokeStyle = css("--text-muted");
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x + 0.5, box.y0); ctx.lineTo(x + 0.5, box.y1); ctx.stroke();
    ctx.setLineDash([]);
  }
}

function bindHover(canvas, curveGetter, onMove) {
  const move = (event) => {
    const curve = curveGetter();
    if (!curve) return;
    const rect = canvas.getBoundingClientRect();
    const frac = clamp((event.clientX - rect.left - 14) / (rect.width - 28), 0, 1);
    onMove(Math.round(frac * (curve.time_s.length - 1)));
  };
  canvas.addEventListener("mousemove", move);
  canvas.addEventListener("touchmove", (e) => move(e.touches[0]), { passive: true });
  canvas.addEventListener("mouseleave", () => onMove(null));
}

function renderIdeaReadout() {
  const el = $("#idea-readout");
  if (!state.ideaCurve) return;
  if (state.hover === null) {
    el.innerHTML = `<span class="readout__hint">Hover the chart to read it</span>`;
    return;
  }
  const c = state.ideaCurve.curve;
  const i = state.hover;
  const inWindow = c.voltage_v[i] >= 3.9 && c.voltage_v[i] <= 4.15;
  el.innerHTML = `
    <span class="readout__cell"><b>${Math.round(c.time_s[i] / 60)}</b> min in</span>
    <span class="readout__cell readout__cell--teal">${fmt(c.voltage_v[i], 2)} V</span>
    <span class="readout__cell readout__cell--blue">${fmt(Math.abs(c.current_a[i]), 2)} A</span>
    <span class="readout__cell readout__cell--violet">${fmt(c.temperature_c[i], 1)} °C</span>
    <span class="readout__flag">${inWindow ? "inside the strip the model reads" : "outside the strip"}</span>`;
}

/* ---- chart 2: the same battery, at an age you choose --------------------- */
function drawAge() {
  const canvas = $("#age-chart");
  if (!canvas || !state.ageRows.length) return;
  const { ctx, w, h } = prepare(canvas);
  const box = { x0: 14, x1: w - 14, y0: 16, y1: h - 26 };
  const row = state.ageRows[state.ageIndex];

  // The battery when it was new, faint behind, so ageing shows up as a gap
  // rather than as a number that changed while you were not looking.
  if (state.ageNewest && state.ageNewest !== row) {
    drawCharge(ctx, box, state.ageNewest.curve, { alpha: 0.16, width: 1.4 });
  }
  const { X } = drawCharge(ctx, box, row.curve);
  drawWindow(ctx, box, row.curve, X, null);
  drawAxisFoot(ctx, box, "start", `${Math.round(row.curve.time_s.at(-1) / 60)} min`,
               "time through the charge");
}

/* ---- chart 3: predicted vs measured across a whole life ------------------ */
function drawLife() {
  const canvas = $("#life-chart");
  if (!canvas || !state.life || !state.lifeCell) return;
  const { ctx, w, h } = prepare(canvas);
  const rows = state.life[state.lifeCell];
  if (!rows || !rows.length) return;

  const box = { x0: 44, x1: w - 14, y0: 16, y1: h - 30 };
  const cycles = rows.map((r) => r.cycle);
  const values = rows.flatMap((r) => [r.measured_soh_pct, r.predicted_soh_pct]);
  const xMin = Math.min(...cycles), xMax = Math.max(...cycles);
  const yMin = Math.floor(Math.min(...values, 78) / 5) * 5;
  const yMax = Math.ceil(Math.max(...values) / 5) * 5;
  const X = (v) => box.x0 + ((v - xMin) / (xMax - xMin || 1)) * (box.x1 - box.x0);
  const Y = (v) => box.y1 - ((v - yMin) / (yMax - yMin || 1)) * (box.y1 - box.y0);

  ctx.font = "10px " + css("--font-mono");
  for (let i = 0; i <= 4; i++) {
    const y = box.y0 + ((box.y1 - box.y0) * i) / 4;
    ctx.strokeStyle = css("--line");
    ctx.beginPath(); ctx.moveTo(box.x0, y + 0.5); ctx.lineTo(box.x1, y + 0.5); ctx.stroke();
    ctx.fillStyle = css("--text-faint");
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(`${Math.round(yMax - ((yMax - yMin) * i) / 4)}%`, box.x0 - 7, y);
  }

  if (yMin <= 80 && yMax >= 80) {
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = css("--rose");
    ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(box.x0, Y(80)); ctx.lineTo(box.x1, Y(80)); ctx.stroke();
    ctx.restore();
  }

  ctx.strokeStyle = css("--teal");
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.beginPath();
  rows.forEach((r, i) => (i ? ctx.lineTo(X(r.cycle), Y(r.measured_soh_pct))
                            : ctx.moveTo(X(r.cycle), Y(r.measured_soh_pct))));
  ctx.stroke();

  ctx.fillStyle = css("--violet") + "cc";
  for (const r of rows) {
    ctx.beginPath(); ctx.arc(X(r.cycle), Y(r.predicted_soh_pct), 2, 0, Math.PI * 2); ctx.fill();
  }

  const mae = rows.reduce((a, r) => a + Math.abs(r.predicted_soh_pct - r.measured_soh_pct), 0)
            / rows.length;
  drawAxisFoot(ctx, box, "first charge", `charge ${xMax}`,
               `${rows.length} charges · ${fmt(mae)} points off on average`);
}

function redrawAll() { drawIdea(); drawAge(); drawLife(); }
addEventListener("resize", redrawAll);

/* ---- progress rail, which is also the navigation ------------------------- */
function buildRail() {
  const steps = $$(".step");
  $("#rail").innerHTML = steps.map((s, i) =>
    `<a class="rail__item" href="#${s.id}" data-for="${s.id}">
       <span class="rail__n">${i + 1}</span>${s.dataset.title}</a>`).join("");

  // An observer rather than a scroll handler, so it costs nothing while the
  // page is still.
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      $$(".rail__item").forEach((a) =>
        a.classList.toggle("is-on", a.dataset.for === entry.target.id));
    }
  }, { rootMargin: "-45% 0px -50% 0px" });
  steps.forEach((s) => observer.observe(s));
}

/* ---- glossary ------------------------------------------------------------
 * Jargon is defined where it first appears, on hover and on keyboard focus, so
 * a reader never has to leave the sentence to find out what a word means. */
function bindGlossary() {
  const tip = $("#tip");
  const show = (el) => {
    tip.textContent = el.dataset.def;
    tip.hidden = false;
    const r = el.getBoundingClientRect();
    tip.style.left = `${clamp(r.left + r.width / 2 - 130, 10, innerWidth - 270)}px`;
    tip.style.top = `${r.bottom + window.scrollY + 8}px`;
  };
  const hide = () => { tip.hidden = true; };
  $$("abbr[data-def]").forEach((el) => {
    el.tabIndex = 0;
    el.addEventListener("mouseenter", () => show(el));
    el.addEventListener("focus", () => show(el));
    el.addEventListener("mouseleave", hide);
    el.addEventListener("blur", hide);
  });
}

/* ---- step 2: the age scrubber ------------------------------------------- */
function buildScrubber() {
  const cells = Object.keys(state.demo.cells).sort();
  $("#cell-pick").innerHTML = cells.map((c) => `<option value="${c}">${c}</option>`).join("");
  $("#cell-pick").addEventListener("change", (e) => loadCell(e.target.value));

  $("#age-slider").addEventListener("input", (e) => {
    state.ageIndex = Number(e.target.value);
    drawAge();
    renderAgeMeta();
    scheduleAgePrediction();
  });
  loadCell(cells[0]);
}

function loadCell(cellId) {
  state.ageCell = cellId;
  // Newest first, so dragging left to right runs forwards through the
  // battery's life — the direction people expect a timeline to go.
  state.ageRows = [...state.demo.cells[cellId]]
    .sort((a, b) => b.measured_soh_pct - a.measured_soh_pct);
  state.ageNewest = state.ageRows[0];
  state.ageIndex = 0;

  const slider = $("#age-slider");
  slider.max = String(state.ageRows.length - 1);
  slider.value = "0";

  drawAge();
  renderAgeMeta();
  runAgePrediction();
}

function renderAgeMeta() {
  const row = state.ageRows[state.ageIndex];
  $("#scrub-pos").textContent = `charge ${row.cycle_number} of this battery's life`;
  $("#age-note").textContent = `${row.samples} readings · room ${fmt(row.ambient_c, 0)}°C`;
  $("#age-hint").hidden = state.ageIndex === 0;
}

let ageTimer = null;
function scheduleAgePrediction() {
  // Debounced: dragging fires input on every pixel, and the model does not need
  // asking sixty times about charges the user swept straight past.
  clearTimeout(ageTimer);
  $("#age-result").classList.add("is-working");
  ageTimer = setTimeout(runAgePrediction, 90);
}

async function runAgePrediction() {
  const row = state.ageRows[state.ageIndex];
  let result;
  try {
    result = await predictCurve(row);
  } catch (err) {
    $("#age-result").classList.remove("is-working");
    $("#age-result").innerHTML =
      `<p class="muted">The model refused to read this charge.</p>
       <div class="note">${err.detail?.reason || err.message}</div>`;
    return;
  }
  // A slow response for a charge the user has already dragged past must not
  // overwrite the one they are actually looking at.
  if (state.ageRows[state.ageIndex] !== row) return;

  state.agePrediction = result;
  state.whyBase = { ...result.features };
  state.whyNow = { ...result.features };
  state.whyPrediction = result.prediction;

  renderAgeResult(result, row);
  $("#why-reset").hidden = true;
  renderWhy();
  renderTally();
  $("#age-result").classList.remove("is-working");
}

function renderAgeResult(result, row) {
  const p = result.prediction;
  const truth = row.measured_soh_pct;
  const miss = p.soh_pct - truth;
  const inside = Math.abs(miss) <= p.interval_high_pct - p.soh_pct;

  $("#age-result").innerHTML = `
    <div class="verdict-num">
      <span class="verdict-num__v">${fmt(p.soh_pct, 1)}</span><span class="verdict-num__pc">% healthy</span>
    </div>
    <div class="bar">
      <div class="bar__fill" style="width:${clamp(p.soh_pct, 0, 100)}%"></div>
      <div class="bar__eol"></div>
    </div>
    <p class="verdict-band">
      9 times in 10 the truth lands between
      <b>${fmt(p.interval_low_pct, 1)}%</b> and <b>${fmt(p.interval_high_pct, 1)}%</b>
    </p>
    <span class="verdict ${p.end_of_life ? "verdict--eol" : "verdict--ok"}">
      ${p.end_of_life ? "below 80% — treat as worn out" : "still good to use"}
    </span>
    <div class="truth">
      <span class="truth__label">What draining it actually measured</span>
      <span class="truth__v">${fmt(truth, 1)}%</span>
      <span class="truth__miss ${inside ? "is-ok" : "is-off"}">
        model was ${miss >= 0 ? "+" : ""}${fmt(miss, 1)} points out —
        ${inside ? "inside the range it promised" : "outside the range it promised"}
      </span>
    </div>
    ${p.notes.map((n) => `<div class="note">${n}</div>`).join("")}`;
}

/* ---- step 3: the calculation, and the what-if sliders -------------------- */
const NAMES = {
  window_charge_ah: "Charge taken in",
  ic_peak_height: "Biggest charge jump",
  ic_peak_voltage: "Voltage of that jump",
  voltage_slope_v_per_s: "How fast voltage rose",
  cv_phase_seconds: "Time spent topping up",
  temp_rise_c: "Warm-up during the strip",
  temp_max_c: "Hottest it got",
};
const prettyName = (n) => NAMES[n] || n;
const valueDigits = (n) =>
  (n === "voltage_slope_v_per_s" ? 6 : n === "cv_phase_seconds" ? 0 : 2);

function renderWhy() {
  if (!state.whyPrediction || !state.card) return;
  const meta = Object.fromEntries(state.card.features.map((f) => [f.name, f]));
  const max = Math.max(
    ...state.whyPrediction.contributions.map((c) => Math.abs(c.effect_pct)), 0.5);

  $("#why-rows").innerHTML = state.whyPrediction.contributions.map((c) => {
    const f = meta[c.feature] || {};
    const range = f.range || { low: c.value * 0.6, high: c.value * 1.4 };
    const span = (range.high - range.low) || 1;
    const step = span / 100;
    const changed = Math.abs(state.whyNow[c.feature] - state.whyBase[c.feature]) > step / 2;
    const pos = c.effect_pct >= 0;
    const width = (Math.abs(c.effect_pct) / max) * 50;
    return `
      <div class="whyrow ${changed ? "is-changed" : ""}">
        <div class="whyrow__top">
          <span class="whyrow__name">${prettyName(c.feature)}</span>
          <span class="whyrow__val">${fmt(state.whyNow[c.feature], valueDigits(c.feature))} ${c.unit}</span>
          <span class="whyrow__eff ${pos ? "is-pos" : "is-neg"}">${pos ? "+" : ""}${fmt(c.effect_pct)}</span>
        </div>
        <div class="whyrow__track">
          <div class="whyrow__mid"></div>
          <div class="whyrow__fill ${pos ? "is-pos" : "is-neg"}"
               style="left:${pos ? 50 : 50 - width}%;width:${width}%"></div>
        </div>
        <input class="whyrow__slider" type="range"
               data-feature="${c.feature}"
               min="${range.low}" max="${range.high}" step="${step}"
               value="${clamp(state.whyNow[c.feature], range.low, range.high)}"
               aria-label="${prettyName(c.feature)}">
        <p class="whyrow__why">${f.physics || c.physics || ""}</p>
      </div>`;
  }).join("");

  $$(".whyrow__slider").forEach((el) => {
    el.addEventListener("input", () => {
      state.whyNow[el.dataset.feature] = Number(el.value);
      $("#why-reset").hidden = false;
      scheduleWhatIf();
    });
  });
}

let whyTimer = null;
function scheduleWhatIf() {
  clearTimeout(whyTimer);
  whyTimer = setTimeout(runWhatIf, 90);
}

async function runWhatIf() {
  try {
    const result = await postJSON("/api/predict/features",
                                  { features: state.whyNow, confidence: 0.9 });
    state.whyPrediction = result.prediction;
    renderWhy();
    renderTally();
  } catch (err) {
    console.warn("what-if failed", err);
  }
}

$("#why-reset").addEventListener("click", () => {
  state.whyNow = { ...state.whyBase };
  state.whyPrediction = state.agePrediction.prediction;
  $("#why-reset").hidden = true;
  renderWhy();
  renderTally();
});

function renderTally() {
  const p = state.whyPrediction;
  if (!p) return;
  const total = p.contributions.reduce((a, c) => a + c.effect_pct, 0);
  const real = state.agePrediction?.prediction.soh_pct;
  const edited = real !== undefined && Math.abs(p.soh_pct - real) > 0.05;

  $("#tally").innerHTML = `
    <div class="tally__row">
      <span>An average battery starts at</span><b>${fmt(p.baseline_pct, 1)}%</b>
    </div>
    <div class="tally__row">
      <span>These seven measurements add</span>
      <b class="${total >= 0 ? "is-pos" : "is-neg"}">${total >= 0 ? "+" : ""}${fmt(total, 1)}</b>
    </div>
    <div class="tally__rule"></div>
    <div class="tally__row tally__row--total">
      <span>This battery</span><b>${fmt(p.soh_pct, 1)}%</b>
    </div>
    ${edited ? `<div class="note">You have changed a measurement. The real charge
       gives ${fmt(real, 1)}%.</div>` : ""}
    ${p.notes.map((n) => `<div class="note">${n}</div>`).join("")}
    <div class="note note--quiet">${p.contribution_caveat}</div>`;
}

/* ---- step 4: proof ------------------------------------------------------- */
function renderSplits() {
  const m = state.card?.evaluation?.models?.ridge;
  if (!m) return;
  const defs = [
    { key: "random", name: "Shuffle the charges", tag: "flattering",
      why: "Mix all the charges together and hide 20%. The hidden ones sit right next to charges the model trained on, so it can copy its neighbour." },
    { key: "by_cell", name: "Hide a whole battery", tag: "the real test · chosen on this",
      honest: true,
      why: "Train on every battery but one, then predict that one. Repeat for each. This is what happens when you meet a new battery." },
    { key: "forward_time", name: "Young to old", tag: "hardest",
      why: "Train on the first 60% of each battery's life and test on the last 40%. Asks whether it copes with damage it has never seen." },
  ];
  $("#splits").innerHTML = defs.map((d) => {
    const s = m[d.key];
    return `<div class="split ${d.honest ? "split--honest" : ""}">
      <span class="split__tag">${d.tag}</span>
      <p class="split__name">${d.name}</p>
      <div><span class="split__mae">${fmt(s.mae_pct)}</span><span class="split__unit">points off, on average</span></div>
      <div class="split__r2">worst miss ${fmt(s.max_abs_err_pct)} points</div>
      <p class="split__why">${d.why}</p>
    </div>`;
  }).join("");
}

function buildCellSwitch() {
  const cells = Object.keys(state.life).sort();
  $("#cellswitch").innerHTML = cells.map((c) =>
    `<button class="pick" data-cell="${c}">${c}</button>`).join("");
  $$("#cellswitch .pick").forEach((b) => b.addEventListener("click", () => {
    state.lifeCell = b.dataset.cell;
    $$("#cellswitch .pick").forEach((x) => x.classList.toggle("is-on", x === b));
    $("#life-title").textContent = `Battery ${state.lifeCell}, charge by charge`;
    drawLife();
  }));
  $("#cellswitch .pick").click();
}

function renderLimits() {
  $("#limits").innerHTML = (state.card.limitations || [])
    .map((l) => `<li>${l}</li>`).join("");
}

/* ---- guided walkthrough -------------------------------------------------- */
const TOUR = [
  { at: "#step-problem",
    say: "The problem. To know what a battery really holds you normally drain it flat — four hours with the bike off the road. Every battery gets charged anyway, so read that instead." },
  { at: "#step-idea",
    say: "This is one real charge. Voltage climbs, current holds then tapers, the cell warms slightly. The shaded strip is the only part we use: a few minutes, not the whole thing." },
  { at: "#step-age",
    say: "Now drag the slider. Every stop is a real charge from the same battery, newest to oldest. Watch the line pull away from the faint one behind it — that gap is the battery ageing.",
    run: () => { const s = $("#age-slider"); s.value = s.max; s.dispatchEvent(new Event("input")); } },
  { at: "#step-why",
    say: "And it shows its working. Seven measurements, each pushing the answer up or down from an average battery. Drag any slider and the answer moves — there is no black box here." },
  { at: "#step-proof",
    say: "The number that matters. Shuffling the charges flatters the model, because the hidden ones sit beside ones it trained on. Hiding a whole battery is the real test, and that is the figure quoted throughout." },
  { at: "#step-limits",
    say: "And where it stops. Typical error is 1.6 points but the worst miss was 9.3, so this ranks and screens batteries rather than settling a warranty claim. Better said up front than asked." },
];

let tourAt = -1;

function tourShow(index) {
  if (index < 0 || index >= TOUR.length) return tourEnd();
  $$(".tour-focus").forEach((el) => el.classList.remove("tour-focus"));
  tourAt = index;
  const stop = TOUR[index];
  const target = $(stop.at);

  $("#tour").hidden = false;
  $("#tour-step").textContent = `${index + 1} / ${TOUR.length}`;
  $("#tour-text").textContent = stop.say;
  $("#tour-prev").disabled = index === 0;
  $("#tour-next").textContent = index === TOUR.length - 1 ? "Done" : "Next";

  if (stop.run) stop.run();
  if (target) {
    target.classList.add("tour-focus");
    window.scrollTo({ top: Math.max(0, target.getBoundingClientRect().top + scrollY - 110),
                      behavior: "smooth" });
  }
}

function tourEnd() {
  tourAt = -1;
  $("#tour").hidden = true;
  $$(".tour-focus").forEach((el) => el.classList.remove("tour-focus"));
}

$("#tour-start").addEventListener("click", () => tourShow(0));
$("#tour-next").addEventListener("click", () => tourShow(tourAt + 1));
$("#tour-prev").addEventListener("click", () => tourShow(tourAt - 1));
$("#tour-end").addEventListener("click", tourEnd);
addEventListener("keydown", (e) => {
  if (tourAt < 0) return;
  if (e.key === "ArrowRight" || e.key === " ") { e.preventDefault(); tourShow(tourAt + 1); }
  if (e.key === "ArrowLeft") { e.preventDefault(); tourShow(tourAt - 1); }
  if (e.key === "Escape") tourEnd();
});

/* ---- boot ---------------------------------------------------------------- */
(async function boot() {
  buildRail();
  bindGlossary();

  try {
    [state.card, state.demo, state.life] = await Promise.all([
      getJSON("/api/model"),
      getJSON("/api/demo/charges"),
      getJSON("/api/demo/cells").then((d) => d.cells),
    ]);
  } catch (err) {
    $("main").insertAdjacentHTML("afterbegin",
      `<div class="wrap"><div class="callout"><h3>Nothing to show yet</h3>
       <p>${err.message}. Build the model first — see the README.</p></div></div>`);
    return;
  }

  // Step 1 uses a healthy charge: it is showing what a charge IS, and a worn
  // one would be teaching two things at once.
  const first = Object.values(state.demo.cells)[0];
  state.ideaCurve = [...first].sort((a, b) => b.measured_soh_pct - a.measured_soh_pct)[0];
  $("#idea-note").textContent =
    `${state.ideaCurve.cell_id} · ${state.ideaCurve.samples} readings`;
  drawIdea();
  bindHover($("#idea-chart"), () => state.ideaCurve?.curve, (i) => {
    state.hover = i; drawIdea(); renderIdeaReadout();
  });

  buildScrubber();
  renderSplits();
  buildCellSwitch();
  renderLimits();
})();
