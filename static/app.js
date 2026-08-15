/* =============================================================================
 *  Battery SoH — front end
 * =============================================================================
 *  No framework and no chart library. Two charts, both hand-rolled on Canvas2D:
 *  it is about 200 lines, it is device-pixel-ratio aware so the lines are crisp
 *  on a retina screen, and it re-reads the CSS custom properties on theme
 *  toggle so the charts recolour in place. A charting library would have been
 *  more code to configure than to write, and would still have needed the theme
 *  hook written by hand.
 * ========================================================================== */

const $ = (sel) => document.querySelector(sel);
const fmt = (n, d = 2) => Number(n).toFixed(d);

const state = {
  demo: null,        // real charge curves, from /api/demo/charges
  card: null,        // model card, from /api/model
  life: null,        // leave-one-cell-out predictions, from /api/demo/cells
  selected: null,    // the charge currently on screen
  lifeCell: null,    // which cell the life chart is showing
  lastPrediction: null,
};

/* ---- theme ---------------------------------------------------------------
 * Stored so a reload keeps the choice, and defaults to the OS preference the
 * first time rather than assuming dark. */
const root = document.documentElement;
// Paper is this report's primary theme, so an unset preference opens light —
// only an explicit OS preference for dark switches it. The fleet console next
// door does the opposite, because it is a different kind of thing.
const savedTheme = localStorage.getItem("soh-theme");
root.dataset.theme = savedTheme
  || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

$("#theme-toggle").addEventListener("click", () => {
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem("soh-theme", root.dataset.theme);
  redrawAll();
});

const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();

/* ---- canvas helper -------------------------------------------------------
 * Sizes the backing store to the device pixel ratio. Without this every line
 * is soft on a retina display, which on a chart reads as imprecision. */
function prepare(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  // Read from data-h, never from the height attribute. Assigning
  // `canvas.height` *writes* that attribute, so reading it back gives the
  // device-pixel value from the previous call — and the chart doubles in height
  // on every redraw. It only shows up on the second render, which is why the
  // first paint looked fine and a theme toggle or a resize did not.
  const height = Number(canvas.dataset.h) || 300;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.height = height + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, height);
  return { ctx, w: rect.width, h: height };
}

function axes(ctx, box, { xLabel, yLabel, xMin, xMax, yMin, yMax, yTicks = 5,
                          showYValues = true }) {
  ctx.strokeStyle = css("--line");
  ctx.fillStyle = css("--text-faint");
  ctx.font = "11px " + css("--font-mono");
  ctx.lineWidth = 1;

  for (let i = 0; i <= yTicks; i++) {
    const frac = i / yTicks;
    const y = box.y0 + (box.y1 - box.y0) * frac;
    ctx.beginPath();
    ctx.moveTo(box.x0, y + 0.5);
    ctx.lineTo(box.x1, y + 0.5);
    ctx.stroke();
    // Suppressed on the charge chart: each channel there is scaled to its own
    // range, so a shared numeric axis would invite reading volts off a line
    // that is showing amps.
    if (showYValues) {
      const value = yMax - (yMax - yMin) * frac;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(fmt(value, Math.abs(yMax - yMin) > 20 ? 0 : 1), box.x0 - 8, y);
    }
  }

  ctx.textBaseline = "top";
  // Anchored to the ends rather than centred on them, or the last label hangs
  // half off the canvas and reads as a truncated number.
  ctx.textAlign = "left";
  ctx.fillText(fmt(xMin, 0), box.x0, box.y1 + 8);
  ctx.textAlign = "right";
  ctx.fillText(fmt(xMax, 0), box.x1, box.y1 + 8);
  ctx.textAlign = "center";
  ctx.fillText(xLabel, (box.x0 + box.x1) / 2, box.y1 + 8);

  ctx.save();
  ctx.translate(12, (box.y0 + box.y1) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

/* ---- chart 1: the charge curve ------------------------------------------
 * Three channels on one axis pair would be meaningless (volts, amps and
 * degrees share no scale), so each is normalised to its own range and the
 * shape is what carries the information. The shaded band is the voltage
 * window the model reads, drawn so the "few minutes of an ordinary charge"
 * claim is visible rather than asserted. */
function drawCurve() {
  const canvas = $("#curve-chart");
  const { ctx, w, h } = prepare(canvas);
  if (!state.selected) return;

  const c = state.selected.curve;
  const box = { x0: 34, x1: w - 14, y0: 14, y1: h - 30 };

  const t = c.time_s;
  const tMin = t[0], tMax = t[t.length - 1];
  const X = (v) => box.x0 + ((v - tMin) / (tMax - tMin || 1)) * (box.x1 - box.x0);

  axes(ctx, box, {
    xLabel: "seconds into the charge", yLabel: "each line has its own scale",
    xMin: 0, xMax: tMax - tMin, yMin: 0, yMax: 1, yTicks: 4, showYValues: false,
  });

  // The window band: from the first sample at/above 3.90 V to the first at/above 4.15 V.
  const iLow = c.voltage_v.findIndex((v) => v >= 3.9);
  const iHigh = c.voltage_v.findIndex((v) => v >= 4.15);
  if (iLow >= 0 && iHigh > iLow) {
    ctx.fillStyle = css("--teal") + "26";
    ctx.fillRect(X(t[iLow]), box.y0, X(t[iHigh]) - X(t[iLow]), box.y1 - box.y0);
    ctx.fillStyle = css("--text-faint");
    ctx.font = "10px " + css("--font-mono");
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("the part the model reads", X(t[iLow]) + 4, box.y0 + 3);
  }

  const series = [
    { data: c.voltage_v, color: css("--teal") },
    { data: c.current_a.map(Math.abs), color: css("--blue") },
    { data: c.temperature_c, color: css("--violet") },
  ];
  for (const s of series) {
    const lo = Math.min(...s.data), hi = Math.max(...s.data);
    const Y = (v) => box.y1 - ((v - lo) / (hi - lo || 1)) * (box.y1 - box.y0) * 0.92;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.8;
    ctx.lineJoin = "round";
    ctx.beginPath();
    s.data.forEach((v, i) => (i ? ctx.lineTo(X(t[i]), Y(v)) : ctx.moveTo(X(t[i]), Y(v))));
    ctx.stroke();
  }
}

/* ---- chart 2: predicted vs measured over a cell's life ------------------- */
function drawLife() {
  const canvas = $("#life-chart");
  const { ctx, w, h } = prepare(canvas);
  if (!state.life || !state.lifeCell) return;

  const rows = state.life[state.lifeCell];
  if (!rows || !rows.length) return;

  const box = { x0: 52, x1: w - 14, y0: 14, y1: h - 34, pad: 0 };
  const cycles = rows.map((r) => r.cycle);
  const values = rows.flatMap((r) => [r.measured_soh_pct, r.predicted_soh_pct]);
  const xMin = Math.min(...cycles), xMax = Math.max(...cycles);
  const yMin = Math.floor(Math.min(...values, 78) / 5) * 5;
  const yMax = Math.ceil(Math.max(...values) / 5) * 5;

  const X = (v) => box.x0 + ((v - xMin) / (xMax - xMin || 1)) * (box.x1 - box.x0);
  const Y = (v) => box.y1 - ((v - yMin) / (yMax - yMin || 1)) * (box.y1 - box.y0);

  axes(ctx, box, {
    xLabel: "charge number", yLabel: "health  %",
    xMin, xMax, yMin, yMax, yTicks: 5,
  });

  // End-of-life threshold.
  if (yMin <= 80 && yMax >= 80) {
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = css("--rose");
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(box.x0, Y(80));
    ctx.lineTo(box.x1, Y(80));
    ctx.stroke();
    ctx.restore();
  }

  ctx.strokeStyle = css("--teal");
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.beginPath();
  rows.forEach((r, i) => (i
    ? ctx.lineTo(X(r.cycle), Y(r.measured_soh_pct))
    : ctx.moveTo(X(r.cycle), Y(r.measured_soh_pct))));
  ctx.stroke();

  ctx.fillStyle = css("--violet") + "cc";
  for (const r of rows) {
    ctx.beginPath();
    ctx.arc(X(r.cycle), Y(r.predicted_soh_pct), 2.1, 0, Math.PI * 2);
    ctx.fill();
  }

  const mae = rows.reduce((a, r) =>
    a + Math.abs(r.predicted_soh_pct - r.measured_soh_pct), 0) / rows.length;
  ctx.fillStyle = css("--text-faint");
  ctx.font = "11px " + css("--font-mono");
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(`${rows.length} charges · ${fmt(mae)} pts off on average`, box.x1, box.y0 + 2);
}

function redrawAll() { drawCurve(); drawLife(); }
addEventListener("resize", redrawAll);

/* ---- rendering ----------------------------------------------------------- */

function renderSplits() {
  const models = state.card?.evaluation?.models?.ridge;
  if (!models) return;
  const defs = [
    { key: "random", name: "Shuffle the charges", tag: "flattering",
      why: "Mix all the charges up and hide 20%. The hidden ones sit right next to charges the model trained on, so it can copy its neighbour." },
    { key: "by_cell", name: "Hide a whole battery", tag: "the real test · chosen on this", honest: true,
      why: "Train on every battery but one, then predict that one. Repeat for each. This is what happens when you meet a new battery." },
    { key: "forward_time", name: "Young to old", tag: "hardest",
      why: "Train on the first 60% of each battery's life and test on the last 40%. Asks whether it copes with damage it has never seen." },
  ];
  const worst = Math.max(...defs.map((d) => models[d.key].mae_pct));
  $("#splits").innerHTML = defs.map((d) => {
    const m = models[d.key];
    return `<div class="split ${d.honest ? "split--honest" : ""}">
      <span class="split__tag">${d.tag}</span>
      <p class="split__name">${d.name}</p>
      <div><span class="split__mae">${fmt(m.mae_pct)}</span><span class="split__unit">points off, on average</span></div>
      <div class="split__r2">worst miss ${fmt(m.max_abs_err_pct)} pts</div>
      <div class="split__bar"><span style="width:${(m.mae_pct / worst) * 100}%"></span></div>
      <p class="split__why">${d.why}</p>
    </div>`;
  }).join("");
}

function renderHeroStats() {
  const perf = state.card?.honest_performance;
  const trained = state.card?.trained_on;
  if (!perf || !trained) return;
  const set = (k, v) => { const el = document.querySelector(`[data-stat="${k}"]`); if (el) el.textContent = v; };
  set("mae", "±" + fmt(perf.mae_pct) + " pts");
  set("cells", trained.cells.length);
  set("cycles", trained.cycles);
  set("features", state.card.features.length);
}

function renderPicker() {
  const cells = state.demo?.cells || {};
  const items = [];
  for (const [cell, rows] of Object.entries(cells)) {
    for (const r of rows) items.push(r);
  }
  items.sort((a, b) => a.cell_id.localeCompare(b.cell_id) || a.cycle_number - b.cycle_number);
  $("#picker").innerHTML = items.map((r, i) =>
    `<button class="pick" data-i="${i}">${r.cell_id} · charge ${r.cycle_number} · ${fmt(r.measured_soh_pct, 0)}%</button>`).join("");
  $("#picker").querySelectorAll(".pick").forEach((btn) => {
    btn.addEventListener("click", () => select(items[+btn.dataset.i], btn));
  });
  if (items.length) select(items[0], $("#picker .pick"));
}

async function select(row, btn) {
  state.selected = row;
  document.querySelectorAll(".pick").forEach((b) => b.classList.toggle("is-on", b === btn));
  $("#curve-note").textContent =
    `${row.samples} readings · room ${fmt(row.ambient_c, 0)}°C`;
  drawCurve();
  await predict(row);
}

async function predict(row) {
  $("#result").innerHTML = `<p class="muted">Reading the charge…</p>`;
  let res, body;
  try {
    res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...row.curve, confidence: 0.9 }),
    });
    body = await res.json();
  } catch (err) {
    $("#result").innerHTML = `<p class="muted">Could not reach the model: ${err}</p>`;
    return;
  }
  if (!res.ok) {
    const reason = body?.detail?.reason || "unknown";
    $("#result").innerHTML =
      `<p class="muted">The model refused to answer for this charge.</p>
       <div class="note">${reason}</div>`;
    $("#contributions").innerHTML = `<p class="muted">—</p>`;
    return;
  }

  const p = body.prediction;
  state.lastPrediction = p;
  const truth = row.measured_soh_pct;
  const miss = p.soh_pct - truth;

  $("#result").innerHTML = `
    <div class="soh"><span class="soh__v">${fmt(p.soh_pct, 1)}</span><span class="soh__pc">% SoH</span></div>
    <div class="soh__band">9 times out of 10 the true value lands between ${fmt(p.interval_low_pct, 1)}% and ${fmt(p.interval_high_pct, 1)}%</div>
    <span class="verdict ${p.end_of_life ? "verdict--eol" : "verdict--ok"}">
      ${p.end_of_life ? "below 80% — treat as worn out" : "still good to use"}</span>
    <div class="truth">
      What the full drain test measured afterwards: <b>${fmt(truth, 1)}%</b><br>
      The model was <b>${miss >= 0 ? "+" : ""}${fmt(miss, 1)} points</b> out
      ${Math.abs(miss) <= p.interval_high_pct - p.soh_pct
        ? "— inside the range it promised." : "— outside the range it promised."}
    </div>
    ${p.notes.map((n) => `<div class="note">${n}</div>`).join("")}`;

  renderContributions(p, body.features);
}

function renderContributions(p, features) {
  const max = Math.max(...p.contributions.map((c) => Math.abs(c.effect_pct)), 0.01);
  const byName = Object.fromEntries((state.card?.features || []).map((f) => [f.name, f]));
  $("#contributions").innerHTML = p.contributions.map((c) => {
    const frac = Math.abs(c.effect_pct) / max;
    const pos = c.effect_pct >= 0;
    const width = frac * 50;
    const left = pos ? 50 : 50 - width;
    return `<div class="contrib">
      <div>
        <span class="contrib__name">${c.feature}</span>
        <span class="contrib__val">${fmt(features[c.feature], 3)} ${c.unit}</span>
      </div>
      <div class="contrib__track">
        <div class="contrib__mid"></div>
        <div class="contrib__fill contrib__fill--${pos ? "pos" : "neg"}"
             style="left:${left}%;width:${width}%"></div>
      </div>
      <div class="contrib__num">${pos ? "+" : ""}${fmt(c.effect_pct)}</div>
      <p class="contrib__why">${byName[c.feature]?.physics || c.physics || ""}</p>
    </div>`;
  }).join("")
    + `<p class="muted">An average battery starts at ${fmt(p.baseline_pct, 1)}%.
       Add every bar above to that and you get the answer exactly — this is the
       arithmetic itself, not a rough guide to it.</p>`
    // The caveat rides with every prediction rather than appearing only when
    // something looks off, because it is true of every prediction.
    + (p.contribution_caveat ? `<div class="note">${p.contribution_caveat}</div>` : "");
}

function renderCard() {
  $("#feature-list").innerHTML = (state.card.features || []).map((f) =>
    `<div class="feature">
      <span class="feature__n">${f.name}</span>
      <span class="feature__d">${f.unit} · ${f.direction}</span>
      <p class="feature__p">${f.physics}</p>
    </div>`).join("");
  $("#limits").innerHTML = (state.card.limitations || [])
    .map((l) => `<li>${l}</li>`).join("");
}

function renderLifeSwitch() {
  const cells = Object.keys(state.life || {}).sort();
  $("#cellswitch").innerHTML = cells.map((c) =>
    `<button class="pick" data-cell="${c}">${c}</button>`).join("");
  $("#cellswitch").querySelectorAll(".pick").forEach((b) => {
    b.addEventListener("click", () => {
      state.lifeCell = b.dataset.cell;
      $("#cellswitch").querySelectorAll(".pick")
        .forEach((x) => x.classList.toggle("is-on", x === b));
      $("#life-title").textContent = `Battery ${state.lifeCell}`;
      drawLife();
    });
  });
  if (cells.length) $("#cellswitch .pick").click();
}

/* ---- boot ---------------------------------------------------------------- */
(async function boot() {
  const get = async (url) => {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url} -> ${r.status}`);
    return r.json();
  };
  try {
    [state.card, state.demo, state.life] = await Promise.all([
      get("/api/model"),
      get("/api/demo/charges"),
      get("/api/demo/cells").then((d) => d.cells),
    ]);
  } catch (err) {
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      `<div class="wrap"><div class="note">The model artifacts are not built yet
       (${err.message}). Run the pipeline — see the README.</div></div>`);
    return;
  }
  renderHeroStats();
  renderSplits();
  renderCard();
  renderPicker();
  renderLifeSwitch();
})();

/* ---- guided walkthrough --------------------------------------------------
 * Built for showing this to a room. One button steps through the page in
 * order, scrolling to each part and putting a plain sentence under it, so the
 * story does not depend on the presenter remembering what to click next.
 *
 * It drives the real page rather than a slideshow: step 3 actually picks a
 * worn battery and runs it through the live model. Arrow keys and Escape work,
 * because clicking a small button while talking is awkward.
 */
const TOUR = [
  {
    at: ".hero",
    say: "The problem: to know what a battery really holds, you normally drain it flat and count. That is about four hours with the bike off the road. This reads a charge you were doing anyway.",
  },
  {
    at: "#finding",
    say: "The same model scored three ways. Shuffling the charges flatters it, because the hidden charges sit next to ones it trained on. Hiding a whole battery is the real test — that is the middle number, and it is the one quoted everywhere here.",
  },
  {
    at: "#try",
    say: "Here is a real charge from the lab. The shaded strip is the only part the model reads — a few minutes, not the whole thing.",
    run: () => {
      // A worn battery, so the number on screen is one worth talking about.
      const buttons = [...document.querySelectorAll("#picker .pick")];
      const worn = buttons.find((b) => /· 6\d%/.test(b.textContent)) || buttons[0];
      worn?.click();
    },
  },
  {
    at: "#contributions",
    say: "And it shows its working. Each bar is how far one measurement pushed the answer. Add them to the starting point and you get the result exactly — nothing is hidden in a black box.",
  },
  {
    at: "#life",
    say: "One battery's whole life. The line is what draining it actually measured; the dots are what the charge alone predicted. The model had never seen this battery. Watch where it crosses 80%, the line where a battery is usually retired.",
  },
  {
    at: "#card",
    say: "And where it stops. Typical error is 1.6 points but the worst miss was 9.3, so this ranks and screens batteries — it does not settle a warranty claim. Say that before someone else asks it.",
  },
];

let tourAt = -1;

function tourShow(index) {
  if (index < 0 || index >= TOUR.length) return tourEnd();
  document.querySelectorAll(".tour-focus").forEach((el) => el.classList.remove("tour-focus"));

  tourAt = index;
  const stop = TOUR[index];
  const target = document.querySelector(stop.at);

  $("#tour").hidden = false;
  $("#tour-step").textContent = `${index + 1} / ${TOUR.length}`;
  $("#tour-text").textContent = stop.say;
  $("#tour-prev").disabled = index === 0;
  $("#tour-next").textContent = index === TOUR.length - 1 ? "Done" : "Next";

  if (stop.run) stop.run();
  if (target) {
    target.classList.add("tour-focus");
    // Leave room for the strip at the bottom so the section is not hidden
    // behind the thing describing it.
    const y = target.getBoundingClientRect().top + window.scrollY - 90;
    window.scrollTo({ top: Math.max(0, y), behavior: "smooth" });
  }
}

function tourEnd() {
  tourAt = -1;
  $("#tour").hidden = true;
  document.querySelectorAll(".tour-focus").forEach((el) => el.classList.remove("tour-focus"));
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
