# Battery State-of-Health Prediction

**How much capacity is left in this pack — without discharging it?**

Measuring a battery's true capacity means running it flat and counting the
amp-hours. That takes hours and takes the vehicle off the road, so nobody does
it to a customer's e-bike. But every pack gets *charged*, and a charger already
measures voltage, current and temperature.

This project estimates state of health from a **single ordinary charge**, with
no knowledge of the pack's history. Trained on NASA's cell-ageing dataset and
scored on cells it had never seen:

| | |
|---|---|
| **Mean error on unseen cells** | **±1.63 percentage points of SoH** |
| 90% of predictions land within | ±3.80 points |
| R² on unseen cells | 0.920 |
| Trained on | 898 measured cycles, 8 cells, 64–101% SoH |
| Model | Ridge regression, 7 features, ~2 KB of JSON |

Live demo: pick a real charge curve, watch the model read it, and compare
against the full discharge that actually followed.

---

## The part worth reading

Most published accuracy on this dataset is measured the wrong way, and the gap
is larger than the difference between any two algorithms.

Run the same model three ways:

| Split | What it holds out | MAE | R² |
|---|---|---|---|
| **Random cycles** | 20% of cycles, shuffled | 1.23 | 0.957 |
| **Leave one cell out** | an entire cell | **1.63** | **0.920** |
| **Early life → late life** | the last 40% of every cell's life | 1.70 | 0.848 |

The random split is the one usually reported, and it is close to meaningless
here. Cycle 100 and cycle 101 of the same cell are nearly the same measurement
an hour apart; with one in train and one in test, a model scores well by
interpolating between neighbours it has already seen. It answers *"can you fill
a gap in an experiment you already watched"* — which is not a question anyone
has.

Holding out a whole cell asks the real one: here is a pack you have never met.

**This is also why the simplest model ships.** Ranked on the naive split you
would pick gradient boosting. Ranked on held-out cells it is the worst of the
three:

| Model | Random MAE | By-cell MAE | Flattered by |
|---|---|---|---|
| **Ridge** | 1.23 | **1.63** | +0.39 |
| Random forest | 0.84 | 2.55 | +1.71 |
| Gradient boosting | **0.76** | 2.67 | +1.91 |

Gradient boosting is the best model on the split that does not matter and the
worst on the one that does. With ~900 rows from 8 cells there is more than
enough freedom to memorise eight degradation trajectories, and the random split
pays for that rather than punishing it. The selection rule — lowest by-cell MAE
— is written down in `pipeline/train.py` before the numbers are seen, so the
choice cannot drift to whichever split flatters a preferred model.

---

## What the model is allowed to look at

A feature qualifies only if a charger could compute it from **one charge
event**, with no memory of the cell:

| Feature | Physics |
|---|---|
| `window_charge_ah` | Charge accepted between 3.90 V and 4.15 V. Lithium lost to SEI growth means the cell crosses the same voltage window on less charge. |
| `ic_peak_height` | Peak of dQ/dV in the window. Each incremental-capacity peak is an electrode phase transition; its height scales with how much active material still participates. |
| `ic_peak_voltage` | Where that peak sits. Growing internal resistance adds an IR offset to every measured voltage, sliding the feature upward with age. |
| `voltage_slope_v_per_s` | How fast voltage climbs through the window, by least squares over every sample rather than two endpoints. |
| `cv_phase_seconds` | Time in the constant-voltage tail. A degraded cell hits the voltage limit early on resistance alone, then takes the rest of its charge slowly. |
| `temp_rise_c` | Warming across the window. Ohmic heating goes as I²R, and R grows with age. |
| `temp_max_c` | Peak temperature over the charge — partly health, partly ambient. Kept separate from the rise so the two can be told apart. |

**Deliberately excluded**, and there is a test that fails if any of them
reappears:

- **Cycle number** — the strongest predictor in the dataset and useless in the
  field. Capacity falls with cycle count by construction, so a model given it
  learns the experiment's schedule rather than the battery's physics. A
  second-hand pack does not arrive with its cycle count written on it.
- **Previous capacity** — same problem, worse. It is what we are predicting.
- **Cell identity** — 8 cells is not enough to learn per-cell offsets that
  generalise. It is exactly enough to memorise them.

The window (3.90–4.15 V) sits inside the constant-current phase, where the
charger holds current fixed and lets voltage climb. That is what makes the
timing a clean read on charge acceptance; in the constant-voltage phase the
logic inverts and the two would average into mush.

---

## Two things that went wrong, and what they cost

Both are in the code with the reasoning attached, because they are the parts
worth stealing.

### 1. Three part-charged curves were poisoning the model

The extractor rejects a charge that starts above the measurement window — the
pack is already part-charged, so the window is truncated. The original margin
was 0.01 V, which satisfied the letter of that rule and missed its point.

Three charges in the NASA set begin at **3.865–3.873 V**, some 30 mV under the
window. They passed. They also cross the window roughly **six times faster**
than the next cycle of the same cell at the same measured capacity — the cell
had not aged between those two charges, it simply started most of the way full.

Left in, they were the model's three worst predictions by a wide margin: one
cell's *healthiest* cycle, at 101.8% SoH, was read as 74.9%.

Requiring 0.15 V of run-up (~5% state of charge) before the window costs 3
cycles out of 902 and moved:

- by-cell MAE **1.74 → 1.63**
- early-life-to-late-life MAE **3.11 → 1.70**

### 2. Two features were the same measurement twice

The first version also carried `cc_window_seconds` — time to cross the window.
It correlated with `window_charge_ah` at **r = +1.00**. Not "highly", exactly:
inside the constant-current phase, charge is current × time, so the two columns
were the same number in different units.

`window_charge_ah` is the one kept, and not because it scored better (0.05
points apart, which is noise). It survives contact with a real charger:
integrated charge stays meaningful when current wobbles, where a
voltage-crossing time quietly becomes a measurement of the charger.

---

## The limit of the explanation

The app shows a per-feature breakdown of every prediction, and because the
model is linear on standardised features those contributions sum to the answer
*exactly* — it is an accounting of the number, not an approximation of it.
There is a test that asserts the sum.

That is not the same as each bar being a physical claim, and the app says so on
every prediction rather than only when something looks odd.

`window_charge_ah`, `ic_peak_height` and `voltage_slope_v_per_s` all measure how
much charge the cell accepts, and they correlate with each other above 0.9. When
inputs are that entangled the split of credit *between* them is not identified:
weight can move from one to another, and a sign can flip, with almost no change
to the prediction. `voltage_slope_v_per_s` does exactly that — it takes a
positive coefficient although its own correlation with SoH is negative.

Regularising until every sign agreed with its physics was tried. It takes
alpha ≈ 1000, and by-cell MAE goes from 1.7 to 4.0. Buying a tidier story with
more than twice the error is the wrong trade; pretending the story was tidy
would be worse. So the group is interpretable, the individual split is not, and
the measured correlations ship inside `artifacts/model.json` so nobody has to
take that on trust.

---

## Why 8 cells out of 34

The archive holds 34 cells and it is tempting to use all of them. It would also
be wrong, quietly.

These cells were not run under one protocol — varying the protocol was the point
of the experiment. Across the archive the discharge rate is 1 A, 2 A or 4 A; the
cut-off runs from 1.84 V to 2.69 V; ambient is 4, 24, 43 or 44 °C; some cells
switch protocol partway through their own life.

That breaks the label. `Capacity` is the amp-hours delivered by *that*
discharge, so it measures the cell's health only when the discharge conditions
are fixed. Pull 4 A instead of 2 A and an identical cell measures lower. A model
trained across all 34 would spend most of its capacity predicting the
experimenter's settings, and the error would look like ordinary model noise
rather than a category error.

`pipeline/ingest.py` admits a cycle only under the reference protocol — 24 ± 3 °C,
1.5 ± 0.2 A charge, 2.0 ± 0.25 A discharge, capacity within a plausible
0.8–2.2 Ah — then requires a cell to have ≥15 admissible cycles and a capacity
series that actually declines (Spearman ρ ≤ −0.30). That last gate rejects on
the *label*, before any model is fitted, so it cannot be tuning for a nicer
score.

Every rejection and its reason is written to `data/processed/admissions.json`.
Two thirds of the archive does not qualify, and saying so is the result.

Included: `B0005 B0006 B0007 B0018 B0036 B0042 B0043 B0044`.

> The duplicated `B0025`–`B0028` files, which ship twice in different
> sub-archives, are de-duplicated by cell id. A repeated cell would otherwise
> sit on both sides of a leave-one-cell-out split and silently undo the whole
> evaluation.

---

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

The trained model and the demo curves are committed, so the app runs
immediately:

```bash
.venv/bin/uvicorn app.main:app --reload --port 8002
```

Then open <http://localhost:8002>.

### Rebuilding the model from raw data

The raw dataset is a 209 MB download and is **not** in the repository.

```bash
curl -L -o nasa.zip "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
unzip nasa.zip -d data/raw && (cd data/raw/*/ && for z in *.zip; do unzip -q "$z" -d "${z%.zip}"; done)
```

```bash
.venv/bin/python -m pipeline.ingest         # .mat -> paired, protocol-filtered cycles
.venv/bin/python -m pipeline.build_dataset  # cycles -> features.csv
.venv/bin/python -m pipeline.train          # evaluate 3 splits x 3 models, export
.venv/bin/python -m pipeline.export_demo    # a few real curves for the UI
```

`data/processed/features.csv` is committed (86 KB), so the model is
reproducible from a clone without the download. The 63 MB parsed intermediate
(`cycles.npz`) is not — it is regenerated by the ingest step in about a minute.

### Tests

```bash
.venv/bin/python -m pytest
```

64 tests. They cover the physics (turn one dial on a simulated cell and assert
the feature moves the way its docstring claims), the refusals, the
charge-to-discharge join, the leakage tripwires — the standardiser must be
fitted inside the fold; metadata columns must never reach the model — and the
committed artifacts, which fail if `model.json` and `app/features.py` have
drifted apart.

The simulated cell in `tests/synthetic.py` is used **only** for testing. Nothing
in the shipped model was trained or evaluated on it.

---

## How it is put together

```
app/
  features.py   charge curve -> 7 numbers. Imported by training AND serving,
                which is the whole defence against train/serve skew.
  model.py      loads model.json, predicts, explains. No scikit-learn import.
  main.py       FastAPI. Refuses what it cannot read.
pipeline/
  ingest.py     NASA .mat -> paired cycles, with the protocol filter
  build_dataset.py  cycles -> features.csv
  train.py      the three splits, model selection, JSON export
  export_demo.py    real curves for the demo, verified after decimation
artifacts/      model.json, evaluation.json, demo curves — all committed
static/         no framework, no build step, two hand-rolled Canvas charts
```

Three decisions worth calling out:

**Feature extraction is shared between training and serving.** The classic way
to ruin a deployed model is to extract features one way in the notebook and
subtly differently in the service — a different smoothing window, seconds vs
minutes, a filter applied in one place. Every answer is then slightly wrong and
nothing crashes. One function makes that impossible rather than unlikely.

**The model is JSON, not a pickle.** Unpickling executes code, so a model file
would become an attack surface the moment anyone could replace it; a pickle is
also welded to the exact scikit-learn that wrote it. As JSON it is 7
coefficients and a standardiser — which means the *deployed service needs numpy
and nothing else*. No scikit-learn, no scipy. `requirements.txt` is three lines;
the training stack lives in `requirements-dev.txt`.

**It refuses charges it cannot read.** A truncated or part-charged curve returns
`422` with the reason, not a 200 with a guess. A caller automating pack
replacement needs to tell those apart, and the distinction has to survive the
network boundary.

---

## What it is not

- Eight 18650 cells from one experiment, all at room temperature under one
  protocol. A different chemistry, format or ambient temperature is out of scope
  until measured, not assumed.
- Trained between 64% and 101% SoH. It has never seen a brand-new cell or one
  well below end-of-life, and it flags predictions that fall outside that band.
- It reads a **cell**, not a pack. A module in series is a different
  measurement, and its weakest cell is what matters.
- **Not a safety system.** It estimates capacity. It says nothing about internal
  shorts, swelling, or thermal runaway risk.

Per-cell error is uneven — 0.90 points on B0005, 3.49 on B0044 — and the spread
is reported rather than averaged away. B0042–B0044 contribute 39 cycles each
over a narrow ~10-point SoH range, which is why their R² is poor even where
their MAE is fine: R² is unkind to a narrow target range, and MAE is the number
an operator would actually act on.

## What I would do next

- **Impedance is in the dataset and unused.** Every cell carries EIS sweeps, and
  charge-transfer resistance is a direct read on ageing. It is excluded because
  it is not something an ordinary charger measures, which would break the
  project's premise — but a workshop tool with an EIS rig is a different product
  and probably a better one.
- **Serving a tree ensemble** would need a tree-to-JSON exporter, which this repo
  does not have. `train.py` deliberately fails rather than silently falling back
  to a pickle if a tree model ever beats ridge by more than the stated margin.
- **More cells under one protocol.** Eight is enough to show the by-cell gap is
  real; it is not enough to characterise how wide it gets across manufacturing
  batches.

---

## Data

NASA Ames Prognostics Center of Excellence — Battery Data Set (B. Saha and
K. Goebel). Public domain, from the
[PCoE datasets repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

MIT licensed. Built by [Jayaprakash M](https://jayaprakash627.github.io).
