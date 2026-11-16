# Design decisions and how the system actually works

Written as a record of *why* the code looks like it does, and — more usefully —
what was tried and rejected. The rejected things are the expensive part to
rediscover, because most of them sound obviously correct until measured.

Every number here came from this project's own 13-photo set and 12-slot drawer.
Re-measure before trusting any of them on a different drawer.

---

## 1. How the app operates, end to end

```
camera frame
   1. LOCATE      find the drawer outline        -> live / held / closed
   2. RECTIFY     warp to a fixed canonical size (1000x700)
   3. REGISTER    ECC-align onto baseline.png    -> cancels residual drift
   4. MOTION GATE anything moving? -> freeze everything, evaluate nothing
   5. SCORE       per slot, normalised correlation against TWO references
   6. DECIDE      nearer reference wins; too close -> UNSURE, hold last state
   7. DEBOUNCE    reading must hold for HOLD_S while static
   8. LOG         append to tools.db (SQLite, WAL)
   9. DRIFT       a long-settled, unambiguous slot nudges its own reference
```

**The contract everything depends on:** every image reaching step 5 must land in
the same coordinate frame as the references. Steps 1–3 exist only to guarantee
that. `warp_size` must never change after references are built.

### Files and their roles

| File | Role |
|---|---|
| `tracker.py` | Runtime *and* the shared library the other scripts import. `--selftest` runs every logic check with no camera. |
| `calibrate.py` | Define slots; `--refit` auto-fits them from the bare drawer (see §6 — measured worse, off by default). |
| `tune.py` | Measures `grad_weight` and `uncertain_margin` from labelled photos. |
| `diagnose.py` | Camera probe. Delete once running. |
| `baseline.png` / `empty.png` | The two references. **Both required** — see §3. |
| `config.json` | Localizer choice, slot geometry, scoring knobs. |
| `diagnostics/` | Evidence images with their own README. Not loaded at runtime. |

---

## 2. Localization: markers vs markerless

`config.json: "localizer"` selects `"aruco"` (four printed markers) or `"auto"`
(markerless). **Shipped default is `aruco`** — it is cheaper and cannot be fooled
by a cluttered bench.

Markerless took five attempts. The four failures are worth recording:

| Attempt | Result | Why it failed |
|---|---|---|
| Canny -> `approxPolyDP` quad | 2/13 located | Needs a *closed* contour. Shadows break the drawer outline, so it never closes. Largest convex quad was a label strip at 1% of frame. |
| `HoughLinesP` + extreme lines | 0/13 | Global vote threshold discards faint fragmented edges: 19–40 lines found, **0–1 vertical**. Two verticals are the minimum for a corner. |
| HSV dark mask | 13/13 "located", all wrong | Dark blobs bridge into chair, cabinet, shadows; the region spans the whole frame. |
| Dark mask + hole fill + rectangularity | Closest failure | Still merges the drawer with the cabinet it is mounted in. |

**What worked (`lsd_corners`):** `FastLineDetector` instead of Hough — it works
locally on the gradient field and merges collinear fragments, giving ~900
segments and **14 usable verticals** on the same frames. Then, instead of picking
the extreme line per group (which grabs whatever desk edge is furthest out),
every pair of horizontals is crossed with every pair of verticals and each
candidate rectangle is **verified on evidence it did not choose itself**:

- *perimeter support* — how much of its own outline is really drawn by segments
- *interior darkness* — a drawer cavity is darker than its surround

Result: 13/13 located, quality indistinguishable from ArUco
(canonical-frame consistency 0.81 median / 0.75 worst, vs ArUco 0.82 / 0.76).

### Performance: 4,100 ms -> 30 ms

Profile before optimising — the first guess was wrong. LSD detection was only
62 ms; **81% of the time was the hypothesis search.**

| Change | Effect |
|---|---|
| Detect at `work_width` (640) not full resolution | 4100 -> 263 ms |
| Prune line pairs by spacing before any geometry | 263 -> 238 ms |
| **Support precomputed per *line*, not scanned per candidate edge** | 238 -> 145 ms |
| Verify the existing lock instead of re-searching (`track_support`) | 145 -> **30 ms tracked** |

The third is the one to remember: thousands of candidate edges all lie along the
same ~28 lines, so coverage becomes a per-line prefix sum and each edge is a
lookup instead of a 500-segment scan.

`work_width` is **not only a speed knob** — 640 was also more *accurate* than
960, because downsampling suppresses the fine texture that spawns competing short
segments. Do not assume bigger is better; re-measure.

### Liveness (explicit requirement)

`auto` re-locates every frame and reports `live` / `held` / `closed`. A shut
drawer stops being warped entirely and every debouncer freezes — reading "all
twelve tools taken" off a drawer front would be far worse than reading nothing.
The `track_support` fast path is evidence-based, not a cache: the old edges must
still be *drawn* in the new frame, so a closing drawer falls through to a full
search and then to `closed`.

### Known limitation: orientation

**Markerless recovers the rectangle but not which way is up.** ArUco marker *ids*
are pinned to physical corners; a bare quad has no such identity. A drawer shot
from the far side yields a 180°-rotated canonical frame, which ECC cannot rescue
(a local optimiser has no path from there). This is why switching localizers
requires rebuilding references and slots, not flipping a flag. Fix, when wanted:
score all four rotations against the baseline once at calibration and store it.

---

## 3. Scoring: nearest of two references, never a threshold

Replaced the original mean-absolute-BGR-difference-vs-threshold scheme.

Each slot is matched against **both** references (`baseline.png` = full,
`empty.png` = bare) using `TM_CCOEFF_NORMED`, and whichever correlates more
strongly wins. Ties count as present; absent must win outright.

Why normalised correlation: it subtracts each window's own mean and divides by
its own energy, so dimming the room barely moves the score. A raw pixel
difference moves with every lighting change, which is what forced per-slot magic
numbers that then drifted. **There are no per-slot thresholds any more.**

Four cached templates per slot — grey and Sobel gradient, for full and empty —
blended by `grad_weight`, matched over a `±search_radius` window (ECC aligns
globally, but a 3px residue invalidates a comparison).

Without `empty.png` nothing can be classified: the decision *is* the comparison,
and inventing an absolute threshold to fill the gap is exactly what this design
removes. The tracker reports `UNSURE` for everything and says so at startup.

**`UNSURE` is a feature.** A gap under `uncertain_margin` holds the last state
rather than guessing. An audit trail cannot express "probably", so a late event
beats a wrong one.

---

## 4. The two defects found by measuring, not reading

**`empty.png` was misaligned with `baseline.png`.** ECC had to apply a 1.38x
x-scale and up to 95px of corner displacement. Runtime frames were only ever
registered onto the *baseline*, so every "empty" template was reading the wrong
patch of foam and no tool could ever score as missing. Fixed at the root by
registering the two references to each other at load time
(`load_references`). This took slots that could not be told apart from 7 to 1.

**Two photos were mislabelled.** Filenames written by the `s` key record what the
tracker *believed*, which is wrong whenever the tracker was wrong. One claimed 4
tools out when the drawer was completely bare; one claimed a tool gone while it
sat in its slot. Between them they made 7 of 12 slots look inseparable. Renaming
them took overlaps to zero. **Always eyeball a photo before trusting an OVERLAP
verdict from `tune.py`.**

---

## 5. Current measured state

Configuration: `localizer: aruco`, `grad_weight: 0.2`, `uncertain_margin: 0.073`,
`search_radius: 5`, slots as hand-drawn rects.

**13 photos x 12 slots = 156 readings: 0 wrong, 11 UNSURE, min margin 0.049.**

The 11 unsure are three slots whose clusters sit near zero — `301` worst, whose
535x114 rect is mostly shared background, so both references match it and the
full one wins even on a bare drawer. It honestly reports "don't know" rather than
guessing.

`grad_weight` ships at 0.2 because that is what measured best *here*, not because
0.2 is principled — 0.6 is the more theoretically sensible value (silhouettes are
edge-defined). Re-run `tune.py` on your own drawer.

---

## 6. Tool detection, and a negative result worth keeping

`detect_pockets()` finds slot outlines in a rectified drawer. The method is
"not foam": black matte foam has neither saturation nor brightness, so
`(S > 60) | (V > 110)` catches coloured handles, blue pocket bases and chrome
heads alike. On this drawer the saturation threshold works anywhere from **60 to
170** and brightness never matters — far less tuned than it looks.

**Run it on the bare drawer, never the full one.** Tools lie touching each other
and merge into blobs; the pockets they came out of are separated by foam. Bare:
exactly 12 regions, 1:1 with the 12 slots, first try, repeatable across all four
independent empty shots. Full: 13 regions, two pairs merged.

To segment tools in a *full* drawer, five methods were scored:

| Method | Matched | Merged |
|---|---|---|
| Colour threshold | 11/12 | 1 |
| Full minus empty | 9/12 | 0 |
| Distance watershed (no prior) | 2/12 | 0 |
| Pocket-seeded watershed | 12/12 | 0 |
| **Per-pocket masked** | **12/12** | **0** |

The winner works because it stops treating the full drawer as a segmentation
problem. Two touching tools are genuinely ambiguous in isolation — but the bare
drawer already gave us the layout, so masking to one pocket at a time turns one
hard problem into twelve easy ones. The distance-watershed control (2/12) proves
shape alone cannot do it.

This survives distance: from a raw frame where the drawer occupied only **10% of
the image** (463px across, upsampled to 1000x700), the full chain gave **12/12
tools outlined and 12/12 occupancy correct**.

### The negative result: do NOT auto-refit slots to these polygons

The auto-detected polygons are 3–67% tighter than the hand-drawn rects, and the
obvious conclusion — tighter outlines mean less neighbour bleed, so better
scoring — **is false on this drawer.** Measured:

| Slot geometry | Wrong | Unsure | Min margin |
|---|---|---|---|
| **Hand-drawn rects (shipped)** | **0** | **11** | **0.049** |
| Auto pocket polygons | 4 | 12 | 0.000 |
| ...dilated 10px | 4 | 12 | 0.002 |
| ...dilated 30px | 1 | 14 | 0.003 |
| ...dilated 45px | 0 | 12 | 0.000 |

Dilation only crawls back toward the baseline; by 45px the polygon is
effectively the rect again and merely ties it. The likely mechanism is that the
loose rect includes surrounding foam whose *structure* helps `TM_CCOEFF_NORMED`
lock on, while a tight pocket is mostly flat coloured base with little internal
structure — so correlation gets noisier, not cleaner.

`calibrate.py --refit` is kept because it works mechanically and may well help a
drawer with different geometry, but it is **report-only unless `--write`**, and
the numbers above should be reproduced before writing. Validate with the
before/after check, never assume.

---

## 6b. Texture as a second channel — measured, rejected

Colour is not a general cue. Measured per pocket on this drawer, full vs empty:

| Cue | Full | Empty | Ratio | Consistent |
|---|---|---|---|---|
| Laplacian variance | 4927 | 1664 | 2.96x | 12/12 |
| Edge energy | 69.3 | 35.0 | 1.98x | 12/12 |
| Saturation | 90.3 | 133.7 | 0.68x | **0/12, inverted** |

So texture carries real, colour-independent signal — an empty pocket is smooth,
an occupied one is not — and it is the cue most likely to survive foam that has
no coloured pocket bases.

It was still **rejected as a tiebreak for UNSURE readings**, and the way it
failed is instructive:

- As an *absolute* per-slot value it separates 10/12 slots (292 and 297 overlap),
  but that requires exactly the per-slot magic numbers this design removed.
- As a *ratio to the slot's own empty reference* — self-calibrating, no magic
  numbers — present spans 1.11–3.62 and absent spans 1.00–1.80. **They overlap**,
  and 5 of 12 slots overlap individually.

An initial test appeared to give a perfect 0 wrong / 0 unsure across a wide
threshold plateau of 1.5–6.0. That was an artifact: **all 11 UNSURE readings are
absent-cases**, so any threshold high enough to call them all absent scored
perfectly while proving nothing about discrimination. A plateau that wide should
have been the tell — a real decision boundary does not survive a 4x change.

The deeper reason the ratio fails is worth keeping: slot 301's empty reference is
the misaligned one (§7), so its spurious marker and printed-text texture inflate
the denominator and compress every one of its ratios toward 1. **The bad
reference poisons the cue that was meant to route around the bad reference.**

## 6c. Why the tooling diagnoses slots now

Four attempts to improve scoring by algorithm all measured worse or turned out
unjustified (§6, §6b). The pattern is the finding: **the binding constraint on
this drawer is reference quality, not the algorithm.** Nine of twelve slots are
flawless with the shipped configuration; every UNSURE reading comes from three
slots, and one of those is unfixable in software.

So the diagnosis that found it was folded into `tune.py --slots` rather than left
as throwaway scripts. It separates two failures that look identical in the
tracker — both surface as UNSURE — but need opposite responses:

- **BORDERLINE** — separates correctly, but lands inside `uncertain_margin`.
  Tuning helps.
- **BROKEN** — reads present even when bare, so it contradicts its own
  references. No threshold anywhere helps.

and reports `ref shift` per slot: the local ECC translation needed to line that
slot's empty reference up with its baseline. On the current drawer this reads
1.0–26.2px, which is the drift that makes slots BROKEN in the first place. The
next drawer should diagnose itself in one command instead of by hand.

## 7. Open items

- **Slot 301 — root cause found: the two references are locally misaligned.**
  Cropping 301's region from both references and diffing them shows the ArUco
  marker and the printed "NET 55062" drawer text lighting up as *changed*. Those
  are static features; they cannot differ between full and empty. One global
  homography fits the drawer as a whole, but 301 sits in the extreme top-left
  corner where lens distortion and slight non-planarity leave residue a single
  homography cannot absorb. Its "empty" template therefore encodes a
  misalignment rather than an empty pocket, which is exactly the
  never-goes-negative signature (`+0.050 .. +0.576`).

  Per-slot ECC re-alignment was tried and **rejected**: it confirms the cause
  (slots needed 1.2–33.8px of local correction, and 301's margin does reach
  −0.011 afterwards) but it is not trustworthy. ECC is being asked to align a
  tool to the empty pocket it came out of — different content — so `cc` only
  reaches 0.47–0.55, and the result swings on padding:

  | Variant | Wrong | Unsure |
  |---|---|---|
  | current, global only | 0 | 11 |
  | slot crop, translation, pad30 | 0 | 9 |
  | ring only, euclidean, pad60 | **4** | 12 |
  | ring only, translation, pad90 | **5** | 9 |

  A change that lands anywhere between +2 and −5 depending on a padding constant
  is luck, not an improvement.

  **The real fix is data, not code: re-shoot `baseline.png` and `empty.png` back
  to back without moving the drawer or camera.** Until then `uncertain_margin`
  must stay at 0.073 — it is tuned precisely to keep 301 in "don't know", and any
  value at or below 0.050 makes it confidently *wrong* on all four bare-drawer
  photos.
- **Orientation resolution** for markerless (§2) if `auto` is ever adopted.
- **Generalisation is untested.** Every number here comes from one drawer whose
  foam is black with *blue pocket bases*. That colour contrast is why saturation
  segments so cleanly. A single-colour drawer (black recess, no coloured base)
  would need shadow- or edge-based detection instead, and is the single most
  informative test still outstanding.
