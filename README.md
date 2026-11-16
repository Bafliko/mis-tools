# Shadow-Foam Tool Tracker

Watches a tool drawer through an overhead camera and logs every tool taken or
returned into a SQLite database.

No AI models, no training. The drawer is located and flattened into a fixed
canonical frame, each slot is matched against two reference photos by normalised
correlation, and a motion gate plus a debounce keeps a hand reaching into the
drawer from triggering false events.

Schematics of the whole flow: [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## How it works

Each frame runs the same stages:

```
camera frame
   ↓  1. LOCATE      find the drawer outline → live / held / closed
   ↓  2. RECTIFY     → flat top-down view at a fixed canonical size
   ↓  3. REGISTER    ECC-align that onto baseline.png → cancels residual drift
   ↓  4. MOTION GATE anything moving? → pause all evaluation
   ↓  5. SCORE       each slot correlated against baseline.png and empty.png,
   ↓                 on grey and Sobel gradients, over a ±5px search window
   ↓  6. DECIDE      nearer of the two references → present, missing, or — when
   ↓                 they are too close — no answer at all
   ↓  7. DEBOUNCE    same reading held while static → state change
   ↓  8. LOG         write event to tools.db, draw the debug overlay
   ↓  9. DRIFT       a long-settled, unambiguous slot nudges its own reference
```

**The contract everything rests on.** Every image entering the scoring stage must
land in the same coordinate frame as the references. That is the whole job of
stages 1–3; everything after them is detail. `canonical_size` (`warp_size` in
`config.json`) must therefore never change after references are built.

### Locating the drawer

Two interchangeable localizers, chosen by `"localizer"` in `config.json`:

| Mode | How | When |
|---|---|---|
| `auto` | Line-segment detection → verified rectangle hypotheses, or an HSV foam mask when `detect.hsv` is set | Markerless. Needs the drawer to be a distinguishable rectangle in frame. |
| `aruco` | Four printed corner markers | The safer default: cheaper, and immune to a cluttered bench. |

`auto` re-locates every frame, so the drawer can be slid and re-opened without
recalibration, and reports one of three states:

- **live** — found in this frame.
- **held** — momentarily lost, coasting on the last outline for `grace_frames`.
  A hand over a corner must not blank the feed.
- **closed** — lost for longer than that. No image is passed downstream and every
  slot freezes. Shutting the drawer is the ordinary way this happens, and reading
  "all twelve tools taken" off the drawer front would be far worse than reading
  nothing.

A detection that jumps more than `detect.jump_px` from the last accepted one is
refused — real drawers slide, they do not teleport — but refusals count toward
the grace period, so a drawer that genuinely *was* moved is re-acquired a few
frames later from scratch.

**How the markerless path was arrived at.** Four earlier approaches were measured
against this project's photo set and all failed, which is worth recording because
the obvious ones are the ones that fail:

| Approach | Result |
|---|---|
| Canny → `approxPolyDP` quad | Drawer found in 2/13. Its outline never closes into a 4-vertex contour — the largest convex quad is 1% of the frame (a label strip). |
| `HoughLinesP` line intersection | 19–40 long lines per frame, of which **0–1 vertical**. Two are needed to intersect a corner. |
| HSV dark mask | Finds something every frame, but it spans the whole image — dark blobs bridge into the chair, cabinet and shadows. |
| Dark mask + hole filling + rectangularity | Closest of the four, but still merges the drawer with the cabinet it is mounted in. |

The fix was not a better threshold, it was a better detector plus a verification
step. `FastLineDetector` recovers ~900 segments where Hough found 19–40, and
**14 usable verticals where Hough found 0–1**, because it works locally on the
gradient field and merges collinear fragments instead of demanding one long,
well-populated line. Then, rather than taking the extreme line of each group —
which grabs whatever desk edge lies furthest out — every pair of horizontals is
crossed with every pair of verticals, and each candidate rectangle is verified on
evidence it did not pick for itself: how much of its own perimeter is really
drawn by detected segments, and whether its interior is darker than its surround.

Result on the same 13 photos: **13/13 located**, correctly.

**How `auto` compares to `aruco` here.** Consistency of the canonical frame,
measured as correlation between rectified full-drawer frames after ECC:

| Mode | Located | Median | Worst |
|---|---|---|---|
| `aruco` | 13/13 | 0.82 | 0.76 |
| `auto` | 13/13 | 0.81 | 0.75 |

Effectively equivalent. `aruco` is still the shipped default because it is
cheaper and cannot be fooled by a cluttered bench, but markerless is no longer
the compromise it looks like.

**Cost, and how it got there.** ~30 ms/frame tracked, ~100 ms to acquire a lock
cold. The first working version took **4,100 ms**, and closing that gap took four
changes, each measured:

| Change | Effect |
|---|---|
| Detect at `work_width` instead of full resolution | 4100 → 263 ms |
| Line-pair spacing pruned before any geometry | 263 → 238 ms |
| Perimeter support precomputed per *line*, not scanned per candidate edge | 238 → 145 ms |
| Verify the existing lock instead of re-searching (`track_support`) | 145 → 30 ms tracked |

The third is the one worth remembering: the search asks about thousands of
candidate edges, but they all lie along the same couple of dozen lines, so
coverage is a per-line prefix sum and each edge is a lookup.

`work_width` is not only a speed knob. Measured across 480–1280 it was also the
most *accurate* at 640, because downsampling suppresses the fine texture that
spawns competing short segments — 960 was worse on both counts. Re-measure if you
change it rather than assuming bigger is better. `detect.top_n` drives an O(n⁴)
search; below 14 it starts picking wrong rectangles.

Switching localizers changes the rectification, so `baseline.png`, `empty.png`
and every slot polygon must be rebuilt afterwards.

**Why rectification alone is not enough.** In `aruco` mode the homography is
pinned to four marker corners, so the warp *is* the markers. Knock one a
millimetre, or peel and re-stick it slightly off, and the whole image slides and
skews underneath slot rects that never move — every tool reads missing at once.
Nothing in the marker path can notice: the markers are its only reference. In
`auto` mode the same happens whenever the detected outline shifts by a pixel or
two, which it does constantly.

So `register()` re-anchors on the drawer itself. One ECC pass against
`baseline.png` recovers the residual homography and puts the frame back where the
slots expect it. It has to be a homography, not a shift: a moved corner changes
scale and skew too, and a translation-only fix leaves most of the error behind.

**The two references are registered to each other too.** `empty.png` is aligned
onto `baseline.png` once at load. Runtime frames are only ever registered onto
the baseline, so an empty reference shot at a slightly different drawer position
leaves every "empty" template looking at the wrong patch of foam — and no tool
can then ever score as missing. This is the common case, not the exotic one: on
this project's own references ECC had to apply a 1.38× x-scale and up to 95px of
corner displacement, and adding that one pass took the slots that could not be
told apart from 7 to 1.

Estimated on a quarter-size grey and warm-started from the previous frame, so it
costs ~12ms on the first frame and ~2ms after. A lock scoring under
`REGISTER_MIN_CC` is rejected and the previous transform kept — a wrong alignment
is indistinguishable from every tool vanishing, so no answer beats a guess.

Flat, featureless foam is what makes this degrade: ECC needs structure to lock
onto. Pockets, labels and drawer edges are plenty; a uniform sheet is not.

**Why the motion gate matters.** A hand crossing a slot looks exactly like a
missing tool. During motion the tracker freezes every pending change rather than
merely ignoring the frame — a hand hovering for 5 seconds accumulates no time
toward a state change. Evaluation resumes 0.5s after the frame goes still.

**Nearest of two references, never a threshold.** `baseline.png` is the full
drawer, `empty.png` the bare foam. Each slot is matched against both by
`TM_CCOEFF_NORMED` — normalised cross-correlation, `+1` identical, `0` unrelated
— and whichever it correlates with more strongly wins. Ties count as present:
absent has to win outright.

The normalisation is the entire point. `TM_CCOEFF_NORMED` subtracts each window's
own mean and divides by its own energy, so dimming the room or adding a lamp
barely moves the score. A raw pixel difference moves with every one of them,
which is what forces per-slot magic numbers that then drift. **There are no
per-slot thresholds any more.**

**Two channels, and a search window.** Each slot carries four cached templates:
grey and Sobel gradient, for full and for empty. `grad_weight` blends the two —
gradients carry the tool's silhouette and shrug off a highlight crawling across
chrome, grey keeps discrimination a bare outline loses. Every match runs over a
`±search_radius` window and takes the best offset, because ECC aligns globally
but a pixel or two of residue is normal and a 3px misalignment alone invalidates
a comparison.

**Slots are polygons, not boxes.** A bounding box around a wrench lying at 40°
contains a corner of each neighbouring slot, so a rectangular comparison partly
scores the tool next door — and removing that neighbour reads as this tool
moving too. Each template is masked to the slot's own outline; masked pixels are
set to the mean of what is kept, so they contribute nothing to the match rather
than a large fake deviation.

**There is no model.** Photos cannot teach it anything; they can only *measure*
`grad_weight` and `uncertain_margin` for your drawer and lighting (see `tune.py`).

**Why a slot is allowed to answer "don't know".** The gap between the two scores
is how much evidence there is. A gap under `uncertain_margin` is a coin flip, so
the tracker holds the last state, paints the box amber and counts it in the
status line as `UNSURE n`. An audit trail cannot express "probably", so a late
event beats a wrong one. Press `d` to see the live scores and margin per slot.
Without `empty.png` there is no comparison to make at all, and the tracker
reports `UNSURE` for everything rather than inventing an absolute threshold to
fill the gap.

**Why references drift toward the frame.** `baseline.png` is frozen at
calibration time and the sun is not. A slot that has been settled, static and
unambiguous for `drift_after` frames blends `drift_alpha` of itself into the
reference it matches, and its templates are recut. Correlation already shrugs off
uniform brightness changes, so this is catching what it cannot: a shadow edge
creeping across one corner, a lamp changing colour temperature. Slow enough that
a tool left out overnight can't teach the baseline that its empty pocket is
normal — the debounce logged it minutes earlier. In memory only: restarting
reloads the file and re-learns.

**Choosing `grad_weight`.** It ships at `0.6` on the theory that silhouettes
carry the signal, but that is a starting point, not a law — on this drawer's
photo set `0.2` measured best (worst gap `+0.146` against `+0.107` at `1.0`).
Press `g` to A/B it live, watch the `d` margins, and re-run `tune.py` if you
change it — `uncertain_margin` measured under one weight is meaningless under
another.

---

## Files

| File | Purpose |
|---|---|
| `gen_markers.py` | Generates `markers.png`, the printable corner markers. Only needed in `aruco` mode. |
| `calibrate.py` | Outline and name one drawer's tool slots. Merges into `config.json`, writes that drawer's baseline. |
| `tracker.py` | The runtime, and the shared library the others import. Also `--selftest`. |
| `tune.py` | Measures `grad_weight` and `uncertain_margin` from a folder of labelled photos. |
| `diagnose.py` | Throwaway camera probe. Delete once running. |
| `markers.png` | Printable marker sheet. Generated. |
| `config.json` | Localizer choice, slot polygons, names, scoring knobs, camera index. Generated. |
| `baseline.png` | Reference photo of the full drawer. Generated. |
| `empty.png` | Reference photo of the bare drawer. **Required** — nothing can be classified without it. |
| `tools.db` | SQLite event log. Generated. |
| `photos/` | Labelled photo set for `tune.py`. |
| `training/` | Where new drawer and tool photos go, to extend the knowledge base. Has its own README with the shooting and naming rules — read it before shooting. |
| `diagnostics/` | Measured evidence behind the localization and tool-detection design, with its own README. Not loaded at runtime. |
| `DECISIONS.md` | Why the system works the way it does, what was tried and rejected, and the numbers behind both. Read before changing the pipeline. |

### More than one drawer

Every drawer uses the *same* printed marker sheet. Calibrate each one under its
own name:

```bash
.venv/bin/python calibrate.py --drawer top      # -> baseline_top.png
.venv/bin/python calibrate.py --drawer middle   # -> baseline_middle.png
```

Each run adds or replaces one entry in `config.json`'s `drawers` list and leaves
the others alone. An old single-drawer `config.json` (slots at the top level,
`baseline.png`) still loads as the one drawer `main`.

At runtime the tracker decides which drawer is in shot by comparing a
thumbnail of the warped frame against every drawer's baseline **and** empty
reference, then swaps in that drawer's slots and labels. The active drawer's name
is shown in the status line, and `b`/`e` re-shoot references for that drawer only.

Two caveats that come with identifying by appearance rather than by marker id:

- **Shoot an empty reference (`e`) for every drawer.** Without it a drawer with
  all its tools out matches nothing and hands itself to whichever other drawer
  happens to be less unlike it.
- **Two drawers that genuinely look alike cannot be told apart.** Identical foam
  layouts need distinct marker sheets instead — the identification is a
  similarity score, not an id.

Tool ids must be unique across all drawers; they key the event log, so a reused
id would merge two physical tools into one history. `normalize_drawers` refuses
to load a config that does this. Identification only re-runs on settled frames,
so a hand in the drawer cannot flip the active drawer mid-reach, and a rival
drawer must beat the active one by 20% (`SWITCH_MARGIN`) to take over.

`calibrate.py` and `diagnose.py` each hard-code `CAMERA_INDEX` and the marker
ids at the top of the file; `tracker.py` reads `camera_index` from
`config.json`, which `calibrate.py` writes from its own constant. Change all of
them together.

---

## Setup

### 1. Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Every command below uses `.venv/bin/python`. Or `source .venv/bin/activate` once
and use plain `python` afterwards.

On macOS the terminal needs camera permission: **System Settings → Privacy &
Security → Camera**. Restart the terminal after granting it.

### 2. Pick the right camera

macOS does **not** guarantee the built-in camera is index 0 — a paired iPhone
(Continuity Camera) is ranked ahead of it, so index 0 is often a phone lying
face-down on a desk, dutifully streaming a blurry grey nothing.

```bash
system_profiler SPCameraDataType     # lists devices in index order
```

Confirm which index is which:

```bash
.venv/bin/python -c "
import cv2, numpy as np
for i in range(3):
    c = cv2.VideoCapture(i)
    if c.isOpened():
        for _ in range(15): ok, f = c.read()
        if ok: print(i, f.shape, 'std', round(float(np.std(f)), 1))
    c.release()"
```

A `std` under ~1 is a blank buffer; a flat low `std` with a soft grey image is a
camera pointed at nothing. Set `CAMERA_INDEX` in `calibrate.py` and
`diagnose.py` to the one you want.

The index is **not stable** — unpairing or moving the iPhone out of range
renumbers the devices. If the feed goes strange later, re-run this before
assuming the code broke.

### 3. Print the markers (`aruco` mode only)

Skip this if you are running `"localizer": "auto"` — but read the markerless
caveat under *Locating the drawer* first. If the drawer is not the clearest
quadrilateral in your frame, markers are the working answer, not a fallback.


```bash
.venv/bin/python gen_markers.py     # writes markers.png
```

Markers are 25mm by default (`MARKER_MM` in `gen_markers.py`), 33mm square once
the 4mm quiet zone is included. Shrink `MARKER_MM` if the drawer is tight — the
floor is set by how many pixels a marker occupies in the *captured frame*, not
by its physical size:

- `DICT_4X4_50` is 6 modules across (4 data + 1 module of border each side).
- Detection needs ~3px per module, ~5px for stable corners.
- So ~30px minimum in frame, 40+ comfortable.

At 1920px across a 550mm drawer (3.5 px/mm) that gives:

| Marker | px in frame | Verdict |
|---|---|---|
| 25mm | 88 | Very safe |
| 15mm | 52 | Fine |
| 10mm | 35 | Marginal — expect dropouts |

Marker size does not affect warp accuracy; corner *spread* does, and that is
fixed by the drawer. Print quality becomes the real limit as you shrink: at 15mm
each module is 2.5mm, and inkjet bleed blurs the module edges faster than the
pixel maths suggests. Laser print if you can.

Print at 100% / "actual size". **Cut along the grey line, not the black edge** —
that white border is the quiet zone the detector needs; cutting flush stops
detection. If space is tight, tape the marker onto white card rather than
trimming closer.

Tape one marker in each drawer corner matching its printed label (0 top-left,
1 top-right, 2 bottom-right, 3 bottom-left). Keep each one square to the drawer
edges — a rotated marker skews the whole warp.

All four must stay visible at all times, so place them where a hand reaching in
will not cover them.

### 4. Check the camera sees them

```bash
.venv/bin/python diagnose.py
```

Want `aruco ids detected : [0, 1, 2, 3]`. It also saves `diagnose_frame.png` so
you can confirm focus and framing.

For live aiming, `calibrate.py` doubles as a viewfinder — press `q` to quit
without saving. In `aruco` mode, while markers are missing, `diagnose.py` names which ids it
does see, so you can tape them up one at a time and watch them register.

### 5. Mount the camera

Overhead, facing straight down, whole drawer in frame with a little margin. It
must not move again after step 6 — the baseline photo is taken from this exact
position, and moving the camera invalidates it.

A laptop's built-in camera points wherever the screen points, so it cannot look
into a drawer without propping the whole machine face-down. A phone on an arm or
a USB webcam is the practical answer.

### 6. Outline the slots

Fill the drawer completely, turn on whatever lighting the drawer will normally
have, then:

```bash
.venv/bin/python calibrate.py                 # the drawer named "main"
.venv/bin/python calibrate.py --drawer top    # one run per drawer
```

Once the drawer is located the red header disappears and you get the flat
top-down view.

| Action | Result |
|---|---|
| click, click, … then `c` | a polygon, one vertex per click — then type its name |
| drag a rectangle | over a tool, then type its name in the terminal |
| `x` | discard the polygon in progress |
| `u` | undo the last slot |
| `s` | merge this drawer into `config.json`, save its baseline, then exit |
| `q` | quit without saving |

**Prefer polygons.** A bounding box around a wrench lying at 40° contains a corner
of each neighbouring slot, and the tracker then partly scores the tool next door
— so removing that neighbour reads as this tool moving too. Trace the tool's
actual outline and the mask confines the score to it.

Outline the **tool itself**, not the whole pocket — the tighter the outline, the
larger the change when the foam beneath is exposed.

The tool ID is derived from the name (`10mm Wrench` → `10mm_wrench`), so
re-running calibration keeps IDs stable and your history intact. Renaming a tool
does mint a new ID and orphans its old events.

Press `s` only with the drawer **completely full** — that frame becomes the
baseline every future comparison depends on.

Then run `tracker.py`, empty the drawer completely and press `e`. Nothing can be
classified until `empty.png` exists: the decision is "which of the two references
does this slot resemble", and until then there is no second reference. Shoot it
without moving the drawer between the two — they are auto-registered to each
other at load, but ECC can only absorb so much.

### 7. Run

```bash
.venv/bin/python tracker.py
```

Green outline = present, red = missing, amber = too close to call. The status line
shows `static | missing 2/8`, `MOTION - paused`, or `HELD` while the drawer
outline is being coasted. With no drawer in view at all the window shows
`NO DRAWER - closed or out of shot` and every slot holds its state. Events print
to the terminal as they're written.

| Key | Action |
|---|---|
| `q` | quit |
| `b` | re-shoot the baseline — drawer **full**, static frame only |
| `e` | re-shoot the empty reference — drawer **empty**, static frame only |
| `d` | per-slot scores and margins overlay |
| `g` | toggle the edge channel on/off |
| `s` | save the current frame into `photos/`, labelled from the live state |
| `[` `]` | uncertain margin down / up |
| `-` `=` | motion limit down / up |
| `w` | write the live values back to `config.json` |

`s` is the fast way to build a `tune.py` photo set: set the drawer, wait for
`static`, press `s`. The filename records which tools were out.

---

## Testing from photos

Every tool accepts a still image instead of the camera, so localization, slots and
scoring can be checked without mounting anything. `open_camera()` returns a
`StillCapture` for any non-numeric source — a photo served as an endless frozen
stream — so the real pipeline runs unchanged.

Take two photos from the *same* position, drawer full and with one tool out:

```bash
.venv/bin/python calibrate.py full.jpg           # outline slots, press s
.venv/bin/python tracker.py --image missing.jpg  # should log that tool Taken
```

A frozen frame reads as a static scene, so the motion gate stays open and the 2s
debounce fires normally. Tools you did not remove staying green while the removed
one goes red means the scoring suits your lighting.

**Ceiling:** a frozen frame cannot exercise the motion gate or the
hand-occlusion path — nothing ever moves. Those need live video and a hand.

### Measuring the scoring knobs from a photo set

`tune.py` reads `grad_weight` and `uncertain_margin` off real photos instead of
you guessing them. Again: no model, no training — there are no per-slot
thresholds left to measure, only these two global knobs.

Shoot every photo from the baseline camera position and label by filename:

```
photos/full_1.jpg                        every tool present
photos/missing_10mm_wrench_1.jpg         that tool absent, rest present
photos/missing_10mm_wrench+pliers.jpg    both absent
```

The part after `missing_` is matched against known tool ids longest-first, so a
trailing counter (`missing_10mm_wrench_2.jpg`) is handled — necessary because
tool ids contain underscores themselves. Anything that matches no known id is
reported as a warning rather than silently miscounted.

```bash
.venv/bin/python tune.py photos/            # report
.venv/bin/python tune.py photos/ --write    # update config.json
```

It sweeps `grad_weight` and reports the worst separation over every slot at each:

```
 grad_weight   worst gap  overlapping  measured
         0.0      +0.132            0        12
         0.2      +0.146            0        12
         0.6      +0.142            0        12
         1.0      +0.107            0        12

tool                               present              absent       gap
292                         +0.323..+0.340      -0.189..-0.180    +0.503
299                         +0.109..+0.228      -0.056..-0.049    +0.159
```

The margin is `full score − empty score`: positive means the slot looks more like
the full drawer. Present readings should all be positive, absent all negative, and
the **gap** between the worst of each is the room `uncertain_margin` has to sit
inside. The minimum over every slot is what matters, not the average — one slot
whose clusters touch is one tool the tracker cannot report on.

| Output | Meaning |
|---|---|
| Wide gap (`+0.5`) | Rock solid; the exact margin barely matters. |
| Narrow gap (`+0.15`) | Works, but lighting drift will break it sooner. |
| `OVERLAP` | The clusters touch — **no** margin can separate them. Redraw the polygon so it stops including a neighbour, or re-shoot the references. |
| `need both` | That tool has photos in only one state. Shoot the other. |

`--safety` sets how much of the worst gap to claim as the uncertain band
(default 0.5).

Photos where no drawer is found are skipped and reported, never averaged in — a
bad rectification would poison every reading in that frame.

About 3 full photos and 2 per tool-out state is enough. Vary nothing but the
tools.

**Where this drawer currently stands.** 13 photos × 12 slots = 156 readings, **0
wrong, 11 `UNSURE`**. The unsure readings are three slots whose clusters sit close
to zero — `301` most of all, whose 535×114 rect is mostly shared background, so
both references match it and the full one wins even on a bare drawer. Redrawing
those three as tight polygons is the fix; until then they honestly report
"don't know" rather than guessing.

**Check your labels before trusting an OVERLAP.** Filenames written by `s` record
what the tracker *believed* at the time, which is wrong whenever the tracker was
wrong. Two of this project's own photos were mislabelled that way — one claiming
4 tools out when the drawer was completely empty, one claiming a tool was gone
while it sat in its slot — and between them they made 7 of 12 slots look
inseparable. Fixing the two filenames took the overlaps to zero.

---

## Tuning

Defaults are a starting point; real optics and lighting need adjustment.

| Symptom | Knob | Direction |
|---|---|---|
| Everything reads `UNSURE`, always | `empty.png` is missing — press `e` with the drawer bare | — |
| Every tool reads missing at once | the two references are misaligned, or the drawer moved; re-shoot `empty.png` with `e` | — |
| Real changes ignored, `UNSURE` in the status line | `uncertain_margin` | lower — but a persistent UNSURE means the slot has no real evidence either way; tighten the polygon first |
| Wrong flips still get through | `uncertain_margin` | raise |
| Removing one tool flips its neighbour too | redraw both as polygons, not boxes | — |
| Scores wobble frame to frame | `search_radius` | raise a little; past ~8px it starts finding false matches |
| Everything slowly drifts toward missing over a day | `drift_alpha` | raise (or `drift_after` lower) |
| A reference has learned something it shouldn't | `drift_alpha` to 0, then `b`/`e` to re-shoot | — |
| Chrome tools flicker as the light moves | `grad_weight` | raise, then re-run `tune.py` |
| `NO DRAWER` with the drawer plainly open | `detect.canny` lower, `detect.min_area_frac` lower — or switch to `"localizer": "aruco"` | — |
| The outline flickers between two positions | `detect.jump_px` | lower, so the spurious one is refused |
| Drawer shut but still being scored | `detect.grace_frames` | lower |
| Everything reads missing after a marker was bumped | registration should absorb it; if not, `REGISTER_MIN_CC` in `tracker.py` | lower — or re-baseline with `b` |
| Readings jump around on plain foam | `REGISTER_MIN_CC` | raise, so weak locks are rejected rather than applied |
| Hand movement doesn't pause evaluation | `MOTION_AREA_FRAC` in `tracker.py` | lower |
| Tracker pauses constantly, never evaluates | `MOTION_AREA_FRAC` | raise |
| Changes register too slowly / too eagerly | `HOLD_S` in `tracker.py` | adjust from 2.0s |

Run `tune.py photos/ --slots` first when anything looks wrong. It gives a verdict
per slot — `OK`, `BORDERLINE`, `BROKEN` — and separates the two failures that
both look like `UNSURE` in the tracker but need opposite responses:

- **BORDERLINE** — the slot separates correctly but lands inside the uncertain
  band. Tuning helps.
- **BROKEN** — the slot reads present even when bare, so it disagrees with its own
  references. **No threshold anywhere fixes this**; the references have to be
  re-shot back to back.

It also reports a `ref shift` per slot: how many pixels of local correction that
slot's empty reference needs to line up with its baseline. Anything above a few
px means the two references drifted, which is the usual root cause of BROKEN.

Prefer `tune.py --write` over hand-editing — it sets the knobs from measurements
rather than guesses.

Re-run `calibrate.py` after any change to lighting, camera position, or the foam
layout — the baseline must match what the camera actually sees.

---

## Database

`tools.db`, SQLite in WAL mode so a dashboard can read while the tracker writes.

```sql
CREATE TABLE events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,     -- ISO-8601 UTC, '2026-08-12T14:03:11Z'
    tool_id   TEXT NOT NULL,     -- stable slug, '10mm_wrench'
    tool_name TEXT NOT NULL,     -- display name, '10mm Wrench'
    action    TEXT NOT NULL      -- 'Taken' | 'Returned'
);
```

Append-only. One row per confirmed state change; nothing is written while a hand
is in the drawer.

**For the dashboard:**

```sql
-- current state of every tool, one row each
SELECT * FROM current_state;

-- recent activity
SELECT * FROM events ORDER BY id DESC LIMIT 50;

-- history for one tool
SELECT * FROM events WHERE tool_id = ? ORDER BY id DESC;
```

`current_state` is a view over the latest event per tool, written to avoid a
correlated subquery so it stays cheap to poll as history grows.

Open it **read-only** from Next.js — `file:tools.db?mode=ro` — so a dashboard bug
can't corrupt the audit trail.

State survives restarts: the tracker seeds itself from `current_state`, so
restarting while tools are out does not re-log them as taken.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Blank/white window | Camera aimed at a wall or out of focus. Run `diagnose.py` and look at `diagnose_frame.png`. |
| Picture never changes, no camera LED lit | Wrong device — see Setup step 2. macOS ranks a paired iPhone ahead of the built-in camera, so index 0 may be a phone face-down somewhere. |
| Picture looks frozen but isn't | Check `std` and frame-to-frame diff before blaming the code: ~0.2 diff is sensor noise on a genuinely static scene, 0.000 is a stuck buffer, `std < 1` is a blank one. |
| `SEARCHING` never clears | Not all four markers visible. The header names which ids are missing. Check for glare, occlusion, or a flush-cut quiet zone. |
| `cannot open camera 0` | Permission not granted, or another app holds the camera. |
| `cannot read image foo.jpg` | Bad path, or a format OpenCV can't decode (HEIC — convert to JPEG). |
| `baseline.png is 800x600 but config warp_size is 1000x700` | Config edited after calibration. Re-run `calibrate.py`. |
| `slot x rect [...] is outside the 1000x700 frame` | Hand-edited `config.json`. Fix the polygon or re-calibrate. |
| Everything reads missing at once | Lighting drifted from the baseline, or registration is failing to lock (flat foam, half the drawer occluded). Re-run `calibrate.py`. |
| Continuity Camera feed dies after a while | The phone dropped the session. Wake it and re-run. |

```bash
.venv/bin/python tracker.py --selftest
```

Checks corner ordering, localization states, correlation scoring, polygon
masking, drift, the debounce logic, slot validation, and the database layer without a
camera. Says nothing about optics.

---

## Known limitations

- **Drift compensation only handles slow drift.** A gradual change is tracked
  out; flipping the room lights, or leaving the tracker off across a lighting
  change, moves both references at once and the first frames back are simply
  wrong. Re-shoot with `b`/`e` after any abrupt change.
- **Occupancy, not identity.** A different tool dropped in the right pocket
  reads as present. The `d` overlay is the tell: both distances large means
  "something is there, but not what belongs there". Nothing acts on that yet.
- **Not yet validated against a real drawer.** Marker detection has been
  confirmed live on a real camera, and the warp/compare path on generated
  photos. The full loop — real foam, real tools, real hands — has not been run.
  Expect reference quality and photo labelling to be where the time goes.
- **Registration needs the drawer to still look like its baseline.** It absorbs
  a bumped marker, not a re-arranged drawer. With most tools out at once the
  lock weakens (correlation ~0.55 against ~0.92 on a full drawer) and the
  previous transform is held instead — fine, because markers do not move while
  nobody is touching the cart, but it means a marker knocked *during* a
  mass-removal is not corrected until the drawer is refilled.
- **Occlusion is all-or-nothing.** Anything parked over the drawer holds the
  entire system paused for as long as it sits there.
- **Camera index is not stable on macOS.** Devices renumber when a phone pairs
  or drops. Nothing detects this; the feed just becomes the wrong camera.
- **No retention policy.** `events` grows forever. Fine for years at typical
  volume, but nothing prunes it.
