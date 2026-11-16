# Diagnostics

Evidence behind the design decisions in the main README. Nothing here is loaded
at runtime — these are records of what was measured and what failed, kept because
the failures are the part that is expensive to rediscover.

Regenerate nothing automatically; these are snapshots of specific experiments.

## localization/

How the drawer gets found. Read in this order — the first four are the
approaches that **did not work**, and they are the reason the fifth exists.

| File | Shows |
|---|---|
| `focus_test.jpg` | `HoughLinesP`: 19–40 long lines, **0–1 vertical**. Two verticals are the minimum to intersect a corner, so this cannot work. Panel 1 is the Canny map — the drawer's own outline barely registers. |
| `focus_test_empty.jpg` | The same on a bare drawer. Same verdict. |
| `focus_test_segment.jpg` | Dark mask + hole filling + rectangularity scoring. Closest of the failures, but it merges the drawer into the cabinet it is mounted in. |
| `focus_test_lsd.jpg` | **The one that works.** `FastLineDetector` finds ~900 segments and 14 usable verticals on the same frames; every pair of horizontals is crossed with every pair of verticals and each candidate rectangle is verified on perimeter support and interior darkness. 13/13 photos located. |
| `drawer_detected.jpg` | Production `Localizer` in `auto` mode on a raw frame: the quad with corners labelled TL/TR/BR/BL. |
| `drawer_rectified.jpg` | The canonical 1000×700 output of that same frame. Note it is upside-down relative to `baseline.png` — see the orientation limitation below. |

## tools/

Finding the tools inside a rectified drawer. The method is "not foam":
black matte foam has neither saturation nor brightness, so `(S > 60) | (V > 110)`
catches coloured handles, blue pocket bases *and* chrome heads.

| File | Shows |
|---|---|
| `tool_detect.jpg` | mask / auto-detected / hand-labelled, for the full and bare drawer. The bare drawer segments into **exactly the 12 slots, 1:1**; the full drawer does not, because tools touch each other and pockets do not. |
| `tool_score_1_colour_threshold.jpg` | Colour threshold alone on the full drawer: 11/12, two pairs merged. |
| `tool_score_2_full_minus_empty.jpg` | Difference against the bare reference: 9/12. |
| `tool_score_3_distance_watershed.jpg` | Distance-transform watershed, no prior: 2/12. The control that proves shape alone cannot split touching tools. |
| `tool_score_4_pocket_seeded_watershed.jpg` | Watershed seeded from the bare drawer's pocket layout: **12/12, 0 merged**. |
| `tool_score_5_per_pocket_masked.jpg` | Segment inside each known pocket separately: **12/12, 0 merged, best IoU (0.52)**. The winner. |
| `tool_faraway.jpg` | Whole chain from a raw distant frame. Top: ArUco path, **12/12 tools outlined and 12/12 occupancy correct** off a drawer occupying only 10% of the frame. Bottom: the `auto` path on the same frame, broken. |

## Two findings worth not rediscovering

**Auto-detected polygons beat hand-drawn rects.** They are 2–59% tighter, which
is exactly the neighbour-bleed the polygon masking exists to prevent. The bare
drawer is the right source: it gives all twelve, first try, with a threshold band
so wide (`s_thr` 60–170) that it is effectively untuned.

**Markerless recovers the rectangle but not its orientation.** ArUco marker *ids*
are pinned to physical corners, so it knows which way is up. A bare quad carries
no such identity: `order_corners` returns image-space TL/TR/BR/BL faithfully, but
a drawer shot from the far side has its true top at image-bottom. The result is a
180°-rotated canonical frame, which ECC cannot rescue — a local optimizer has no
path from there. This is the `auto` failure in `tool_faraway.jpg`, and it is why
switching localizers requires rebuilding the references and slots rather than
flipping a flag. The fix, when wanted, is to resolve orientation once at
calibration by scoring all four rotations against the baseline and storing the
winner.
