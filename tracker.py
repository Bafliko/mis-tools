"""Real-time shadow-foam tool tracker.

Pipeline, per frame:
  camera -> locate the drawer -> rectify to the canonical frame -> ECC re-anchor
         -> motion gate -> drawer identification
         -> per-slot normalised correlation against the full and empty templates
         -> debounce -> SQLite event log -> overlay

Localization is markerless by default: Canny edges (or an HSV foam mask) find the
drawer outline every frame, so an open drawer is tracked as it slides and a
*closed* drawer stops being warped altogether rather than freezing on the last
homography. The ArUco path is still here and selected with "localizer": "aruco".

A slot is classified by which of two references it correlates with better - the
drawer full and the drawer bare - never by an absolute number. Correlation is
scored on grey *and* Sobel gradients over a small search window, so a specular
highlight sliding over chrome, or a residual pixel of misalignment, does not read
as a missing tool. A slot too close to call reports nothing rather than guessing,
and one that has been confidently settled for a while nudges its reference toward
the current frame, so slow lighting drift never accumulates.

Several drawers may be watched by one camera, so the drawer in shot is identified
by which drawer's reference images the rectified frame resembles.

Run calibrate.py first to produce config.json + baseline.png + empty.png.

  python tracker.py                # live
  python tracker.py --selftest     # run the logic checks, no camera needed
"""

import argparse
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
CONFIG_PATH = HERE / "config.json"
BASELINE_PATH = HERE / "baseline.png"
EMPTY_PATH = HERE / "empty.png"     # optional: same drawer, every tool removed
DB_PATH = HERE / "tools.db"

# --- tuning knobs (real optics/lighting need tuning; defaults are a starting point) ---
MOTION_PIXEL_DELTA = 25     # per-pixel grey delta counted as "changed"
MOTION_AREA_FRAC = 0.005    # >0.5% of the drawer changed => something is moving
MOTION_SETTLE_S = 0.5       # frame must be quiet this long before evaluation resumes
HOLD_S = 1.0                # consistent reading required to register a state change
                            # (override per-drawer with "hold_s" in config.json;
                            # total latency is this + MOTION_SETTLE_S)

# --- scoring (normalised correlation: +1 identical, 0 unrelated, -1 inverted) ---
GRAD_WEIGHT = 0.6           # 0 = grey only, 1 = edges only. Tool silhouettes are
                            # edge-defined and foam is smooth, so edges carry most
                            # of the signal and shrug off a highlight sliding over
                            # chrome; grey keeps the discrimination that a bare
                            # outline loses. Press 'g' to A/B it live and re-run
                            # tune.py against your own photo set before trusting
                            # a different split.
SEARCH_RADIUS = 5           # px of slack the template may slide before scoring.
                            # Absorbs the sub-pixel residue ECC leaves behind; a
                            # 3px misalignment otherwise invalidates every score.
UNCERTAIN_MARGIN = 0.08     # |full - empty| below this is too close to call
DRIFT_MARGIN = 0.30         # ...and above this the reading is safe to learn from
DRIFT_AFTER = 60            # confident static frames before a reference is nudged
DRIFT_ALPHA = 0.02          # how much of the live ROI each nudge blends in

# --- drawer localization (markerless; see Localizer) ---
DETECT = {
    "work_width": 640,      # detection runs at this width; the answer is scaled
                            # back up. Not merely a speed knob - measured across
                            # 480-1280 this also came out the most *accurate*
                            # setting, because downsampling suppresses the fine
                            # texture that spawns competing short segments. 960
                            # was worse on both counts. Re-measure if you change
                            # it; do not assume bigger is better
    "canny": [30, 90],      # gradient thresholds feeding the segment detector
    "min_seg_len": 16,      # px at work_width; shorter fragments are texture
    "angle_tol": 22,        # degrees a segment may lean and still count h or v
    "top_n": 14,            # longest segments kept per orientation. The search is
                            # O(n^4) in this, so it buys accuracy at a steep price
    "near_px": 12,          # how close a segment must lie to count as supporting
    "min_support": 0.20,    # every side must be at least this covered by segments
    "track_support": 0.45,  # ...and an existing lock survives a frame while its
                            # own four edges stay this covered. Above min_support
                            # deliberately: keeping a stale outline is worse than
                            # paying for a fresh search
    "verify_top": 8,        # candidates reaching the (costly) photometric check
    "min_area_frac": 0.04,  # a quad smaller than this share of the frame is noise
    "max_area_frac": 0.60,  # ...and a larger one has swallowed the whole bench
    "aspect_tol": 0.30,     # +/- tolerance on warp_size's aspect ratio
    "clahe": False,         # local contrast equalisation before edges - only worth
                            # it when the lighting falls off across the drawer
    "hsv": None,            # [[h,s,v],[h,s,v]] - if the foam is a distinct colour
                            # this beats the segment path outright: set it and the
                            # segment path is skipped
    "jump_px": 40,          # a corner set this far off the last accepted one is a
                            # spurious detection, not a drawer that teleported
    "grace_frames": 5,      # frames the last good outline survives a miss, so a
                            # hand over one corner does not blank the feed - past
                            # this the drawer counts as gone and warping stops
}

# --- registration (cancels marker movement relative to the drawer) ---
REGISTER_SCALE = 0.25       # estimate on a quarter-size grey: 12ms not 700ms, and
                            # the blur actually helps ECC find the basin
REGISTER_ITERS = 30         # converges by ~20 at that scale; warm starts take 2
REGISTER_MIN_CC = 0.5       # below this the lock is not trusted and the previous
                            # transform is kept - a hand over half the drawer, or
                            # every tool out at once, correlates poorly

# --- drawer identification (same markers on every drawer, so it goes by looks) ---
ID_SIZE = (100, 70)         # frames are matched at this size: shape, not detail
SWITCH_MARGIN = 0.8         # a rival drawer must score this much better than the
                            # active one to take over, so a borderline frame does
                            # not flap the ROI set back and forth

# --- camera resilience ---
READ_RETRY_DELAY_S = 0.2    # pause after a failed read before trying again
REOPEN_AFTER_FAILS = 10     # consecutive failures before reopening the device
MAX_READ_FAILS = 100        # consecutive failures before giving up entirely

TAKEN, RETURNED = "Taken", "Returned"

PHOTO_DIR = HERE / "photos"
MARGIN_STEP = 0.01
HINT = ("q quit | b re-baseline (drawer FULL) | e empty ref (drawer EMPTY) | "
        "d scores | g grad weight | s save photo | [ ] uncertain margin | "
        "- = motion limit | w write cfg")


# ---------------------------------------------------------------- localization

def order_corners(pts):
    """Four unordered points -> [top-left, top-right, bottom-right, bottom-left].

    Non-negotiable, and the single most expensive thing to get wrong: an
    inconsistent corner order silently rotates the rectified image 90 or 180
    degrees, every slot then looks at the wrong part of the drawer, and nothing
    downstream can tell that from every tool being missing at once.

    Sum and difference place the corners without any angle arithmetic: the
    top-left minimises x+y, the bottom-right maximises it, and y-x separates the
    other two. Holds while the drawer is within ~45 degrees of upright in frame,
    which a fixed overhead camera always is.
    """
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def quad_of(component_mask):
    """Tightest rotated rectangle around a binary mask, as 4 ordered corners."""
    pts = cv2.findNonZero(component_mask)
    if pts is None:
        return None
    return order_corners(cv2.boxPoints(cv2.minAreaRect(pts)))


def foam_corners(frame, det):
    """Locate the drawer by the colour of its foam.

    Preferred whenever the foam is a distinct colour: one threshold and one
    connected component, with no dependence on the drawer's rim being unbroken or
    its edges being in focus. Everything the edge path below needs to guess at -
    which contour is the drawer, whether a shadow closed the outline - is decided
    by the colour itself.
    """
    lo, hi = (np.array(b, np.uint8) for b in det["hsv"])
    mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lo, hi)
    # Close the gaps the tools themselves punch in the foam, so the drawer comes
    # back as one component rather than a constellation of islands.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    # label 0 is the background, which is otherwise always the largest thing here
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[big, cv2.CC_STAT_AREA] < det["min_area_frac"] * frame.shape[0] * frame.shape[1]:
        return None
    return quad_of((labels == big).astype(np.uint8))


def line_segments(grey, det):
    """Straight edge fragments in the frame, collinear ones merged.

    A line *segment* detector rather than HoughLinesP, which cannot see this
    problem at all. Hough votes in a global accumulator and needs a long,
    well-populated line to clear its threshold, so a drawer edge broken by a
    shadow contributes nothing: on this project's own frames it recovered 19-40
    lines of which 0-1 were vertical, and two verticals are the minimum needed to
    intersect a corner. FastLineDetector works locally on the gradient field,
    returns short low-contrast fragments, and merges the collinear ones - the same
    frames give ~900 segments with 14 usable verticals.
    """
    lo, hi = det["canny"]
    fld = cv2.ximgproc.createFastLineDetector(
        length_threshold=det["min_seg_len"], distance_threshold=1.41,
        canny_th1=lo, canny_th2=hi, canny_aperture_size=3, do_merge=True)
    segs = fld.detect(grey)
    if segs is None:
        return np.zeros((0, 4), np.float32)
    return np.asarray(segs, np.float32).reshape(-1, 4)


def split_orientation(segs, det):
    """The longest near-horizontal and near-vertical segments, longest first."""
    horiz, vert = [], []
    for s in segs:
        x1, y1, x2, y2 = s
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        length = float(np.hypot(x2 - x1, y2 - y1))
        tol = det["angle_tol"]
        if ang < tol or ang > 180 - tol:
            horiz.append((length, s))
        elif abs(ang - 90) < tol:
            vert.append((length, s))
    horiz.sort(key=lambda t: -t[0])
    vert.sort(key=lambda t: -t[0])
    n = det["top_n"]
    return [s for _, s in horiz[:n]], [s for _, s in vert[:n]]


def _line_intersect(a, b):
    """Where the infinite lines through segments a and b cross, or None."""
    p1 = np.array(a[:2], np.float64)
    d1 = np.array(a[2:], np.float64) - p1
    p2 = np.array(b[:2], np.float64)
    d2 = np.array(b[2:], np.float64) - p2
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
    return p1 + t * d1


def prepare_segments(segs):
    """Segment endpoints and unit directions as arrays, for side_support().

    Precomputed once per frame rather than per candidate edge: the rectangle
    search asks about thousands of edges and rebuilding this each time was, by
    itself, most of the runtime.
    """
    p = segs[:, :2].astype(np.float64)
    q = segs[:, 2:].astype(np.float64)
    d = q - p
    length = np.linalg.norm(d, axis=1)
    keep = length >= 1.0
    p, q, d, length = p[keep], q[keep], d[keep], length[keep]
    return p, q, d / length[:, None]


COVER_SPAN = 4000       # px of line parameter the coverage table spans, centred


def line_coverage(line, prepared, near_px):
    """Where along an infinite line real segments actually lie.

    Built once per candidate *line*, not once per candidate rectangle, and that
    distinction is the whole performance story: the rectangle search asks about
    thousands of edges but they all lie along the same couple of dozen lines.
    Scanning every segment per edge cost ~250ms a frame; scanning per line and
    answering each edge from a prefix sum costs a lookup.

    Returns (origin, direction, prefix) where prefix[i] is how many pixels of
    line parameter below bin i are covered. Overlapping segments coalesce into
    the same bins, so coverage is a union rather than a sum - summing would let
    one heavily-fragmented edge score above 1.0 and outrank a complete one.
    """
    p, q, dirs = prepared
    A = np.array(line[:2], np.float64)
    d = np.array(line[2:], np.float64) - A
    L = np.linalg.norm(d)
    u = d / L if L else np.array([1.0, 0.0])
    n = np.array([-u[1], u[0]])
    cov = np.zeros(COVER_SPAN, bool)
    if len(p):
        near = ((np.abs((p - A) @ n) <= near_px) & (np.abs((q - A) @ n) <= near_px)
                & (np.abs(dirs @ u) >= 0.94))
        if near.any():
            t0 = (p[near] - A) @ u
            t1 = (q[near] - A) @ u
            lo = np.clip(np.minimum(t0, t1) + COVER_SPAN // 2, 0, COVER_SPAN - 1)
            hi = np.clip(np.maximum(t0, t1) + COVER_SPAN // 2, 0, COVER_SPAN - 1)
            for a, b in zip(lo.astype(int), hi.astype(int)):
                cov[a:b + 1] = True
    return A, u, np.concatenate([[0], np.cumsum(cov)])


def side_support(A, B, coverage):
    """What fraction of the edge A->B is actually drawn by detected segments.

    The check that makes hypothesised rectangles trustworthy. Four lines always
    intersect into a quadrilateral, so a candidate proves nothing by existing -
    but a real drawer edge has segments lying along it, while a coincidence of a
    desk edge and a door frame has three empty sides.
    """
    origin, u, prefix = coverage
    ta = (A - origin) @ u + COVER_SPAN // 2
    tb = (B - origin) @ u + COVER_SPAN // 2
    lo, hi = sorted((ta, tb))
    lo = int(np.clip(lo, 0, COVER_SPAN - 1))
    hi = int(np.clip(hi, 0, COVER_SPAN - 1))
    if hi - lo < 1:
        return 0.0
    return float(prefix[hi] - prefix[lo]) / (hi - lo)


def lsd_corners(frame, size, det, prev=None, stats=None):
    """Locate the drawer by hypothesising rectangles and verifying them.

    Two pairs of detected lines - two horizontal, two vertical - intersect into a
    candidate rectangle. Every such pair is tried, which is the part that matters:
    taking the *extreme* line of each orientation group (the obvious approach, and
    what the spec's Hough fallback suggests) grabs whatever desk edge happens to
    lie furthest out and never even considers the drawer.

    Candidates are then judged on evidence they did not choose for themselves:
    how much of their own perimeter is really drawn by detected segments, and
    whether their interior is darker than the surround - a drawer cavity is.

    Cheap filters first, and the photometric test only on the few survivors: the
    ring dilation costs more than everything else combined and running it on every
    candidate would make this far too slow to do per frame.
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect on a smaller image and scale the answer back up. The drawer is a
    # metre-scale object whose edges survive downsampling perfectly well, the
    # search below is O(pixels) only through the segment count, and ECC refines
    # the result afterwards regardless - so full resolution buys nothing here and
    # costs several seconds a frame.
    scale = min(1.0, det["work_width"] / float(frame.shape[1]))
    if scale < 1.0:
        grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if det["clahe"]:
        grey = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)
    segs = line_segments(grey, det)
    prepared = prepare_segments(segs)

    # Already locked on? Ask only whether the outline we had is still drawn in
    # this frame, which is four table lookups against the ~8000-candidate search
    # below. The drawer is stationary in the overwhelming majority of frames, so
    # this is the path almost every frame takes.
    #
    # Liveness survives it because the check is evidence-based, not a cache: a
    # drawer that shuts, is pushed out of shot, or slides more than near_px stops
    # supporting its own old edges and falls through to the full search. The
    # deliberate ceiling is that a drawer creeping slower than near_px per frame
    # can hold a lock up to that far behind - which registration then absorbs,
    # since ECC exists precisely to take out a few px of residue.
    if prev is not None:
        p = np.asarray(prev, np.float64) * scale
        held = [side_support(p[a], p[(a + 1) % 4],
                             line_coverage(np.concatenate([p[a], p[(a + 1) % 4]]),
                                           prepared, det["near_px"]))
                for a in range(4)]
        if min(held) >= det["track_support"]:
            return np.asarray(prev, np.float32)

    horiz, vert = split_orientation(segs, det)
    if stats is not None:
        stats.update(segs=len(segs), horiz=len(horiz), vert=len(vert))
    if len(horiz) < 2 or len(vert) < 2:
        return None

    H, W = grey.shape
    want = max(size) / min(size)
    # Legal spacing between the two parallel sides, derived from the area and
    # aspect limits. Pairing lines 3px apart into a "drawer" and only rejecting
    # it after four support scans was most of the runtime: a segment detector
    # returns both sides of every thick edge, so near-duplicate pairs are the
    # common case, not the rare one.
    # Never below 1.0: aspect is measured as long/short, so 1.0 is a square and
    # nothing can be under it. Without the clamp a tolerance of 1.0 or more puts
    # a zero in the denominator below.
    lo_asp = max(1.0, want * (1 - det["aspect_tol"]))
    hi_asp = max(lo_asp + 1e-6, want * (1 + det["aspect_tol"]))
    lo_area = det["min_area_frac"] * H * W
    hi_area = det["max_area_frac"] * H * W
    short_lo, short_hi = np.sqrt(lo_area / hi_asp), np.sqrt(hi_area / lo_asp)
    long_lo, long_hi = lo_asp * short_lo, hi_asp * short_hi

    def pairs(lines, axis, lo, hi):
        """Index pairs whose perpendicular spacing could bound the drawer."""
        mid = np.array([(l[axis] + l[axis + 2]) / 2.0 for l in lines])
        out = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if lo <= abs(mid[i] - mid[j]) <= hi:
                    out.append((i, j))
        return out

    h_pairs = pairs(horiz, 1, short_lo, short_hi)     # spaced apart in y
    v_pairs = pairs(vert, 0, long_lo, long_hi)        # spaced apart in x
    h_cov = [line_coverage(l, prepared, det["near_px"]) for l in horiz]
    v_cov = [line_coverage(l, prepared, det["near_px"]) for l in vert]
    hyps = []
    shaped = 0
    for i, j in h_pairs:
        for k, m in v_pairs:
            pts = [_line_intersect(horiz[i], vert[k]),
                   _line_intersect(horiz[i], vert[m]),
                   _line_intersect(horiz[j], vert[m]),
                   _line_intersect(horiz[j], vert[k])]
            if any(p is None for p in pts):
                continue
            q = np.array(pts)
            # A drawer may overhang the frame edge a little; a "rectangle" far
            # outside it is two unrelated lines crossing off in space.
            if (q.min() < -0.15 * W or q[:, 0].max() > 1.15 * W
                    or q[:, 1].max() > 1.15 * H):
                continue
            # Shoelace and edge lengths straight from the corners: cv2's
            # contourArea/isContourConvex/minAreaRect are three Python-to-C round
            # trips per candidate, and there are thousands of candidates.
            x, y = q[:, 0], q[:, 1]
            frac = abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2 / (H * W)
            if not (det["min_area_frac"] <= frac <= det["max_area_frac"]):
                continue
            e = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
            wid, hei = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
            if min(wid, hei) <= 0:
                continue
            if abs(max(wid, hei) / min(wid, hei) - want) > want * det["aspect_tol"]:
                continue
            # Sides in corner order: top (h[i]), right (v[m]), bottom (h[j]),
            # left (v[k]) - each queried against its own line's coverage table.
            shaped += 1
            sup = []
            for a, cov in enumerate((h_cov[i], v_cov[m], h_cov[j], v_cov[k])):
                s = side_support(q[a], q[(a + 1) % 4], cov)
                if s < det["min_support"]:
                    break               # one bare side is already disqualifying
                sup.append(s)
            if len(sup) < 4:
                continue
            hyps.append((float(np.mean(sup)), q, q.astype(np.int32)))
    if stats is not None:
        stats.update(h_pairs=len(h_pairs), v_pairs=len(v_pairs),
                     shaped=shaped, supported=len(hyps))
    if not hyps:
        return None

    hyps.sort(key=lambda t: -t[0])
    best, best_score = None, -1e9
    ring_k = np.ones((61, 61), np.uint8)
    for support, q, poly in hyps[:det["verify_top"]]:
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        ring = cv2.dilate(mask, ring_k) - mask
        inside = grey[mask > 0].mean() if mask.any() else 255.0
        outside = grey[ring > 0].mean() if ring.any() else inside
        score = support + max(0.0, (outside - inside) / 255.0)
        if score > best_score:
            best, best_score = q, score
    return order_corners(best / scale)      # back to full-resolution coordinates


class Localizer:
    """Finds the drawer in each frame and rectifies it to the canonical size.

    The contract the whole system rests on: every image handed downstream lands in
    the same coordinate frame as the references, or no image is handed downstream
    at all. So this reports one of three things per frame:

      "live"   - the drawer was found in this frame
      "held"   - momentarily lost, coasting on the last outline (a hand over a
                 corner must not blank the feed)
      "closed" - lost for longer than that. Returns no image, and the caller
                 stops evaluating rather than scoring whatever is now under the
                 slot rects. Shutting the drawer is the ordinary way this
                 happens, and reading "every tool taken" off the drawer front
                 would be worse than reading nothing.

    Re-locating every frame is the point: the drawer moves, it is opened to
    different depths, and the outline has to follow it. A detection that jumps
    further than `jump_px` from the last accepted one is refused instead - real
    drawers slide, they do not teleport - but refusals count toward the grace
    period, so a drawer that genuinely was moved is re-acquired a few frames later
    from scratch.
    """

    def __init__(self, cfg):
        self.mode = cfg.get("localizer", "auto")
        self.size = tuple(cfg["warp_size"])
        self.det = {**DETECT, **cfg.get("detect", {})}
        self.marker_ids = cfg.get("marker_ids", [0, 1, 2, 3])
        self.detector = make_detector() if self.mode == "aruco" else None
        self.stats = {}         # last detection's stage counts, for hint()
        self.corners = None     # last accepted, in source-frame coordinates
        self.H = None
        self.misses = 0

    def find(self, frame):
        """Raw detection for this frame: 4 ordered corners, or None.

        Markerless order of preference: a colour mask if the foam has a distinct
        one (cheapest and hardest to fool), otherwise line segments.
        """
        if self.mode == "aruco":
            return aruco_corners(frame, self.detector, self.marker_ids)
        if self.det["hsv"]:
            return foam_corners(frame, self.det)
        self.stats = {}
        return lsd_corners(frame, self.size, self.det, self.corners, self.stats)

    def locate(self, frame):
        """(rectified image or None, state) - see the class docstring."""
        corners = self.find(frame)
        if corners is not None and self.corners is not None:
            if np.abs(corners - self.corners).max() > self.det["jump_px"]:
                corners = None      # spurious: treat exactly like a miss
        if corners is None:
            self.misses += 1
            if self.corners is None or self.misses > self.det["grace_frames"]:
                self.corners, self.H = None, None
                return None, "closed"
            return self.rectify(frame), "held"
        self.misses = 0
        self.corners = corners
        w, h = self.size
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        self.H = cv2.getPerspectiveTransform(corners, dst)
        return self.rectify(frame), "live"

    def rectify(self, frame):
        return cv2.warpPerspective(frame, self.H, self.size)

    def hint(self, frame):
        """Why the drawer was not found, in terms the operator can act on.

        "NO DRAWER" alone is the least useful thing to show someone: it cannot
        distinguish a shut drawer from the wrong camera, and in marker mode the
        detector already knows exactly which ids it can see.
        """
        if self.mode != "aruco":
            s = self.stats
            if not s:
                return "no line segments at all - check focus and lighting"
            # Name the stage that emptied, because each one implicates a
            # different knob and they are not interchangeable.
            if s.get("horiz", 0) < 2 or s.get("vert", 0) < 2:
                return (f"{s.get('segs', 0)} segments but only h{s.get('horiz', 0)}"
                        f"/v{s.get('vert', 0)} usable - drawer edges too faint, "
                        f"lower detect.min_seg_len or detect.canny")
            if not s.get("h_pairs") or not s.get("v_pairs"):
                return (f"h{s['horiz']}/v{s['vert']} lines, no pair spaced like a "
                        f"drawer - check detect.min_area_frac / max_area_frac")
            if not s.get("shaped"):
                return (f"{s['h_pairs']}x{s['v_pairs']} line pairs, none the right "
                        f"shape - check detect.aspect_tol vs warp_size")
            return (f"{s['shaped']} rectangles found, none with supported edges - "
                    f"lower detect.min_support")
        corners, ids, _ = self.detector.detectMarkers(frame)
        found = sorted(int(i) for i in ids.flatten()) if ids is not None else []
        missing = [m for m in self.marker_ids if m not in found]
        if not found:
            return "no markers at all - wrong camera? try diagnose.py"
        return f"markers found {found}, missing {missing}"


# ---------------------------------------------------------------- aruco
# Kept as an alternative to markerless localization: set "localizer": "aruco" in
# config.json and print the sheet with gen_markers.py. Markers cannot be fooled
# by a cluttered bench, but they have to be stuck on and stay stuck - a corner
# knocked a millimetre skews the whole rectification underneath slot rects that
# never move, and only register() below can see that happen.

def aruco_dict():
    """The one place the marker dictionary is chosen.

    gen_markers.py prints from this and the tracker detects with it, so the
    printed sheet cannot drift out of sync with what the detector accepts.
    """
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def make_detector():
    """ArUco detector for the 4 corner markers (opencv-contrib >= 4.7 API)."""
    return cv2.aruco.ArucoDetector(aruco_dict(), cv2.aruco.DetectorParameters())


def aruco_corners(frame, detector, marker_ids):
    """The drawer outline as seen by four printed markers, or None.

    marker_ids is [top-left, top-right, bottom-right, bottom-left]. For each we
    take that marker's *outer* corner, so the rectified image encloses the
    markers. Each marker must be stuck down rotationally aligned with the drawer -
    a marker rotated 90 degrees makes "its top-left corner" the wrong point and
    subtly skews everything.

    All four must be present exactly once. A duplicate id (a reflection in the
    drawer metal, a second cart in shot) would otherwise silently win the dict and
    produce a plausible-but-wrong outline. Already in TL/TR/BR/BL order by
    construction, so order_corners() is neither needed nor wanted here.
    """
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return None
    flat = [int(i) for i in ids.flatten()]
    if not all(flat.count(m) == 1 for m in marker_ids):
        return None
    found = {int(i): c[0] for i, c in zip(flat, corners)}
    # corner index i of marker i => TL of TL marker, TR of TR marker, ...
    return np.array([found[m][i] for i, m in enumerate(marker_ids)], np.float32)


def register(warped, drawer, last_M=None):
    """Re-anchor a warped frame onto the drawer's own contents.

    The homography above is pinned to four marker corners, so the warp *is* the
    markers. A marker knocked a millimetre - or peeled and re-stuck a little off
    - slides and skews the entire warped image underneath ROI rects that never
    move, and every tool reads missing at once. Nothing in the marker path can
    detect this: the markers are the only reference it has.

    So the drawer itself becomes the reference. One ECC pass against the
    baseline recovers the residual homography (translation *and* the scale/skew
    a moved corner introduces - a pure translation fix leaves most of the error
    behind) and puts the frame back where the ROIs expect it.

    Estimated on a quarter-size grey and scaled back up: coordinates scale by s,
    so a transform between them is S-inverse @ M @ S. Warm-started from the last
    frame's answer, which is why this costs ~2ms in the steady state.

    Returns (aligned, M). A poor lock keeps `last_M` rather than accepting a
    guess - a wrong alignment is indistinguishable from every tool vanishing.
    """
    ref_small = drawer.get("baseline_small")
    if ref_small is None:
        return warped, last_M
    s = REGISTER_SCALE
    small = cv2.resize(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s)
    M = np.eye(3, 3, dtype=np.float32) if last_M is None else last_M.copy()
    try:
        cc, M = cv2.findTransformECC(
            ref_small, small, M, cv2.MOTION_HOMOGRAPHY,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, REGISTER_ITERS, 1e-6),
            None, 5)
    except cv2.error:
        # ECC raises rather than returning when it cannot converge at all.
        cc, M = 0.0, last_M
    if M is None or cc < REGISTER_MIN_CC:
        M = last_M
    if M is None:
        return warped, None
    S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], np.float32)
    full = np.linalg.inv(S) @ M @ S
    return cv2.warpPerspective(warped, full, warped.shape[1::-1],
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE), M


# ---------------------------------------------------------------- detection

def channels(img):
    """(grey, Sobel gradient magnitude) of a rectified frame, both float32.

    The two channels every score is computed on. Grey separates a tool from its
    pocket by tone; gradients separate them by outline, which is what survives a
    highlight crawling across chrome. One Sobel for the whole frame, not one per
    slot - the slots overlap the same rows and it is the same cost either way.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return grey, cv2.magnitude(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3),
                               cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3))


def slot_mask(slot):
    """Binary mask of a slot's own shape, in its bounding rect's coordinates.

    Slots are not rectangles. A wrench lies diagonally and its bounding box
    swallows a corner of each neighbour, so a rectangular comparison partly scores
    the tool next door - and reports the wrong one missing when that neighbour
    moves. The polygon confines each score to the slot's actual outline.

    A slot with no polygon (the shape calibrate.py records for a plain drag) is
    all-ones, which is exactly the old rectangular behaviour.
    """
    x, y, w, h = slot["rect"]
    mask = np.full((h, w), 255, np.uint8)
    poly = slot.get("polygon")
    if poly:
        mask[:] = 0
        cv2.fillPoly(mask, [np.array(poly, np.int32) - (x, y)], 255)
    return mask


POCKETS = {
    "s_thr": 60,            # saturation above which a pixel is not bare foam
    "v_thr": 110,           # ...or brightness, which catches chrome with no hue
    "close_px": 9,          # bridge the label stickers that split one pocket
    "min_area_frac": 0.004, # a region smaller than this is speckle, not a slot
}


def detect_pockets(img, cfg=None):
    """Outline every moulded pocket in a bare, rectified drawer.

    Shadow foam is black and matte, so the segmentation is "not foam" rather than
    "tool": saturation catches the coloured pocket bases, brightness catches
    anything chrome, and the foam itself has neither. Measured on this drawer the
    saturation threshold works anywhere from 60 to 170 and brightness never
    matters at all, so this is far less tuned than it looks.

    Run it on the *bare* drawer, never the full one. Tools lie touching each
    other and segment into merged blobs; the pockets they came out of are
    separated by foam and segment into exactly one region per slot.
    """
    p = {**POCKETS, **(cfg or {})}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (((hsv[:, :, 1] > p["s_thr"]) | (hsv[:, :, 2] > p["v_thr"]))
         .astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["close_px"],) * 2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = img.shape[0] * img.shape[1]
    out = []
    for c in contours:
        if cv2.contourArea(c) < p["min_area_frac"] * area:
            continue
        # Convex hull, not the raw contour: a pocket's jaws and finger cutouts
        # produce concavities that would exclude the very pixels a tool sitting
        # in them occupies, which is what the mask is meant to measure.
        hull = cv2.convexHull(c)
        poly = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True)
        out.append(poly.reshape(-1, 2))
    return sorted(out, key=lambda p: -cv2.contourArea(p))


def tool_outline(img, pocket, cfg=None):
    """Trace whatever object is lying in one pocket, or None if nothing is.

    Shape only - this deliberately does NOT decide occupancy. On foam with
    coloured pocket bases an empty pocket is *more* "not foam" than a full one
    (measured 0.57-0.84 empty against 0.46-0.76 full: a tool covers the bright
    base with dark rubber and chrome). Segmentation therefore says where a thing
    is, and the correlation score in classify() says whether it is there.

    Masking to one pocket at a time is what makes this work at all: tools lying
    against each other merge into one blob when the whole drawer is thresholded,
    but a neighbour cannot merge with a tool it is not in the same image as.
    """
    p = {**POCKETS, **(cfg or {})}
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.asarray(pocket, np.int32)], 255)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    nf = (((hsv[:, :, 1] > p["s_thr"]) | (hsv[:, :, 2] > p["v_thr"]))
          .astype(np.uint8) * 255)
    local = cv2.morphologyEx(cv2.bitwise_and(nf, m), cv2.MORPH_CLOSE,
                             np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < p["min_area_frac"] * img.shape[0] * img.shape[1]:
        return None
    hull = cv2.convexHull(c)
    return cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True).reshape(-1, 2)


def match_pockets(pockets, rois, shape, min_iou=0.25):
    """slot id -> the detected pocket that overlaps it, matched once at startup.

    By overlap rather than by order: tool ids key the event log, so a slot that
    silently picked up a different pocket would draw one tool's outline over
    another's history.
    """
    out, taken = {}, set()
    for r in rois:
        scored = sorted(((polygon_iou(p, r["rect"], shape), i)
                         for i, p in enumerate(pockets)), reverse=True)
        if not scored:
            continue
        best, idx = scored[0]
        if best >= min_iou and idx not in taken:
            taken.add(idx)
            out[r["id"]] = pockets[idx]
    return out


def polygon_iou(poly, rect, shape):
    """Overlap between a detected polygon and an existing slot rect, 0..1."""
    a = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(a, [np.asarray(poly, np.int32)], 1)
    b = np.zeros(shape[:2], np.uint8)
    x, y, w, h = rect
    b[y:y + h, x:x + w] = 1
    union = (a | b).sum()
    return float((a & b).sum()) / float(union or 1)


def template(plane, rect, mask):
    """Cut one reference template out of a channel, neutralising what is masked.

    Masked-out pixels are set to the *mean of what is kept* rather than to zero.
    TM_CCOEFF_NORMED scores deviations from the template mean, so a pixel sitting
    exactly at that mean contributes nothing to the match instead of contributing
    a large fake deviation - which is what a black border would do, and it would
    dominate the very score it was supposed to stay out of.
    """
    x, y, w, h = rect
    tpl = plane[y:y + h, x:x + w].copy()
    keep = mask > 0
    if not keep.all():
        tpl[~keep] = tpl[keep].mean() if keep.any() else 0.0
    return tpl


def build_templates(drawer):
    """Per slot, the four templates every frame is scored against.

    Grey and gradient, for the drawer full and the drawer bare. Cut once at load
    time because they never change except when drift() nudges one, and re-cutting
    them per frame would be the most expensive thing in the loop.
    """
    planes = {which: (None if drawer.get(which) is None else channels(drawer[which]))
              for which in ("baseline", "empty")}
    drawer["templates"] = {}
    for slot in drawer["rois"]:
        mask = slot_mask(slot)
        t = {"mask": mask}
        for which, pair in planes.items():
            if pair is None:
                continue
            grey, grad = pair
            t[which + "_grey"] = template(grey, slot["rect"], mask)
            t[which + "_grad"] = template(grad, slot["rect"], mask)
        drawer["templates"][slot["id"]] = t
    return drawer["templates"]


def slot_score(grey, grad, rect, tpl, which, grad_weight=GRAD_WEIGHT,
               radius=SEARCH_RADIUS):
    """How strongly one slot resembles a reference: +1 identical, 0 unrelated.

    Normalised cross-correlation rather than a pixel difference, because the
    normalisation is the whole point: TM_CCOEFF_NORMED subtracts each window's own
    mean and divides by its own energy, so dimming the room, or a lamp arriving,
    moves the score by almost nothing. A raw difference moves with every one of
    them, which is what forces per-slot magic numbers that then drift.

    Scored over a +/-`radius` window and taking the best offset: ECC has already
    aligned the frame globally, but a pixel or two of residue is normal and a
    3-pixel misalignment alone can invalidate the comparison.
    """
    x, y, w, h = rect
    x0, y0 = max(0, x - radius), max(0, y - radius)
    x1, y1 = min(grey.shape[1], x + w + radius), min(grey.shape[0], y + h + radius)
    s = 0.0
    for chan, plane, weight in (("grey", grey, 1.0 - grad_weight),
                                ("grad", grad, grad_weight)):
        if weight <= 0:
            continue
        r = cv2.matchTemplate(plane[y0:y1, x0:x1], tpl[which + "_" + chan],
                              cv2.TM_CCOEFF_NORMED)
        s += weight * float(r.max())
    return s


def classify(s_full, s_empty, uncertain_margin=0.0):
    """True (tool out), False (tool there), or None (too close to call).

    Nearest of two references, never a threshold on one score. "Does this look
    more like the full drawer or the bare drawer" needs no per-slot number and
    self-calibrates across lighting, where "is this score above 40" is a knife
    edge whenever the pocket is the same colour as the tool it holds. Ties count
    as present: absent has to win outright.

    Without the bare reference there is no comparison to make, and inventing an
    absolute threshold to fill the gap is exactly what this avoids - so the answer
    is None until the operator shoots one with 'e'.

    When the two scores are within `uncertain_margin` the answer is None rather
    than a coin flip. The caller holds the last known state instead of logging a
    change it cannot justify: a wrong Taken event is worse than a late one,
    because the audit trail has no way to express "probably".
    """
    if s_empty is None:
        return None
    if abs(s_full - s_empty) < uncertain_margin:
        return None
    return s_empty > s_full


def drift(drawer, which, slot, warped, alpha):
    """Blend one slot of the live frame into a reference, in place.

    The baseline is a photo frozen at calibration time, so an hour of the sun
    moving across the room walks every slot away from it at once. Nudging the
    reference toward frames we are confident about tracks that drift out; the
    rate is slow enough that a tool removed and left out cannot teach the
    baseline that its empty pocket is normal before the debounce has logged it.

    Correlation already shrugs off uniform brightness changes, so this is
    catching what it cannot: a shadow edge creeping across one corner of the
    drawer, a lamp changing colour temperature.
    """
    x, y, w, h = slot["rect"]
    sl = (slice(y, y + h), slice(x, x + w))
    ref = drawer[which]
    ref[sl] = cv2.addWeighted(ref[sl], 1.0 - alpha, warped[sl], alpha, 0)
    grey, grad = channels(ref[sl])
    tpl, mask = drawer["templates"][slot["id"]], drawer["templates"][slot["id"]]["mask"]
    local = [0, 0, w, h]
    tpl[which + "_grey"] = template(grey, local, mask)
    tpl[which + "_grad"] = template(grad, local, mask)
    # ponytail: the gradient is recut from the blended patch alone, so its one-pixel
    # border sees the slot edge instead of the drawer around it. Immaterial at
    # alpha 0.02 against a template of thousands of pixels, and it keeps drift a
    # slot-local operation instead of a whole-frame Sobel per nudge.
    # The drawer fingerprint is left alone too - a few tool-sized patches barely
    # move a 100x70 thumbnail, and identification only needs to rank drawers.


# ---------------------------------------------------------------- drawers

def slugify(name):
    """"Top Drawer" -> "top_drawer". Ids and filenames derive from names."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "tool"


def fingerprint(img):
    """Small grey thumbnail used to compare a frame against a drawer reference.

    Downsized because identification is a question about layout - which foam
    cutouts are where - and matching at full resolution just makes every drawer
    equally distant once the light shifts.
    """
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), ID_SIZE).astype(np.int16)


def drawer_score(warped_fp, drawer):
    """How unlike this drawer the frame looks: lower is a better match.

    Scored against the full *and* (when shot) the empty reference, taking
    whichever is closer. A drawer with every tool out looks nothing like its own
    baseline, and without the empty reference it would hand itself to whatever
    other drawer happened to be less unlike it.
    """
    scores = [float(np.mean(np.abs(warped_fp - drawer["fp"])))]
    if drawer.get("fp_empty") is not None:
        scores.append(float(np.mean(np.abs(warped_fp - drawer["fp_empty"]))))
    return min(scores)


def identify(warped, drawers, active=None):
    """Return the drawer now in shot, sticking with `active` unless clearly beaten.

    One drawer configured means no identification to do (and no reference to be
    wrong about), so the answer is always that drawer.
    """
    if len(drawers) == 1:
        return drawers[0]
    fp = fingerprint(warped)
    scored = sorted(((drawer_score(fp, d), d) for d in drawers), key=lambda s: s[0])
    best_score, best = scored[0]
    if active is None or best is active:
        return best
    return best if best_score < SWITCH_MARGIN * drawer_score(fp, active) else active


def photo_name(missing_ids, stamp):
    """Filename tune.py can parse back into labels.

    Named from what the tracker currently believes is out, so the training set
    labels itself. Sorted for a stable name; the stamp keeps repeat shots of the
    same state from overwriting each other.
    """
    if not missing_ids:
        return f"full_{stamp}.jpg"
    return "missing_" + "+".join(sorted(missing_ids)) + f"_{stamp}.jpg"


class Debouncer:
    """Emits a state change only after `hold` seconds of a consistent reading.

    update() returns "Taken", "Returned" or None - never a bare bool, because a
    falsy-but-meaningful return is exactly the kind of thing a later `if
    d.update(...)` silently swallows (it would drop every Returned event).

    Feed readings only while the frame is static; call freeze() during motion so
    a hand crossing the ROI never accumulates time toward a change.
    """

    def __init__(self, missing=False, hold=HOLD_S):
        self.missing = missing
        self.candidate = missing
        self.since = None
        self.hold = hold
        self.steady = 0     # consecutive confident frames agreeing with `missing`

    def update(self, reading, now):
        """Return TAKEN / RETURNED if the state just changed, else None.

        `reading` is True/False, or None when the frame is too close to call -
        which is treated exactly like motion: proof of nothing, so hold.
        """
        if reading is None:
            self.freeze()
            return None
        if reading == self.missing:
            self.candidate, self.since = self.missing, None
            self.steady += 1
            return None
        self.steady = 0
        if reading != self.candidate:
            self.candidate, self.since = reading, now
        if now - self.since >= self.hold:
            self.missing = reading
            self.since = None
            return TAKEN if reading else RETURNED
        return None

    def freeze(self):
        """Drop any pending change, and any confidence built up with it.

        Motion, or a reading too close to call, proves nothing - and a reference
        must never learn from a frame with a hand in it, so the drift counter
        goes back to zero too.
        """
        self.candidate, self.since, self.steady = self.missing, None, 0


# ---------------------------------------------------------------- storage

SCHEMA = """
PRAGMA journal_mode=WAL;               -- dashboard can read while we write
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,           -- ISO-8601 UTC, e.g. 2026-08-12T14:03:11Z
    tool_id   TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    action    TEXT NOT NULL CHECK (action IN ('Taken', 'Returned'))
);
CREATE INDEX IF NOT EXISTS idx_events_ts   ON events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_tool ON events(tool_id, id DESC);

-- Current state of every tool, one row each: SELECT * FROM current_state;
-- One grouped index scan, not a correlated subquery per row, so a dashboard
-- polling this stays cheap as the event history grows.
CREATE VIEW IF NOT EXISTS current_state AS
SELECT tool_id, tool_name, action, ts FROM events
WHERE id IN (SELECT MAX(id) FROM events GROUP BY tool_id);
"""


def open_db(path=DB_PATH):
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.commit()
    return db


def log_event(db, tool_id, tool_name, action):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    db.execute(
        "INSERT INTO events (ts, tool_id, tool_name, action) VALUES (?,?,?,?)",
        (ts, tool_id, tool_name, action),
    )
    db.commit()
    print(f"{ts}  {tool_id:<20} {tool_name:<24} {action}")


def seed_debouncers(db, rois, hold=HOLD_S):
    """Start each tool from its last persisted state, not from 'present'.

    Without this, restarting the tracker while three tools are legitimately out
    logs three fresh 'Taken' events with the wrong timestamp - the audit trail
    would then claim they were removed at every restart.
    """
    persisted = dict(db.execute("SELECT tool_id, action FROM current_state"))
    return {r["id"]: Debouncer(persisted.get(r["id"]) == TAKEN, hold) for r in rois}


def seed_all(db, drawers, hold=HOLD_S):
    """Debouncers for every drawer, not just the one first in shot.

    Each drawer keeps its own set, so the drawers currently out of frame simply
    hold their last state instead of being re-seeded (and re-logged) every time
    the camera swings back to them.
    """
    for d in drawers:
        d["debouncers"] = seed_debouncers(db, d["rois"], hold)
    return drawers


# ---------------------------------------------------------------- config

def normalize_drawers(cfg):
    """Config -> list of drawer dicts, accepting the old single-drawer layout.

    A pre-multi-drawer config has its ROIs at the top level and its references in
    baseline.png/empty.png; it keeps working untouched as the one drawer "main".
    """
    drawers = cfg.get("drawers")
    if not drawers:
        if not cfg.get("rois"):
            raise ValueError("config defines no drawers and no ROIs - run calibrate.py")
        drawers = [{"name": "main", "rois": cfg["rois"],
                    "baseline_file": BASELINE_PATH.name,
                    "empty_file": EMPTY_PATH.name}]
    seen_names, seen_ids = set(), {}
    for d in drawers:
        if not d.get("name"):
            raise ValueError("every drawer needs a name")
        if not d.get("rois"):
            raise ValueError(f"drawer {d['name']!r} defines no ROIs")
        if d["name"] in seen_names:
            raise ValueError(f"duplicate drawer name {d['name']!r}")
        seen_names.add(d["name"])
        slug = slugify(d["name"])
        d.setdefault("baseline_file", f"baseline_{slug}.png")
        d.setdefault("empty_file", f"empty_{slug}.png")
        # Tool ids key the event log, so a id reused in a second drawer would
        # merge two physical tools into one history.
        for r in d["rois"]:
            if r["id"] in seen_ids:
                raise ValueError(f"tool id {r['id']!r} used in both drawer "
                                 f"{seen_ids[r['id']]!r} and {d['name']!r}")
            seen_ids[r["id"]] = d["name"]
    return drawers


def validate_rois(rois, size):
    """Raise ValueError on any slot the main loop could not survive.

    An out-of-bounds rect slices to an empty array and matchTemplate raises on it,
    killing the loop mid-shift. A rect that merely overruns an edge is worse - it
    silently truncates and watches the wrong region with no error at all.

    Where a polygon is given it is authoritative and the rect is recomputed from
    it. The two are stored separately (the rect is what gets sliced, the polygon
    is what gets masked) and a hand-edited config that moved one without the other
    would otherwise mask the wrong part of the slot with no complaint.
    """
    w, h = size
    seen = set()
    for r in rois:
        poly = r.get("polygon")
        if poly:
            if len(poly) < 3:
                raise ValueError(f"slot {r['id']} polygon needs at least 3 points")
            r["rect"] = list(cv2.boundingRect(np.array(poly, np.int32)))
        x, y, rw, rh = r["rect"]
        if rw <= 0 or rh <= 0 or x < 0 or y < 0 or x + rw > w or y + rh > h:
            raise ValueError(f"slot {r['id']} rect {r['rect']} is outside the {w}x{h} frame")
        if r["id"] in seen:
            raise ValueError(f"duplicate tool id {r['id']!r} - ids must be unique")
        seen.add(r["id"])
        # Left over from the absolute-threshold scoring this replaced; a stale
        # number in the file invites someone to tune a knob nothing reads.
        r.pop("threshold", None)
    return rois


def load_config(config_path=CONFIG_PATH):
    """Load config + every drawer's references, failing fast on anything the
    loop can't survive.

    Every check here corresponds to a failure that is otherwise silent or that
    surfaces as an unreadable numpy/cv2 traceback hours into a shift.
    """
    if not config_path.exists():
        raise SystemExit(f"{config_path.name} missing - run calibrate.py first")
    cfg = json.loads(config_path.read_text())

    w, h = cfg["warp_size"]
    if w <= 0 or h <= 0:
        raise SystemExit(f"warp_size {cfg['warp_size']} must be positive")

    try:
        drawers = normalize_drawers(cfg)
        for d in drawers:
            validate_rois(d["rois"], (w, h))
            load_references(d, (w, h), config_path.parent)
    except ValueError as e:
        raise SystemExit(str(e))
    cfg["drawers"] = drawers
    return cfg


# Attached at load time, meaningless (and unserialisable) in the file itself.
RUNTIME_KEYS = ("baseline", "empty", "fp", "fp_empty", "baseline_small",
                "templates", "debouncers", "baseline_path", "empty_path")


def save_config(cfg, path=CONFIG_PATH):
    """Write config.json back without the loaded images and debouncers.

    Also drops the legacy top-level "rois": normalize_drawers has already moved
    them under drawers, and leaving both would let the two copies drift apart
    with no rule about which one wins.
    """
    out = {k: v for k, v in cfg.items() if k != "rois"}
    out["drawers"] = [{k: v for k, v in d.items() if k not in RUNTIME_KEYS}
                      for d in cfg["drawers"]]
    path.write_text(json.dumps(out, indent=2))


def load_references(drawer, size, here=HERE):
    """Attach baseline/empty images (and their fingerprints) to a drawer.

    A wrong-sized reference is a stale shot from before a re-calibration; using
    it would compare each ROI against the wrong part of the drawer, so refuse it
    loudly rather than degrade into confident nonsense.
    """
    w, h = size
    drawer["baseline_path"] = here / drawer["baseline_file"]
    drawer["empty_path"] = here / drawer["empty_file"]

    baseline = cv2.imread(str(drawer["baseline_path"]))
    if baseline is None:
        raise ValueError(f"{drawer['baseline_file']} missing for drawer "
                         f"{drawer['name']!r} - run calibrate.py")
    if baseline.shape[:2] != (h, w):
        raise ValueError(
            f"{drawer['baseline_file']} is {baseline.shape[1]}x{baseline.shape[0]} "
            f"but config warp_size is {w}x{h} - re-run calibrate.py")
    set_reference(drawer, "baseline", baseline)

    empty = None
    if not drawer["empty_path"].exists():
        # Not fatal, but nothing can be classified without it: the decision is
        # "which of the two references does this slot resemble", and there is no
        # second reference. Say so once here rather than let the operator watch
        # every slot sit at UNSURE and guess why.
        print(f"warning: {drawer['empty_file']} missing for drawer "
              f"{drawer['name']!r} - empty the drawer and press 'e'; until then "
              f"nothing can be classified")
    if drawer["empty_path"].exists():
        # Shot before the current baseline: possibly stale, possibly fine (a
        # runtime 'b' re-baseline moves the baseline forward on its own). Warn,
        # never refuse - an older empty reference still beats no empty reference,
        # and a hard failure here locks the operator out of the 'e' key that
        # fixes it.
        if drawer["empty_path"].stat().st_mtime < drawer["baseline_path"].stat().st_mtime:
            print(f"warning: {drawer['empty_file']} is older than "
                  f"{drawer['baseline_file']} - if readings look random, re-shoot "
                  f"it with 'e'")
        empty = cv2.imread(str(drawer["empty_path"]))
        if empty is None or empty.shape != baseline.shape:
            raise ValueError(f"{drawer['empty_file']} does not match "
                             f"{drawer['baseline_file']} - delete it and re-shoot "
                             f"with 'e'")
        # The two references must share one coordinate frame or nothing works:
        # runtime frames are registered onto the baseline, so an empty reference
        # shot at a different drawer position leaves every "empty" template
        # looking at the wrong patch of foam - and no tool can then ever score
        # as missing. They are usually shot minutes and one nudge apart, so
        # this is the common case, not the exotic one. Same ECC pass the loop
        # runs, cold, once.
        empty, M = register(empty, drawer)
        if M is None:
            print(f"warning: {drawer['empty_file']} could not be aligned to "
                  f"{drawer['baseline_file']} - re-shoot it with 'e' without "
                  f"moving the drawer")
    set_reference(drawer, "empty", empty)
    return drawer


def set_reference(drawer, which, img):
    """Store a reference image and keep everything derived from it in step.

    Identification reads the fingerprints and classification reads the templates,
    so a re-baseline that updated only the image would keep matching the drawer -
    and scoring its slots - on what it used to look like.
    """
    drawer[which] = img
    if which == "baseline":
        # register() aligns against this every frame; resizing it there instead
        # would redo the same work 30 times a second.
        drawer["baseline_small"] = None if img is None else cv2.resize(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None,
            fx=REGISTER_SCALE, fy=REGISTER_SCALE)
    drawer["fp" if which == "baseline" else "fp_empty"] = (
        None if img is None else fingerprint(img))
    if drawer.get("rois"):
        build_templates(drawer)


# ---------------------------------------------------------------- main loop

class StillCapture:
    """A photo served as an endless frozen stream, quacking like VideoCapture.

    Lets the whole pipeline run off saved images so the warp, ROI comparison and
    thresholds can be checked without rigging a camera over a real drawer.
    """

    def __init__(self, frame):
        self.frame = frame

    def read(self):
        return True, self.frame.copy()   # copy: callers draw overlays on it

    def isOpened(self):
        return True

    def release(self):
        pass


def open_camera(source):
    """Open a camera index, or a still image path for photo testing."""
    if isinstance(source, str) and not source.isdigit():
        frame = cv2.imread(source)
        if frame is None:
            raise SystemExit(f"cannot read image {source}")
        return StillCapture(frame)
    cap = cv2.VideoCapture(int(source))
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {source}")
    return cap


def preview_detect(source=None, mode=None, aspect_tol=0.6, size=None):
    """Live: find the drawer, then outline whatever is lying in it. Nothing else.

    No configured slots, no tool ids, no reference comparison - so what is drawn
    is only ever what this frame actually contains. That makes it the honest view
    of how well detection works on a drawer nobody has calibrated, which is the
    question that matters for every drawer after the first.

    The ceiling is that segmentation alone cannot say whether a tool is *missing*
    (an empty pocket with a coloured base is as "not foam" as a tool is), and
    touching tools merge into one outline. Both are inherent here; occupancy is
    what the configured pipeline in main() is for.
    """
    cfg = load_config()
    cfg["localizer"] = mode or "auto"
    if size:
        cfg["warp_size"] = list(size)
    # An uncalibrated drawer is not the shape this config was built around, and
    # rejecting it on proportions would look exactly like detection failing.
    cfg["detect"] = {**cfg.get("detect", {}), "aspect_tol": aspect_tol}
    localizer = Localizer(cfg)
    cap = open_camera(source if source is not None else cfg.get("camera_index", 0))
    print(f"live detection: localizer={cfg['localizer']}  "
          f"canonical {cfg['warp_size']}  aspect_tol {aspect_tol}  q quits")
    fails = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails >= MAX_READ_FAILS:
                    raise SystemExit(f"no frames after {fails} reads - see diagnose.py")
                if cv2.waitKey(int(READ_RETRY_DELAY_S * 1000)) & 0xFF == ord("q"):
                    break
                continue
            fails = 0
            t = time.perf_counter()
            warped, state = localizer.locate(frame)
            if warped is None:
                cv2.putText(frame, "NO DRAWER", (20, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame, localizer.hint(frame), (20, 82),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.imshow("drawer", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            cv2.polylines(frame, [np.round(localizer.corners).astype(np.int32)],
                          True, (0, 255, 0) if state == "live" else (0, 165, 255), 4)
            found = detect_pockets(warped)
            view = warped.copy()
            for p in found:
                cv2.polylines(view, [p.astype(np.int32)], True, (0, 255, 0), 2)
            ms = (time.perf_counter() - t) * 1000
            cv2.putText(view, f"{len(found)} objects detected   {ms:.0f} ms",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.imshow("drawer", frame)
            cv2.imshow("live detection", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def preview_tools(source=None, mode=None):
    """Live: drawer localization and per-slot tool detection in one view.

    Both halves of the system at once, each doing the job it is actually good at.
    Localization finds the drawer every frame; segmentation traces the shape of
    whatever lies in each pocket; correlation decides whether it is the tool.
    Nothing is logged and no state is kept - this is for watching the pipeline,
    not for running a shift.
    """
    cfg = load_config()
    if mode:
        cfg["localizer"] = mode
    drawer = cfg["drawers"][0]
    if drawer["empty"] is None:
        raise SystemExit("no empty reference - run tracker.py, empty the drawer "
                         "and press 'e'")
    pockets = detect_pockets(drawer["empty"])
    by_slot = match_pockets(pockets, drawer["rois"], drawer["empty"].shape)
    gw = cfg.get("grad_weight", GRAD_WEIGHT)
    um = cfg.get("uncertain_margin", UNCERTAIN_MARGIN)
    rad = cfg.get("search_radius", SEARCH_RADIUS)
    localizer = Localizer(cfg)
    cap = open_camera(source if source is not None else cfg.get("camera_index", 0))
    print(f"preview+tools: localizer={cfg.get('localizer')}  "
          f"{len(pockets)} pockets, {len(by_slot)} matched to slots  |  q quits")
    fails, M = 0, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails >= MAX_READ_FAILS:
                    raise SystemExit(f"no frames after {fails} reads - see diagnose.py")
                if cv2.waitKey(int(READ_RETRY_DELAY_S * 1000)) & 0xFF == ord("q"):
                    break
                continue
            fails = 0
            t = time.perf_counter()
            warped, state = localizer.locate(frame)
            if warped is None:
                M = None
                cv2.putText(frame, "NO DRAWER - closed or out of shot", (20, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame, localizer.hint(frame), (20, 82),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.imshow("preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            cv2.polylines(frame, [np.round(localizer.corners).astype(np.int32)],
                          True, (0, 255, 0) if state == "live" else (0, 165, 255), 4)

            aligned, M = register(warped, drawer, M)
            grey, grad = channels(aligned)
            view = aligned.copy()
            present = missing = unsure = 0
            for r in drawer["rois"]:
                tpl = drawer["templates"][r["id"]]
                s_full = slot_score(grey, grad, r["rect"], tpl, "baseline", gw, rad)
                s_empty = slot_score(grey, grad, r["rect"], tpl, "empty", gw, rad)
                verdict = classify(s_full, s_empty, um)
                colour = ((0, 200, 255) if verdict is None else
                          (0, 0, 255) if verdict else (0, 255, 0))
                unsure += verdict is None
                missing += verdict is True
                present += verdict is False
                # The traced shape, drawn only where the score says a tool is
                # there - segmentation cannot make that call itself.
                pocket = by_slot.get(r["id"])
                if pocket is not None and verdict is False:
                    shape = tool_outline(aligned, pocket)
                    if shape is not None:
                        cv2.polylines(view, [shape.astype(np.int32)], True, colour, 2)
                x, y, w, h = r["rect"]
                cv2.rectangle(view, (x, y), (x + w, y + h), colour, 1)
                cv2.putText(view, r["name"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, colour, 1)
            ms = (time.perf_counter() - t) * 1000
            cv2.putText(view, f"{state} {ms:.0f}ms | present {present} "
                              f"missing {missing} unsure {unsure}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("preview", frame)
            cv2.imshow("drawer + tools", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def preview(source=None, mode=None):
    """Localization only: draw the outline and the rectified view, score nothing.

    The one way to try a localizer against a real drawer without committing to it.
    Switching localizers changes the rectification, so the slots and references
    built for the old one are meaningless under the new - which makes "just flip
    it and see" impossible, and leaves an operator with no way to find out whether
    markerless would work on their bench short of recalibrating the whole thing
    first. This answers that question in isolation and costs nothing to abandon.
    """
    cfg = load_config()
    if mode:
        cfg["localizer"] = mode
    localizer = Localizer(cfg)
    cap = open_camera(source if source is not None else cfg.get("camera_index", 0))
    print(f"preview: localizer={cfg.get('localizer', 'auto')}  q quits")
    fails = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # macOS hands back nothing for the first several reads while the
                # device wakes up, so a single failure means "not yet", not "no
                # camera". Bailing on it looks exactly like the program exiting
                # for no reason.
                fails += 1
                if fails >= MAX_READ_FAILS:
                    raise SystemExit(
                        f"no frames after {fails} reads - check the camera index "
                        f"(diagnose.py lists them) and, on macOS, that the "
                        f"terminal has camera permission in System Settings > "
                        f"Privacy & Security > Camera")
                if cv2.waitKey(int(READ_RETRY_DELAY_S * 1000)) & 0xFF == ord("q"):
                    break
                continue
            fails = 0
            t = time.perf_counter()
            warped, state = localizer.locate(frame)
            ms = (time.perf_counter() - t) * 1000
            colour = (0, 255, 0) if state == "live" else (0, 165, 255)
            if warped is None:
                cv2.putText(frame, "NO DRAWER - closed or out of shot", (20, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame, localizer.hint(frame), (20, 82),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            else:
                cv2.polylines(frame, [np.round(localizer.corners).astype(np.int32)],
                              True, colour, 4)
                cv2.putText(frame, f"{state}  {ms:.0f} ms", (20, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
                cv2.imshow("rectified", warped)
            cv2.imshow("preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(source=None):
    cfg = load_config()
    drawers = cfg["drawers"]
    localizer = Localizer(cfg)

    db = open_db()
    cap = None
    try:
        src = source if source is not None else cfg.get("camera_index", 0)
        cap = open_camera(src)
        seed_all(db, drawers, cfg.get("hold_s", HOLD_S))
        drawer = drawers[0]
        prev_grey = None
        last_motion = 0.0
        M = None           # last registration, warm-starts the next frame
        fails = 0
        show_diffs = False
        motion_area_frac = cfg.get("motion_area_frac", MOTION_AREA_FRAC)
        settle_s = cfg.get("motion_settle_s", MOTION_SETTLE_S)
        grad_weight = cfg.get("grad_weight", GRAD_WEIGHT)
        radius = cfg.get("search_radius", SEARCH_RADIUS)
        unsure_margin = cfg.get("uncertain_margin", UNCERTAIN_MARGIN)
        drift_margin = cfg.get("drift_margin", DRIFT_MARGIN)
        drift_after = cfg.get("drift_after", DRIFT_AFTER)
        drift_alpha = cfg.get("drift_alpha", DRIFT_ALPHA)
        note = ""          # last action, shown briefly in the corner

        while True:
            ok, frame = cap.read()
            if not ok:
                # A USB re-enumeration or hub blip must not kill an unattended
                # tracker: a silent tracker looks exactly like a quiet drawer.
                fails += 1
                if fails >= MAX_READ_FAILS:
                    raise SystemExit(f"camera read failed {fails} times - giving up")
                if fails % REOPEN_AFTER_FAILS == 0:
                    print(f"camera read failing ({fails}) - reopening")
                    cap.release()
                    # A reopen that itself fails is the expected case while the
                    # device is still gone - keep retrying, don't exit here.
                    cap = cv2.VideoCapture(src)
                # waitKey doubles as the retry delay so the window stays alive
                # and 'q' still quits while the camera is down.
                if cv2.waitKey(int(READ_RETRY_DELAY_S * 1000)) & 0xFF == ord("q"):
                    break
                continue
            fails = 0
            now = time.monotonic()

            # --- where is the drawer? ---------------------------------------
            warped, found = localizer.locate(frame)
            if warped is None:
                # Drawer shut, or pushed out of shot. Every slot holds its last
                # state: with no drawer to look at, "all twelve tools taken" is
                # the one reading we can be sure is wrong.
                for d in drawers:
                    for deb in d["debouncers"].values():
                        deb.freeze()
                M = None                # nothing to warm-start from any more
                cv2.putText(frame, "NO DRAWER - closed or out of shot", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("tool tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # --- re-anchor onto the drawer, not the markers ----------------
            # Against the drawer that was active last frame, since identify()
            # has not run yet. On a multi-drawer rig the wrong baseline simply
            # fails to correlate and register() keeps the previous transform.
            warped, M = register(warped, drawer, M)

            # --- motion gate ---------------------------------------------
            grey = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            moved = 0.0
            if prev_grey is not None:
                moved = float((cv2.absdiff(grey, prev_grey) > MOTION_PIXEL_DELTA).mean())
                if moved > motion_area_frac:
                    last_motion = now
            prev_grey = grey
            static = (now - last_motion) > settle_s

            # --- which drawer is this? -----------------------------------
            # Only while the frame is settled: a hand across the foam changes
            # exactly the layout identification reads, and switching drawers
            # mid-reach would evaluate one drawer's ROIs against another's
            # baseline and log a fistful of phantom Taken events.
            if static:
                drawer = identify(warped, drawers, drawer)
            rois, debouncers = drawer["rois"], drawer["debouncers"]
            templates = drawer["templates"]

            # --- per-slot evaluation --------------------------------------
            clean = warped.copy()   # pre-overlay, for the re-baseline snap
            grey, grad = channels(clean)
            unsure_n = 0
            for r in rois:
                d = debouncers[r["id"]]
                tpl = templates[r["id"]]
                # `grey`/`grad` come from `clean`, not `warped`: the overlay boxes
                # of slots already drawn this frame would otherwise be scored as
                # part of a neighbour.
                s_full = slot_score(grey, grad, r["rect"], tpl, "baseline",
                                    grad_weight, radius)
                s_empty = (None if drawer["empty"] is None else
                           slot_score(grey, grad, r["rect"], tpl, "empty",
                                      grad_weight, radius))
                reading = classify(s_full, s_empty, unsure_margin)
                margin = 0.0 if s_empty is None else abs(s_full - s_empty)
                unsure_n += reading is None
                if static:
                    action = d.update(reading, now)
                    if action is not None:
                        log_event(db, r["id"], r["name"], action)
                    # Only a settled slot, well clear of the boundary and holding
                    # the same answer for a while, is allowed to teach a
                    # reference. Anything less and a hand or a knife-edge
                    # reading gets baked in permanently.
                    if (drift_alpha > 0 and d.steady >= drift_after
                            and margin >= drift_margin):
                        which = "empty" if d.missing else "baseline"
                        if drawer[which] is not None:
                            drift(drawer, which, r, clean, drift_alpha)
                else:
                    d.freeze()  # hand in the drawer: hold everything as-is

                x, y, w, h = r["rect"]
                colour = ((0, 200, 255) if reading is None else
                          (0, 0, 255) if d.missing else (0, 255, 0))
                if r.get("polygon"):
                    cv2.polylines(warped, [np.array(r["polygon"], np.int32)],
                                  True, colour, 2)
                else:
                    cv2.rectangle(warped, (x, y), (x + w, y + h), colour, 2)
                label = r["name"] if not show_diffs else (
                    f"{r['name']} full{s_full:+.2f} "
                    f"empty{'  n/a' if s_empty is None else f'{s_empty:+.2f}'} "
                    f"m{margin:.2f}")
                cv2.putText(warped, label, (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

            missing_n = sum(d.missing for d in debouncers.values())
            # The motion number is the one to watch when detection feels slow:
            # if it never drops under the limit, nothing is ever evaluated.
            # "held" means the outline is coasting - a few more frames of that
            # and the drawer counts as gone.
            held = "" if found == "live" else " | HELD"
            status = (f"[{drawer['name']}] MOTION {moved * 100:.2f}% > "
                      f"{motion_area_frac * 100:.2f}% - paused{held}"
                      if not static else
                      f"[{drawer['name']}] static | motion {moved * 100:.2f}% | "
                      f"missing {missing_n}/{len(rois)}" +
                      (f" | UNSURE {unsure_n}" if unsure_n else "") + held)
            cv2.putText(warped, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 165, 255) if not static else (255, 255, 255), 2)
            wh = warped.shape[0]
            cv2.putText(warped, HINT, (10, wh - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1)
            if note:
                cv2.putText(warped, note, (10, wh - 34), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 1)

            cv2.imshow("tool tracker", warped)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                show_diffs = not show_diffs
            elif key == ord("g"):
                # A/B the edge channel against a live drawer: watch the margins
                # in the 'd' overlay and keep whichever separates better.
                grad_weight = 0.0 if grad_weight else cfg.get("grad_weight",
                                                              GRAD_WEIGHT)
                note = f"grad weight {grad_weight:.2f}"
            elif key == ord("b"):
                # Re-baseline is only meaningful on a settled frame; snapping
                # mid-motion bakes a hand into the reference for good.
                if static:
                    cv2.imwrite(str(drawer["baseline_path"]), clean)
                    set_reference(drawer, "baseline", clean)
                    # `clean` is already registered, so the new baseline lands in
                    # the same coordinate frame the ROIs were drawn in. The old
                    # transform is meaningless against it: re-lock from scratch.
                    M = None
                    # ponytail: no state fixup - every ROI now reads ~0 and the
                    # normal debounce path logs Returned for whatever was out,
                    # so the DB agrees with the new reference within HOLD_S.
                    note = f"re-baselined {drawer['name']}"
                else:
                    note = "re-baseline needs a static frame"
            elif key == ord("e"):
                if static:
                    cv2.imwrite(str(drawer["empty_path"]), clean)
                    set_reference(drawer, "empty", clean)
                    note = f"empty reference saved for {drawer['name']}"
                else:
                    note = "empty reference needs a static frame"
            elif key == ord("s"):
                PHOTO_DIR.mkdir(exist_ok=True)
                # drawer in the stamp, not the prefix: tune.py keys off the
                # leading "missing_"/"full_" and would stop parsing the name.
                name = photo_name(
                    [i for i, d in debouncers.items() if d.missing],
                    f"{slugify(drawer['name'])}_{datetime.now():%H%M%S}")
                # raw frame, not the warp: tune.py does its own warp and needs
                # the markers in shot.
                cv2.imwrite(str(PHOTO_DIR / name), frame)
                note = f"saved photos/{name}"
            elif key in (ord("["), ord("]")):
                unsure_margin = max(0.0, unsure_margin +
                                    MARGIN_STEP * (1 if key == ord("]") else -1))
                note = f"uncertain margin {unsure_margin:.2f}"
            elif key in (ord("-"), ord("=")):
                motion_area_frac = max(0.0005, motion_area_frac *
                                       (2.0 if key == ord("=") else 0.5))
                note = f"motion limit {motion_area_frac * 100:.2f}%"
            elif key == ord("w"):
                cfg.update(motion_area_frac=motion_area_frac,
                           motion_settle_s=settle_s,
                           grad_weight=grad_weight,
                           search_radius=radius,
                           uncertain_margin=unsure_margin,
                           drift_margin=drift_margin,
                           drift_after=drift_after,
                           drift_alpha=drift_alpha)
                save_config(cfg)
                note = (f"wrote config.json (margin {unsure_margin:.2f}, "
                        f"grad {grad_weight:.2f}, "
                        f"motion {motion_area_frac * 100:.2f}%)")
    finally:
        # The camera stays claimed by a dead process on macOS otherwise.
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        db.close()


def selftest():
    # --- debounce timing ---
    d = Debouncer(False, hold=2.0)
    assert d.update(True, 0.0) is None            # change starts the timer
    assert d.update(True, 1.9) is None            # not held long enough
    assert d.update(True, 2.0) == TAKEN           # committed
    assert d.update(True, 3.0) is None            # no repeat event
    d.update(False, 4.0)
    d.freeze()                                    # motion wipes the pending change
    assert d.update(False, 4.5) is None
    assert d.update(False, 6.4) is None           # timer restarted at 4.5, not 4.0
    assert d.update(False, 6.5) == RETURNED       # falsy-safe: a real action string
    # flapping reading never commits
    d2 = Debouncer(False, hold=2.0)
    for i in range(20):
        assert d2.update(i % 2 == 0, i * 0.5) is None

    # --- ROI validation: every one of these used to crash or silently truncate ---
    ok = [{"id": "a", "rect": [10, 10, 50, 50]}]
    assert validate_rois(ok, (1000, 700)) is ok
    for bad in ([-5, 10, 50, 50], [10, 10, 0, 0], [980, 10, 100, 50],
                [10, 690, 50, 50], [10, -1, 50, 50]):
        try:
            validate_rois([{"id": "a", "rect": bad}], (1000, 700))
            raise AssertionError(f"rect {bad} should have been rejected")
        except ValueError:
            pass
    try:
        validate_rois([{"id": "dup", "rect": [0, 0, 10, 10]},
                       {"id": "dup", "rect": [0, 0, 10, 10]}], (1000, 700))
        raise AssertionError("duplicate id should have been rejected")
    except ValueError:
        pass
    # a polygon owns its rect: a hand-edited config where the two disagree gets
    # the rect recomputed rather than quietly masking the wrong region
    poly = [{"id": "p", "polygon": [[10, 20], [60, 20], [60, 90], [10, 90]],
             "rect": [0, 0, 5, 5]}]
    assert validate_rois(poly, (1000, 700))[0]["rect"] == [10, 20, 51, 71]
    # ...and one that escapes the frame is still caught
    for bad in ([[10, 20], [1200, 20], [10, 90]], [[10, 20], [60, 20]]):
        try:
            validate_rois([{"id": "p", "polygon": bad}], (1000, 700))
            raise AssertionError(f"polygon {bad} should have been rejected")
        except ValueError:
            pass
    # the dead per-slot threshold is stripped, not carried forward
    legacy_slot = [{"id": "t", "rect": [0, 0, 10, 10], "threshold": 40}]
    assert "threshold" not in validate_rois(legacy_slot, (1000, 700))[0]

    # --- storage: view correctness, plan, and restart seeding ---
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
    log_event(db, "10mm_wrench", "10mm Wrench", TAKEN)
    log_event(db, "10mm_wrench", "10mm Wrench", RETURNED)
    log_event(db, "torque_driver", "Torque Driver", TAKEN)
    assert dict(db.execute("SELECT tool_id, action FROM current_state")) == {
        "10mm_wrench": RETURNED, "torque_driver": TAKEN}
    plan = " ".join(str(r) for r in
                    db.execute("EXPLAIN QUERY PLAN SELECT * FROM current_state"))
    assert "CORRELATED" not in plan, plan

    rois = [{"id": "10mm_wrench"}, {"id": "torque_driver"}, {"id": "never_seen"}]
    seeded = seed_debouncers(db, rois)
    assert seeded["torque_driver"].missing is True    # still out across a restart
    assert seeded["10mm_wrench"].missing is False
    assert seeded["never_seen"].missing is False      # unknown tools start present
    # a restart must not re-log what the DB already knows
    assert seeded["torque_driver"].update(True, 99.0) is None

    # --- missing decision: nearest of two references, never a threshold ---
    assert classify(0.2, 0.9) is True             # correlates with empty: tool out
    assert classify(0.9, 0.2) is False            # correlates with full: still there
    assert classify(0.5, 0.5) is False            # a tie is not evidence of absence
    # both references fit badly - the wrong tool in the slot, say - and the
    # answer still comes from which fits less badly, not from any fixed number
    assert classify(0.10, 0.31) is True
    # no empty reference: refuse to answer rather than fall back on a threshold
    assert classify(0.9, None) is None
    assert classify(0.1, None) is None

    # --- uncertainty: a reading too close to call must not become an event ---
    assert classify(0.30, 0.35, 0.08) is None     # .05 apart, margin needs .08
    assert classify(0.30, 0.45, 0.08) is True     # clear of it again
    assert classify(0.45, 0.30, 0.08) is False
    # None is held, never logged, and it wipes any pending change
    d3 = Debouncer(False, hold=2.0)
    assert d3.update(True, 0.0) is None
    assert d3.update(None, 1.0) is None
    assert d3.update(True, 2.5) is None           # timer restarted at 2.5
    assert d3.update(True, 4.5) == TAKEN

    # --- drift confidence counter ---
    d4 = Debouncer(False, hold=2.0)
    for i in range(5):
        d4.update(False, float(i))
    assert d4.steady == 5
    d4.freeze()                                   # motion: confidence is gone
    assert d4.steady == 0
    d4.update(True, 10.0)                         # a pending change is not steady
    assert d4.steady == 0

    # --- corner ordering: the mistake that rotates the whole frame ---
    box = [[10, 8], [90, 12], [88, 60], [12, 56]]     # TL, TR, BR, BL
    for roll in range(4):
        got = order_corners(box[roll:] + box[:roll])
        assert (got == np.array(box, np.float32)).all(), (roll, got)
    # a rotated rect from OpenCV comes back in whatever order it likes
    assert (order_corners(cv2.boxPoints(((50, 30), (80, 40), 0.0)))
            == np.array([[10, 10], [90, 10], [90, 50], [10, 50]], np.float32)).all()

    # --- perimeter support: the test that makes a hypothesised rectangle real ---
    edge = np.array([0, 0, 100, 0], np.float32)
    A, B = np.array([0.0, 0.0]), np.array([100.0, 0.0])

    def support(*segs):
        prep = prepare_segments(np.array(segs, np.float32).reshape(-1, 4))
        return side_support(A, B, line_coverage(edge, prep, 12))

    assert abs(support([0, 0, 100, 0]) - 1.0) < 0.02
    assert abs(support([0, 0, 50, 0]) - 0.5) < 0.02
    # overlapping fragments are unioned, not summed - summing would let one
    # heavily-fragmented edge score above 1.0 and outrank a genuinely complete one
    assert abs(support([0, 0, 60, 0], [40, 0, 100, 0]) - 1.0) < 0.02
    # ...and a gap in the middle is honestly reported as a gap
    assert abs(support([0, 0, 30, 0], [70, 0, 100, 0]) - 0.6) < 0.02
    # a segment running alongside but too far out supports nothing
    assert support([0, 40, 100, 40]) == 0.0
    # ...nor does one crossing it at an angle
    assert support([0, -50, 100, 50]) == 0.0
    # a segment beyond the edge's own extent does not count toward it
    assert support([200, 0, 300, 0]) == 0.0
    empty = prepare_segments(np.zeros((0, 4), np.float32))
    assert side_support(A, B, line_coverage(edge, empty, 12)) == 0.0

    # a wide-open aspect tolerance must widen the search, never divide by zero
    for tol in (0.0, 0.3, 1.0, 5.0):
        wide = Localizer({"warp_size": [1000, 700], "detect": {"aspect_tol": tol}})
        assert wide.locate(np.zeros((400, 600, 3), np.uint8))[1] == "closed"

    # --- localization: find a drawer-shaped thing, and notice when it goes ---
    scene_frame = np.full((400, 600, 3), 30, np.uint8)
    cv2.rectangle(scene_frame, (120, 80), (460, 318), (190, 185, 175), -1)
    loc = Localizer({"warp_size": [1000, 700], "detect": {"grace_frames": 2}})
    got, state = loc.locate(scene_frame)
    assert state == "live" and got.shape[:2] == (700, 1000), state
    # the drawer's own corners, within a couple of px of where they were drawn
    assert np.abs(loc.corners - [[120, 80], [460, 80], [460, 318], [120, 318]]).max() < 3
    # a blank frame is a shut drawer: coast briefly, then stop warping entirely
    blank = np.full((400, 600, 3), 30, np.uint8)
    assert loc.locate(blank)[1] == "held"
    assert loc.locate(blank)[1] == "held"
    assert loc.locate(blank) == (None, "closed")
    # ...and re-acquires from scratch when it comes back
    assert loc.locate(scene_frame)[1] == "live"
    # a detection that teleports is refused, not followed
    moved_frame = np.full((400, 600, 3), 30, np.uint8)
    cv2.rectangle(moved_frame, (10, 80), (350, 318), (190, 185, 175), -1)
    assert loc.locate(moved_frame)[1] == "held"
    # but a drawer that really did move is picked up once the grace runs out
    assert loc.locate(moved_frame)[1] == "held"
    assert loc.locate(moved_frame) == (None, "closed")
    assert loc.locate(moved_frame)[1] == "live"
    # foam colour, when there is a colour to use, finds the same drawer
    foam = np.full((400, 600, 3), 30, np.uint8)
    cv2.rectangle(foam, (120, 80), (460, 318), (200, 40, 40), -1)
    fl = Localizer({"warp_size": [1000, 700], "detect": {"hsv": [[100, 80, 80],
                                                                [140, 255, 255]]}})
    assert fl.locate(foam)[1] == "live"
    assert np.abs(fl.corners - [[120, 80], [460, 80], [460, 318], [120, 318]]).max() < 3

    # --- scoring: correlation against two references, and what it shrugs off ---
    def pocket(tool=True, bright=0):
        """A slot: pale foam, a dark pocket, and maybe a tool lying in it."""
        img = np.full((60, 60, 3), 170, np.uint8)
        cv2.rectangle(img, (10, 10), (50, 50), (60, 60, 60), -1)
        if tool:
            cv2.rectangle(img, (16, 20), (44, 40), (210, 205, 195), -1)
            cv2.circle(img, (30, 30), 6, (90, 90, 90), -1)
        # keywords, not positions: convertScaleAbs takes `dst` second, and
        # passing alpha there silently scales the whole image to zero
        return cv2.convertScaleAbs(cv2.GaussianBlur(img, (3, 3), 0),
                                   alpha=1.0, beta=bright)

    full_ref, empty_ref = pocket(True), pocket(False)
    slot = {"id": "s", "rect": [5, 5, 50, 50]}
    dd = {"rois": [slot], "baseline": full_ref, "empty": empty_ref}
    build_templates(dd)
    tpl = dd["templates"]["s"]

    def read(img, w=GRAD_WEIGHT):
        g, e = channels(img)
        return (slot_score(g, e, slot["rect"], tpl, "baseline", w),
                slot_score(g, e, slot["rect"], tpl, "empty", w))

    assert classify(*read(pocket(True))) is False        # tool there
    assert classify(*read(pocket(False))) is True        # tool gone
    # a lamp arriving changes every pixel; normalised correlation barely notices,
    # which is the entire reason this is not a pixel difference
    lit_full, lit_empty = read(pocket(True, bright=60))
    assert lit_full > 0.9, lit_full
    assert classify(lit_full, lit_empty, UNCERTAIN_MARGIN) is False
    dim_full, dim_empty = read(pocket(False, bright=-40))
    assert classify(dim_full, dim_empty, UNCERTAIN_MARGIN) is True
    # a couple of px of residual misalignment is absorbed by the search window
    shifted = np.roll(pocket(True), (2, -2), axis=(0, 1))
    assert classify(*read(shifted), UNCERTAIN_MARGIN) is False
    # the edge channel carries the silhouette on its own
    assert classify(*read(pocket(True), 1.0), UNCERTAIN_MARGIN) is False
    assert classify(*read(pocket(False), 1.0), UNCERTAIN_MARGIN) is True

    # --- the polygon mask keeps a neighbour out of the score ---
    diag = {"id": "d", "polygon": [[8, 8], [50, 8], [50, 50], [8, 50]],
            "rect": [8, 8, 43, 43]}
    m = slot_mask(diag)
    assert m.shape == (43, 43) and m.all()              # a rect polygon masks nothing
    tri = {"id": "t", "polygon": [[0, 0], [40, 0], [0, 40]], "rect": [0, 0, 41, 41]}
    tm = slot_mask(tri)
    assert tm[2, 2] == 255 and tm[38, 38] == 0         # inside kept, outside cut
    # a masked template ignores what happens outside the polygon: paint the
    # excluded half of the bbox and the score must not move
    tri_ref = pocket(True)
    dd2 = {"rois": [tri], "baseline": tri_ref, "empty": pocket(False)}
    build_templates(dd2)
    noisy = tri_ref.copy()
    cv2.rectangle(noisy, (30, 30), (41, 41), (0, 0, 255), -1)   # outside the triangle
    unmasked = {k: v for k, v in tri.items() if k != "polygon"}
    dd3 = {"rois": [unmasked], "baseline": tri_ref, "empty": None}
    build_templates(dd3)
    g2, e2 = channels(noisy)
    clean = channels(tri_ref)

    def drop(templates):
        """How much the out-of-polygon vandalism costs this slot's score."""
        t = templates["t"]
        return (slot_score(*clean, tri["rect"], t, "baseline") -
                slot_score(g2, e2, tri["rect"], t, "baseline"))

    assert drop(dd2["templates"]) < drop(dd3["templates"]) / 2, (
        drop(dd2["templates"]), drop(dd3["templates"]))

    # --- registration: a marker that moved must not read as tools missing ---
    rng = np.random.default_rng(0)
    # a stand-in drawer: pockets on foam. Blurred noise will not do - it washes
    # out to nothing at REGISTER_SCALE and ECC has no structure to lock onto,
    # which is also the real-world warning that flat foam registers badly.
    scene = np.full((280, 400, 3), 40, np.uint8)
    for cx in range(60, 400, 80):
        for cy in range(50, 280, 70):
            cv2.rectangle(scene, (cx - 24, cy - 18), (cx + 24, cy + 18),
                          (200, 190, 170), -1)
            cv2.circle(scene, (cx, cy), 9, (60, 90, 200), -1)
    scene = cv2.GaussianBlur(scene, (5, 5), 0)
    tool = [150, 100, 60, 60]
    # what a nudged corner does: shift the whole frame and skew it slightly
    moved = cv2.warpPerspective(
        scene, np.array([[1.0, 0.0, 9.0], [0.0, 1.0, 6.0], [1e-5, 0, 1.0]]),
        scene.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)
    drawer = {"rois": [{"id": "tool", "rect": tool}]}
    set_reference(drawer, "baseline", scene)
    set_reference(drawer, "empty", None)

    def corr(img):
        g, e = channels(img)
        return slot_score(g, e, tool, drawer["templates"]["tool"], "baseline")

    before = corr(moved)
    fixed, M = register(moved, drawer)
    assert M is not None
    after = corr(fixed)
    # the misalignment alone was reading like a missing tool; registration
    # has to take out nearly all of it, not merely improve on it. 9px of shift
    # is well past what the +/-5px search window can absorb on its own.
    assert before < 0.7, before
    assert after > 0.95, (before, after)
    # a frame with nothing in common keeps the caller's transform rather than
    # inventing one - a bad lock looks exactly like every tool vanishing at once
    noise = cv2.GaussianBlur(rng.integers(0, 255, scene.shape, dtype=np.uint8), (9, 9), 0)
    _, kept = register(noise, drawer, M)
    assert kept is M
    # ...and with no transform to fall back on, the frame passes through untouched
    passthrough, none_m = register(noise, drawer)
    assert none_m is None and passthrough is noise
    # a drawer with no baseline yet must not crash the loop
    assert register(moved, {})[0] is moved

    # --- drift: a reference nudges toward the frame, and only in that slot ---
    dark, bright = pocket(True), pocket(True, bright=40)
    small = {"id": "s", "rect": [0, 0, 20, 20]}
    dd4 = {"rois": [small], "baseline": dark.copy(), "empty": None}
    build_templates(dd4)
    tpl_before = dd4["templates"]["s"]["baseline_grey"].copy()
    outside = dd4["baseline"][40, 40].copy()
    drift(dd4, "baseline", small, bright, 0.5)
    assert abs(int(dd4["baseline"][2, 2, 0]) - (int(dark[2, 2, 0]) + 20)) <= 1
    assert (dd4["baseline"][40, 40] == outside).all()         # outside the rect
    # the template moved with the image - a nudge that updated only the picture
    # would keep scoring against the reference it no longer is
    assert not np.allclose(dd4["templates"]["s"]["baseline_grey"], tpl_before)
    for _ in range(20):
        drift(dd4, "baseline", small, bright, 0.5)
    assert abs(int(dd4["baseline"][2, 2, 0]) -
               int(bright[2, 2, 0])) <= 1                     # converges, no overshoot

    # --- drawer config normalisation ---
    legacy = {"rois": [{"id": "a", "rect": [0, 0, 10, 10]}]}
    d0, = normalize_drawers(legacy)
    assert d0["name"] == "main" and d0["rois"] is legacy["rois"]
    assert d0["baseline_file"] == BASELINE_PATH.name    # legacy files still used
    multi = {"drawers": [{"name": "Top Drawer", "rois": [{"id": "a"}]},
                         {"name": "bottom", "rois": [{"id": "b"}]}]}
    top, bottom = normalize_drawers(multi)
    assert top["baseline_file"] == "baseline_top_drawer.png"
    assert bottom["empty_file"] == "empty_bottom.png"
    for bad in ({"drawers": [{"name": "x", "rois": [{"id": "a"}]},
                             {"name": "y", "rois": [{"id": "a"}]}]},   # shared id
                {"drawers": [{"name": "x", "rois": [{"id": "a"}]},
                             {"name": "x", "rois": [{"id": "b"}]}]},   # same name
                {"drawers": [{"name": "x", "rois": []}]},              # no ROIs
                {"drawers": [{"rois": [{"id": "a"}]}]},                # no name
                {}):                                                   # nothing
        try:
            normalize_drawers(bad)
            raise AssertionError(f"{bad} should have been rejected")
        except ValueError:
            pass

    # --- identification: which drawer is the camera looking at ---
    def fake(name, fill, empty_fill=None):
        d = {"name": name, "rois": [{"id": name, "rect": [0, 0, 20, 20]}]}
        set_reference(d, "baseline", np.full((70, 100, 3), fill, np.uint8))
        set_reference(d, "empty", None if empty_fill is None else
                      np.full((70, 100, 3), empty_fill, np.uint8))
        return d

    a, b = fake("a", 10, empty_fill=200), fake("b", 100)
    assert identify(np.full((70, 100, 3), 12, np.uint8), [a, b]) is a
    assert identify(np.full((70, 100, 3), 98, np.uint8), [a, b]) is b
    # drawer a emptied out looks like its empty reference, not like drawer b
    assert identify(np.full((70, 100, 3), 195, np.uint8), [a, b]) is a
    # a frame between the two does not yank the active drawer away
    mid = np.full((70, 100, 3), 60, np.uint8)
    assert identify(mid, [a, b], active=b) is b
    assert identify(mid, [a, b], active=a) is a
    # ...but a decisive frame does
    assert identify(np.full((70, 100, 3), 100, np.uint8), [a, b], active=a) is b
    # one drawer configured: always that drawer, references or not
    assert identify(mid, [a]) is a

    # --- saving must drop the loaded images, not choke on them ---
    out = Path(HERE / "_selftest_cfg.json")
    try:
        save_config({"warp_size": [1000, 700], "rois": legacy["rois"],
                     "drawers": [dict(a, debouncers={})]}, out)
        written = json.loads(out.read_text())
        assert "rois" not in written                    # legacy copy dropped
        assert set(written["drawers"][0]) == {"name", "rois"}
    finally:
        out.unlink(missing_ok=True)

    # --- saved photo names must round-trip through tune.py's parser ---
    from tune import labels
    known = {"10mm_wrench", "pliers"}
    for missing in ([], ["pliers"], ["pliers", "10mm_wrench"]):
        # the live stamp carries the drawer slug, which must not be read as a tool
        for stamp in ("120001", "top_drawer_120001"):
            stem = Path(photo_name(missing, stamp)).stem
            absent, bad = labels(stem, known)
            assert absent == set(missing), (stem, absent)
            assert not bad, (stem, bad)

    print("selftest ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--image", help="run against a photo instead of the camera")
    p.add_argument("--camera", type=int, metavar="N",
                   help="use camera N for this run, ignoring config.json - the "
                        "index is not stable on macOS, so trying one is faster "
                        "than editing config to find out")
    p.add_argument("--preview", action="store_true",
                   help="localization only - no scoring, no config to rebuild")
    p.add_argument("--tools", action="store_true",
                   help="live drawer localization + per-slot tool detection")
    p.add_argument("--detect", action="store_true",
                   help="live drawer localization + raw tool outlines, using no "
                        "slots or references at all - for a drawer nobody has "
                        "calibrated")
    p.add_argument("--aspect-tol", type=float, default=0.6,
                   help="with --detect, how far the drawer may differ from "
                        "warp_size's proportions (default 0.6, wide on purpose)")
    p.add_argument("--size", type=int, nargs=2, metavar=("W", "H"),
                   help="with --detect, the canonical size to rectify to")
    p.add_argument("--localizer", choices=("auto", "aruco"),
                   help="override config.json for this run (preview only)")
    a = p.parse_args()
    source = a.image if a.image else (str(a.camera) if a.camera is not None else None)
    if a.selftest:
        selftest()
    elif a.detect:
        preview_detect(source, a.localizer, a.aspect_tol, a.size)
    elif a.tools:
        preview_tools(source, a.localizer)
    elif a.preview:
        preview(source, a.localizer)
    else:
        main(source)
