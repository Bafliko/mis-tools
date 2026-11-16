"""Measure how well the detector separates present from absent, on real photos.

  .venv/bin/python tune.py photos/            # report only
  .venv/bin/python tune.py photos/ --slots    # ...plus a per-slot diagnosis
  .venv/bin/python tune.py photos/ --write    # also update config.json

There is no per-slot threshold to tune any more: a slot is classified by which of
the two references it correlates with better, which needs no number at all. What
still needs choosing is how much of the score comes from edges rather than grey
("grad_weight"), and how big a gap counts as too close to call
("uncertain_margin"). Both are global, and this reads them off real photos.

For every slot it collects the margin - full-reference score minus
empty-reference score - across the photos where the tool was present and the ones
where it was absent. A healthy slot shows two clusters either side of zero. The
worst gap over every slot is what `uncertain_margin` has to sit inside; a slot
whose clusters overlap is reported as OVERLAP and no margin can save it, so
re-shoot its references or redraw its polygon.

Label by filename:
    full_*.jpg                     every tool present
    missing_10mm_wrench_*.jpg      that tool absent, the rest present
    missing_10mm_wrench+pliers.jpg both absent

Each photo must show the drawer from roughly the camera position the references
were shot from.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from tracker import (GRAD_WEIGHT, SEARCH_RADIUS, Localizer, channels, identify,
                     load_config, register, save_config, slot_score)

WEIGHTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]   # grad_weight values the sweep tries


def labels(stem, known):
    """Tool ids marked absent by a filename, plus any chunk that matched nothing.

    'missing_10mm_wrench+pliers' -> {'10mm_wrench', 'pliers'}; 'full_3' -> set().

    Chunks are matched against `known` longest-first rather than taken whole:
    tool ids contain underscores, so 'missing_10mm_wrench_2.jpg' is otherwise
    read as a tool called '10mm_wrench_2' and every photo silently lands in the
    present bucket.
    """
    if not stem.startswith("missing_"):
        return set(), set()
    absent, bad = set(), set()
    for chunk in stem[len("missing_"):].split("+"):
        match = next((k for k in sorted(known, key=len, reverse=True)
                      if chunk == k or chunk.startswith(k + "_")), None)
        (absent if match else bad).add(match or chunk)
    return absent, bad


def gap(seen, weight):
    """Worst separation over every slot at this weight, and how many overlap.

    The minimum is what matters, not the average: one slot whose clusters touch
    is one tool the tracker cannot report on, however cleanly the other eleven
    separate.
    """
    worst, overlaps, measured = float("inf"), 0, 0
    for buckets in seen.values():
        pres, abst = buckets[weight]["present"], buckets[weight]["absent"]
        if not pres or not abst:
            continue
        measured += 1
        # present margins should be positive (nearer the full reference), absent
        # ones negative. The gap is between the worst of each.
        g = min(pres) - max(abst)
        overlaps += g <= 0
        worst = min(worst, g)
    return worst, overlaps, measured


def reference_shift(drawer, slot, pad=30):
    """How far this slot's empty reference sits from its baseline, in pixels.

    The whole-drawer registration fits one homography, which a slot out at a
    corner does not obey exactly - lens distortion and slight non-planarity leave
    residue there that a centred slot never sees. Where that residue lands on
    static features, the "empty" template stops describing an empty pocket and
    starts describing a misalignment, and the slot can never score as absent.

    This is the measurement that identified exactly that failure on slot 301, so
    it is worth reporting for every slot rather than rediscovering by hand.
    """
    base, empty = drawer.get("baseline"), drawer.get("empty")
    if base is None or empty is None:
        return None
    H, W = base.shape[:2]
    x, y, w, h = slot["rect"]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    a = cv2.cvtColor(base[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(empty[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    M = np.eye(2, 3, dtype=np.float32)
    try:
        cc, M = cv2.findTransformECC(
            a, b, M, cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-6), None, 5)
    except cv2.error:
        return None
    if cc < 0.3:
        return None
    return float(np.hypot(M[0, 2], M[1, 2]))


def diagnose_slot(present, absent, margin):
    """(verdict, advice) for one slot, from its signed margins.

    The distinction that matters is between a slot that is merely *close* and one
    that is *wrong*. Both show up as UNSURE in the tracker, but only the first is
    fixed by tuning: a slot whose absent readings never go negative disagrees with
    its own references, and no threshold anywhere can rescue it.
    """
    if not present or not absent:
        return "NO DATA", "shoot a photo of the other state"
    # present should score positive (nearer the full reference), absent negative
    if min(present) <= 0:
        return "BROKEN", "reads absent while present - check the label on those photos"
    if max(absent) >= 0:
        return ("BROKEN",
                "reads present even when absent - its two references disagree, "
                "re-shoot them back to back")
    gap = min(present) - max(absent)
    closest = min(min(present), abs(max(absent)))
    if closest < margin:
        return ("BORDERLINE",
                f"separates, but lands inside the uncertain band "
                f"(closest {closest:.3f} < {margin:.3f}) so it reports UNSURE")
    if gap < 2 * margin:
        return "TIGHT", f"gap {gap:+.3f} leaves little room for drift"
    return "OK", ""


VERDICT_COLOUR = {"OK": (0, 200, 0), "TIGHT": (0, 200, 255),
                  "BORDERLINE": (0, 165, 255), "BROKEN": (0, 0, 255),
                  "NO DATA": (150, 150, 150)}


def render_report(drawer, verdicts, weight, margin, path, base=None):
    """Draw the diagnosis onto the drawer it is about.

    A table of numbers tells you slot 301 is broken; a picture tells you *where*
     301 is and that it sits in the corner - which is what made its misalignment
    obvious in the first place. The two together are what a person actually needs.
    """
    base = drawer.get("baseline") if base is None else base
    if base is None:
        return None
    view = base.copy()
    for r in drawer["rois"]:
        v = verdicts.get(r["id"])
        if v is None:
            continue
        verdict, _, pres, abst, n_unsure, shift = v
        colour = VERDICT_COLOUR.get(verdict, (200, 200, 200))
        thick = 2 if verdict == "OK" else 4
        if r.get("polygon"):
            cv2.polylines(view, [np.array(r["polygon"], np.int32)], True, colour, thick)
            x, y = np.array(r["polygon"]).min(axis=0)
        else:
            x, y, w, h = r["rect"]
            cv2.rectangle(view, (x, y), (x + w, y + h), colour, thick)
        label = r["id"] if verdict == "OK" else f"{r['id']} {verdict}"
        cv2.putText(view, label, (int(x) + 3, int(y) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    # A legend panel under the drawer, so the picture is readable on its own.
    rows = [r for r in drawer["rois"] if r["id"] in verdicts]
    pad, line = 14, 22
    panel = np.full((pad * 3 + line * (len(rows) + 2), view.shape[1], 3), 30, np.uint8)
    cv2.putText(panel, f"grad_weight {weight:.1f}   uncertain_margin {margin:.3f}"
                       f"   ({sum(v[4] for v in verdicts.values())} UNSURE readings)",
                (pad, pad + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel, f"{'tool':<10}{'verdict':<13}{'present':>15}{'absent':>15}"
                       f"{'unsure':>9}{'ref shift':>11}",
                (pad, pad + 16 + line), cv2.FONT_HERSHEY_DUPLEX, 0.45, (170, 170, 170), 1)
    for i, r in enumerate(rows):
        verdict, _, pres, abst, n_unsure, shift = verdicts[r["id"]]
        txt = (f"{r['id']:<10}{verdict:<13}"
               f"{f'{min(pres):+.2f}..{max(pres):+.2f}' if pres else '-':>15}"
               f"{f'{min(abst):+.2f}..{max(abst):+.2f}' if abst else '-':>15}"
               f"{n_unsure:>9}{('-' if shift is None else f'{shift:.1f}px'):>11}")
        cv2.putText(panel, txt, (pad, pad + 16 + line * (i + 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45,
                    VERDICT_COLOUR.get(verdict, (200, 200, 200)), 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([view, panel]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def slot_report(drawers, seen, weight, margin, save=None, base=None):
    """Per-slot verdict table: which slots produce the UNSURE readings, and why."""
    print(f"\n--- per-slot diagnosis at grad_weight {weight:.1f}, "
          f"uncertain_margin {margin:.3f} ---")
    print(f"{'tool':<10}{'verdict':<12}{'present':>16}{'absent':>16}"
          f"{'unsure':>8}{'ref shift':>11}")
    problems, verdicts = [], {}
    for d in drawers:
        for r in d["rois"]:
            buckets = seen[r["id"]][weight]
            pres, abst = buckets["present"], buckets["absent"]
            verdict, advice = diagnose_slot(pres, abst, margin)
            n_unsure = sum(abs(m) < margin for m in pres + abst)
            shift = reference_shift(d, r)
            verdicts[r["id"]] = (verdict, advice, pres, abst, n_unsure, shift)
            pres_s = f"{min(pres):+.2f}..{max(pres):+.2f}" if pres else "-"
            abst_s = f"{min(abst):+.2f}..{max(abst):+.2f}" if abst else "-"
            shift_s = "-" if shift is None else f"{shift:.1f}px"
            print(f"{r['id']:<10}{verdict:<12}{pres_s:>16}{abst_s:>16}"
                  f"{n_unsure:>8}{shift_s:>11}")
            if advice:
                problems.append((r["id"], verdict, advice))
    if problems:
        print("\nwhat to do:")
        for tool_id, verdict, advice in problems:
            print(f"  {tool_id:<10}{verdict:<12}{advice}")
    else:
        print("\nevery slot separates cleanly and clear of the uncertain band.")
    if save:
        # Only the drawer the photos are actually of; a multi-drawer rig would
        # need one image each and the caller can ask again.
        out = render_report(drawers[0], verdicts, weight, margin, save, base)
        if out:
            full = Path(out).resolve()
            print(f"\nwrote {full}  ({full.stat().st_size // 1024} KB)")
    big = [p for p in problems if p[1] == "BROKEN"]
    if big:
        print(f"\n{len(big)} slot(s) disagree with their own references. No amount "
              f"of tuning fixes that:\n  re-shoot the drawer full ('b') and bare "
              f"('e') back to back, without moving the drawer or camera between.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder of labelled photos")
    ap.add_argument("--write", action="store_true",
                    help="update grad_weight and uncertain_margin in config.json")
    ap.add_argument("--on", metavar="PHOTO",
                    help="render the slot report onto this photo instead of the "
                         "baseline reference - useful when the baseline is not "
                         "the clearest shot of the drawer")
    ap.add_argument("--slots", action="store_true",
                    help="per-slot diagnosis: which slots cause the UNSURE "
                         "readings, and whether tuning can fix them")
    ap.add_argument("--safety", type=float, default=0.5,
                    help="fraction of the worst gap to claim as the uncertain "
                         "band (0 = trust the photos exactly, 1 = refuse to "
                         "answer anywhere near the worst case seen)")
    args = ap.parse_args()

    # Everything printed from here also lands in the report file, so the numbers
    # behind a decision survive the terminal being closed.
    cfg = load_config()
    drawers = cfg["drawers"]
    localizer = Localizer(cfg)
    radius = cfg.get("search_radius", SEARCH_RADIUS)

    photos = sorted(p for p in Path(args.folder).iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not photos:
        raise SystemExit(f"no photos in {args.folder}")

    # slot id -> weight -> {"present": [margins], "absent": [margins]}; ids are
    # unique across drawers, so one flat table covers the lot.
    rois = [r for d in drawers for r in d["rois"]]
    seen = {r["id"]: {w: {"present": [], "absent": []} for w in WEIGHTS}
            for r in rois}
    unknown, skipped = set(), 0

    for p in photos:
        frame = cv2.imread(str(p))
        if frame is None:
            print(f"  {p.name}: unreadable, skipped")
            skipped += 1
            continue
        # Cold every photo: they are unordered, so there is no previous outline
        # to be consistent with and the jump guard would reject half of them.
        localizer.corners, localizer.misses = None, 0
        warped, _ = localizer.locate(frame)
        if warped is None:
            # Silently averaging a photo with no drawer in it would poison every
            # reading in the table.
            print(f"  {p.name}: no drawer found, skipped")
            skipped += 1
            continue
        absent, bad = labels(p.stem, seen.keys())
        unknown |= bad
        # The photo shows one drawer; measuring the others against it would dump
        # a wall of meaningless scores into their buckets.
        drawer = identify(warped, drawers)
        if drawer["empty"] is None:
            raise SystemExit(f"drawer {drawer['name']!r} has no empty reference - "
                             f"there is nothing to measure a margin against")
        # The same ECC re-anchoring the tracker runs. Photos are shot over days
        # and the drawer sits differently each time; without this the slots land
        # off the tools and every reading measures misalignment, not occupancy.
        warped, _ = register(warped, drawer)
        grey, grad = channels(warped)
        for r in drawer["rois"]:
            bucket = "absent" if r["id"] in absent else "present"
            tpl = drawer["templates"][r["id"]]
            for w in WEIGHTS:
                margin = (slot_score(grey, grad, r["rect"], tpl, "baseline", w, radius)
                          - slot_score(grey, grad, r["rect"], tpl, "empty", w, radius))
                seen[r["id"]][w][bucket].append(margin)

    if unknown:
        print(f"warning: filenames name unknown tool ids {sorted(unknown)}")
    print(f"\n{len(photos) - skipped} photos used, {skipped} skipped")

    # --- which mix of grey and edges separates this drawer best? ---
    print(f"\n{'grad_weight':>12}{'worst gap':>12}{'overlapping':>13}{'measured':>10}")
    best_w, best_gap = None, -float("inf")
    for w in WEIGHTS:
        worst, overlaps, measured = gap(seen, w)
        shown = "-" if measured == 0 else f"{worst:+.3f}"
        print(f"{w:>12.1f}{shown:>12}{overlaps:>13}{measured:>10}")
        if measured and worst > best_gap:
            best_w, best_gap = w, worst
    if best_w is None:
        raise SystemExit("\nno slot has photos of both states - shoot a 'full_' "
                         "photo and a 'missing_<id>_' photo for each tool")

    # --- per-slot detail at the winning weight ---
    print(f"\nat grad_weight {best_w:.1f}   (margin = full score - empty score; "
          f"present should be positive)")
    print(f"{'tool':<22}{'present':>20}{'absent':>20}{'gap':>10}")
    for r in rois:
        pres = seen[r["id"]][best_w]["present"]
        abst = seen[r["id"]][best_w]["absent"]
        pres_s = f"{min(pres):+.3f}..{max(pres):+.3f}" if pres else "-"
        abst_s = f"{min(abst):+.3f}..{max(abst):+.3f}" if abst else "-"
        if not pres or not abst:
            print(f"{r['id']:<22}{pres_s:>20}{abst_s:>20}{'need both':>10}")
            continue
        g = min(pres) - max(abst)
        print(f"{r['id']:<22}{pres_s:>20}{abst_s:>20}"
              f"{('OVERLAP' if g <= 0 else f'{g:+.3f}'):>10}")

    if args.slots:
        base = None
        if args.on:
            frame = cv2.imread(args.on)
            if frame is None:
                raise SystemExit(f"cannot read {args.on}")
            localizer.corners, localizer.misses = None, 0
            base, _ = localizer.locate(frame)
            if base is None:
                raise SystemExit(f"no drawer found in {args.on}")
            # Into the same canonical frame the slots are defined in, or the
            # boxes would be drawn over the wrong part of the drawer.
            base, _ = register(base, drawers[0])
        # Name the output after its source. Overwriting one fixed filename every
        # run means an image viewer shows a cached copy of the previous report and
        # the new numbers are invisible.
        stem = Path(args.on).stem if args.on else "baseline"
        slot_report(drawers, seen, best_w, cfg.get("uncertain_margin", 0.073),
                    save=f"diagnostics/slot_report_{stem}.jpg", base=base)

    margin = round(max(0.0, best_gap) * args.safety, 3)
    print(f"\nworst gap {best_gap:+.3f} over every slot -> "
          f"uncertain_margin {margin:.3f} at safety {args.safety}")
    if best_gap <= 0:
        print("  OVERLAP: at least one slot cannot be told apart at any margin. "
              "Re-shoot its references, or redraw its polygon so it stops "
              "including a neighbour.")

    if args.write and best_gap > 0:
        cfg["grad_weight"], cfg["uncertain_margin"] = best_w, margin
        save_config(cfg)
        print(f"\nwrote config.json (grad_weight {best_w:.1f}, "
              f"uncertain_margin {margin:.3f})")
    elif args.write:
        print("\nnothing written - fix the overlap first")
    else:
        print("\n(report only - pass --write to update config.json)")




if __name__ == "__main__":
    main()
