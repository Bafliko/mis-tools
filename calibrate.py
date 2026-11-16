"""Calibration helper: define the tool slots on the rectified drawer view.

  python calibrate.py                      # calibrate the drawer named "main"
  python calibrate.py --drawer top         # add/replace the drawer "top"
  python calibrate.py --drawer top shot.jpg   # ...from a photo instead of live

One drawer per run. Saving adds that drawer to config.json and leaves the other
drawers - and their references - alone.

Point the camera at a FULL drawer, then outline each tool:
  drag             a rectangle, for a tool that fills its box
  click, click...  a polygon, one vertex per click - for anything diagonal,
                   which is most of them. `c` closes it and asks for the name.
  x  discard the polygon in progress
  u  undo last slot
  s  save config.json + the baseline (baseline = current rectified frame)
  q  quit without saving

Prefer polygons. A bounding box around a wrench lying at 40 degrees contains a
corner of each neighbouring slot, and the tracker then partly scores the tool
next door - so removing that neighbour reads as this tool moving too.

Localization follows config.json ("localizer": "auto" finds the drawer outline
itself; "aruco" uses the printed markers), so the frame here is the same one the
tracker will see.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import tracker
from tracker import (CONFIG_PATH, DETECT, HERE, Localizer, normalize_drawers,
                     open_camera, save_config, slugify, validate_rois)

CAMERA_INDEX = 0   # iPhone via Continuity Camera; 1 is the built-in MacBook cam
MARKER_IDS = [0, 1, 2, 3]   # TL, TR, BR, BL, only read in "aruco" mode
WARP_SIZE = [1000, 700]     # output pixels; keep the drawer's aspect ratio
CLICK_SLOP = 5              # px of movement below which a drag is really a click

drag = {"start": None, "cur": None, "rect": None}
poly = {"points": [], "close": False}


def clamp(p):
    """Keep a point inside the rectified frame.

    Backends report coordinates outside the window while a drag is in progress,
    so an unclamped rect can carry a negative origin - which slices to an empty
    array and crashes the tracker's main loop.
    """
    x, y = p
    return max(0, min(x, WARP_SIZE[0])), max(0, min(y, WARP_SIZE[1]))


def base_config():
    """The config keys calibrate.py is responsible for seeding."""
    return {"camera_index": CAMERA_INDEX, "localizer": "auto",
            "marker_ids": MARKER_IDS, "warp_size": WARP_SIZE, "detect": dict(DETECT)}


def save(rois, warped, drawer_name):
    """Merge this drawer into config.json and write its baseline.

    Read-modify-write rather than overwrite: the other drawers' slots live in the
    same file, and re-calibrating one drawer must not delete the rest of the
    cabinet. Tool ids are derived from names, never from drag order - a positional
    id silently reassigns T01 to a different physical tool on the next run,
    corrupting every past event in the database.
    """
    cfg = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    for k, v in base_config().items():
        cfg.setdefault(k, v)
    existing = normalize_drawers(cfg) if (cfg.get("drawers") or cfg.get("rois")) else []
    if cfg["warp_size"] != WARP_SIZE:
        raise ValueError(f"config warp_size is {cfg['warp_size']} but this run used "
                         f"{WARP_SIZE} - every drawer must share one canonical size")

    slug = slugify(drawer_name)
    prior = next((d for d in existing if d["name"] == drawer_name), None)
    # Keep the old filenames when replacing a drawer, so a legacy config that
    # points "main" at baseline.png keeps pointing there.
    drawer = {"name": drawer_name,
              "baseline_file": (prior or {}).get("baseline_file", f"baseline_{slug}.png"),
              "empty_file": (prior or {}).get("empty_file", f"empty_{slug}.png"),
              "rois": rois}
    cfg["drawers"] = [d for d in existing if d["name"] != drawer_name] + [drawer]
    normalize_drawers(cfg)      # re-check: ids must stay unique across drawers
    save_config(cfg)
    cv2.imwrite(str(HERE / drawer["baseline_file"]), warped)

    empty = HERE / drawer["empty_file"]
    if empty.exists():
        print(f"note: {empty.name} is from the previous calibration - re-shoot it "
              f"with 'e' in tracker.py")
    else:
        print(f"next: run tracker.py, empty the drawer completely and press 'e' - "
              f"nothing can be classified until {empty.name} exists")
    print(f"saved drawer {drawer_name!r} ({len(rois)} slots), "
          f"{CONFIG_PATH.name} now has {len(cfg['drawers'])} drawer(s) "
          f"and {drawer['baseline_file']}")


def on_mouse(event, x, y, flags, _):
    """Drag = rectangle, click = polygon vertex. One button, told apart by
    whether the mouse actually moved between press and release."""
    if event == cv2.EVENT_LBUTTONDOWN:
        drag["start"] = clamp((x, y))
    elif event == cv2.EVENT_MOUSEMOVE and drag["start"]:
        drag["cur"] = clamp((x, y))
    elif event == cv2.EVENT_LBUTTONUP and drag["start"]:
        x0, y0 = drag["start"]
        x, y = clamp((x, y))
        drag["start"], drag["cur"] = None, None
        if abs(x - x0) > CLICK_SLOP and abs(y - y0) > CLICK_SLOP:
            drag["rect"] = [min(x0, x), min(y0, y), abs(x - x0), abs(y - y0)]
        else:
            poly["points"].append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN and len(poly["points"]) >= 3:
        poly["close"] = True    # right-click closes, same as 'c'


def name_slot(rois, shape):
    """Prompt for a name and turn it into a slot, or None if discarded.

    `shape` is a rect or a polygon; the polygon's rect is filled in by
    validate_rois, which treats the polygon as authoritative.
    """
    name = input("tool name (blank = discard): ").strip()
    if not name:
        return None
    tool_id = slugify(name)
    if any(r["id"] == tool_id for r in rois):
        print(f"  '{name}' already defined (id {tool_id}) - skipped")
        return None
    slot = {"id": tool_id, "name": name}
    slot.update(shape)
    print(f"  added {tool_id}  {name}")
    return slot


def refit(drawer_name="main", write=False, min_iou=0.25):
    """Replace hand-drawn slot outlines with ones detected off the bare drawer.

    Hand-drawn boxes are loose - a bounding box around a diagonal tool swallows a
    corner of each neighbour, and a slot whose outline is mostly shared
    background cannot be told apart from the full drawer at all. The bare
    reference already contains the exact answer: one moulded pocket per slot,
    separated by foam.

    Matching by overlap rather than by order is the whole point of doing it this
    way: tool ids key the event log, so a slot that silently picked up a
    different id would rewrite the history of two physical tools. Anything that
    does not match confidently is left exactly as it was and reported.
    """
    cfg = tracker.load_config()
    drawer = next((d for d in cfg["drawers"] if d["name"] == drawer_name), None)
    if drawer is None:
        raise SystemExit(f"no drawer named {drawer_name!r} in config.json")
    if drawer.get("empty") is None:
        raise SystemExit(f"drawer {drawer_name!r} has no empty reference - run "
                         f"tracker.py, empty the drawer and press 'e'")

    pockets = tracker.detect_pockets(drawer["empty"])
    shape = drawer["empty"].shape
    print(f"{len(pockets)} pockets detected in {drawer['empty_file']}, "
          f"{len(drawer['rois'])} slots configured\n")

    taken, changed, skipped = {}, [], []
    for r in drawer["rois"]:
        scored = sorted(((tracker.polygon_iou(p, r["rect"], shape), i)
                         for i, p in enumerate(pockets)), reverse=True)
        best, idx = scored[0] if scored else (0.0, -1)
        if best < min_iou:
            skipped.append((r["id"], f"no pocket overlaps it (best {best:.2f})"))
            continue
        if idx in taken:
            # Two slots claiming one pocket means the detection merged them;
            # rewriting both to the same outline would make them indistinguishable.
            skipped.append((r["id"], f"pocket {idx} already taken by {taken[idx]}"))
            continue
        taken[idx] = r["id"]
        old = r["rect"][2] * r["rect"][3]
        new = abs(cv2.contourArea(np.asarray(pockets[idx], np.int32)))
        changed.append((r, pockets[idx], old, new, best))

    print(f"{'tool':<8}{'IoU':>6}{'old rect px':>13}{'new poly px':>13}{'tighter':>9}")
    for r, poly, old, new, best in changed:
        print(f"{r['id']:<8}{best:>6.2f}{old:>13,}{new:>13,}{(1 - new / old) * 100:>8.0f}%")
    for tool_id, why in skipped:
        print(f"{tool_id:<8}  left alone - {why}")
    extra = len(pockets) - len(taken)
    if extra > 0:
        print(f"\n{extra} detected pocket(s) matched no configured slot "
              f"(likely foam texture, or a tool you have not labelled)")

    if not write:
        print("\n(report only - pass --write to update config.json)")
        return
    for r, poly, _, _, _ in changed:
        r["polygon"] = [[int(x), int(y)] for x, y in poly]
    # Recomputes each rect from its new polygon and rejects anything out of bounds
    # before it can reach the tracker.
    tracker.validate_rois(drawer["rois"], cfg["warp_size"])
    tracker.save_config(cfg)
    print(f"\nwrote config.json - {len(changed)} slot(s) refitted to polygons")


def frame_transform(src, dst):
    """Homography taking points in `src`'s frame to `dst`'s, or None.

    Both images are rectifications of the same flat drawer to the same canonical
    size, so one homography relates them exactly - which means a slot outlined
    under one localizer can be carried over to another instead of being traced
    again by hand.

    The rotation is resolved by brute force first. A markerless localizer knows
    the drawer's rectangle but not which way is up (marker *ids* are what carry
    that, and there are none here), so the two frames can differ by a quarter
    turn - and ECC is a local optimiser with no path across one.
    """
    best, best_cc, best_k = None, -2.0, 0
    for k in range(4):
        cand = np.rot90(src, k).copy()
        if cand.shape != dst.shape:
            continue
        a = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY).astype(np.float32)
        b = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cc = float(cv2.matchTemplate(b, a, cv2.TM_CCOEFF_NORMED)[0, 0])
        if cc > best_cc:
            best, best_cc, best_k = cand, cc, k
    if best is None or best_cc < 0.2:
        return None, best_cc, best_k
    a = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(best, cv2.COLOR_BGR2GRAY).astype(np.float32)
    M = np.eye(3, 3, dtype=np.float32)
    try:
        cc, M = cv2.findTransformECC(
            b, a, M, cv2.MOTION_HOMOGRAPHY,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7), None, 5)
    except cv2.error:
        return None, best_cc, best_k
    return M, cc, best_k


def migrate(drawer_name="main", source=None, write=False):
    """Carry an existing drawer's slots into the current localizer's frame.

    Switching localizers changes the rectification, which normally invalidates
    every slot and both references at once. Rather than re-outline twelve tools
    by hand - and re-key twelve tool ids, any of which would orphan its own event
    history - this measures the transform between the two frames and moves the
    slots through it.
    """
    cfg = tracker.load_config()
    drawer = next((d for d in cfg["drawers"] if d["name"] == drawer_name), None)
    if drawer is None:
        raise SystemExit(f"no drawer named {drawer_name!r}")
    old_baseline = drawer["baseline"]

    live = dict(cfg)
    live["localizer"] = "auto"
    localizer = Localizer(live)
    cap = open_camera(source if source is not None else
                      cfg.get("camera_index", CAMERA_INDEX))
    print("point the camera at the FULL drawer; capturing when the outline is "
          "steady...  q aborts")
    warped = None
    try:
        steady = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            got, state = localizer.locate(frame)
            steady = steady + 1 if state == "live" else 0
            if got is not None:
                cv2.putText(got, f"{state}  steady {steady}/15", (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("migrate - rectified by the new localizer", got)
            else:
                cv2.putText(frame, localizer.hint(frame), (20, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("migrate - rectified by the new localizer", frame)
            if steady >= 15:
                warped, _ = localizer.locate(frame)
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if warped is None:
        raise SystemExit("no steady drawer outline - nothing captured")

    M, cc, k = frame_transform(warped, old_baseline)
    print(f"\nrotation {k * 90} degrees, correlation {cc:.2f}")
    if M is None:
        raise SystemExit("could not relate the two frames - is this the same drawer, "
                         "full, and lit as it was when the old baseline was shot?")
    if cc < 0.5:
        print("warning: weak match; check the result before trusting it")

    # Old-frame points -> new-frame points is the inverse of what ECC returned.
    inv = np.linalg.inv(M)
    w, h = cfg["warp_size"]
    rot = {0: lambda p: p,
           1: lambda p: np.stack([p[:, 1], w - 1 - p[:, 0]], 1),
           2: lambda p: np.stack([w - 1 - p[:, 0], h - 1 - p[:, 1]], 1),
           3: lambda p: np.stack([h - 1 - p[:, 1], p[:, 0]], 1)}[(4 - k) % 4]
    moved = 0
    for r in drawer["rois"]:
        x, y, rw, rh = r["rect"]
        pts = np.array(r.get("polygon") or
                       [[x, y], [x + rw, y], [x + rw, y + rh], [x, y + rh]], np.float64)
        p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), inv).reshape(-1, 2)
        p = rot(p)
        p[:, 0] = np.clip(p[:, 0], 0, w - 1)
        p[:, 1] = np.clip(p[:, 1], 0, h - 1)
        r["polygon"] = [[int(a), int(b)] for a, b in p]
        moved += 1
    validate_rois(drawer["rois"], cfg["warp_size"])

    preview = warped.copy()
    for r in drawer["rois"]:
        cv2.polylines(preview, [np.array(r["polygon"], np.int32)], True, (0, 255, 0), 2)
        px, py = np.array(r["polygon"]).min(axis=0)
        cv2.putText(preview, r["id"], (int(px), int(py) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    out = Path("diagnostics/migrate_preview.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{moved} slots carried over - check {out.resolve()}")

    if not write:
        print("\n(preview only - pass --write to switch the config over)")
        return
    cfg["localizer"] = "auto"
    cv2.imwrite(str(HERE / drawer["baseline_file"]), warped)
    save_config(cfg)
    print(f"\nwrote {drawer['baseline_file']} and config.json (localizer=auto).\n"
          f"NOW, without moving the drawer or camera: run tracker.py, empty the "
          f"drawer completely and press 'e'.\nShooting both references in one "
          f"session is what fixes slots 296 and 301.")


def main(drawer_name="main", source=None):
    cfg = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    localizer = Localizer({**base_config(), **cfg, "warp_size": WARP_SIZE})
    # An image path argument calibrates off a photo instead of the live camera.
    cap = open_camera(source if source is not None else
                      cfg.get("camera_index", CAMERA_INDEX))

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)

    rois, warped = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        found, state = localizer.locate(frame)
        if found is None:
            cv2.putText(frame, "NO DRAWER FOUND - open it, fill the frame, "
                               "check the lighting", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("calibrate", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break   # must be able to quit before the drawer is ever found
            continue
        warped = found

        shape = None
        if drag["rect"]:
            x, y, w, h = drag["rect"]
            drag["rect"] = None
            shape = {"rect": [x, y, w, h]}
        elif poly["close"]:
            poly["close"] = False
            shape = {"polygon": [list(p) for p in poly["points"]]}
            poly["points"] = []
        if shape:
            cv2.imshow("calibrate", warped)   # freeze the view while typing
            cv2.waitKey(1)
            slot = name_slot(rois, shape)
            if slot:
                rois.append(slot)

        view = warped.copy()
        for r in rois:
            if r.get("polygon"):
                pts = np.array(r["polygon"], np.int32)
                cv2.polylines(view, [pts], True, (0, 255, 0), 2)
                x, y = pts.min(axis=0)
            else:
                x, y, w, h = r["rect"]
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(view, f"{r['id']} {r['name']}", (int(x), int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if poly["points"]:
            pts = np.array(poly["points"], np.int32)
            cv2.polylines(view, [pts], False, (255, 255, 0), 2)
            for p in poly["points"]:
                cv2.circle(view, tuple(p), 3, (255, 255, 0), -1)
        if drag["start"] and drag["cur"]:
            cv2.rectangle(view, drag["start"], drag["cur"], (255, 255, 0), 1)
        cv2.putText(view, f"[{drawer_name}] {len(rois)} slots | "
                          f"drag=rect  click=polygon point ({len(poly['points'])})  "
                          f"c=close  x=discard  u=undo  s=save  q=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if state != "live":
            cv2.putText(view, "HELD - drawer outline lost", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.imshow("calibrate", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c") and len(poly["points"]) >= 3:
            poly["close"] = True
        if key == ord("x"):
            poly["points"] = []
        if key == ord("u") and rois:
            print(f"  removed {rois.pop()['name']}")
        if key == ord("s"):
            if not rois:
                print("no slots to save")
                continue
            try:
                # Never write a config the tracker would refuse to load.
                validate_rois(rois, WARP_SIZE)
                save(rois, warped, drawer_name)
            except ValueError as e:
                print(f"not saving: {e}")
                continue
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="calibrate off a photo, not the camera")
    ap.add_argument("--drawer", default="main", help="drawer to add or replace")
    ap.add_argument("--migrate", action="store_true",
                    help="carry this drawer's slots into the markerless (auto) "
                         "frame and switch config over, instead of re-outlining "
                         "every tool by hand")
    ap.add_argument("--camera", type=int, metavar="N", help="camera index override")
    ap.add_argument("--refit", action="store_true",
                    help="refit existing slots to polygons detected off the "
                         "bare drawer, keeping their ids")
    ap.add_argument("--write", action="store_true",
                    help="with --refit, actually update config.json")
    a = ap.parse_args()
    src = a.image if a.image else (str(a.camera) if a.camera is not None else None)
    if a.migrate:
        migrate(a.drawer, src, a.write)
    elif a.refit:
        refit(a.drawer, a.write)
    else:
        main(a.drawer, src)
