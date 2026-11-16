"""Camera probe - no GUI, prints facts and saves a frame per working device.

  .venv/bin/python diagnose.py

Answers, in order: which camera indices exist, do frames actually arrive, are
they real image data or a blank buffer, and which one can see the drawer.

Scans every index rather than a hardcoded one, because on macOS index 0 is often
a paired iPhone via Continuity Camera - it opens, it streams, and it is pointed
at a desk, which looks identical to broken code from the tracker's side.
"""

import cv2
import numpy as np

from tracker import CONFIG_PATH, make_detector, DETECT, lsd_corners

MAX_INDEX = 4          # how many device slots to try
WARMUP_READS = 30      # first frames off a Mac camera are often blank


def probe(index, detector):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    frame = None
    for i in range(WARMUP_READS):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
            if i and float(np.std(f)) > 1.0:      # real image data, not flat
                break
    cap.release()
    if frame is None:
        return {"index": index, "frames": False}

    corners, ids, _ = detector.detectMarkers(frame)
    found = sorted(int(i) for i in ids.flatten()) if ids is not None else []
    # Markerless too, so a rig with no markers still gets a useful answer.
    quad = lsd_corners(frame, (1000, 700), DETECT)
    return {"index": index, "frames": True, "shape": frame.shape,
            "mean": float(np.mean(frame)), "std": float(np.std(frame)),
            "markers": found, "quad": quad is not None, "frame": frame}


detector = make_detector()
results = [r for r in (probe(i, detector) for i in range(MAX_INDEX)) if r]
if not results:
    print("no cameras opened at all.")
    print("  -> on macOS this is almost always permission: System Settings >")
    print("     Privacy & Security > Camera, enable your terminal, restart it.")
    raise SystemExit(1)

print(f"{'idx':<5}{'frames':<8}{'resolution':<13}{'mean':>6}{'std':>7}"
      f"{'markers':>16}{'drawer?':>9}")
for r in results:
    if not r["frames"]:
        print(f"{r['index']:<5}{'NO':<8}{'-':<13}{'-':>6}{'-':>7}{'-':>16}{'-':>9}")
        continue
    h, w = r["shape"][:2]
    marks = ",".join(str(m) for m in r["markers"]) or "none"
    print(f"{r['index']:<5}{'yes':<8}{f'{w}x{h}':<13}{r['mean']:>6.0f}{r['std']:>7.1f}"
          f"{marks:>16}{'yes' if r['quad'] else 'no':>9}")
    cv2.imwrite(f"diagnose_cam{r['index']}.png", r["frame"])

print("\nstd under ~1 means a blank buffer; a low std with a grey image means the"
      "\ncamera is pointed at nothing. A frame per working camera was saved as"
      "\ndiagnose_cam<N>.png - open them and see which is your drawer.")

usable = [r for r in results if r.get("markers") == [0, 1, 2, 3]]
markerless = [r for r in results if r["frames"] and r.get("quad")]
if usable:
    print(f"\n-> use camera_index {usable[0]['index']} in {CONFIG_PATH.name} "
          f"(all four markers visible)")
elif markerless:
    print(f"\n-> no camera sees all four markers. Camera {markerless[0]['index']} "
          f"does find a drawer-shaped rectangle, so \"localizer\": \"auto\" would "
          f"work there - but note that switching localizers needs the references "
          f"and slots rebuilt.")
else:
    print("\n-> no camera sees the drawer. Check that it is open, in frame, lit,"
          "\n   and that all four markers are unobstructed.")
