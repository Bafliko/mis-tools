"""Generate a printable sheet of the four corner ArUco markers.

  .venv/bin/python gen_markers.py

Writes markers.png (A4 @ 150dpi). Print at 100% / "actual size" - any scaling
is fine as long as all four end up the SAME size, since the warp only uses
their corner positions, not their scale.
"""

import cv2
import numpy as np

from tracker import HERE, aruco_dict

DPI = 150
MARKER_MM = 25                   # black square edge; see README for the size floor
QUIET_MM = 4                     # white border ArUco needs; also the cut line

MARKER_PX = round(MARKER_MM / 25.4 * DPI)
QUIET_PX = round(QUIET_MM / 25.4 * DPI)
MARGIN, GAP, LABEL_PX = 90, 60, 70

# Sheet position matches where each marker goes on the drawer, so the printout
# reads as a map rather than a list.
PLACES = {0: (0, 0, "0  TOP-LEFT"), 1: (0, 1, "1  TOP-RIGHT"),
          3: (1, 0, "3  BOTTOM-LEFT"), 2: (1, 1, "2  BOTTOM-RIGHT")}

CELL = MARKER_PX + QUIET_PX * 2
sheet_w = MARGIN * 2 + CELL * 2 + GAP
sheet_h = MARGIN * 2 + (CELL + LABEL_PX) * 2 + GAP
sheet = np.full((sheet_h, sheet_w, 3), 255, np.uint8)

d = aruco_dict()
for i, (row, col, label) in PLACES.items():
    x = MARGIN + col * (CELL + GAP)
    y = MARGIN + row * (CELL + LABEL_PX + GAP)
    img = cv2.aruco.generateImageMarker(d, i, MARKER_PX)
    sheet[y + QUIET_PX:y + QUIET_PX + MARKER_PX,
          x + QUIET_PX:x + QUIET_PX + MARKER_PX] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Cut on this line, not on the black edge: cutting flush removes the quiet
    # zone and the detector stops seeing the marker.
    cv2.rectangle(sheet, (x, y), (x + CELL, y + CELL), (160, 160, 160), 1)
    cv2.putText(sheet, label, (x, y + CELL + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

out = HERE / "markers.png"
cv2.imwrite(str(out), sheet)
print(f"wrote {out.name}  ({sheet_w}x{sheet_h}px @ {DPI}dpi, "
      f"{MARKER_PX / DPI * 25.4:.0f}mm markers)")
print("Cut along the GREY line, not the black edge - that white border is the")
print("quiet zone the detector needs. Tape one in each corner per its label.")
print("Keep each marker square to the drawer edges - a rotated marker skews the warp.")
