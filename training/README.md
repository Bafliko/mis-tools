# Training data — put your photos here

This folder is the input to the tool/drawer knowledge base. Nothing here is read
at runtime by `tracker.py`; it is consumed by the ingestion step to build
descriptors, and by validation to prove those descriptors actually work.

**Read the naming rules before shooting.** Two mislabelled photos in the existing
`photos/` set made 7 of 12 slots look inseparable and cost real time to find. The
filename *is* the label — there is no other source of truth.

---

## What "training" means here

No neural network. The system is classic CV by design, so this builds a
**descriptor library**: per tool, a set of measurements (outline shape, colour
distribution, keypoints, size in mm-equivalent) that a live slot can be matched
against. Consequences worth knowing before you shoot hundreds of photos:

- It generalises to **new drawers of the same construction** reasonably well —
  localization and pocket detection are geometric and already do this.
- It generalises to **unseen tool types** poorly. A descriptor library recognises
  what it has seen. Ten photos of a tool it has never met will not help.
- **Small tools are the hard case.** Sockets, bits and allen keys are small,
  round, and low-texture — few keypoints, weak shape signature. Expect these to
  be the first thing that fails, and shoot them deliberately.

---

## `drawers/` — one folder per physical drawer

```
training/drawers/<drawer_name>/
    full_001.jpg                       every tool present
    empty_001.jpg                      every tool removed
    missing_1333-292_001.jpg           that one tool absent
    missing_1333-292+1333-294_001.jpg  those two absent
```

Rules that matter:

1. **`full_` and `empty_` are both mandatory**, and must be shot **back to back
   without moving the drawer or the camera**. This is the single most important
   rule in the whole file. The existing drawer's two references were shot apart
   and drifted by a 1.38x scale — it silently broke every "missing" reading until
   it was found, and slot `301` is *still* unusable because of residual local
   misalignment in the corner.
2. Shoot from a **fixed camera position** for all photos of one drawer. Vary the
   tools, not the viewpoint.
3. Name tools by their printed ID (`1333-292`) where one exists. IDs key the
   event log; renaming a tool later orphans its history.
4. Include a few frames with **normal working light variation** — a lamp on, a
   shadow across the drawer. Not extreme darkness; realistic conditions.
5. **Check your labels against the photo before saving.** If you use the `s` key
   in `tracker.py`, the filename records what the tracker *believed*, which is
   wrong whenever the tracker was wrong.

Roughly 5 `full_`, 3 `empty_`, and 2 per single-tool-removed state is plenty per
drawer. More photos of the same thing add nothing; more *variety* does.

### The most valuable drawer you can add

A drawer with **single-colour foam** — black pockets with no coloured base. All
current pocket detection leans on your blue pocket bases being saturated against
black foam. A mono-colour drawer would need shadow- or edge-based detection
instead, and until one exists we do not know whether any of this generalises.

Also valuable: a drawer of **small parts** (sockets, bits), and one with
**light-coloured foam** (which inverts the polarity entirely).

---

## `tools/` — one folder per tool, for identity

```
training/tools/<tool_id>/
    inpocket_001.jpg     the tool lying in its own pocket, shot as the camera sees it
    inpocket_002.jpg     ...again under different light
    loose_001.jpg        the tool alone on a plain surface, top-down
    loose_002.jpg        ...rotated 180 degrees
```

- `inpocket_` photos are what identity matching will actually run against, so
  these matter most. Shoot them from the real camera position.
- `loose_` photos give a clean silhouette for the shape descriptor.
- **Include a 180°-rotated view.** A tool put back the other way round is a known
  failure mode and will otherwise score as the wrong tool.
- 4–6 photos per tool is enough. Consistency beats volume.

If you want the "wrong tool in the right slot" case to work, you must shoot the
tools that could plausibly be confused — a set of similar pliers is far more
useful than one photo each of twelve unrelated tools.

---

## What happens after you add photos

Nothing automatically. The agreed autonomy model is **propose, never apply**:
ingestion writes a report of what it learned and what it would change, and no
config or reference is modified without you approving it.

That rule exists because this session produced three separate "obviously
correct" improvements — tighter auto-fitted slot polygons, difference-based
masks, and per-slot re-alignment — and **all three measured worse** than what
they replaced. Anything learned here gets validated against a held-out set
before it is trusted. See `DECISIONS.md` §6 for the numbers.
