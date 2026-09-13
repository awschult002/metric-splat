# STATUS — Metric splat pilot (3DGS room → house)

Last updated: 2026-09-13 ~13:20 (America/New_York)

## Current milestone

**Phase 1 scaffold up; Francis dry-run on this box stops at SfM inspect.** Entrypoint `scripts/run_phase1.py`. Dry-run: 120 frames + HLoc/GLOMAP **recipes** (tools not on box — no pretend SfM). Real HLoc+GLOMAP → `sparse/0` and MCMC only on Alex’s NVIDIA. No house until Tester’s SfM + scale + train-smoke gates clear.

| Owner | Work |
| --- | --- |
| Dev | Scaffold landed; `work/francis_tanks_dryrun_run/` |
| Senior | Lean one-path lock; Vid2Scene only after hand-run |
| Oracle | AprilTag scale notes; **`1000×n` confirmed** (120000 @ n=120) |
| Tester | Recipe gate OK (`max_num_tracks=120000` for n=120); SfM/scale/train await NVIDIA `sparse/0` |
| Scribe | This STATUS |

## Phase 0 stack (locked)

| Piece | Choice |
| --- | --- |
| Frames | ffmpeg extract + thin script (not naive every-Nth) |
| SfM front-end | HLoc: **ALIKED + LightGlue + EigenPlaces** |
| SfM back-end | **GLOMAP** `mapper` |
| Trainer | **gsplat**, MCMC (or AbsGrad), packed |
| Appearance | **PPISP** on |
| Scale | AprilTags **`tagStandard41h12`** |
| Skip | FastGS; VGGT except 30s preview |
| Later | Vid2Scene only after pieces run by hand |

## Lean rules (Senior)

One path: `frames_sel → hloc → glomap → scale → gsplat`. Cap tracks `1000×n` (Oracle locked). Frustum clump → fix selection before MCMC. **When NVIDIA emits `sparse/0`, frustum inspect before any train command.** This box: recipes only; **no CUDA train**.

## Scaffold (Dev)

- `python3 scripts/run_phase1.py --help`
- Francis: `--dry-run-francis --max-frames 120` → `work/francis_tanks_dryrun_run/`
- On box: ffmpeg ✓ · colmap/glomap/hloc/CUDA ✗
- Stages callable alone; recipes written when SfM tools missing

## Scenes

`/workspace/repos/data/Tanks/` (Francis dry-run via `work/francis_tanks_dryrun`). Real: `capture/room01/` → `work/room01/`.

## Phase 1 pass criteria (Tester)

1. **SfM** — continuous frustums; tracks ≈ `1000×n`; fail → no train
2. **Scale** — AprilTag `measured_m / recon_size`; tape within a few % (Francis dry-run **skips** — no tags)
3. **Train smoke** — short MCMC+PPISP; corners hold; no AE blobs
4. **Stop** — any fail → no house/neighborhood


## Oracle — house / yard / switch analysis (2026-09-13)

**Verdict:** Phase 0 stack is a **good foundation** for a metric house, **not enough alone** for yard → interior → light-switch in one flat gsplat. Order: (1) one tagged room, (2) house via connected blocks + tags, (3) LOD/hierarchy for macro↔micro. One mega-model → OOM, smeared switch, or blown doorway.

**AE / WB:** PPISP is the right tool (per-frame exposure/color + per-cam vignette/CRF). Prefer two sessions (exterior / interior). If doorway is two photometric worlds after short train → **two models sharing tags + doorway cams**, not one PPISP inventing physics.

**AprilTags — keep.** \(\lambda = s_{meas}/s_{recon}\) for meters + room consistency. Nest: ~0.15 m matte indoors; ~0.6–1.0 m yard/aerial; unique IDs/room. Tags do **not** buy switch microgeometry — Phase 3 close-up → register → short MCMC fine-tune.

**Multi-scale:** Do **not** one GLOMAP on yard+rooms+switch. Partition exterior / interior with doorway overlap; merge via tags or `model_merger`/`image_registrator`. Tracks `1000×n`. Frustum inspect before train (gated).

### Jump-starts (hand path first)
| Project | Use |
| --- | --- |
| `nv-tlabs/ppisp` + gsplat MCMC/PPISP | AE/WB (already in stack) |
| Inria Hierarchical 3D Gaussians | Chunk → hierarchy/LOD for house+yard |
| Octree-GS / A LoD of Gaussians | View-dependent detail (switch vs yard) |
| HLoc + GLOMAP (keep); watch GLUEMAP later | Multi-room SfM; don’t swap pilot |
| Vid2Scene / stechdrive-3dgs-utils AprilTag step | **After** we run detect→λ→apply ourselves |

**Plan for Alex’s house goal:** Phase 2 = interior block + exterior block + nested tags, PPISP on; Phase 3 = switch; Hierarchical/Octree only when one room `.ply` is clean. No Vid2Scene wrap yet.

## Tester gates — beyond Phase 1

**Phase 2** — doorway walk: no scale pop (pairwise tag distances across rooms); no black/white wall on one-model claim (else two-model composite is the pass path).

**Phase 3** — switch clip registers into existing sparse before fine-tune; if poses pull the house sideways, re-BA (local-only register fails).

**Hard stop** — no Hierarchical/Octree/house train until Phase-1 SfM+scale+smoke clear on a **real tagged room**.

## Written config

[configs/pilot.yaml](../configs/pilot.yaml)

## Open

- [x] Phase 0 + layout + pilot.yaml
- [x] Dev scaffold + Francis dry-run recipes (n=120, tracks=120000)
- [x] Tester recipe gate OK; NVIDIA holds real SfM/train
- [ ] NVIDIA HLoc+GLOMAP → `sparse/0`; Tester frustum inspect
- [ ] Oracle scale notes + tagged room01 / `tags.json`
- [ ] Short gsplat on NVIDIA after SfM pass
- [x] Oracle house-scale analysis + jump-start list folded
- [ ] Phase 2 (interior+exterior blocks + nested tags) only after Phase 1 tagged-room pass
- [ ] Hierarchical/Octree only after clean room `.ply`

## Decision log

| When | Lock |
| --- | --- |
| 2026-09-13 | Stack: ffmpeg → HLoc ALIKED+LG+EigenPlaces → GLOMAP → gsplat MCMC+PPISP → tagStandard41h12 |
| 2026-09-13 | Skip FastGS/VGGT-for-keeps; Francis first; no city scale |
| 2026-09-13 | Project `/workspace/repos/metric-splat`; `scripts/run_phase1.py` |
| 2026-09-13 | Box = SfM inspect/recipes only; NVIDIA = real SfM + MCMC; house blocked until gates |
| 2026-09-13 | Oracle: 1000×n stands; Senior: frustum inspect on sparse/0 before any train command |
| 2026-09-13 | Oracle: Phase0 good for metric house but not flat yard→switch; PPISP+nested tags+blocks; Hierarchical/Octree after room ply |
| 2026-09-13 | Tester: Phase2 doorway/tag gates; Phase3 register-before-finetune; hard stop Hierarchical until Phase1 tagged room |

## Dev note (2026-09-13)
- Added `frustum_inspect` stage; `train_gsplat` blocked until inspect passes on real `sparse/0` (Senior lock). Override: `--force-train`.
