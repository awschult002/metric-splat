# Metric splat pilot

Repeatable path: video → metric high-quality Gaussian splat. Grow room → house → blocks later. **Not** city-scale first.

See [docs/STATUS.md](docs/STATUS.md) and [configs/pilot.yaml](configs/pilot.yaml).

## Locked stack (Phase 1)

| Piece | Choice |
| --- | --- |
| Frames | ffmpeg extract + thin (blur / motion stubs; not every-Nth) |
| SfM | HLoc **ALIKED + LightGlue + EigenPlaces** → **GLOMAP** mapper |
| Export | COLMAP-compatible `sparse/0` for gsplat |
| Scale | AprilTag `tagStandard41h12` (stub; Francis has no tags) |
| Train | gsplat MCMC + packed + PPISP on **Alex’s NVIDIA box** (this box: no CUDA) |
| Skip | FastGS; VGGT for keeps |

## Run Phase 1

Single entrypoint:

```bash
cd /workspace/repos/metric-splat
python3 scripts/run_phase1.py --help
```

### Francis dry-run (Tanks & Temples stand-in)

Uses existing Francis images (no video extract). SfM tools may be missing — still writes manifests, HLoc/GLOMAP recipes, and stage markers.

```bash
python3 scripts/run_phase1.py --dry-run-francis
# equivalent:
python3 scripts/run_phase1.py \
  --config configs/pilot.yaml \
  --work work/francis_tanks_dryrun_run \
  --images-dir work/francis_tanks_dryrun/images
```

Outputs under `work/francis_tanks_dryrun_run/`: `frames_sel/`, `frames_sel_manifest.json`, `hloc/README`, `stage_markers/`, `gsplat/cfg.yaml`, `run_summary.json`.

Pointer: `work/francis_tanks_dryrun` → `/workspace/repos/data/Tanks/Francis`.

### Real room01 video

```bash
# place capture
# capture/room01/video.mp4
# fill work/room01/tags.json with measured tag sizes (meters)

python3 scripts/run_phase1.py \
  --config configs/pilot.yaml \
  --work work/room01 \
  --video capture/room01/video.mp4
```

Or run stages alone:

```bash
python3 scripts/run_phase1.py --work work/room01 --video capture/room01/video.mp4 extract_frames thin_frames
python3 scripts/run_phase1.py --work work/room01 run_hloc run_glomap export_colmap
python3 scripts/run_phase1.py --work work/room01 scale_apriltag train_gsplat
```

### Full SfM install (this box or NVIDIA box)

1. `ffmpeg` — present on this box  
2. HLoc + ALIKED + LightGlue + EigenPlaces (prefer GPU box)  
3. COLMAP (for DB / optional) + **GLOMAP** `mapper`  
4. gsplat train only on NVIDIA box — see `work/*/gsplat/train_command.txt`

Never pretends GLOMAP/HLoc ran if binaries/imports are missing; check `stage_markers/*.json` and `tools_detected.json`.
