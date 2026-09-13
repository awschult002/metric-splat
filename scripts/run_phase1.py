#!/usr/bin/env python3
"""
Phase-1 metric-splat pipeline (lean single entrypoint).

Stages: extract_frames → thin_frames → run_hloc → run_glomap →
        export_colmap → frustum_inspect → scale_apriltag → train_gsplat

Senior lock: frustum_inspect on real sparse/0 must pass before train_gsplat
writes a train command (unless --force-train).

Box has no CUDA. SfM tools may be missing — scaffold still writes
manifests / command recipes / stage markers. Train is a stub for
Alex's NVIDIA box.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from yaml_lite import load as load_yaml  # noqa: E402

STAGE_NAMES = [
    "extract_frames",
    "thin_frames",
    "run_hloc",
    "run_glomap",
    "export_colmap",
    "frustum_inspect",
    "scale_apriltag",
    "train_gsplat",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def detect_tools() -> dict[str, Any]:
    tools = {
        "ffmpeg": which("ffmpeg"),
        "ffprobe": which("ffprobe"),
        "colmap": which("colmap"),
        "glomap": which("glomap"),
        "python": sys.executable,
    }
    hloc_ok = False
    hloc_err = None
    try:
        subprocess.run(
            [sys.executable, "-c", "import hloc"],
            check=True,
            capture_output=True,
            text=True,
        )
        hloc_ok = True
    except Exception as e:  # noqa: BLE001
        hloc_err = str(e)
    tools["hloc_importable"] = hloc_ok
    tools["hloc_error"] = hloc_err
    return tools


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    files = [
        p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return files


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_marker(work: Path, stage: str, status: str, detail: dict[str, Any] | None = None) -> None:
    markers = work / "stage_markers"
    markers.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "status": status,
        "ts": utc_now(),
        "detail": detail or {},
    }
    write_json(markers / f"{stage}.json", payload)
    print(f"[marker] {stage}: {status}")


def ensure_work_layout(work: Path) -> None:
    for sub in (
        "frames_raw",
        "frames_sel",
        "hloc",
        "sparse/0",
        "gsplat",
        "stage_markers",
    ):
        (work / sub).mkdir(parents=True, exist_ok=True)


def resolve_path(p: str | Path, base: Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def laplacian_variance_stub(path: Path) -> float:
    """
    Blur score stub. Prefer numpy byte-std proxy when OpenCV unavailable.
    Higher ≈ sharper (rough proxy, not true Laplacian).
    """
    try:
        import numpy as np

        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return 0.0
        # sample to keep CPU cheap
        if data.size > 2_000_000:
            data = data[:: data.size // 2_000_000]
        return float(np.std(data))
    except Exception:  # noqa: BLE001
        return float(path.stat().st_size)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_extract_frames(
    work: Path,
    cfg: dict[str, Any],
    *,
    video: Path | None,
    images_dir: Path | None,
    tools: dict[str, Any],
) -> None:
    raw = work / "frames_raw"
    raw.mkdir(parents=True, exist_ok=True)

    if images_dir is not None:
        imgs = list_images(images_dir)
        print(f"[extract_frames] --images-dir given ({images_dir}); skip ffmpeg extract ({len(imgs)} images).")
        write_marker(
            work,
            "extract_frames",
            "skipped_images_dir",
            {"images_dir": str(images_dir), "n_images": len(imgs)},
        )
        return

    if video is None:
        # try capture path from config conventions
        cand = REPO_ROOT / "capture" / "room01" / "video.mp4"
        video = cand if cand.is_file() else None

    if video is None or not Path(video).is_file():
        print("[extract_frames] No video provided and no capture/room01/video.mp4 — skip.")
        write_marker(work, "extract_frames", "skipped_no_video", {})
        return

    ffmpeg = tools.get("ffmpeg")
    if not ffmpeg:
        print("[extract_frames] ffmpeg not found; cannot extract.")
        write_marker(work, "extract_frames", "missing_ffmpeg", {"video": str(video)})
        return

    # target ~3–5 fps from pilot.yaml (use 4 as default)
    fps = "4"
    out_pattern = str(raw / "frame_%06d.jpg")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        out_pattern,
    ]
    print("[extract_frames] running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        n = len(list_images(raw))
        write_marker(
            work,
            "extract_frames",
            "ok",
            {"video": str(video), "fps": fps, "n_frames": n, "cmd": cmd},
        )
    except subprocess.CalledProcessError as e:
        write_marker(work, "extract_frames", "failed", {"error": str(e), "cmd": cmd})
        raise


def stage_thin_frames(
    work: Path,
    cfg: dict[str, Any],
    *,
    images_dir: Path | None,
    max_frames: int | None,
) -> None:
    """Blur + motion-gate stubs; write frames_sel + manifest."""
    sel = work / "frames_sel"
    sel.mkdir(parents=True, exist_ok=True)

    # Source priority: --images-dir > frames_raw > existing frames_sel
    source: Path | None = None
    if images_dir is not None and images_dir.is_dir():
        source = images_dir
    elif list_images(work / "frames_raw"):
        source = work / "frames_raw"
    elif list_images(sel):
        # already populated (e.g. prior dry-run symlink)
        source = sel

    if source is None:
        print("[thin_frames] No source images found.")
        write_marker(work, "thin_frames", "skipped_no_images", {})
        return

    candidates = list_images(source)
    if not candidates:
        write_marker(work, "thin_frames", "skipped_no_images", {"source": str(source)})
        return

    # Parse selected_frames_room range from config (e.g. "80-200")
    sel_cfg = str(cfg.get("capture", {}).get("selected_frames_room", "80-200"))
    lo, hi = 80, 200
    if "-" in sel_cfg:
        try:
            a, b = sel_cfg.split("-", 1)
            lo, hi = int(a.strip()), int(b.strip())
        except ValueError:
            pass
    target_max = max_frames if max_frames is not None else hi
    target_max = min(target_max, hi)
    target_min = lo

    # Score: blur stub (higher better). Motion gate: stub keeps temporal order,
    # drops near-duplicates by index stride if we need to thin.
    scored: list[tuple[float, Path]] = []
    for p in candidates:
        scored.append((laplacian_variance_stub(p), p))
    scored.sort(key=lambda t: t[0], reverse=True)

    # If already in range and source is external images_dir, prefer uniform temporal sample
    n = len(candidates)
    if n <= target_max:
        chosen = candidates  # keep temporal order
    else:
        # uniform temporal subsample to target_max, bias toward sharper when ties
        step = n / float(target_max)
        idxs = sorted({min(n - 1, int(i * step)) for i in range(target_max)})
        chosen = [candidates[i] for i in idxs]
        # optional: among neighbors, pick sharper — stub notes motion_gap_max_s
        print(
            f"[thin_frames] motion_gate stub: uniform subsample {n} → {len(chosen)} "
            f"(motion_gap_max_s={cfg.get('capture', {}).get('motion_gap_max_s', 0.5)})"
        )

    if len(chosen) < target_min:
        print(
            f"[thin_frames] WARNING: only {len(chosen)} frames "
            f"(pilot wants {target_min}-{hi}). Proceeding."
        )

    # Populate frames_sel: symlink when possible, else copy
    # Clear previous selection files (not dirs)
    for old in list_images(sel):
        if old.is_symlink() or old.is_file():
            old.unlink()

    manifest_files = []
    for src in chosen:
        dst = sel / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.symlink(src.resolve(), dst)
            link = True
        except OSError:
            shutil.copy2(src, dst)
            link = False
        manifest_files.append(
            {
                "name": src.name,
                "source": str(src.resolve()),
                "selected": str(dst),
                "symlink": link,
                "blur_proxy": laplacian_variance_stub(src),
            }
        )

    manifest = {
        "stage": "thin_frames",
        "ts": utc_now(),
        "source": str(source),
        "n_candidates": n,
        "n_selected": len(manifest_files),
        "target_range": [target_min, hi],
        "blur_method": "byte_std_proxy_stub (Laplacian variance when OpenCV present — not installed)",
        "motion_gate": "stub_uniform_temporal",
        "files": manifest_files,
    }
    write_json(work / "frames_sel_manifest.json", manifest)
    print(f"[thin_frames] selected {len(manifest_files)} → {sel}")
    write_marker(
        work,
        "thin_frames",
        "ok",
        {"n_selected": len(manifest_files), "manifest": str(work / "frames_sel_manifest.json")},
    )


def stage_run_hloc(work: Path, cfg: dict[str, Any], tools: dict[str, Any]) -> None:
    hloc_dir = work / "hloc"
    hloc_dir.mkdir(parents=True, exist_ok=True)
    images = work / "frames_sel"
    n = len(list_images(images))
    sfm = cfg.get("sfm", {})
    features = sfm.get("features", "ALIKED")
    matcher = sfm.get("matcher", "LightGlue")
    retrieval = sfm.get("retrieval", "EigenPlaces")
    pairs = sfm.get("pairs", "sequential_neighbors + retrieval")

    readme = hloc_dir / "README"
    recipe = f"""# HLoc recipe (Phase-1 pilot)

Locked stack: {features} + {matcher} + {retrieval}
Pairs: {pairs}
Images: {images}  (n={n})
Outputs expected under: {hloc_dir}

## Pair strategy
1. Sequential neighbors (window ~10–20) for video continuity.
2. Retrieval pairs via {retrieval} (top-k ~20–50) for loop closures.
3. Merge unique pairs; feed to {matcher}.

## Exact commands (when hloc is installed)

```bash
# Example using hloc CLI / python API (adjust to your hloc checkout)
python -m hloc.extract_features \\
  --image_dir {images} \\
  --export_dir {hloc_dir} \\
  --conf {features}

python -m hloc.pairs_from_sequential \\
  --image_list {hloc_dir}/image_list.txt \\
  --output {hloc_dir}/pairs-seq.txt \\
  --features {hloc_dir}/feats-{features}.h5 \\
  --num_matched 20

python -m hloc.pairs_from_retrieval \\
  --output {hloc_dir}/pairs-retr.txt \\
  --num_matched 30 \\
  # + EigenPlaces retrieval db/query conf

# merge pairs-seq.txt + pairs-retr.txt → pairs.txt

python -m hloc.match_features \\
  --pairs {hloc_dir}/pairs.txt \\
  --export_dir {hloc_dir} \\
  --conf {matcher} \\
  --features {hloc_dir}/feats-{features}.h5

# Then triangulation / export to COLMAP database for GLOMAP, e.g.:
# python -m hloc.triangulation ...  OR import matches into COLMAP db
```

Install: `pip install hloc` (or clone cvg/Hierarchical-Localization) + LightGlue + ALIKED + EigenPlaces deps.
This agent box has **no CUDA** — feature extraction / matching is slow on CPU; prefer Alex's NVIDIA box for real runs.
"""
    readme.write_text(recipe, encoding="utf-8")

    cmds = [
        f"{sys.executable} -m hloc.extract_features --image_dir {images} --export_dir {hloc_dir} --conf {features}",
        f"{sys.executable} -m hloc.match_features --pairs {hloc_dir}/pairs.txt --export_dir {hloc_dir} --conf {matcher}",
    ]
    write_json(
        hloc_dir / "commands.json",
        {
            "features": features,
            "matcher": matcher,
            "retrieval": retrieval,
            "pairs": pairs,
            "n_images": n,
            "commands": cmds,
            "readme": str(readme),
        },
    )

    if tools.get("hloc_importable"):
        print("[run_hloc] hloc importable — thin wrapper would invoke here.")
        print("[run_hloc] Not auto-running full SfM (lean scaffold); see hloc/README.")
        write_marker(
            work,
            "run_hloc",
            "detected_not_executed",
            {"commands": cmds, "readme": str(readme)},
        )
    else:
        print("[run_hloc] hloc NOT installed. Wrote recipe →", readme)
        for c in cmds:
            print("  RECIPE:", c)
        write_marker(
            work,
            "run_hloc",
            "missing_hloc",
            {"commands": cmds, "readme": str(readme), "error": tools.get("hloc_error")},
        )


def stage_run_glomap(work: Path, cfg: dict[str, Any], tools: dict[str, Any]) -> None:
    images = work / "frames_sel"
    n = len(list_images(images))
    max_tracks = 1000 * max(n, 1)
    database = work / "hloc" / "database.db"  # expected after hloc→colmap import
    sparse_out = work / "sparse"

    # GLOMAP typical invocation
    cmd = [
        "glomap",
        "mapper",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--output_path",
        str(sparse_out),
        "--TrackEstablishment.max_num_tracks",
        str(max_tracks),
    ]
    cmd_str = " ".join(cmd)
    recipe_path = work / "hloc" / "glomap_command.txt"
    recipe_path.write_text(
        cmd_str
        + "\n\n"
        + f"# max_num_tracks = 1000 * n_images = 1000 * {n} = {max_tracks}\n"
        + "# Requires COLMAP-format database with matches from HLoc.\n"
        + "# Output: sparse/0/ with cameras.bin images.bin points3D.bin\n",
        encoding="utf-8",
    )
    print("[run_glomap] recipe:", cmd_str)

    glomap_bin = tools.get("glomap")
    if not glomap_bin:
        print("[run_glomap] glomap NOT found on PATH — did not run.")
        write_marker(
            work,
            "run_glomap",
            "missing_glomap",
            {"command": cmd, "max_num_tracks": max_tracks, "n_images": n, "recipe": str(recipe_path)},
        )
        return

    if not database.is_file():
        print(f"[run_glomap] glomap found but database missing ({database}) — not running.")
        write_marker(
            work,
            "run_glomap",
            "missing_database",
            {"command": cmd, "database": str(database)},
        )
        return

    print("[run_glomap] running:", cmd_str)
    try:
        subprocess.run(cmd, check=True)
        write_marker(
            work,
            "run_glomap",
            "ok",
            {"command": cmd, "max_num_tracks": max_tracks},
        )
    except subprocess.CalledProcessError as e:
        write_marker(work, "run_glomap", "failed", {"error": str(e), "command": cmd})
        # do not pretend success
        print("[run_glomap] FAILED — marker written; sparse may be incomplete.")


def stage_export_colmap(work: Path, cfg: dict[str, Any], tools: dict[str, Any]) -> None:
    """Ensure sparse/0 layout for gsplat."""
    sparse = work / "sparse"
    sparse0 = sparse / "0"
    sparse0.mkdir(parents=True, exist_ok=True)

    # If GLOMAP wrote directly into sparse/ without 0/, move/link
    bins = ["cameras.bin", "images.bin", "points3D.bin"]
    txts = ["cameras.txt", "images.txt", "points3D.txt"]

    def has_model(d: Path) -> bool:
        return any((d / b).is_file() for b in bins) or any((d / t).is_file() for t in txts)

    moved = False
    if has_model(sparse) and not has_model(sparse0):
        for name in bins + txts + ["project.ini"]:
            src = sparse / name
            if src.is_file():
                shutil.move(str(src), str(sparse0 / name))
                moved = True

    present = [p.name for p in sparse0.iterdir() if p.is_file()] if sparse0.is_dir() else []
    status = "ok_layout" if has_model(sparse0) else "empty_placeholder"
    # Always leave a README so gsplat path is clear
    (sparse0 / "README").write_text(
        "COLMAP-compatible sparse model for gsplat.\n"
        "Expect: cameras.bin, images.bin, points3D.bin (or .txt).\n"
        "Filled by GLOMAP mapper when SfM tools + HLoc database are available.\n",
        encoding="utf-8",
    )
    print(f"[export_colmap] sparse/0 present={present or '(empty)'} status={status}")
    write_marker(
        work,
        "export_colmap",
        status,
        {"sparse0": str(sparse0), "files": present, "moved_into_0": moved},
    )



def stage_frustum_inspect(work: Path, cfg: dict[str, Any], tools: dict[str, Any]) -> None:
    """Dump a frustum/SfM inspect report for sparse/0. Gate before train.

    Senior lock: inspect real sparse/0 before anyone writes a train command.
    This stage does not require CUDA. If colmap is present, optionally run
    `colmap model_analyzer`; always write a human checklist + counts.
    """
    sparse0 = work / "sparse" / "0"
    out_dir = work / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)
    bins = ["cameras.bin", "images.bin", "points3D.bin"]
    txts = ["cameras.txt", "images.txt", "points3D.txt"]
    has_bin = all((sparse0 / b).is_file() for b in bins)
    has_txt = all((sparse0 / t).is_file() for t in txts)
    has_model = has_bin or has_txt

    report: dict[str, Any] = {
        "sparse0": str(sparse0),
        "has_model": has_model,
        "has_bin": has_bin,
        "has_txt": has_txt,
        "files": [p.name for p in sparse0.iterdir() if p.is_file()] if sparse0.is_dir() else [],
        "checklist": [
            "Frustums continuous (no wall-clump / broken loop)",
            "Cameras cover the room (not one wall)",
            f"Track cap recipe ≈ 1000 × n_images (see glomap_command.txt)",
            "If inspect fails → STOP; do not train",
        ],
    }

    # Optional colmap model_analyzer
    colmap = tools.get("colmap")
    if has_model and colmap:
        try:
            proc = subprocess.run(
                [colmap, "model_analyzer", "--path", str(sparse0)],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            report["colmap_model_analyzer"] = {
                "returncode": proc.returncode,
                "stdout": (proc.stdout or "")[-4000:],
                "stderr": (proc.stderr or "")[-2000:],
            }
        except Exception as e:
            report["colmap_model_analyzer"] = {"error": str(e)}
    elif has_model:
        report["colmap_model_analyzer"] = "colmap not on PATH — open sparse/0 in COLMAP GUI / viser on NVIDIA box"

    status = "pass_files_present" if has_model else "fail_no_sparse_model"
    report_path = out_dir / "frustum_inspect.json"
    write_json(report_path, report)
    checklist_path = out_dir / "frustum_checklist.md"
    checklist_path.write_text(
        "# Frustum inspect (required before train)\n\n"
        f"sparse/0 model present: **{has_model}**\n\n"
        "Manual checks (Tester gate):\n"
        + "\n".join(f"- [ ] {c}" for c in report["checklist"])
        + "\n\n"
        "Senior lock: do not write/run train until this inspect passes.\n",
        encoding="utf-8",
    )
    print(f"[frustum_inspect] status={status} report→{report_path}")
    write_marker(work, "frustum_inspect", status, report)
    if not has_model:
        print("[frustum_inspect] FAIL — no cameras/images/points3D in sparse/0. Train must not run.")


def stage_scale_apriltag(work: Path, cfg: dict[str, Any]) -> None:
    scale_cfg = cfg.get("scale", {})
    family = scale_cfg.get("family", "tagStandard41h12")
    tags_rel = scale_cfg.get("tags_json", "work/room01/tags.json")
    tags_path = resolve_path(tags_rel, REPO_ROOT)
    # Also accept work-local tags.json
    local_tags = work / "tags.json"
    if local_tags.is_file():
        tags_path = local_tags

    tags_data: dict[str, Any] = {}
    if tags_path.is_file():
        tags_data = json.loads(tags_path.read_text(encoding="utf-8"))
    tags = tags_data.get("tags") or {}

    if not tags:
        msg = (
            f"[scale_apriltag] DRY-RUN no-op: no tags in {tags_path} "
            f"(family={family}). Francis / Tanks has no AprilTags."
        )
        print(msg)
        write_marker(
            work,
            "scale_apriltag",
            "dryrun_noop",
            {
                "family": family,
                "tags_json": str(tags_path),
                "formula": scale_cfg.get("formula", "measured_m / reconstructed_tag_size"),
                "n_tags": 0,
            },
        )
        return

    # Stub: would detect tags in frames_sel and compute scale
    print(f"[scale_apriltag] stub: {len(tags)} tag size(s) loaded from {tags_path}")
    print("[scale_apriltag] Detection/scale math not executed in this scaffold.")
    write_marker(
        work,
        "scale_apriltag",
        "stub_tags_present",
        {"family": family, "tags": tags, "tags_json": str(tags_path)},
    )


def stage_train_gsplat(work: Path, cfg: dict[str, Any], force: bool = False) -> None:
    """Stub cfg + command for NVIDIA box (MCMC, packed, PPISP).

    Refuses to write train_command.txt unless frustum_inspect passed
    (real sparse/0 present) or force=True.
    """
    marker = work / "stage_markers" / "frustum_inspect.json"
    inspect_ok = False
    if marker.is_file():
        try:
            inspect_ok = json.loads(marker.read_text(encoding="utf-8")).get("status") == "pass_files_present"
        except Exception:
            inspect_ok = False
    if not inspect_ok and not force:
        print(
            "[train_gsplat] BLOCKED — frustum_inspect did not pass on sparse/0. "
            "Run frustum_inspect after GLOMAP, or pass --force-train to override."
        )
        write_marker(
            work,
            "train_gsplat",
            "blocked_needs_frustum_inspect",
            {"marker": str(marker), "force": force},
        )
        return

    train_cfg = cfg.get("train", {})
    gsplat_dir = work / "gsplat"
    gsplat_dir.mkdir(parents=True, exist_ok=True)

    stub_cfg = {
        "engine": train_cfg.get("engine", "gsplat"),
        "data_dir": str(work),
        "images": str(work / "frames_sel"),
        "sparse": str(work / "sparse" / "0"),
        "densify": train_cfg.get("densify", "MCMC"),
        "packed": train_cfg.get("packed", True),
        "ppisp": train_cfg.get("ppisp", True),
        "max_steps": train_cfg.get("short_schedule_steps", "15000-30000"),
        "export": train_cfg.get("export", ["ckpt.ply", "cameras"]),
        "note": "Run on Alex's NVIDIA box — this agent box has no CUDA.",
        "hardware": cfg.get("hardware", {}),
    }
    cfg_path = gsplat_dir / "cfg.yaml"
    # write simple YAML by hand
    lines = ["# gsplat train stub — copy to NVIDIA box\n"]
    for k, v in stub_cfg.items():
        if isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}\n")
        elif isinstance(v, (list, dict)):
            lines.append(f"{k}: {json.dumps(v)}\n")
        else:
            lines.append(f"{k}: {v}\n")
    cfg_path.write_text("".join(lines), encoding="utf-8")

    cmd = (
        f"CUDA_VISIBLE_DEVICES=0 python -m gsplat.examples.simple_trainer "
        f"--data_dir {work} "
        f"--result_dir {gsplat_dir} "
        f"--max_steps 30000 "
        f"# densify=MCMC packed=true ppisp=true  (map to current gsplat flags)\n"
        f"# Exact flags vary by gsplat version — set MCMC / packed / PPISP on the NVIDIA box.\n"
        f"# Input: frames_sel + sparse/0 COLMAP model.\n"
    )
    (gsplat_dir / "train_command.txt").write_text(cmd, encoding="utf-8")
    (gsplat_dir / "README").write_text(
        "Train stub only. This box has no CUDA.\n"
        "Copy work dir (frames_sel + sparse/0 + this cfg) to Alex's NVIDIA box and run train_command.txt.\n"
        "Locked: MCMC (or AbsGrad), packed, PPISP on. Skip FastGS.\n",
        encoding="utf-8",
    )
    print("[train_gsplat] stub written →", gsplat_dir)
    print("[train_gsplat] NOT executed (no CUDA on this box).")
    write_marker(
        work,
        "train_gsplat",
        "stub_nvidia_only",
        {"cfg": str(cfg_path), "command_file": str(gsplat_dir / "train_command.txt")},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_phase1.py",
        description="Phase-1 metric-splat: ffmpeg thin → HLoc → GLOMAP → COLMAP → AprilTag stub → gsplat stub",
    )
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "pilot.yaml"),
        help="Path to pilot.yaml (default: configs/pilot.yaml)",
    )
    p.add_argument(
        "--work",
        required=False,
        default=None,
        help="Work directory (e.g. work/francis_tanks_dryrun_run or work/room01)",
    )
    p.add_argument(
        "--video",
        default=None,
        help="Source video for extract_frames (room01). Ignored if --images-dir set.",
    )
    p.add_argument(
        "--images-dir",
        default=None,
        help="Existing images dir → skip extract; used as thin_frames source "
        "(Francis: work/francis_tanks_dryrun/images)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap selected frames (default: pilot selected_frames_room high end)",
    )
    p.add_argument(
        "--dry-run-francis",
        action="store_true",
        help="Convenience: work=work/francis_tanks_dryrun_run, images from francis_tanks_dryrun/images",
    )
    p.add_argument(
        "--force-train",
        action="store_true",
        help="Write train stub even if frustum_inspect did not pass (escape hatch)",
    )
    p.add_argument(
        "stages",
        nargs="*",
        default=["all"],
        help=f"Stages to run: all | {' | '.join(STAGE_NAMES)}",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = load_yaml(str(cfg_path))
    if not isinstance(cfg, dict):
        print("ERROR: config root must be a mapping", file=sys.stderr)
        return 2

    if args.dry_run_francis:
        work = REPO_ROOT / "work" / "francis_tanks_dryrun_run"
        images_dir = REPO_ROOT / "work" / "francis_tanks_dryrun" / "images"
        if not images_dir.is_dir():
            # symlink target
            images_dir = Path("/workspace/repos/data/Tanks/Francis/images")
    else:
        if not args.work:
            print("ERROR: --work is required (or pass --dry-run-francis)", file=sys.stderr)
            return 2
        work = resolve_path(args.work, REPO_ROOT)
        images_dir = Path(args.images_dir).resolve() if args.images_dir else None

    video = Path(args.video).resolve() if args.video else None

    ensure_work_layout(work)
    tools = detect_tools()
    write_json(work / "tools_detected.json", tools)
    print("[tools]", json.dumps({k: v for k, v in tools.items() if k != "hloc_error"}, indent=2))

    stages = args.stages
    if not stages or stages == ["all"] or "all" in stages:
        stages = list(STAGE_NAMES)

    unknown = [s for s in stages if s not in STAGE_NAMES]
    if unknown:
        print(f"ERROR: unknown stages: {unknown}", file=sys.stderr)
        return 2

    print(f"[run_phase1] work={work}")
    print(f"[run_phase1] config={cfg_path}")
    print(f"[run_phase1] stages={stages}")

    runners: dict[str, Callable[[], None]] = {
        "extract_frames": lambda: stage_extract_frames(
            work, cfg, video=video, images_dir=images_dir, tools=tools
        ),
        "thin_frames": lambda: stage_thin_frames(
            work, cfg, images_dir=images_dir, max_frames=args.max_frames
        ),
        "run_hloc": lambda: stage_run_hloc(work, cfg, tools),
        "run_glomap": lambda: stage_run_glomap(work, cfg, tools),
        "export_colmap": lambda: stage_export_colmap(work, cfg, tools),
        "frustum_inspect": lambda: stage_frustum_inspect(work, cfg, tools),
        "scale_apriltag": lambda: stage_scale_apriltag(work, cfg),
        "train_gsplat": lambda: stage_train_gsplat(work, cfg, force=args.force_train),
    }

    for name in stages:
        print(f"\n=== {name} ===")
        runners[name]()

    summary = {
        "ts": utc_now(),
        "work": str(work),
        "config": str(cfg_path),
        "stages": stages,
        "tools": {k: bool(v) if k != "hloc_error" else v for k, v in tools.items()},
        "markers_dir": str(work / "stage_markers"),
    }
    write_json(work / "run_summary.json", summary)
    print("\n[run_phase1] done. summary →", work / "run_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
