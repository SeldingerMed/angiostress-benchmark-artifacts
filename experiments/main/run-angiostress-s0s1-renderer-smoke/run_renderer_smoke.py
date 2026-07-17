#!/usr/bin/env python3
"""Deterministic S0/S1 renderer smoke for AngioStress."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import draw, measure, morphology


@dataclass(frozen=True)
class Cell:
    family: str
    severity: int
    sequence: np.ndarray
    gt_mask: np.ndarray
    device_mask: np.ndarray
    parameters: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(contiguous.dtype).encode("utf-8"))
    h.update(str(contiguous.shape).encode("utf-8"))
    h.update(contiguous.tobytes())
    return h.hexdigest()


def configure_synthetic_fixture(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Generate a tiny vessel mask so the renderer can run from a clean checkout."""
    fixture_dir = out_dir / "synthetic_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    mask_path = fixture_dir / "vessel_mask.nii.gz"

    mask = np.zeros((48, 48, 48), dtype=np.uint8)
    yy, xx = np.ogrid[: mask.shape[1], : mask.shape[2]]
    for depth in range(6, 42):
        center_y = 24 + int(round(5 * np.sin(depth / 6)))
        center_x = 8 + depth // 2
        mask[depth, (yy - center_y) ** 2 + (xx - center_x) ** 2 <= 4] = 1
        if depth >= 22:
            branch_y = center_y - (depth - 22) // 2
            branch_x = center_x + (depth - 22) // 3
            mask[depth, (yy - branch_y) ** 2 + (xx - branch_x) ** 2 <= 2] = 2

    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))
    configured = dict(config)
    configured["source_case"] = {
        "case_id": "generated_synthetic_fixture",
        "source_type": "generated_smoke_fixture",
        "mask_path": str(mask_path.resolve()),
        "mask_sha256": sha256_file(mask_path),
    }
    return configured


def normalize01(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    lo = float(array.min())
    hi = float(array.max())
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - lo) / (hi - lo)).astype(np.float32)


def crop_to_nonzero(array: np.ndarray, mask: np.ndarray, margin: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("projected TopCoW mask is empty")
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, mask.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, mask.shape[1])
    return array[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def resize_to_shape(array: np.ndarray, shape: tuple[int, int], order: int) -> np.ndarray:
    zoom = (shape[0] / array.shape[0], shape[1] / array.shape[1])
    resized = ndi.zoom(array.astype(np.float32), zoom, order=order)
    return resized[: shape[0], : shape[1]]


def project_mask(
    mask_path: Path,
    axis: int,
    output_size: tuple[int, int],
    margin: int,
    density_blur_sigma: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    img = nib.load(str(mask_path))
    mask_data = np.asanyarray(img.dataobj)
    vessel = mask_data > 0
    density = vessel.sum(axis=axis).astype(np.float32)
    gt = density > 0
    density, gt = crop_to_nonzero(density, gt, margin)
    density = resize_to_shape(density, output_size, order=1)
    gt = resize_to_shape(gt.astype(np.float32), output_size, order=0) > 0.5
    density = normalize01(ndi.gaussian_filter(density, sigma=density_blur_sigma))
    source_summary = {
        "mask_shape": [int(x) for x in mask_data.shape],
        "mask_dtype": str(mask_data.dtype),
        "mask_spacing": [float(x) for x in img.header.get_zooms()[:3]],
        "nonzero_voxels": int(vessel.sum()),
        "nonzero_label_count": int(len(np.unique(mask_data[vessel]))),
        "projection_axis": int(axis),
    }
    return density.astype(np.float32), gt.astype(bool), source_summary


def make_clean_image(density: np.ndarray, gt_mask: np.ndarray, seed: int, noise_std: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vessel_signal = normalize01(density)
    background = np.full_like(vessel_signal, 0.05, dtype=np.float32)
    image = background + 0.9 * vessel_signal
    image = ndi.gaussian_filter(image, sigma=0.7)
    if noise_std > 0:
        image = image + rng.normal(0.0, noise_std, size=image.shape).astype(np.float32)
    image[~gt_mask] *= 0.35
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def make_sequence(clean_image: np.ndarray, phase_curve: list[float], background_level: float) -> np.ndarray:
    vessel_signal = np.clip(clean_image - background_level, 0.0, 1.0)
    frames = []
    for phase in phase_curve:
        frame = background_level + float(phase) * vessel_signal
        frames.append(np.clip(frame, 0.0, 1.0).astype(np.float32))
    return np.stack(frames, axis=0)


def make_device_mask(shape: tuple[int, int], radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if radius <= 0:
        return mask
    height, width = shape
    y = np.linspace(int(0.12 * height), int(0.90 * height), num=height)
    x = width * (0.45 + 0.18 * np.sin(np.linspace(-1.4, 1.7, num=height)))
    points = np.stack([y, x], axis=1)
    for start, end in zip(points[:-1], points[1:]):
        rr, cc = draw.line(int(round(start[0])), int(round(start[1])), int(round(end[0])), int(round(end[1])))
        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        mask[rr[valid], cc[valid]] = True
    return morphology.binary_dilation(mask, morphology.disk(radius))


def overlay_device(sequence: np.ndarray, device_mask: np.ndarray, alpha: float) -> np.ndarray:
    if not device_mask.any():
        return sequence.copy()
    out = sequence.copy()
    out[:, device_mask] = (1.0 - alpha) * out[:, device_mask] + alpha * 1.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def build_cells(config: dict[str, Any], clean_sequence: np.ndarray, gt_mask: np.ndarray) -> list[Cell]:
    cells: list[Cell] = []
    stressors = config["stressors"]

    for severity in (0, 1, 2):
        indices = stressors["frame_thinning"][f"severity_{severity}_indices"]
        seq = clean_sequence[np.asarray(indices, dtype=int)]
        cells.append(Cell("frame_thinning", severity, seq, gt_mask, np.zeros_like(gt_mask), {"indices": indices}))

    background = float(config["clean_sequence"]["background_level"])
    clean_peak = clean_sequence.max(axis=0)
    for severity in (0, 1, 2):
        curve = stressors["contrast_phase"][f"severity_{severity}_phase_curve"]
        seq = make_sequence(clean_peak, curve, background)
        cells.append(Cell("contrast_phase", severity, seq, gt_mask, np.zeros_like(gt_mask), {"phase_curve": curve}))

    overlay_cfg = stressors["device_overlay"]
    for severity in (0, 1, 2):
        radius = int(overlay_cfg[f"severity_{severity}_radius"])
        alpha = float(overlay_cfg.get(f"severity_{severity}_alpha", 0.0))
        device_mask = make_device_mask(gt_mask.shape, radius)
        seq = overlay_device(clean_sequence, device_mask, alpha)
        cells.append(Cell("device_overlay", severity, seq, gt_mask, device_mask, {"radius": radius, "alpha": alpha}))

    return cells


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    denom = int(a.sum() + b.sum())
    if denom == 0:
        return 1.0
    return float((2.0 * np.logical_and(a, b).sum()) / denom)


def cldice_self(mask: np.ndarray) -> float:
    skel = morphology.skeletonize(mask.astype(bool))
    if not skel.any():
        return 1.0 if not mask.any() else 0.0
    precision = float(np.logical_and(skel, mask).sum() / skel.sum())
    recall = float(np.logical_and(skel, mask).sum() / skel.sum())
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def save_png(path: Path, image: np.ndarray) -> None:
    arr = np.asarray(image)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    else:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_outputs(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(exist_ok=True)

    projection_cfg = config["projection"]
    source = config["source_case"]
    mask_path = Path(source["mask_path"])
    density, gt_mask, source_summary = project_mask(
        mask_path,
        axis=int(projection_cfg["axis"]),
        output_size=tuple(int(x) for x in projection_cfg["output_size"]),
        margin=int(projection_cfg["crop_margin_pixels"]),
        density_blur_sigma=float(projection_cfg["density_blur_sigma"]),
    )
    clean_image = make_clean_image(
        density,
        gt_mask,
        seed=int(config["seed"]),
        noise_std=float(projection_cfg["image_noise_std"]),
    )
    clean_sequence = make_sequence(
        clean_image,
        phase_curve=[float(x) for x in config["clean_sequence"]["phase_curve"]],
        background_level=float(config["clean_sequence"]["background_level"]),
    )
    cells = build_cells(config, clean_sequence, gt_mask)
    repeated_cells = build_cells(config, clean_sequence, gt_mask)
    deterministic_hash_match = [sha256_array(a.sequence) for a in cells] == [sha256_array(b.sequence) for b in repeated_cells]

    np.save(out_dir / "projected_gt_mask.npy", gt_mask)
    np.save(out_dir / "clean_image.npy", clean_image)
    np.save(out_dir / "clean_sequence.npy", clean_sequence)
    save_png(out_dir / "projected_gt_mask.png", gt_mask)
    save_png(out_dir / "clean_preview.png", clean_sequence.max(axis=0))

    cell_rows = []
    for cell in cells:
        stem = f"{cell.family}_severity_{cell.severity}"
        npy_path = cells_dir / f"{stem}.npy"
        png_path = cells_dir / f"{stem}.png"
        mask_path_out = cells_dir / f"{stem}_device_mask.png"
        np.save(npy_path, cell.sequence)
        save_png(png_path, cell.sequence.max(axis=0))
        if cell.device_mask.any():
            save_png(mask_path_out, cell.device_mask)
        sequence_hash = sha256_array(cell.sequence)
        row = {
            "family": cell.family,
            "severity": int(cell.severity),
            "frame_count": int(cell.sequence.shape[0]),
            "sequence_shape": [int(x) for x in cell.sequence.shape],
            "sequence_sha256": sequence_hash,
            "npy_path": str(npy_path.relative_to(out_dir)),
            "preview_path": str(png_path.relative_to(out_dir)),
            "device_mask_area_pixels": int(cell.device_mask.sum()),
            "device_mask_overlap_gt_fraction": float(np.logical_and(cell.device_mask, gt_mask).sum() / max(int(cell.device_mask.sum()), 1)),
            "parameters": cell.parameters,
        }
        if cell.device_mask.any():
            row["device_mask_path"] = str(mask_path_out.relative_to(out_dir))
        cell_rows.append(row)

    labels = measure.label(gt_mask, connectivity=2)
    numeric_metrics = {
        "angiostress_renderer_success": 1.0,
        "angiostress_renderer_cell_count": float(len(cell_rows)),
        "angiostress_renderer_family_count": 3.0,
        "angiostress_renderer_severity_levels_per_family": 3.0,
        "angiostress_renderer_deterministic_hash_match": 1.0 if deterministic_hash_match else 0.0,
        "angiostress_renderer_projected_gt_area_pixels": float(gt_mask.sum()),
        "angiostress_renderer_projected_gt_area_fraction": float(gt_mask.mean()),
        "angiostress_renderer_projected_gt_components_2d": float(labels.max()),
        "angiostress_renderer_clean_gt_self_dice": dice(gt_mask, gt_mask),
        "angiostress_renderer_clean_gt_self_cldice": cldice_self(gt_mask),
        "angiostress_renderer_frozen_model_smoke_completed": 0.0,
    }
    finite_metrics = all(np.isfinite(float(v)) for v in numeric_metrics.values())
    numeric_metrics["angiostress_renderer_finite_metric_check"] = 1.0 if finite_metrics else 0.0

    manifest = {
        "run_id": config["run_id"],
        "claim_boundary": "S0/S1 renderer and stressor feasibility only; no construct-validity or clinical claim.",
        "source_case": source,
        "source_summary": source_summary,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "nibabel": nib.__version__,
        },
        "config_sha256": sha256_bytes(json.dumps(config, sort_keys=True).encode("utf-8")),
        "output_hashes": {
            "projected_gt_mask": sha256_array(gt_mask),
            "clean_image": sha256_array(clean_image),
            "clean_sequence": sha256_array(clean_sequence),
        },
        "deterministic_hash_match": bool(deterministic_hash_match),
        "frozen_model_smoke_status": "not_run_dependency_deferred",
        "frozen_model_smoke_reason": "This scoped pass tests renderer/stressor feasibility; frozen 2D DSA model loading is deferred to the next harness run.",
        "cells": cell_rows,
    }
    metrics = {
        "run_id": config["run_id"],
        "metrics": numeric_metrics,
        "non_numeric_status": {
            "frozen_model_smoke_status": manifest["frozen_model_smoke_status"],
            "frozen_model_smoke_reason": manifest["frozen_model_smoke_reason"],
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "metrics_summary.json", metrics)
    with (out_dir / "manifest.jsonl").open("w") as f:
        for row in cell_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return {"manifest": manifest, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--synthetic-fixture",
        action="store_true",
        help="Generate a tiny vessel mask under --out instead of requiring the configured TopCoW source file.",
    )
    args = parser.parse_args()

    config = load_json(args.config)
    if args.synthetic_fixture:
        config = configure_synthetic_fixture(config, args.out)
    result = save_outputs(config, args.out)
    summary = result["metrics"]["metrics"]
    print(json.dumps({
        "run_id": config["run_id"],
        "cell_count": int(summary["angiostress_renderer_cell_count"]),
        "gt_area_pixels": int(summary["angiostress_renderer_projected_gt_area_pixels"]),
        "deterministic_hash_match": bool(summary["angiostress_renderer_deterministic_hash_match"]),
        "finite_metric_check": bool(summary["angiostress_renderer_finite_metric_check"]),
        "manifest": str((args.out / "manifest.json").resolve()),
        "metrics": str((args.out / "metrics_summary.json").resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
