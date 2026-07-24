import argparse
from omegaconf import OmegaConf
import sys
import os
import json
from tqdm import tqdm
import torch
import cv2
sys.path = [
    './',
    '../external/OneFormer-Colab',
] + sys.path
from dataset_model import get_scene
from pathlib import Path
import numpy as np
from PIL import Image
from util import read_bounding_boxes_segmentations, crop_object
from scipy.ndimage import label as cc_label
from scipy.ndimage import binary_opening
from detectron2.structures import BoxMode
from batch_scripts.pipeline_loader import resolve_data_backend, setup_pipeline_loop


def _run_get(run_cfg, key, default=None):
    getter = getattr(run_cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    if isinstance(run_cfg, dict):
        return run_cfg.get(key, default)
    return getattr(run_cfg, key, default)


def _seg_get(seg_cfg, key, default=None):
    if seg_cfg is None:
        return default
    if isinstance(seg_cfg, dict):
        return seg_cfg.get(key, default)
    getter = getattr(seg_cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(seg_cfg, key, default)


def _normalize_label(label):
    label = str(label).strip().lower().replace(" ", "_").replace("-", "_")
    return label or "object"


# LA / common aliases → project taxonomy (data-pipeline-4d style).
DEFAULT_LABEL_MAP = {
    "sed_car": "vehicle_sedancar",
    "sedan": "vehicle_sedancar",
    "sedan_car": "vehicle_sedancar",
    "car": "vehicle_sedancar",
    "automobile": "vehicle_sedancar",
    "vehicle_sedancar": "vehicle_sedancar",
    "ltrucks": "vehicle_truck",
    "truck": "vehicle_truck",
    "lorry": "vehicle_truck",
    "pickup": "vehicle_truck",
    "vehicle_truck": "vehicle_truck",
    "bus": "vehicle_bus",
    "minibus": "vehicle_bus",
    "vehicle_bus": "vehicle_bus",
    "pedestrian": "person_person",
    "person": "person_person",
    "person_person": "person_person",
    "cyclist": "non-vehicle_bicycle",
    "bicycle": "non-vehicle_bicycle",
    "bike": "non-vehicle_bicycle",
    "non_vehicle_bicycle": "non-vehicle_bicycle",
    "non-vehicle_bicycle": "non-vehicle_bicycle",
    "tricyclist": "non-vehicle_tricycle",
    "tricycle": "non-vehicle_tricycle",
    "non_vehicle_tricycle": "non-vehicle_tricycle",
    "non-vehicle_tricycle": "non-vehicle_tricycle",
}


def _label_map_from_cfg(seg_cfg):
    raw = _seg_get(seg_cfg, "label_map", None)
    if raw is None:
        la_cfg = _seg_get(seg_cfg, "locateanything", {}) or {}
        if hasattr(la_cfg, "items") and not isinstance(la_cfg, dict):
            try:
                from omegaconf import OmegaConf

                la_cfg = OmegaConf.to_container(la_cfg, resolve=True) or {}
            except Exception:
                la_cfg = {}
        if isinstance(la_cfg, dict):
            raw = la_cfg.get("label_map")
    mapping = dict(DEFAULT_LABEL_MAP)
    if raw:
        if hasattr(raw, "items") and not isinstance(raw, dict):
            try:
                from omegaconf import OmegaConf

                raw = OmegaConf.to_container(raw, resolve=True) or {}
            except Exception:
                raw = dict(raw)
        for k, v in dict(raw).items():
            mapping[_normalize_label(k)] = str(v).strip()
    return mapping


def _map_detection_label(label, label_map=None):
    key = _normalize_label(label)
    mapping = label_map or DEFAULT_LABEL_MAP
    return mapping.get(key, key)


def _allowed_categories(seg_cfg):
    allowed = _seg_get(seg_cfg, "allowed_categories", None)
    if allowed is None:
        return None
    return {_normalize_label(c) for c in list(allowed)}


def _oneformer_hint_from_label(label):
    """Map taxonomy / LA labels to OneFormer ADE hints."""
    key = _normalize_label(label)
    if key in ("vehicle_sedancar", "sedan", "sed_car", "car", "automobile"):
        return "car"
    if key in ("vehicle_truck", "ltrucks", "truck", "lorry", "pickup"):
        return "truck"
    if key in ("vehicle_bus", "bus", "minibus"):
        return "bus"
    if key in ("person_person", "person", "pedestrian"):
        return "person"
    if key in ("non-vehicle_bicycle", "non_vehicle_bicycle", "bicycle", "cyclist", "bike"):
        return "bicycle"
    if key in ("non-vehicle_tricycle", "non_vehicle_tricycle", "tricycle", "tricyclist"):
        return "bicycle"  # closest ADE thing class
    return key


_ONEFORMER_CACHE = {}


def _clear_oneformer_cache():
    _ONEFORMER_CACHE.clear()


def _get_oneformer(device):
    if device not in _ONEFORMER_CACHE:
        try:
            from model_wrappers import initialize_oneformer

            _ONEFORMER_CACHE[device] = initialize_oneformer(device)
        except Exception as exc:
            print(f"OneFormer unavailable ({exc}); using box masks for crop refinement.")
            _ONEFORMER_CACHE[device] = None
    return _ONEFORMER_CACHE[device]


def _letterbox_for_oneformer(image_pil, canvas_size=1024):
    """Pad crop to a square canvas so NATTEN sees balanced spatial axes."""
    w, h = image_pil.size
    scale = canvas_size / float(max(w, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = image_pil.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (canvas_size, canvas_size), (0, 0, 0))
    pad_x, pad_y = (canvas_size - nw) // 2, (canvas_size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, (scale, pad_x, pad_y, nw, nh)


def _unmap_letterbox_sem(sem_canvas, orig_size, layout):
    scale, pad_x, pad_y, nw, nh = layout
    sem_crop = sem_canvas[pad_y : pad_y + nh, pad_x : pad_x + nw]
    return np.array(
        Image.fromarray(sem_crop.astype(np.uint8)).resize(orig_size, Image.NEAREST)
    )


def _oneformer_semantic_map(image_pil, device, canvas_size=1024):
    """Run OneFormer semantic segmentation; return label map aligned to image_pil size."""
    oneformer = _get_oneformer(device)
    if oneformer is None:
        return None, None, None
    predictor, metadata, thing_classes_ids = oneformer
    infer_pil, layout = _letterbox_for_oneformer(image_pil, canvas_size=canvas_size)
    image_np = np.array(infer_pil)[:, :, ::-1]
    predictions = predictor(image_np, "semantic")["sem_seg"].argmax(dim=0).cpu().numpy()
    sem = _unmap_letterbox_sem(predictions, image_pil.size, layout)
    return sem, metadata, thing_classes_ids


def _class_ids_for_label_hint(label_hint, stuff_classes, thing_classes_ids):
    hint = _normalize_label(label_hint)
    synonyms = {
        "person": ("person", "man", "woman", "boy", "girl", "pedestrian", "rider"),
        "car": ("car", "automobile", "jeep", "limousine", "taxi", "ambulance", "minivan"),
        "truck": ("truck", "pickup", "lorry"),
        "bus": ("bus", "minibus"),
        "motorcycle": ("motorcycle", "motorbike", "bike"),
        "bicycle": ("bicycle", "bike", "cycle"),
        "traffic_light": ("traffic", "light", "signal"),
        # taxonomy aliases
        "vehicle_sedancar": ("car", "automobile", "jeep", "limousine", "taxi", "ambulance", "minivan"),
        "vehicle_truck": ("truck", "pickup", "lorry"),
        "vehicle_bus": ("bus", "minibus"),
        "person_person": ("person", "man", "woman", "boy", "girl", "pedestrian", "rider"),
        "non-vehicle_bicycle": ("bicycle", "bike", "cycle"),
        "non_vehicle_bicycle": ("bicycle", "bike", "cycle"),
        "non-vehicle_tricycle": ("bicycle", "bike", "cycle"),
        "non_vehicle_tricycle": ("bicycle", "bike", "cycle"),
    }
    mapped = _normalize_label(_oneformer_hint_from_label(hint))
    keys = synonyms.get(mapped, synonyms.get(hint, (mapped,)))
    ids = []
    for tid in thing_classes_ids:
        if tid < 0 or tid >= len(stuff_classes):
            continue
        primary = _normalize_label(str(stuff_classes[tid]).split(",")[0])
        if primary in keys or any(k in primary or primary in k for k in keys):
            ids.append(tid)
    return ids


def _padded_xyxy(box, width, height, pad_ratio):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * pad_ratio, bh * pad_ratio
    return [
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(width, int(round(x2 + pad_x))),
        min(height, int(round(y2 + pad_y))),
    ]


def _oneformer_mask_in_box(image_pil, box_xyxy, label_hint, device, pad_ratio=0.15, min_component_area=400):
    """Semantic segmentation on a LocateAnything crop; mask pasted to full-image coordinates."""
    w, h = image_pil.size
    px1, py1, px2, py2 = _padded_xyxy(box_xyxy, w, h, pad_ratio)
    if px2 - px1 < 8 or py2 - py1 < 8:
        return None

    crop = image_pil.crop((px1, py1, px2, py2))
    sem, metadata, thing_classes_ids = _oneformer_semantic_map(crop, device)
    if sem is None:
        return None
    class_ids = _class_ids_for_label_hint(label_hint, metadata.stuff_classes, thing_classes_ids)
    if not class_ids:
        class_ids = list(thing_classes_ids)

    binary = np.isin(sem, class_ids).astype(np.uint8)
    if binary.sum() < min_component_area:
        return None

    ix1 = int(round(box_xyxy[0])) - px1
    iy1 = int(round(box_xyxy[1])) - py1
    ix2 = int(round(box_xyxy[2])) - px1
    iy2 = int(round(box_xyxy[3])) - py1
    inner = np.zeros_like(binary, dtype=bool)
    cw, ch = binary.shape[1], binary.shape[0]
    inner[
        max(0, iy1) : min(ch, iy2),
        max(0, ix1) : min(cw, ix2),
    ] = True

    comps, n_comp = cc_label(binary)
    best_mask, best_score = None, -1
    for comp_id in range(1, n_comp + 1):
        m = comps == comp_id
        if m.sum() < min_component_area:
            continue
        overlap = (m & inner).sum()
        score = overlap if overlap > 0 else m.sum() * 0.25
        if score > best_score:
            best_score, best_mask = score, m

    if best_mask is None:
        return None

    full = np.zeros((h, w), dtype=bool)
    full[py1:py2, px1:px2] = best_mask
    return full


def _locateanything_boxes_then_crop_oneformer(image_pil, image_np, seg_cfg, only_foreground=True):
    from integrations.locateanything.detect import detect_boxes

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    la_cfg = _seg_get(seg_cfg, "locateanything", {}) or {}
    if hasattr(la_cfg, "items") and not isinstance(la_cfg, dict):
        la_cfg = OmegaConf.to_container(la_cfg, resolve=True) or {}
    elif not isinstance(la_cfg, dict):
        la_cfg = {}

    categories = la_cfg.get("categories")
    if categories is not None:
        categories = list(categories)

    pad_ratio = float(_seg_get(seg_cfg, "crop_refinement_pad", 0.15))
    min_component = int(_seg_get(seg_cfg, "crop_refinement_min_area", 400))

    labels, boxes = detect_boxes(
        image_pil,
        categories=categories,
        device=device,
        model_path=la_cfg.get("model_path"),
        generation_mode=la_cfg.get("generation_mode", "hybrid"),
        # Allow LA raw names; map + filter to taxonomy below.
        allowed_categories=None,
        min_box_area=int(la_cfg.get("min_mask_area", 1600)),
    )
    from integrations.locateanything.detect import release_worker

    release_worker()
    _clear_oneformer_cache()
    if not boxes:
        h, w = image_np.shape[:2]
        return np.zeros((0, h, w), dtype=bool), [], []

    label_map = _label_map_from_cfg(seg_cfg)
    allowed = _allowed_categories(seg_cfg)
    kept_masks, kept_labels, kept_boxes = [], [], []
    h, w = image_np.shape[:2]

    for box, label in zip(boxes, labels):
        mapped = _map_detection_label(label, label_map)
        if allowed is not None and _normalize_label(mapped) not in allowed:
            continue
        of_hint = _oneformer_hint_from_label(mapped)
        mask = _oneformer_mask_in_box(
            image_pil, box, of_hint, device,
            pad_ratio=pad_ratio,
            min_component_area=min_component,
        )
        if mask is None or mask.sum() < min_component:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            mask = np.zeros((h, w), dtype=bool)
            mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
            if mask.sum() < min_component:
                continue
            print(f"OneFormer mask empty for {mapped}; using box fallback.")
        kept_masks.append(mask)
        kept_labels.append(mapped)
        kept_boxes.append(box)

    if not kept_masks:
        return np.zeros((0, h, w), dtype=bool), [], []

    return np.stack(kept_masks, axis=0), kept_labels, kept_boxes


def _oneformer_instance_masks(image_pil, device):
    sem, metadata, thing_classes_ids = _oneformer_semantic_map(image_pil, device)
    h, w = sem.shape
    masks, labels = [], []
    stuff_classes = list(metadata.stuff_classes)
    for tid in thing_classes_ids:
        if tid < 0 or tid >= len(stuff_classes):
            continue
        binary = (sem == tid).astype(np.uint8)
        if binary.sum() < 100:
            continue
        comps, n_comp = cc_label(binary)
        name = _normalize_label(stuff_classes[tid])
        for comp_id in range(1, n_comp + 1):
            m = comps == comp_id
            if m.sum() < 400:
                continue
            masks.append(m)
            labels.append(name)
    if not masks:
        return np.zeros((0, h, w), dtype=bool), []
    return np.stack(masks, axis=0), labels


def _model_segment_image(image_pil, image_np, seg_cfg, only_foreground=True):
    """Run image-based detection/segmentation; return masks, labels, xyxy boxes."""
    from model_wrappers import (
        run_clipseg,
        run_entityv2,
        run_locateanything,
        run_oneformer,
        run_ovsam,
    )

    holistic = _seg_get(seg_cfg, "holistic", "entityv2")
    fg_bg = _seg_get(seg_cfg, "fg_bg", "clipseg")
    tagger = _seg_get(seg_cfg, "tagger", "ovsam")
    labels = None

    if holistic == "entityv2":
        try:
            masks = run_entityv2(image_np)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "EntityV2/CropFormer is not installed. "
                "Install external/detectron2 CropFormer or set segmentation.holistic to "
                "'oneformer' or 'locateanything'."
            ) from exc
        masks = np.asarray(masks)
    elif holistic == "oneformer":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        masks, labels = _oneformer_instance_masks(image_pil, device)
    elif holistic == "locateanything":
        crop_refinement = _seg_get(seg_cfg, "crop_refinement", None)
        if crop_refinement == "oneformer":
            masks, labels, bboxes = _locateanything_boxes_then_crop_oneformer(
                image_pil, image_np, seg_cfg, only_foreground=only_foreground,
            )
            return masks, labels, bboxes
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        la_cfg = _seg_get(seg_cfg, "locateanything", {}) or {}
        if hasattr(la_cfg, "items") and not isinstance(la_cfg, dict):
            la_cfg = OmegaConf.to_container(la_cfg, resolve=True) or {}
        elif not isinstance(la_cfg, dict):
            la_cfg = {}
        categories = la_cfg.get("categories")
        if categories is not None:
            categories = list(categories)
        masks, labels, _boxes = run_locateanything(
            image_pil,
            categories=categories,
            device=device,
            model_path=la_cfg.get("model_path"),
            generation_mode=la_cfg.get("generation_mode", "hybrid"),
            allowed_categories=None,
            min_mask_area=int(la_cfg.get("min_mask_area", 1600)),
        )
        label_map = _label_map_from_cfg(seg_cfg)
        allowed = _allowed_categories(seg_cfg)
        if labels:
            mapped_labels = []
            keep_idx = []
            for i, lab in enumerate(labels):
                mapped = _map_detection_label(lab, label_map)
                if allowed is not None and _normalize_label(mapped) not in allowed:
                    continue
                mapped_labels.append(mapped)
                keep_idx.append(i)
            if keep_idx:
                masks = np.asarray(masks)[keep_idx]
                labels = mapped_labels
                _boxes = [_boxes[i] for i in keep_idx] if _boxes else _boxes
            else:
                h, w = image_np.shape[:2]
                masks = np.zeros((0, h, w), dtype=bool)
                labels = []
                _boxes = []
    else:
        raise ValueError(
            f"Unsupported segmentation.holistic='{holistic}'. "
            "Use entityv2, oneformer, or locateanything."
        )

    if len(masks) == 0:
        return np.zeros((0, image_np.shape[0], image_np.shape[1]), dtype=bool), [], []

    if only_foreground and fg_bg == "clipseg":
        keep_ids, _ = run_clipseg(image_pil, masks)
        masks = masks[keep_ids]
        if labels is not None:
            labels = [labels[i] for i in keep_ids]
    elif only_foreground and fg_bg == "oneformer":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        keep_ids, _ = run_oneformer(image_pil, masks, device)
        masks = masks[keep_ids]
        if labels is not None:
            labels = [labels[i] for i in keep_ids]

    if labels is None:
        if tagger == "oneformer":
            _, labels = _oneformer_instance_masks(image_pil, "cuda:0" if torch.cuda.is_available() else "cpu")
            if len(labels) != len(masks):
                labels = ["object"] * len(masks)
        elif tagger == "ovsam":
            try:
                labels, _ = run_ovsam(image_pil, masks)
            except Exception:
                labels = ["object"] * len(masks)
        else:
            labels = ["object"] * len(masks)

    allowed = _allowed_categories(seg_cfg)
    bboxes_xyxy = []
    kept_masks = []
    kept_labels = []
    for mask, label in zip(masks, labels):
        label = _normalize_label(label)
        if allowed is not None and label not in allowed:
            continue
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        if w < 2 or h < 2:
            continue
        bboxes_xyxy.append([x, y, x + w, y + h])
        kept_masks.append(mask)
        kept_labels.append(label)

    if not kept_masks:
        h, w = image_np.shape[:2]
        return np.zeros((0, h, w), dtype=bool), [], []

    return np.stack(kept_masks, axis=0), kept_labels, bboxes_xyxy


def _resolve_detection_source(opt, data_backend):
    source = _run_get(opt.run, "detection_source", None)
    if source is not None:
        return source
    # bad_case dumps usually have empty ann_infos → must use model detection.
    if data_backend == "bad_case_pkl":
        return "model"
    if data_backend == "py123d":
        return "gt"
    return "gt"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to the yaml config file", default='configs/image.yaml', type=str)
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--start_index', type=int, default=0, help='Object index to start processing')
    parser.add_argument('--end_index', type=int, default=-1, help='Object index to end processing (-1 = all)')
    parser.add_argument("--split", help="split", default="val", type=str)
    parser.add_argument("--save_dir", help="save directory", default="../experimental_results/COCO/", type=str)
    parser.add_argument(
        "--data_backend",
        choices=["coco", "py123d", "bad_case_pkl"],
        default=None,
        help="coco / py123d / bad_case_pkl scene dirs under save_dir/split",
    )

    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    data_backend = resolve_data_backend(args, opt)
    detection_source = _resolve_detection_source(opt, data_backend)
    use_model_detection = detection_source == "model"
    require_annotations = not use_model_detection

    data_backend, split, loader, indices = setup_pipeline_loop(
        args,
        opt,
        require_files=("input.png",),
        require_annotations=require_annotations,
    )
    crop_size = 512
    seg_cfg = _run_get(opt.run, "segmentation", {})

    for i in tqdm(indices):
        scene_entry = loader.get_scene_by_index(i)
        output_dir = scene_entry["scene_dir"]
        opt.scene.attributes.img_path = scene_entry["image_path"]
        scene = get_scene(opt.scene.type, opt.scene.attributes)
        img_name = scene_entry["file_name"]

        out_dir = Path(output_dir)
        print(f"Saving to {out_dir}")
        out_dir.mkdir(exist_ok=True, parents=True)
        (out_dir / "crops").mkdir(exist_ok=True)
        (out_dir / "object_space").mkdir(exist_ok=True)
        (out_dir / "reconstruction").mkdir(exist_ok=True)

        enhanced_path = out_dir / "enhanced" / "input.png"
        if not enhanced_path.exists():
            print(f"Missing {enhanced_path}; run enhance.py first")
            continue
        enhanced_image = Image.open(enhanced_path)
        scene.image_pil = enhanced_image.convert("RGB")
        scene.image_np = np.array(enhanced_image)
        enh_h, enh_w = scene.image_np.shape[:2]

        # Annotations / depth live in native (input.png) resolution. InvSR enhance is typically 4x.
        native_path = out_dir / "input.png"
        if native_path.exists():
            native_w, native_h = Image.open(native_path).size
        else:
            native_w, native_h = enh_w, enh_h
        scale_x = enh_w / float(native_w)
        scale_y = enh_h / float(native_h)
        if abs(scale_x - scale_y) > 1e-3:
            print(
                f"Warning: non-uniform enhance scale sx={scale_x:.3f} sy={scale_y:.3f}; "
                "crop_params assume isotropic scale"
            )
        enhance_scale = float(0.5 * (scale_x + scale_y))

        if use_model_detection:
            masks, instance_labels, bboxes = _model_segment_image(
                scene.image_pil, scene.image_np, seg_cfg,
                only_foreground=_run_get(opt.run, "only_foreground", True),
            )
            if len(masks) == 0:
                print(f"No model-detected objects in {img_name}")
                continue
            object_ids = np.arange(len(masks))
            # Model masks/boxes are already in enhanced image coordinates.
            bboxes = np.array(bboxes, dtype=np.float32)
            bboxes_native = bboxes / np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
        else:
            annotations = scene_entry["annotations"]
            if not annotations:
                print(f"No annotations found for {img_name}")
                continue
            # GT boxes/polygons are stored in native input.png coordinates.
            bboxes, masks, object_ids, instance_labels = read_bounding_boxes_segmentations(
                annotations, (native_w, native_h)
            )
            if len(masks[object_ids]) == 0:
                print(f"No valid objects found in {img_name}")
                continue
            bboxes = BoxMode.convert(np.array(bboxes), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            bboxes_native = np.array(bboxes, dtype=np.float32)
            # Upsample masks/boxes to the enhanced canvas for cropping.
            scaled_masks = []
            for mask in masks:
                scaled_masks.append(
                    cv2.resize(
                        mask.astype(np.uint8),
                        (enh_w, enh_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                )
            masks = np.array(scaled_masks)
            bboxes = bboxes_native * np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)

        selected_bboxes = []
        for j in range(len(masks) - 1, -1, -1):
            if use_model_detection:
                label = instance_labels[j]
                bbox_native = bboxes_native[j]
            else:
                label = instance_labels[object_ids[j]]
                bbox_native = bboxes_native[object_ids[j]]
            label = label.replace(" (", ", ").replace(")", "")
            obj_id = f"{j}_{label.replace(' ', '_')}"

            mask = binary_opening(masks[j], np.ones((7, 7)))
            if mask.sum() < 6400:
                print(f"Skipped too small object: {obj_id}")
                continue
            ys, xs = np.where(mask)
            if ys.size == 0 or xs.size == 0:
                print(f"Skipped empty mask after opening: {obj_id}")
                continue
            bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
            if bh < 8 or bw < 8:
                print(f"Skipped degenerate bbox: {obj_id} ({bw}x{bh})")
                continue
            selected_bboxes.append(
                bbox_native.tolist() if hasattr(bbox_native, "tolist") else list(bbox_native)
            )
            crop_path = out_dir / "crops" / f"{obj_id}_reproj.png"
            crop_params_path = out_dir / "crops" / f"{obj_id}_crop_params.npy"
            if not crop_path.exists() or not crop_params_path.exists():
                crop_mask = mask
                if crop_mask.shape[0] != enh_h or crop_mask.shape[1] != enh_w:
                    crop_mask = cv2.resize(
                        crop_mask.astype(np.uint8),
                        (enh_w, enh_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                crop, crop_params = crop_object(scene.image_np, crop_mask, crop_size)
                crop.save(crop_path)
                # whole.py restores masks onto native-resolution depth_map.
                crop_params = np.array(
                    [
                        crop_params[0] / enhance_scale,
                        crop_params[1] / enhance_scale,
                        crop_params[2] * enhance_scale,
                    ],
                    dtype=np.float64,
                )
                np.save(crop_params_path, crop_params)
        with open(out_dir / "bboxes.json", "w") as f:
            json.dump(np.array(selected_bboxes).tolist(), f)
