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


def _allowed_categories(seg_cfg):
    allowed = _seg_get(seg_cfg, "allowed_categories", None)
    if not allowed:
        return None
    return {_normalize_label(x) for x in allowed}


_ONEFORMER_CACHE = {}


def _clear_oneformer_cache():
    _ONEFORMER_CACHE.clear()


def _get_oneformer(device):
    if device not in _ONEFORMER_CACHE:
        from model_wrappers import initialize_oneformer

        _ONEFORMER_CACHE[device] = initialize_oneformer(device)
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
    predictor, metadata, thing_classes_ids = _get_oneformer(device)
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
    }
    keys = synonyms.get(hint, (hint,))
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

    n_comp, comps = cc_label(binary)
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
        allowed_categories=_seg_get(seg_cfg, "allowed_categories", None),
        min_box_area=int(la_cfg.get("min_mask_area", 1600)),
    )
    from integrations.locateanything.detect import release_worker

    release_worker()
    _clear_oneformer_cache()
    if not boxes:
        h, w = image_np.shape[:2]
        return np.zeros((0, h, w), dtype=bool), [], []

    allowed = _allowed_categories(seg_cfg)
    kept_masks, kept_labels, kept_boxes = [], [], []
    h, w = image_np.shape[:2]

    for box, label in zip(boxes, labels):
        label = _normalize_label(label)
        if allowed is not None and label not in allowed:
            continue
        mask = _oneformer_mask_in_box(
            image_pil, box, label, device,
            pad_ratio=pad_ratio,
            min_component_area=min_component,
        )
        if mask is None or mask.sum() < min_component:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            mask = np.zeros((h, w), dtype=bool)
            mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
            if mask.sum() < min_component:
                continue
            print(f"OneFormer mask empty for {label}; using box fallback.")
        kept_masks.append(mask)
        kept_labels.append(label)
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
        n_comp, comps = cc_label(binary)
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
            allowed_categories=_seg_get(seg_cfg, "allowed_categories", None),
            min_mask_area=int(la_cfg.get("min_mask_area", 1600)),
        )
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
        choices=["coco", "py123d"],
        default=None,
        help="coco: COCONUT annotations; py123d: scene dirs from depth.py --depth_source py123d",
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

        if use_model_detection:
            masks, instance_labels, bboxes = _model_segment_image(
                scene.image_pil, scene.image_np, seg_cfg,
                only_foreground=_run_get(opt.run, "only_foreground", True),
            )
            if len(masks) == 0:
                print(f"No model-detected objects in {img_name}")
                continue
            object_ids = np.arange(len(masks))
        else:
            annotations = scene_entry["annotations"]
            if not annotations:
                print(f"No annotations found for {img_name}")
                continue
            bboxes, masks, object_ids, instance_labels = read_bounding_boxes_segmentations(
                annotations, scene.image_pil.size
            )
            if len(masks[object_ids]) == 0:
                print(f"No valid objects found in {img_name}")
                continue
            bboxes = BoxMode.convert(np.array(bboxes), BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

        scaled_masks = []
        for mask in masks:
            mask = mask.astype(np.uint8)
            new_size = (mask.shape[1] * 4, mask.shape[0] * 4)
            scaled_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
            scaled_masks.append(scaled_mask)
        masks = np.array(scaled_masks)

        if use_model_detection:
            bboxes = np.array(bboxes, dtype=np.float32)

        selected_bboxes = []
        for j in range(len(masks) - 1, -1, -1):
            if use_model_detection:
                label = instance_labels[j]
                bbox = bboxes[j]
            else:
                label = instance_labels[object_ids[j]]
                bbox = bboxes[object_ids[j]]
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
            selected_bboxes.append(bbox.tolist() if hasattr(bbox, "tolist") else list(bbox))
            crop_path = out_dir / "crops" / f"{obj_id}_reproj.png"
            crop_params_path = out_dir / "crops" / f"{obj_id}_crop_params.npy"
            if not crop_path.exists() or not crop_params_path.exists():
                crop_mask = mask
                img_h, img_w = scene.image_np.shape[:2]
                if crop_mask.shape[0] != img_h or crop_mask.shape[1] != img_w:
                    crop_mask = cv2.resize(
                        crop_mask.astype(np.uint8),
                        (img_w, img_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                crop, crop_params = crop_object(scene.image_np, crop_mask, crop_size)
                crop.save(crop_path)
                scale_xy = mask.shape[1] / img_w
                crop_params = np.array([
                    crop_params[0] / scale_xy,
                    crop_params[1] / scale_xy,
                    crop_params[2] * scale_xy,
                ])
                np.save(crop_params_path, crop_params)
        with open(out_dir / "bboxes.json", "w") as f:
            json.dump(np.array(selected_bboxes).tolist(), f)
