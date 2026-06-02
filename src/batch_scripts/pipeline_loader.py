"""
Unified scene iteration for COCO (COCONUT) and py123d nuScenes output dirs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from batch_scripts.coconut_loader import CoconutLoader, get_dataset_paths
from batch_scripts.py123d_loader import Py123dOutputLoader


def sanitize_image_name(file_name: str) -> str:
    return file_name.split(".")[0].replace("/", "_").replace("-", "_")


def resolve_data_backend(args, opt) -> str:
    backend = getattr(args, "data_backend", None)
    if backend is not None:
        return backend
    return _run_cfg(opt, "data_backend", "coco")


def _run_cfg(opt, key: str, default=None):
    if isinstance(opt, dict):
        run = opt.get("run", {})
    else:
        run = getattr(opt, "run", {})
    if run is None:
        return default
    if isinstance(run, dict):
        return run.get(key, default)
    getter = getattr(run, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(run, key, default)


def resolve_py123d_split(split: str, opt) -> str:
    if split not in ("val", "train", "test"):
        return split
    py_cfg = _run_cfg(opt, "py123d", {}) or {}
    if isinstance(py_cfg, dict):
        dataset = py_cfg.get("dataset", "nuscenes")
        split_type = py_cfg.get("split_type", split)
    else:
        dataset = getattr(py_cfg, "dataset", "nuscenes")
        split_type = getattr(py_cfg, "split_type", split)
    return f"{dataset}_{split_type}"


def resolve_split(args, opt, data_backend: str) -> str:
    if data_backend == "py123d":
        return resolve_py123d_split(args.split, opt)
    return args.split


def resolve_end_index(end_index: int, loader_len: int) -> int:
    if end_index < 0:
        return loader_len
    return min(end_index, loader_len)


def build_loader(
    data_backend: str,
    save_dir: str,
    split: str,
    opt,
    require_files: Optional[Tuple[str, ...]] = None,
    require_annotations: bool = False,
):
    if data_backend == "py123d":
        req = require_files or ("input.png",)
        return Py123dOutputLoader(
            save_dir,
            split,
            require_files=req,
            require_annotations=require_annotations,
        )
    dataset_root, annotations_dir = get_dataset_paths(split)
    return CocoPipelineLoader(
        split=split,
        save_dir=save_dir,
        dataset_root=dataset_root,
        annotations_dir=annotations_dir,
        require_annotations=require_annotations,
    )


def iter_scene_entries(
    loader,
    start_index: int,
    end_index: int,
) -> Iterator[Dict[str, Any]]:
    end = resolve_end_index(end_index, len(loader))
    for i in range(start_index, end):
        yield loader.get_scene_by_index(i)


def setup_pipeline_loop(args, opt, require_files=None, require_annotations=False):
    """
    Common setup for batch scripts: backend, split, loader, index range.

    Returns:
        (data_backend, split, loader, indices)
    """
    data_backend = resolve_data_backend(args, opt)
    split = resolve_split(args, opt, data_backend)
    loader = build_loader(
        data_backend,
        args.save_dir,
        split,
        opt,
        require_files=require_files,
        require_annotations=require_annotations,
    )
    end_index = resolve_end_index(args.end_index, len(loader))
    indices = range(args.start_index, end_index)
    return data_backend, split, loader, indices


class CocoPipelineLoader:
    """Wrap CoconutLoader with unified SceneEntry dicts."""

    def __init__(
        self,
        split: str,
        save_dir: str,
        dataset_root: str,
        annotations_dir: str,
        require_annotations: bool = False,
    ):
        self.split = split
        self.save_dir = save_dir
        self.dataset_root = dataset_root
        self.coconut = CoconutLoader(split=split, annotations_dir=annotations_dir)
        self.require_annotations = require_annotations

    def __len__(self) -> int:
        return len(self.coconut)

    def get_scene_by_index(self, index: int) -> Dict[str, Any]:
        image_info = self.coconut.get_image_by_index(index)
        img_name = image_info["file_name"]
        image_id = image_info["id"]
        image_path = os.path.join(self.dataset_root, img_name)
        output_dir = os.path.join(
            self.save_dir,
            self.split,
            sanitize_image_name(img_name),
        )
        annotations = self.coconut.get_annotations(image_id)
        if self.require_annotations and not annotations:
            raise FileNotFoundError(f"No annotations for image {image_id} ({img_name})")
        return {
            "id": str(image_id),
            "scene_dir": output_dir,
            "image_path": image_path,
            "file_name": img_name,
            "annotations": annotations if annotations else None,
            "image_id": image_id,
        }
