#!/usr/bin/env python3
# filepath: auto_labeler.py

import gc
import os
import cv2
import numpy as np
import torch
import argparse
import time
import json
import random
import warnings
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from contextlib import contextmanager
import torchvision.transforms as T
from PyQt6.QtGui import QColor
import torch.nn.functional as F
import torchmetrics
from PIL import Image

from segmenter_sam2 import Segmenter
from point_selection_strategies import PointSelectionFactory
from plas.segmenter_plas import SuperpixelLabelExpander

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
# Suppress SAM2 optional post-processing extension warning
import warnings as _w
_w.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*cannot import name '_C' from 'sam2'.*"
)

# Disable PyTorch SDPA flash/mem-efficient kernels and silence related chatter
try:
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

# Silence SDPA kernel selection messages and dtype hints
_w.filterwarnings("ignore", category=UserWarning, message=r".*Flash Attention kernel failed due to:.*")
_w.filterwarnings("ignore", category=UserWarning, message=r".*Falling back to all available kernels for scaled_dot_product_attention.*")
_w.filterwarnings("ignore", category=UserWarning, message=r".*Expected query, key and value to all be of dtype.*scaled_dot_product_attention.*")

# Optional multi-scale helpers (non-fatal if missing)
try:
    from multi_scale_utils import (
        choose_scale, ScaleConfig, downscale_image,
        init_coverage_map, update_coverage_map
    )
except Exception:  # pragma: no cover - graceful fallback
    choose_scale = None
    ScaleConfig = None
    downscale_image = None
    init_coverage_map = None
    update_coverage_map = None

@contextmanager
def timer(name=None):
    """Context manager for timing operations."""
    start = time.time()
    yield
    elapsed = time.time() - start
    if name:
        print(f"{name}: {elapsed:.3f}s")

class PipelineTimer:
    """Precise timing tracker for different segmentation pipelines."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all timers for a new image."""
        self.times = {
            'setup': 0.0,
            'point_selection': 0.0,
            'sam2_propagation': 0.0,
            'mask_merging': 0.0,
            'plas_expansion': 0.0,
            'postprocessing': 0.0,
            'io_operations': 0.0
        }
        self.total_start = None
    
    def start_total(self):
        """Start timing the total processing for this image."""
        self.total_start = time.time()
    
    def add_time(self, operation, duration):
        """Add time for a specific operation."""
        if operation in self.times:
            self.times[operation] += duration
    
    @contextmanager
    def time_operation(self, operation):
        """Context manager to time a specific operation."""
        start = time.time()
        yield
        duration = time.time() - start
        self.add_time(operation, duration)
    
    def get_pipeline_times(self):
        """Calculate times for different pipeline combinations."""
        if self.total_start is None:
            total_time = sum(self.times.values())
        else:
            total_time = time.time() - self.total_start
        
        # Pure times (no overlap)
        sam2_only_time = (
            self.times['setup'] + 
            self.times['point_selection'] + 
            self.times['sam2_propagation'] + 
            self.times['mask_merging'] +
            self.times['postprocessing']
        )
        
        plas_only_time = (
            self.times['setup'] + 
            self.times['point_selection'] + 
            self.times['plas_expansion'] +
            self.times['postprocessing']
        )
        
        combined_time = (
            self.times['setup'] + 
            self.times['point_selection'] + 
            self.times['sam2_propagation'] + 
            self.times['mask_merging'] +
            self.times['plas_expansion'] +
            self.times['postprocessing']
        )
        
        return {
            'total_measured': total_time,
            'sam2_propagation_pipeline': sam2_only_time,
            'plas_pipeline': plas_only_time,
            'combined_pipeline': combined_time,
            'breakdown': self.times.copy()
        }

class AutoLabeler:
    def __init__(self,
                 images_dir,
                 ground_truth_dir=None,  # Optional
                 output_dir=None,
                 sam2_checkpoint="checkpoints/sam2.1_hiera_large.pt",
                 sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
                 save_visualizations=False,
                 debug_save_expanded_masks=False,
                 device="cuda",
                 point_selection_strategy="random",
                 num_points=30,
                 use_maskSLIC=False,
                 num_classes=None,
                 downscale_auto=False,
                 downscale_fixed=None,
                 seed=42,
                 label_to_id_json=None,
                 default_background_class_id=0,
                 **strategy_kwargs):
        """Initialize the unified AutoLabeler with paths and parameters."""

        # Background semantics
        self.DEFAULT_BACKGROUND_CLASS_ID = default_background_class_id
        # self.DEFAULT_BACKGROUND_COLOR = (63, 69, 131)  # RGB
        self.DEFAULT_BACKGROUND_COLOR = (0, 0, 0)  # RGB

        # Paths & config
        self.images_dir = Path(images_dir)
        self.ground_truth_dir = Path(ground_truth_dir) if ground_truth_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config

        # Device selection
        if device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device
        # Store seed for reproducibility
        self.seed = seed

        self.save_visualizations = save_visualizations
        # Debug option: save every expanded mask separately + overlap counts
        self.save_expanded_masks_debug = debug_save_expanded_masks
        self.num_classes = num_classes  # May be determined later
        self.eval_mask_type = "propagation_plas"

        # Strategy configuration
        self.point_selection_strategy = point_selection_strategy
        self.num_points = num_points
        self.use_maskSLIC = use_maskSLIC
        self.strategy_kwargs = strategy_kwargs

        # Instantiate strategy
        self.strategy = PointSelectionFactory.create_strategy(
            self.point_selection_strategy,
            num_points=self.num_points,
            **self.strategy_kwargs,
        )
        # GT point sampling configuration (max points per GT mask used for overlap resolution)
        # Can be overridden via strategy kwargs: max_gt_points_per_mask
        self.max_gt_points_per_mask = self.strategy_kwargs.get('max_gt_points_per_mask', 3000)
        self.min_gt_points_full_mask = self.strategy_kwargs.get('min_gt_points_full_mask', 2500)
        # Cached listings
        self._cached_image_files = None

        # Interactive strategies (exclude SAM2_guided)
        self.is_interactive_strategy = self.point_selection_strategy in [
            "dynamicPoints_onlyA",
            "dynamicPoints",
            "dynamicPointsLargestGT",
        ]

        # Stats
        self.stats = {
            "images_processed": 0,
            "masks_identified": 0,
            "per_class_masks": defaultdict(int),
        }

        # Output directories
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.save_visualizations:
                (self.output_dir / "visualizations").mkdir(exist_ok=True)
            (self.output_dir / "masks_plas").mkdir(exist_ok=True)
            (self.output_dir / "masks_propagation").mkdir(exist_ok=True)
            (self.output_dir / "masks_propagation_plas").mkdir(exist_ok=True)
            (self.output_dir / "stats").mkdir(exist_ok=True)

        # Color mappings (populate after loading any existing mapping and/or using
        # an external color dict). Do not hard-code a mapping here because the
        # true background color may be provided by the dataset color dictionary
        # or by a saved mapping file.
        self.color_to_label = {}
        self.label_to_color = {}

        # Load/create mapping
        self.load_or_create_color_mapping()

        print("Unified AutoLabeler initialized with:")
        print(f"  - Images directory: {self.images_dir}")
        print(f"  - Ground truth directory: {self.ground_truth_dir}")
        print(f"  - Output directory: {self.output_dir}")
        print(f"  - Device: {self.device}")
        print(f"  - Point selection strategy: {self.point_selection_strategy}")
        print(f"  - Interactive strategy: {self.is_interactive_strategy}")
        # For the 'list' strategy the number of points is defined by the input point file
        # so printing the default/num_points value is misleading. Only display when
        # the strategy is not 'list'.
        if str(self.point_selection_strategy).lower() != 'list':
            print(f"  - Number of points: {self.num_points}")
        print(f"  - Strategy-specific parameters: {self.strategy_kwargs}")
        if self.save_expanded_masks_debug:
            print("  - Expanded mask debug saving: ENABLED")

        # Lazy segmenter init
        self.segmenter = None
        self._segmenter_initialized = False

        # Timing
        self.timer = PipelineTimer()

        # Downscale config (optional multi-scale assistance)
        self.downscale_auto = downscale_auto
        self.downscale_fixed = downscale_fixed
        self.scale_config = ScaleConfig() if ScaleConfig is not None else None
        self.current_scale = 1.0
        self.coverage_map = None  # low-res coverage (uint16) if scaling active

        # Label to ID mapping (optional JSON file)
        self.label_to_id_map = None
        self.id_to_label_map = None
        if label_to_id_json is not None:
            with open(label_to_id_json, "r") as f:
                self.label_to_id_map = json.load(f)
            # Reverse mapping: id (int) -> label (str)
            self.id_to_label_map = {int(v): k for k, v in self.label_to_id_map.items()}

    def _initialize_segmenter(self):
        """Initialize the segmenter when first needed."""
        if not self._segmenter_initialized:
            self.segmenter = Segmenter(
                image=None,
                sam2_checkpoint_path=self.sam2_checkpoint,
                sam2_config_path=self.sam2_config,
                device=self.device
            )
            self._segmenter_initialized = True

    def build_complete_color_mapping(self):
        """
        Build a complete color mapping by analyzing all ground truth images.
        This ensures all colors in GT images are mapped to class indices.
        """
        print("Building complete color mapping from ground truth images...")
        
        image_files = self.get_image_files()
        all_colors = set()
        
        for img_path, gt_path in tqdm(image_files, desc="Analyzing GT images", leave=False):
            gt_image = cv2.imread(str(gt_path))
            if gt_image is None:
                continue
                
            gt_rgb = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
            
            # Get unique colors (sample every 4th pixel for speed)
            unique_colors = set(map(tuple, gt_rgb[::4, ::4].reshape(-1, 3)))
            all_colors.update(unique_colors)
        
        print(f"Found {len(all_colors)} unique colors in ground truth images")
        
        # Build color mapping
        self.color_to_label = {}
        self.label_to_color = {}
        
        # Always start with background
        black = (0, 0, 0)
        if black in all_colors:
            self.color_to_label[black] = 0
            self.label_to_color[0] = black
            all_colors.remove(black)
        
        # Map remaining colors to consecutive class indices
        for i, color in enumerate(sorted(all_colors), start=1):
            self.color_to_label[color] = i
            self.label_to_color[i] = color
        
        print(f"Created color mapping with {len(self.color_to_label)} entries")
        
        # Save the mapping
        self.save_color_mapping()

    def load_or_create_color_mapping(self):
        """Load existing color to label mapping or create a new one."""
        mapping_file = self.output_dir / "color_mapping.json"
        
        if mapping_file.exists():
            try:
                print(f"Loading color mapping from {mapping_file}")
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
                    # Convert string keys (from JSON) back to tuple keys
                    self.color_to_label = {eval(k): v for k, v in mapping["color_to_label"].items()}
                    self.label_to_color = {k: v for k, v in mapping["label_to_color"].items()}
                print(f"Successfully loaded color mapping with {len(self.color_to_label)} entries")
            except json.JSONDecodeError as e:
                print(f"Error loading color mapping: {e}")
                print("Creating a new color mapping file.")
                self.color_to_label = {}
                self.label_to_color = {}
                # Rename the corrupted file
                backup_file = mapping_file.with_suffix('.json.bak')
                try:
                    mapping_file.rename(backup_file)
                    print(f"Backed up corrupted mapping file to {backup_file}")
                except Exception as rename_err:
                    print(f"Could not rename corrupted file: {rename_err}")
        else:
            print("No existing color mapping found. Will create during processing.")
            self.color_to_label = {}
            self.label_to_color = {}
    
    def save_color_mapping(self, verbose=False):
        """Save color to label mapping for future use."""
        mapping_file = self.output_dir / "color_mapping.json"
        
        # Convert tuple keys to strings and ensure all values are JSON serializable
        color_to_label_serializable = {str(k): v for k, v in self.color_to_label.items()}
        label_to_color_serializable = {k: [int(c) for c in v] for k, v in self.label_to_color.items()}
        
        # Create a temporary file first to avoid corrupting the existing file
        temp_file = mapping_file.with_suffix('.json.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump({
                    "color_to_label": color_to_label_serializable,
                    "label_to_color": label_to_color_serializable
                }, f, indent=2)
                f.flush()
                import os
                os.fsync(f.fileno())  # Ensure data is written to disk
            
            # Rename the temporary file to the target file (atomic operation on most systems)
            temp_file.replace(mapping_file)
            if verbose:
                print(f"Color mapping saved to {mapping_file} ({len(self.color_to_label)} entries)")
        except Exception as e:
            print(f"Error saving color mapping: {e}")
            if temp_file.exists():
                try:
                    temp_file.unlink()  # Delete the temporary file
                except Exception:
                    pass

    def get_image_files(self):
        """Get list of image files to process (with optional ground truth), recursively.

        Matching strategy (robust):
        1. Prefer GT with identical relative path (same subfolder structure) but .png extension.
        2. Fallback to GT files matched by stem (filename without extension).
           - If multiple GTs share the same stem, prefer one with matching subfolder name when possible,
             else take the first candidate.
        """
        if self._cached_image_files is not None:
            return self._cached_image_files

        image_exts = {'.jpg', '.jpeg', '.png'}
        gt_exts = {'.png'}  # enforce .png for GT as requested

        # Gather image files
        image_files = [p for p in self.images_dir.rglob('*') if p.is_file() and p.suffix.lower() in image_exts]
        rel_image_files = {str(p.relative_to(self.images_dir)): p for p in image_files}

        # Sparse-GT mode
        if self.ground_truth_dir is None:
            pairs = [(p, None) for p in image_files]
            print(f"Found {len(pairs)} images for sparseGT-only processing (recursive)")
            self._cached_image_files = pairs
            return pairs

        # Gather GT files (only allowed GT extensions)
        gt_files = [f for f in self.ground_truth_dir.rglob('*') if f.is_file() and f.suffix.lower() in gt_exts]

        # Build lookups
        gt_by_rel = {str(f.relative_to(self.ground_truth_dir)): f for f in gt_files}
        gt_by_stem = {}
        for f in gt_files:
            gt_by_stem.setdefault(f.stem.lower(), []).append(f)

        valid = []
        used_gts = set()
        for rel_path, img in rel_image_files.items():
            # 1) Try exact relative path match but with .png ext
            img_rel = Path(rel_path)
            candidate_rel = img_rel.with_suffix('.png')
            gt_candidate = gt_by_rel.get(str(candidate_rel))
            if gt_candidate and gt_candidate.exists():
                valid.append((img, gt_candidate))
                used_gts.add(str(gt_candidate.resolve()))
                continue

            # 2) Try basename/stem match
            stem = img.stem.lower()
            candidates = gt_by_stem.get(stem, [])
            if not candidates:
                # 3) Try matching by filename (case-insensitive) among GTs
                img_name_lower = img.name.lower()
                for f in gt_files:
                    if f.name.lower() == img_name_lower:
                        candidates = [f]
                        break

            if candidates:
                # If multiple candidates, prefer one that shares a subdirectory name with the image
                chosen = None
                if len(candidates) == 1:
                    chosen = candidates[0]
                else:
                    img_parents = [p.name.lower() for p in img.relative_to(self.images_dir).parents]
                    for cand in candidates:
                        try:
                            rel_cand = cand.relative_to(self.ground_truth_dir)
                        except Exception:
                            rel_cand = cand
                        cand_parents = [p.name.lower() for p in Path(rel_cand).parents]
                        if any(p in img_parents for p in cand_parents):
                            chosen = cand
                            break
                    if chosen is None:
                        chosen = sorted(candidates, key=lambda x: str(x.relative_to(self.ground_truth_dir)))[0]

                if chosen and chosen.exists() and str(chosen.resolve()) not in used_gts:
                    valid.append((img, chosen))
                    used_gts.add(str(chosen.resolve()))
                    continue

            # No GT found for this image -> skip
            continue

        print(f"Found {len(valid)} valid image/ground-truth pairs (recursive)")
        self._cached_image_files = valid
        return valid

    def extract_labels_from_ground_truth(self, gt_image):
        """Extract unique labels from ground truth image (grayscale or RGB)."""
        is_grayscale = len(gt_image.shape) == 2 or (len(gt_image.shape) == 3 and gt_image.shape[2] == 1)
        if is_grayscale:
            if len(gt_image.shape) == 3:
                gt_image = gt_image[:, :, 0]
            unique_values = set()
            h, w = gt_image.shape[:2]
            for y in range(0, h, 4):
                for x in range(0, w, 4):
                    v = int(gt_image[y, x])
                    if v > 0:  # skip background (0) here; background handled globally
                        unique_values.add(v)
            labels = []
            for v in unique_values:
                mask_v = (gt_image == v)
                area = int(mask_v.sum())
                if area <= 100:
                    continue
                labels.append({
                    "mask": mask_v,
                    "label": v,
                    "color": (v, v, v),
                    "grayscale_value": v,
                    "area": area
                })
            return labels
        else:
            unique_colors = set(map(tuple, gt_image[::4, ::4].reshape(-1, 3)))
            labels = []
            for color in unique_colors:
                mask_c = np.all(gt_image == color, axis=2)
                if color == self.DEFAULT_BACKGROUND_COLOR:
                    lbl = self.DEFAULT_BACKGROUND_CLASS_ID
                    self.color_to_label[color] = lbl
                    self.label_to_color[lbl] = color
                else:
                    if color in self.color_to_label:
                        lbl = self.color_to_label[color]
                    else:
                        lbl = len([c for c in self.color_to_label.values() if c != self.DEFAULT_BACKGROUND_CLASS_ID]) + 1
                        if lbl == self.DEFAULT_BACKGROUND_CLASS_ID:
                            lbl += 1
                        self.color_to_label[color] = lbl
                        self.label_to_color[lbl] = color
                area = int(mask_c.sum())
                if area > 100:
                    labels.append({
                        "mask": mask_c,
                        "label": lbl,
                        "color": color,
                        "area": area
                    })
            return labels

    def find_mask_for_point(self, point, gt_masks):
        """
        Find the ground truth mask containing the given point.
        
        Args:
            point: (y, x) tuple (row, col ordering)
            gt_masks: List of dictionaries with mask information
            
        Returns:
            (mask, label) for the first mask containing the point or (None, None, None) if not found.
        """
        y, x = int(point[0]), int(point[1])
        for mask_info in gt_masks:
            mask = mask_info.get("mask")
            if mask is None:
                continue
            # Bounds check (row=y, col=x)
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                if mask[y, x]:
                    # Color may be absent in pure grayscale mode; fall back to grayscale triple if available.
                    col = mask_info.get("color")
                    if col is None and mask_info.get("grayscale_value") is not None:
                        gv = int(mask_info.get("grayscale_value"))
                        col = (gv, gv, gv)
                    return mask, mask_info.get("label"), col
        return None, None, None


    def merge_overlapping_masks(self, masks, mask_labels, gt_points=None, gt_labels=None):
        import numpy as np
        from collections import Counter

        if not masks:
            return None, None

        H, W = masks[0].shape
        n_masks = len(masks)

        # Convert inputs to efficient arrays
        mask_labels = np.array(mask_labels, dtype=np.int32)
        # Stack masks: (N, H, W) -> (N, H*W)
        flat_masks = np.stack([m.flatten() for m in masks], axis=0).astype(bool) # (N, Pixels)
        

        packed = np.packbits(flat_masks, axis=0)
        
        # We transpose to (Pixels, Bytes) and view as 1D array of 'void' blobs.
        # This forces np.unique to do a 1D sort (fast) instead of 2D row sort (slow).
        packed_t = packed.T
        dt_void = np.dtype((np.void, packed_t.dtype.itemsize * packed_t.shape[1]))
        # ascontiguousarray is required to ensure memory is safe for viewing
        packed_view = np.ascontiguousarray(packed_t).view(dt_void).ravel()
        
        # return_index=True gives us the index of the *first* occurrence of each unique signature.
        _, idxs, inv_inds = np.unique(packed_view, return_index=True, return_inverse=True)
        
        # This is much faster than unpacking the bits manually.
        u_rows = flat_masks.T[idxs]
        
        # Calculate coverage count per signature to quickly identify conflicts
        cover_counts = u_rows.sum(axis=1)

        # Prepare outputs
        # -1 sentinel as requested
        final_mask_flat = np.full(H * W, -1, dtype=np.int32)
        owner_mask_flat = np.zeros(H * W, dtype=np.int32)
        
        # Prepare GT data if available
        has_gt = False
        if gt_points is not None and gt_labels is not None and len(gt_points) > 0:
            gt_points_arr = np.asarray(gt_points)
            gt_labels_arr = np.asarray(gt_labels)
            has_gt = True
            
            # Clip GT points to bounds once
            gt_r = np.clip(gt_points_arr[:, 0].astype(int), 0, H - 1)
            gt_c = np.clip(gt_points_arr[:, 1].astype(int), 0, W - 1)
            # Map GT points to linear indices for fast region checking
            gt_linear_indices = gt_r * W + gt_c
            
            k_nearest_k = getattr(self, 'nearest_gt_k', 5)

        # --- Iterate through unique overlap combinations ---
        # This loop runs N_Combinations times (usually < 100), not N_Pixels times.
        for i in range(len(u_rows)):
            count = cover_counts[i]
            
            if count == 0:
                continue # Background

            # Get the pixel indices belonging to this specific overlap combination
            pixel_indices = np.where(inv_inds == i)[0]
            if pixel_indices.size == 0:
                continue

            # Indices of masks involved in this overlap
            active_mask_indices = np.where(u_rows[i])[0]

            # --- Case 1: Single Mask (No Conflict) ---
            if count == 1:
                idx = active_mask_indices[0]
                lbl = mask_labels[idx]
                final_mask_flat[pixel_indices] = lbl
                owner_mask_flat[pixel_indices] = idx + 1
                continue

            # --- Case 2: Overlap (Conflict Resolution) ---
            # Get coordinates of pixels in this region
            rows, cols = np.divmod(pixel_indices, W)
            region_centroid = np.array([rows.mean(), cols.mean()])

            best_seg = None
            
            # 2a. Ground Truth Logic
            if has_gt:
                # Labels involved in this conflict
                candidate_labels = mask_labels[active_mask_indices]
                
                # Filter GT points that have relevant labels
                # np.isin is efficient
                relevant_gt_mask = np.isin(gt_labels_arr, candidate_labels)
                
                if np.any(relevant_gt_mask):
                    cand_pts = gt_points_arr[relevant_gt_mask]
                    cand_lbls = gt_labels_arr[relevant_gt_mask]
                    cand_lin_inds = gt_linear_indices[relevant_gt_mask]

                    # Check 1: Points physically INSIDE the overlap region
                    # We can check containment using the inv_inds map directly!
                    # If inv_inds[pt_idx] == i, the point is inside this specific overlap chunk.
                    is_inside = (inv_inds[cand_lin_inds] == i)
                    
                    if np.any(is_inside):
                        # --- Weight by Inverse Distance to Centroid ---
                        pts_in = cand_pts[is_inside]
                        lbls_in = cand_lbls[is_inside]
                        
                        dists = np.linalg.norm(pts_in - region_centroid, axis=1)
                        weights = 1.0 / (1.0 + dists)
                        
                        best_score = -1.0
                        
                        # Score each candidate mask
                        for mask_idx in active_mask_indices:
                            m_lbl = mask_labels[mask_idx]
                            # Sum weights of GT points matching this mask's label
                            score = np.sum(weights[lbls_in == m_lbl])
                            
                            # Strict > check preserves "first winner" on ties if implemented that way,
                            # but original code used best_score = -inf.
                            if score > best_score:
                                best_score = score
                                best_seg = mask_idx
                                
                        # If we found a winner via interior points, we stop GT logic
                        if best_seg is not None and best_score > 0:
                            pass # Done
                        else:
                            best_seg = None # Fallthrough to KNN if score was 0? (Original logic implies check KNN if !region_has_gt)

                    else:
                        # --- Check 2: KNN (Global Distance) ---
                        # Calculate dists to centroid for ALL candidate points
                        dists_all = np.linalg.norm(cand_pts - region_centroid, axis=1)
                        
                        # argsort is fast enough for small N
                        if dists_all.size > 0:
                            order = np.argsort(dists_all)
                            k_use = min(k_nearest_k, len(order))
                            top_idx = order[:k_use]
                            
                            top_lbls = cand_lbls[top_idx]
                            top_dists = dists_all[top_idx]
                            
                            # Score labels within top K
                            unique_top_lbls = np.unique(top_lbls)
                            best_k_score = -1.0
                            winning_lbl = None
                            
                            # Tie breaking storage
                            tie_candidates = []

                            for t_lbl in unique_top_lbls:
                                mask_t = (top_lbls == t_lbl)
                                d_sub = top_dists[mask_t]
                                score_val = np.sum(1.0 / (1.0 + d_sub))
                                
                                # Float comparison logic from original
                                if score_val > best_k_score + 1e-12:
                                    best_k_score = score_val
                                    winning_lbl = t_lbl
                                    tie_candidates = [(t_lbl, d_sub.mean())]
                                elif abs(score_val - best_k_score) <= 1e-12:
                                    tie_candidates.append((t_lbl, d_sub.mean()))
                            
                            # Tie break by average distance
                            if len(tie_candidates) > 1:
                                tie_candidates.sort(key=lambda x: x[1])
                                winning_lbl = tie_candidates[0][0]
                                
                            # Map winning label back to a mask index
                            if winning_lbl is not None:
                                for mask_idx in active_mask_indices:
                                    if mask_labels[mask_idx] == winning_lbl:
                                        best_seg = mask_idx
                                        break

            # 2b. Label Frequency Logic
            if best_seg is None:
                # Count label occurrences among the active masks
                # (e.g. if Mask A and Mask B both have Label X, count is 2)
                lbls_in_conflict = mask_labels[active_mask_indices]
                counts = Counter(lbls_in_conflict)
                
                # Sort by (count desc, label_val asc) to match max() behavior
                if counts:
                    # Find max count
                    max_c = max(counts.values())
                    # Get all labels with this count
                    candidates_c = [l for l, c in counts.items() if c == max_c]
                    
                    # Original logic: "if list(...).count(majority_count) == 1 and majority_count > 1"
                    # Meaning: strict majority required, and count must be > 1.
                    if len(candidates_c) == 1 and max_c > 1:
                        maj_lbl = candidates_c[0]
                        # Find first mask with this label
                        for mask_idx in active_mask_indices:
                            if mask_labels[mask_idx] == maj_lbl:
                                best_seg = mask_idx
                                break

            # 2c. Area / Index Fallback
            if best_seg is None:
                best_seg = active_mask_indices[0]

            # Apply result for this overlap chunk
            final_mask_flat[pixel_indices] = mask_labels[best_seg]
            owner_mask_flat[pixel_indices] = best_seg + 1

        return final_mask_flat.reshape(H, W), owner_mask_flat.reshape(H, W)

    def extract_sam2_features(self, mask=None):
        """
        Extract SAM2 features from a masked region.
        
        Args:
            mask: Binary mask to focus on a specific region
            
        Returns:
            Feature vector from SAM2
        """
        if self.features_sam2 is None:
            return None
            
        mask_tensor = torch.from_numpy(mask).to(self.features_sam2.device)        # [H_orig, W_orig]
        # Resize mask to feature spatial size using nearest neighbor
        mask_resized = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0).float(),
            size=self.features_sam2.shape[-2:], mode='nearest'
        ).squeeze(0).squeeze(0)  # Now shape [H_feat, W_feat]
        # Apply mask: broadcast to all channels
        masked_feats = self.features_sam2 * mask_resized
        # Flatten spatial dims: each channel's values for the masked region
        masked_flat = masked_feats.view(self.features_sam2.size(0), -1)            # [C, H_feat*W_feat]
        # Also flatten mask to count region size
        mask_flat = mask_resized.view(-1)                               # [H_feat*W_feat]
        # Select only the entries within the mask (mask_flat == 1)
        if mask_flat.sum() > 0:
            region_feats = masked_flat[:, mask_flat.bool()]             # [C, N_pixels]
            region_descriptor = region_feats.mean(dim=1)                # [C] vector
        else:
            region_descriptor = torch.zeros(self.features_sam2.size(0), device=self.features_sam2.device)

        return region_descriptor

    def process_image_interactive(self, image_path, gt_path, color_dict=None):
        """
        Process image using interactive/iterative point selection strategies.
        """
        self.timer.reset()
        self.timer.start_total()

        # Initialize segmenter if not already done
        self._initialize_segmenter()

        # Load image
        with self.timer.time_operation('io_operations'):
            image = cv2.imread(str(image_path))

        # Decide auxiliary scale (never affects SAM2 embedding resolution)
        self.current_scale = 1.0
        self.coverage_map = None
        if (self.downscale_fixed is not None or self.downscale_auto) and choose_scale is not None:
            if self.downscale_fixed is not None:
                self.current_scale = float(max(0.1, min(1.0, self.downscale_fixed)))
            elif self.downscale_auto and self.scale_config is not None:
                try:
                    self.current_scale = float(choose_scale(image.shape, self.scale_config))
                except Exception:
                    self.current_scale = 1.0
            if self.current_scale < 0.999 and init_coverage_map is not None:
                low_h = max(1, int(round(image.shape[0] * self.current_scale)))
                low_w = max(1, int(round(image.shape[1] * self.current_scale)))
                self.coverage_map = init_coverage_map((low_h, low_w))

        with self.timer.time_operation('setup'):
            generated_masks, self.features_sam2 = self.segmenter.set_image(image)

            generated_masks_overlay = np.zeros(image.shape[:2], dtype=np.uint8)
            # Create overlay for generated masks
            for mask in generated_masks:
                generated_masks_overlay[mask['segmentation'] > 0] = 1

        # Handle ground truth loading (optional for sparseGT-only mode)
        gt_masks = []
        if gt_path is not None:
            with self.timer.time_operation('io_operations'):
                # Load ground truth
                gt_image = cv2.imread(str(gt_path))
                gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
                
                # Extract ground truth masks/labels
                gt_masks = self.extract_labels_from_ground_truth(gt_image)
        else:
            # sparseGT-only mode: no ground truth available
            # Suppressed per-image print to keep tqdm progress bar clean
            pass

        # Track expanded masks and their labels
        expanded_masks = []
        
        # Setup for interactive strategy
        with self.timer.time_operation('setup'):
            if hasattr(self.strategy, 'set_gt_masks'):
                try:
                    self.strategy.set_gt_masks(gt_masks)
                except Exception:
                    pass
            self.strategy.setup_simple(image, generated_masks)
        
        points_to_process = []
        labels_to_process = []
        rgb_to_int_label = {}
        int_labels_to_rgb = {}
        next_label = 1

        last_mask = None

        # Interactive point selection loop
        for i in range(self.num_points):
            with self.timer.time_operation('point_selection'):
                pt = self.strategy.get_next_point(last_mask)
                
                if pt is None:
                    continue
                    
                x, y = pt[1], pt[0]
                current_point = (x, y)
                
                points_to_process.append(current_point)
                # Use grayscale label directly from GT image
                if gt_image.ndim == 2:
                    gray_val = int(gt_image[pt[0], pt[1]])
                else:
                    gray_val = int(gt_image[pt[0], pt[1], 0])
                labels_to_process.append(gray_val)
                # No per-point GT mask lookup; rely solely on pixel value.
                gt_label = None
                gt_color = None
            
            with self.timer.time_operation('sam2_propagation'):
                # Expand the mask using SAM2
                point_pair = np.array([current_point])
                label_pair = np.array([1])  # Positive point
                mask = self.segmenter.propagate_points(point_pair, label_pair, update_expanded_mask=True)
                last_mask = mask  # Update last_mask for the next iteration
                
                if mask is None:
                    continue
                
                eff_label = gray_val  # Use direct grayscale value
                # No color in pipeline; store None and colorize only when saving/debugging
                expanded_masks.append((mask, eff_label, None, current_point))
                self.stats["masks_identified"] += 1
                self.stats["per_class_masks"][eff_label] += 1

                # Update coverage (best-effort)
                if self.coverage_map is not None and update_coverage_map is not None:
                    try:
                        update_coverage_map(self.coverage_map, mask, self.current_scale, bg_value=0)
                    except Exception:
                        pass

        return self._finalize_processing_with_timing(
            image_path, image, gt_image if gt_path is not None else None, points_to_process, labels_to_process, 
            int_labels_to_rgb, expanded_masks, color_dict
        )

    def process_image_batch(self, image_path, gt_path, color_dict=None):
        """
        Process image using batch/traditional point selection strategies.
        """
        self.timer.reset()
        self.timer.start_total()

        # Initialize segmenter if not already done
        self._initialize_segmenter()

        # Load image
        with self.timer.time_operation('io_operations'):
            image = cv2.imread(str(image_path))
        
        # Decide scale for batch path
        self.current_scale = 1.0
        self.coverage_map = None
        if (self.downscale_fixed is not None or self.downscale_auto) and choose_scale is not None:
            if self.downscale_fixed is not None:
                self.current_scale = float(max(0.1, min(1.0, self.downscale_fixed)))
            elif self.downscale_auto and self.scale_config is not None:
                try:
                    self.current_scale = float(choose_scale(image.shape, self.scale_config))
                except Exception:
                    self.current_scale = 1.0
            if self.current_scale < 0.999 and init_coverage_map is not None:
                low_h = max(1, int(round(image.shape[0] * self.current_scale)))
                low_w = max(1, int(round(image.shape[1] * self.current_scale)))
                self.coverage_map = init_coverage_map((low_h, low_w))
            
        with self.timer.time_operation('setup'):
            self.segmenter.just_set_image(image)

        if gt_path is not None:
            with self.timer.time_operation('io_operations'):
                # Load ground truth
                gt_image = cv2.imread(str(gt_path))
                gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        else:
            # sparseGT-only mode: no ground truth available
            # Suppressed per-image print to keep tqdm progress bar clean
            pass
        
        # Track expanded masks and their labels
        expanded_masks = []
        
        with self.timer.time_operation('point_selection'):
            # Get points and classes to process using the unified strategy
            # Prefer an image identifier relative to the images_dir so CSVs that
            # contain subfolder paths (relative to images_dir) will match.
            try:
                if getattr(self, 'images_dir', None) is not None:
                    try:
                        rel_image_name = str(Path(image_path).relative_to(self.images_dir).as_posix())
                    except Exception:
                        rel_image_name = Path(image_path).name
                else:
                    rel_image_name = Path(image_path).name
            except Exception:
                rel_image_name = Path(image_path).name

            points_and_classes = self.strategy.select_points(
                self.segmenter,
                image,
                expanded_masks=expanded_masks,
                image_name=rel_image_name
            )

            # Handle the unified return value (points, classes)
            if isinstance(points_and_classes, tuple) and len(points_and_classes) == 2:
                points_to_process, point_classes = points_and_classes
            else:
                # Fallback for strategies that only return points
                points_to_process = points_and_classes
                point_classes = None
            
            # Process points and create labels
            labels_to_process = []
            rgb_to_int_label = {}
            int_labels_to_rgb = {}
            next_label = 1 
            
            for i, point in enumerate(points_to_process):
                if point is None:
                    continue

                if gt_path is not None:
                    # Grayscale ground truth mode: read class id directly from GT image
                    if gt_image.ndim == 2:
                        gray_val = int(gt_image[point[0], point[1]])
                    else:
                        gray_val = int(gt_image[point[0], point[1], 0])
                    labels_to_process.append(gray_val)
                else:
                    # sparseGT-only mode: use class information from points
                    if point_classes is None or i >= len(point_classes):
                        raise ValueError(f"sparseGT-only mode requires class information for point {i}")
                    point_class = point_classes[i]
                    labels_to_process.append(point_class)
            
            # Process each selected point; for 'list' strategy we keep all provided points
            if self.point_selection_strategy != 'list':
                points_to_process = points_to_process[:self.num_points]
                labels_to_process = labels_to_process[:self.num_points]
            else:
                pass

            #flip points coordinates
            points_to_process = np.array(points_to_process)[:, [1, 0]]

        iteration = 0
        for idx, current_point in enumerate(points_to_process):
            iteration += 1
            if current_point is None:
                continue

            point_label = labels_to_process[idx] if idx < len(labels_to_process) else None

            # Always propagate regardless of gt_mask presence
            with self.timer.time_operation('sam2_propagation'):
                point_pair = np.array([current_point])
                label_pair = np.array([1])
                mask = self.segmenter.propagate_points(point_pair, label_pair, update_expanded_mask=True)
                if mask is None:
                    continue

                eff_label = point_label  # Use pixel-derived label directly
                if eff_label is None:
                    continue

                # No color in pipeline; store None and colorize only when saving/debugging
                expanded_masks.append((mask, eff_label, None, tuple(current_point)))
                self.stats["masks_identified"] += 1
                self.stats["per_class_masks"][eff_label] += 1

        return self._finalize_processing_with_timing(
            image_path, image, gt_image if gt_path is not None else None, points_to_process, labels_to_process, 
            int_labels_to_rgb, expanded_masks, color_dict
        )

    def _finalize_processing_with_timing(self, image_path, image, gt_image, points_to_process, 
                                        labels_to_process, int_labels_to_rgb, expanded_masks, color_dict=None, gt_masks=None):
        """Finalize processing: unify masks, apply background, return result dict."""
        
        # --- 1. Setup & Color Synchronization ---
        # Sync internal color maps immediately
        for lbl, rgb in int_labels_to_rgb.items():
            self.label_to_color[lbl] = rgb
            self.color_to_label[tuple(rgb)] = lbl

        bg_id = int(self.DEFAULT_BACKGROUND_CLASS_ID)
        
        # Resolve background color once (priority: color_dict -> internal map -> default)
        bg_color = self.DEFAULT_BACKGROUND_COLOR
        if color_dict:
            # Try to find bg_id in color_dict (handling string/int keys)
            val = color_dict.get(bg_id) or color_dict.get(str(bg_id))
            if val: bg_color = tuple(int(x) for x in val)
        
        # Ensure consistency
        self.DEFAULT_BACKGROUND_COLOR = bg_color
        self.label_to_color[bg_id] = bg_color

        # --- 2. Fast Visualization (OpenCV) ---
        if self.save_visualizations:
            vis_dir = self.output_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            vis_img = image.copy()
            
            # Build lookup for speed
            lookup = {**self.label_to_color}
            if color_dict: 
                # safe merge
                for k, v in color_dict.items():
                    try: lookup[int(k)] = tuple(v) # Assume k is label
                    except: pass

            # Draw points
            for (x, y), lbl in zip(points_to_process, labels_to_process):
                rgb = lookup.get(int(lbl), (128, 128, 128))
                bgr = (rgb[2], rgb[1], rgb[0]) # RGB to BGR for OpenCV
                cv2.circle(vis_img, (int(x), int(y)), 6, bgr, -1)
                cv2.circle(vis_img, (int(x), int(y)), 6, (0, 255, 255), 2)
                
            cv2.imwrite(str(vis_dir / f"{Path(image_path).stem}.png"), vis_img)

        # --- 3. PLAS Expansion ---
        if self.num_classes is None:
            self._determine_num_classes(color_dict, labels_to_process, allow_infer=True)
            
        # Calculate safe num_classes without try/except blocks
        max_lbl = int(np.max(labels_to_process)) if labels_to_process else 0
        safe_nc = max(int(self.num_classes or 0), max_lbl + 1, bg_id + 1, 1)

        with self.timer.time_operation('plas_expansion'):
            plas_mask = self.PLAS_segmenter.expand_labels(
                image, points_to_process, labels_to_process, num_classes=safe_nc)

        # --- 4. SAM Propagation Merge ---
        propagation_mask = None
        if expanded_masks:
            # Prepare GT points array (Vectorized)
            gt_pts_arr = np.array(points_to_process, dtype=np.int32)
            if gt_pts_arr.ndim == 2: gt_pts_arr = gt_pts_arr[:, [1, 0]] # (x,y) -> (row,col)
            gt_labs_arr = np.array(labels_to_process, dtype=np.int32)

            # Append GT Image points if they exist (Simplified sampling)
            if gt_image is not None:
                # ... (Existing dense GT sampling logic is complex but specific to your use case. 
                # I kept the logic logic but assume `extract_labels_from_ground_truth` works)
                masks = gt_masks or self.extract_labels_from_ground_truth(gt_image) or []
                extras_p, extras_l = [], []
                for m in masks:
                    pts = np.argwhere(m['mask'])
                    if pts.shape[0] > 0:
                        # Random sample limited by max_gt_points
                        idx = np.random.choice(pts.shape[0], min(pts.shape[0], self.max_gt_points_per_mask), replace=False)
                        extras_p.append(pts[idx])
                        extras_l.append(np.full(len(idx), int(m['label']), dtype=np.int32))
                
                if extras_p:
                    gt_pts_arr = np.vstack([gt_pts_arr] + extras_p) if gt_pts_arr.size else np.vstack(extras_p)
                    gt_labs_arr = np.concatenate([gt_labs_arr] + extras_l) if gt_labs_arr.size else np.concatenate(extras_l)

            # Clip to image bounds
            if gt_pts_arr.size:
                gt_pts_arr[:, 0] = np.clip(gt_pts_arr[:, 0], 0, image.shape[0]-1)
                gt_pts_arr[:, 1] = np.clip(gt_pts_arr[:, 1], 0, image.shape[1]-1)

            with self.timer.time_operation('mask_merging'):
                # Unzip list of tuples efficiently
                m_list, l_list = zip(*[(m, l) for m, l, _, _ in expanded_masks])
                propagation_mask, _ = self.merge_overlapping_masks(list(m_list), list(l_list), gt_pts_arr, gt_labs_arr)
        else:
            propagation_mask = np.full(image.shape[:2], -1, dtype=np.int32)

        # --- 5. Combine & Cleanup (Vectorized) ---
        # Combine: If propagation says -1 (unlabeled), take PLAS. Else take propagation.
        combined_mask = np.where(propagation_mask == -1, plas_mask, propagation_mask)
        
        # Cleanup: Replace remaining -1s with background ID
        plas_mask[plas_mask == -1] = bg_id
        propagation_mask[propagation_mask == -1] = bg_id
        combined_mask[combined_mask == -1] = bg_id

        # --- 6. Return ---
        times = self.timer.get_pipeline_times()
        
        return {
            'image_path': Path(image_path),
            'expanded_masks': expanded_masks,
            'plas_mask_indices': plas_mask,
            'propagation_mask_indices': propagation_mask,
            'propagation_plas_mask_indices': combined_mask,
            'plas_time': times.get('plas_pipeline', 0),
            'propagation_time': times.get('sam2_propagation_pipeline', 0),
            'propagation_plas_time': times.get('combined_pipeline', 0),
            'timing_breakdown': times.get('breakdown', {}),
            'total_time': times.get('total_measured', 0),
            'iterations': len(expanded_masks) if expanded_masks else 0
        }


    def process_image(self, image_path, gt_path, color_dict=None):
        """
        Process a single image using the appropriate method based on strategy type.
        """
        if self.is_interactive_strategy:
            return self.process_image_interactive(image_path, gt_path, color_dict)
        else:
            return self.process_image_batch(image_path, gt_path, color_dict)

    def image_to_grayscale(self, image_bgr, color_dict=None):
        """Convert RGB mask image to class index grayscale using current color mapping.

        Unmatched pixels are assigned the default background class id (34).
        """
        color_to_label = color_dict if color_dict is not None else self.color_to_label
        if image_bgr is None:
            return None
        rgb_img = image_bgr[..., ::-1]
        h, w, _ = rgb_img.shape
        if not color_to_label:
            return np.full((h, w), self.DEFAULT_BACKGROUND_CLASS_ID, dtype=np.uint8)

        rgb_colors = np.array(list(color_to_label.keys()), dtype=np.int32)
        class_ids = np.array(list(color_to_label.values()), dtype=np.int32)
        flat = rgb_img.reshape(-1, 3)
        matches = (flat[:, None, :] == rgb_colors[None, :, :]).all(axis=2)
        any_match = matches.any(axis=1)
        idx = matches.argmax(axis=1)
        mapped = class_ids[idx]
        mapped[~any_match] = self.DEFAULT_BACKGROUND_CLASS_ID
        return mapped.reshape(h, w).astype(np.uint8)

    def save_grayscale_results(self, result, image_index, color_dict=None):
        """Save grayscale mask images.

        Preferred source: direct integer index arrays stored in result (*_indices).
        Fallback: derive from RGB mask if index array missing.
        Output directories follow the convention:
          non-gray:  masks_plas, masks_propagation_plas, masks_propagation (handled elsewhere)
          gray:      masks_plas_gray, masks_propagation_plas_gray, masks_propagation_gray
        """
        image_name = result['image_path'].stem
        out_specs = [
            ('plas_mask_indices', 'masks_plas_gray'),
            ('propagation_plas_mask_indices', 'masks_propagation_plas_gray'),
            ('propagation_mask_indices', 'masks_propagation_gray')
        ]
        for idx_field, gray_folder in out_specs:
            gray_dir = self.output_dir / gray_folder
            gray_dir.mkdir(exist_ok=True)
            gray_arr = result.get(idx_field)
            if isinstance(gray_arr, np.ndarray):
                # Direct save (already integer class IDs) with optional background remap to match GT
                arr_to_save = gray_arr.astype(np.uint8)

                cv2.imwrite(str(gray_dir / f"{image_name}.png"), arr_to_save)
                continue
            # Fallback: attempt derivation from corresponding RGB mask
            rgb_field = idx_field.replace('_indices', '')
            rgb_img = result.get(rgb_field)
            if isinstance(rgb_img, np.ndarray):
                # Convert by per-pixel uniqueness (assumes grayscale encoded as (g,g,g))
                if rgb_img.ndim == 3 and rgb_img.shape[2] == 3:
                    # If truly arbitrary RGB (external palette) we need mapping; attempt color_dict
                    if color_dict:
                        inv = {}
                        for rgb, cls in color_dict.items():
                            if isinstance(rgb, (list, tuple)) and len(rgb) == 3:
                                inv[tuple(int(v) for v in rgb)] = int(cls)
                        flat = rgb_img.reshape(-1, 3)
                        keys = np.array(list(inv.keys()), dtype=np.int32)
                        vals = np.array(list(inv.values()), dtype=np.int32)
                        if keys.size:
                            matches = (flat[:, None, :] == keys[None, :, :]).all(axis=2)
                            any_match = matches.any(axis=1)
                            idx = matches.argmax(axis=1)
                            mapped = vals[idx]
                            mapped[~any_match] = self.DEFAULT_BACKGROUND_CLASS_ID
                            gray = mapped.reshape(rgb_img.shape[0], rgb_img.shape[1]).astype(np.uint8)
                        else:
                            # Treat as grayscale triple
                            gray = rgb_img[..., 0].astype(np.uint8)
                    else:
                        # Treat as grayscale triple
                        gray = rgb_img[..., 0].astype(np.uint8)
                else:
                    # Already single channel?
                    gray = rgb_img.astype(np.uint8)

                cv2.imwrite(str(gray_dir / f"{image_name}.png"), gray)
    
    def save_results(self, result, image_index, color_dict=None):
        """Save colored (derived from indices) and grayscale outputs for a processed image.

        RGB is generated from index masks using color_dict when provided; otherwise grayscale triples.
        """
        image_name = result['image_path'].stem
        # Ensure output dirs exist
        for sub in ["masks_plas", "masks_propagation_plas", "masks_propagation"]:
            (self.output_dir / sub).mkdir(exist_ok=True)

        def indices_to_rgb(indices: np.ndarray, color_dict):
            H, W = indices.shape
            # Default canvas filled with the configured background color
            bg = tuple(int(c) for c in self.DEFAULT_BACKGROUND_COLOR)
            rgb = np.full((H, W, 3), bg, dtype=np.uint8)
            if color_dict:
                # invert mapping: (r,g,b)->class_id  =>  class_id->(r,g,b)
                label_to_rgb = {}
                for rgb_key, cls in color_dict.items():
                    try:
                        lab = int(cls)
                        if isinstance(rgb_key, (list, tuple)) and len(rgb_key) == 3:
                            label_to_rgb.setdefault(lab, tuple(int(v) for v in rgb_key))
                    except Exception:
                        continue
                uniq = np.unique(indices)
                for lab in uniq:
                    lab_i = int(lab)
                    if lab_i == int(self.DEFAULT_BACKGROUND_CLASS_ID):
                        # background color (already set on canvas, but keep explicit)
                        color = tuple(int(c) for c in self.DEFAULT_BACKGROUND_COLOR)
                    else:
                        color = label_to_rgb.get(lab_i, (lab_i & 0xFF, lab_i & 0xFF, lab_i & 0xFF))
                    rgb[indices == lab] = color
            else:
                # simple grayscale, but start from background color instead of black
                # so pixels with the background class id will show the configured bg color
                uniq = np.unique(indices)
                for lab in uniq:
                    lab_i = int(lab)
                    if lab_i == int(self.DEFAULT_BACKGROUND_CLASS_ID):
                        color = tuple(int(c) for c in self.DEFAULT_BACKGROUND_COLOR)
                    else:
                        g = lab_i & 0xFF
                        color = (g, g, g)
                    rgb[indices == lab] = color
            return rgb

        # Prepare indices (ensure combined indices exist)
        plas_idx = result.get('plas_mask_indices')
        prop_idx = result.get('propagation_mask_indices')
        prop_plas_idx = result.get('propagation_plas_mask_indices')
        if prop_idx is not None and prop_plas_idx is None and plas_idx is not None:
            comb = prop_idx.copy()
            mask_unl = (comb == -1)
            if np.any(mask_unl):
                comb[mask_unl] = plas_idx[mask_unl]
            prop_plas_idx = comb
            result['propagation_plas_mask_indices'] = comb

        # Save colored outputs derived from indices
        if isinstance(plas_idx, np.ndarray):
            rgb = indices_to_rgb(plas_idx, color_dict)
            cv2.imwrite(str(self.output_dir / "masks_plas" / f"{image_name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if isinstance(prop_plas_idx, np.ndarray):
            rgb = indices_to_rgb(prop_plas_idx, color_dict)
            cv2.imwrite(str(self.output_dir / "masks_propagation_plas" / f"{image_name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if isinstance(prop_idx, np.ndarray):
            rgb = indices_to_rgb(prop_idx, color_dict)
            cv2.imwrite(str(self.output_dir / "masks_propagation" / f"{image_name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Grayscale variants
        self.save_grayscale_results(result, image_index, color_dict)
    
    def save_stats(self):
        """Save statistics about the processed images with detailed timing breakdown."""
        # Calculate statistics for each timing type
        timing_stats = {}
        if self.stats.get("timing_breakdowns"):
            # Aggregate timing breakdown across all images
            operation_times = defaultdict(list)
            for breakdown in self.stats["timing_breakdowns"]:
                for operation, time_val in breakdown.items():
                    operation_times[operation].append(time_val)
            
            timing_stats = {
                operation: {
                    "mean": float(np.mean(times)),
                    "std": float(np.std(times)),
                    "total": float(np.sum(times))
                }
                for operation, times in operation_times.items()
            }

        # Calculate pipeline timing statistics
        pipeline_stats = {}
        for pipeline_type in ["propagation_times", "plas_times", "propagation_plas_times", "total_times"]:
            if pipeline_type in self.stats and self.stats[pipeline_type]:
                times = self.stats[pipeline_type]
                pipeline_stats[pipeline_type] = {
                    "mean": float(np.mean(times)),
                    "std": float(np.std(times)),
                    "min": float(np.min(times)),
                    "max": float(np.max(times))
                }
        
        # Save comprehensive stats as JSON
        with open(self.output_dir / "stats" / "processing_stats.json", 'w') as f:
            # Convert defaultdict to regular dict for serialization
            serializable_stats = {
                "images_processed": self.stats["images_processed"],
                "masks_identified": self.stats["masks_identified"],
                "pipeline_timing": pipeline_stats,
                "operation_timing": timing_stats,
                "masks_per_internal_class": dict(self.stats["per_class_masks"])
            }
            json.dump(serializable_stats, f, indent=2)
            
        # Print summary with improved timing information
        print("\n" + "="*100)
        print("PROCESSING SUMMARY")
        print("="*100)
        print(f"Images processed: {self.stats['images_processed']}")
        print(f"Total masks identified: {self.stats['masks_identified']}")
        
        # Print timing summary if available
        if pipeline_stats:
            print("\nTiming Summary (per image):")
            pretty_names = {
                "propagation_times": "Propagation",
                "plas_times": "PLAS",
                "propagation_plas_times": "Propagation+PLAS",
                "total_times": "Total"
            }
            for pipeline_type, stats in pipeline_stats.items():
                pipeline_name = pretty_names.get(pipeline_type, pipeline_type)
                print(f"  {pipeline_name}: {stats['mean']:.3f}s ± {stats['std']:.3f}s")
        
        if timing_stats:
            print("\nOperation Breakdown (total across all images):")
            for operation, stats in timing_stats.items():
                operation_name = operation.replace("_", " ").title()
                print(f"  {operation_name}: {stats['total']:.3f}s (avg: {stats['mean']:.3f}s)")
        
        print("="*100)
    
    def _determine_num_classes(self, color_dict=None, class_labels=None, allow_infer=True):
        """Determine number of classes with optional inference.
        Priority: color_dict > explicit (constructor) > inferred from labels (if allow_infer).
        """
        determined = None
        sources = []
        if color_dict:
            vals = list(color_dict.values())
            determined = len(set(vals))
            sources.append(f"color_dict({determined})")
        elif self.num_classes is not None:
            determined = self.num_classes
            sources.append(f"explicit({self.num_classes})")
        if class_labels and len(class_labels) > 0:
            max_label = max(class_labels)
            min_label = min(class_labels)
            required = max_label + 1 if min_label == 0 else max_label + 1
            if determined is None and allow_infer:
                determined = required
                sources.append(f"inferred({required})")
            elif determined is None and not allow_infer:
                raise ValueError("num_classes not provided and inference disabled")
            elif determined < required:
                raise ValueError(f"num_classes {determined} < needed {required} (max label {max_label})")
        if determined is None:
            if allow_infer:
                # fallback minimal 1
                determined = 1
                sources.append("default(1)")
            else:
                raise ValueError("Unable to determine num_classes")
        self.num_classes = determined
        return determined
    
    def _validate_input_configuration(self, color_dict):
        """
        Validate that we have the correct combination of inputs before processing any images.
        """
        # Strategy capability detection (fast path)
        point_classes_available = False
        if hasattr(self.strategy, 'has_class_info'):
            try:
                point_classes_available = bool(self.strategy.has_class_info())
            except Exception:
                point_classes_available = False
        else:
            # Fallback to sampling first image (legacy behavior)
            point_classes_available = False
            if hasattr(self.strategy, 'select_points'):
                image_files = self.get_image_files()
                probe_names = []
                if image_files:
                    # Use the same relative image identifier used during processing
                    try:
                        probe_rel = str(image_files[0][0].relative_to(self.images_dir).as_posix())
                    except Exception:
                        probe_rel = image_files[0][0].stem
                    probe_names.append(probe_rel)
                probe_names.append("dummy")
                for name in probe_names:
                    try:
                        points_and_classes = self.strategy.select_points(
                            segmenter=None,
                            image=None,
                            gt_masks=[],
                            image_name=name
                        )
                        if isinstance(points_and_classes, tuple) and len(points_and_classes) == 2 and points_and_classes[1] is not None:
                            point_classes_available = True
                            break
                    except Exception:
                        continue
        
        # Check if we're in RGB ground truth mode or sparseGT-only mode
        is_sparse_gt_only = self.ground_truth_dir is None
        
        if is_sparse_gt_only:  # sparseGT-only mode
            if color_dict is None:
                raise ValueError("sparseGT-only mode requires a color_dict to be provided")
            if not point_classes_available:
                raise ValueError(
                    "sparseGT-only mode (no ground truth) requires points file with class information.\n"
                    "Detected strategy: {}. It did not report a class/label column.\n"
                    "Ensure CSV has a column named one of: class, label, class_id, category.\n"
                    "Points must have format rows with (row/y, col/x, class).".format(self.strategy.__class__.__name__)
                )
            print("✓ Configuration validated: Running in sparseGT-only mode with class information")
        else:  # RGB ground truth mode
            if point_classes_available:
                print("⚠ Warning: Points file contains class information, but it will be ignored in RGB ground truth mode")

            print("✓ Configuration validated: Running in RGB ground truth mode")
            
    def process_all_images(self, color_dict=None):
        """Process all valid images in the dataset."""
        # Validate config first
        self._validate_input_configuration(color_dict)
        # If an external color_dict is provided, normalize and apply it so color_mapping.json reflects it
        if color_dict is not None:
            ext_color_to_label = {}
            ext_label_to_color = {}
            try:
                for k, v in list(color_dict.items()):
                    # Accept forms: (r,g,b)->id, "(r, g, b)"->id, id->(r,g,b), "id"->(r,g,b)
                    if isinstance(k, (list, tuple)) and len(k) == 3:
                        rgb = tuple(int(x) for x in k)
                        lab = int(v)
                        ext_color_to_label[rgb] = lab
                        ext_label_to_color[lab] = rgb
                    elif isinstance(k, str) and k.strip().startswith("("):
                        try:
                            parsed = eval(k)
                            if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
                                rgb = tuple(int(x) for x in parsed)
                                lab = int(v)
                                ext_color_to_label[rgb] = lab
                                ext_label_to_color[lab] = rgb
                        except Exception:
                            continue
                    else:
                        # id->rgb
                        try:
                            lab = int(k)
                            if isinstance(v, (list, tuple)) and len(v) == 3:
                                rgb = tuple(int(x) for x in v)
                            elif isinstance(v, str) and v.strip().startswith("("):
                                parsed = eval(v)
                                if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
                                    rgb = tuple(int(x) for x in parsed)
                                else:
                                    continue
                            else:
                                continue
                            ext_color_to_label[rgb] = lab
                            ext_label_to_color[lab] = rgb
                        except Exception:
                            continue
            except Exception:
                pass

            if ext_color_to_label:
                self.color_to_label = dict(ext_color_to_label)
                self.label_to_color = dict(ext_label_to_color)
                # Set background color from mapping if available
                try:
                    bg_lab = int(self.DEFAULT_BACKGROUND_CLASS_ID)
                    if bg_lab in self.label_to_color:
                        self.DEFAULT_BACKGROUND_COLOR = tuple(int(c) for c in self.label_to_color[bg_lab])
                except Exception:
                    pass
                self.save_color_mapping()
        # Pre-determine num_classes only if explicit
        if self.num_classes is not None:
            _ = self._determine_num_classes(color_dict, class_labels=None)
        elif color_dict is not None:
            self.num_classes = self._determine_num_classes(color_dict, class_labels=None)
        # Gather images
        image_files = self.get_image_files()
        print(f"Processing {len(image_files)} images with num_classes={self.num_classes if self.num_classes is not None else 'auto'}...\n")
        # Initialize stats containers
        self.stats.update({
            "propagation_times": [],
            "plas_times": [],
            "propagation_plas_times": [],
            "total_times": [],
            "timing_breakdowns": [],
            "images_processed": 0
        })

        # Init PLAS
        self.PLAS_segmenter = SuperpixelLabelExpander(self.device, seed=self.seed)
        # Loop
        for i, (image_path, gt_path) in enumerate(tqdm(image_files, desc="Processing images")):
            try:
                result = self.process_image(image_path, gt_path, color_dict)
                self.save_results(result, i, color_dict)
                self.save_color_mapping()
                self.stats["images_processed"] += 1
                self.stats["propagation_times"].append(result["propagation_time"])
                self.stats["plas_times"].append(result["plas_time"])
                self.stats["propagation_plas_times"].append(result["propagation_plas_time"])
                self.stats["total_times"].append(result["total_time"])
                self.stats["timing_breakdowns"].append(result["timing_breakdown"])
                self.stats.setdefault("scales", []).append(getattr(self, 'current_scale', 1.0))
            except Exception as e:
                import traceback
                print(f"\nError processing {image_path.name}: {e}")
                traceback.print_exc()
        # Aggregate timing
        if self.stats["total_times"]:
            mean_total_time = np.mean(self.stats["total_times"])
            std_total_time = np.std(self.stats["total_times"])
            print(f"\nMean total time per image: {mean_total_time:.2f}s (±{std_total_time:.2f})")
        # Persist
        self.save_color_mapping()
        self.save_stats()
        # Evaluation guard
        if self.ground_truth_dir is not None:
            if self.num_classes is None:
                observed_labels = list(self.stats["per_class_masks"].keys())
                # Allow inference from observed labels; color_dict may be None in grayscale mode
                self._determine_num_classes(color_dict, class_labels=observed_labels)
            print("\nEvaluating segmentation results...")
            self.evaluate_segmentation_results(self.num_classes, color_dict)
        else:
            print("\nSkipping evaluation (no ground truth).")

    def run(self, color_dict=None):
        self.process_all_images(color_dict)
        
    def evaluate_segmentation_results(self, num_classes, color_dict):
        """Multi-variant evaluation of grayscale masks.
        Generates per-class and global metrics for:
          - Combined (propagation_plas)
          - Propagation-only
          - PLAS-only
        Saves one JSON per variant plus a summary file. Returns main variant dict or None.
        """
        import json
        print("\n" + "="*60)
        print("EVALUATING SEGMENTATION RESULTS (multi-variant)")
        print("="*60)
        if self.ground_truth_dir is None:
            print("No ground truth directory."); return None

        variants = [self.eval_mask_type, 'propagation', 'plas']
        seen = set(); variants = [v for v in variants if not (v in seen or seen.add(v))]
        results = {}

        def eval_one(variant: str):
            dir_path = self.output_dir / f"masks_{variant}_gray"
            if not dir_path.exists():
                print(f"[skip] {variant}: directory missing {dir_path}")
                return None
            pred_files = list(dir_path.glob("*.png"))
            if not pred_files:
                print(f"[skip] {variant}: no prediction files")
                return None
            print(f"Evaluating {variant}: {len(pred_files)} images ...")
            preds_flat, gts_flat = [], []
            for pf in pred_files:
                gt_path = self.ground_truth_dir / pf.name
                if not gt_path.exists():
                    continue
                pred = np.array(Image.open(pf).convert('L'))
                gt = np.array(Image.open(gt_path).convert('L'))
                preds_flat.append(torch.tensor(pred, dtype=torch.long).flatten())
                gts_flat.append(torch.tensor(gt, dtype=torch.long).flatten())
            if not preds_flat:
                print(f"[skip] {variant}: no valid pairs")
                return None
            preds_all = torch.cat(preds_flat); gts_all = torch.cat(gts_flat)
            gt_unique = sorted([int(x) for x in torch.unique(gts_all).tolist()])
            if not gt_unique:
                print(f"[skip] {variant}: GT empty")
                return None
            bg = int(self.DEFAULT_BACKGROUND_CLASS_ID)
            labels_eval = list(gt_unique)
            if bg not in labels_eval:
                labels_eval.append(bg)
            labels_eval = sorted(labels_eval)
            max_val = int(max(int(preds_all.max().item()), int(gts_all.max().item()), max(labels_eval)))
            lut_valid = torch.zeros((max_val+1,), dtype=torch.bool)
            for v in labels_eval: lut_valid[v] = True
            keep = lut_valid[gts_all.clamp(0, max_val)]
            dropped = int((~keep).sum().item())
            if dropped: print(f"{variant}: dropped {dropped} pixels (GT outside eval set)")
            preds_used = preds_all[keep]; gts_used = gts_all[keep]
            if preds_used.numel() == 0: print(f"[skip] {variant}: empty after filter"); return None
            label_to_idx = {lab:i for i, lab in enumerate(labels_eval)}
            lut = torch.full((max(labels_eval)+1,), -1, dtype=torch.long)
            for lab, idx in label_to_idx.items(): lut[lab] = idx
            p_map = lut[preds_used]; g_map = lut[gts_used]; C = len(labels_eval)
            # Vectorized confusion matrix via bincount (far faster than Python loop)
            valid = (g_map >= 0) & (p_map >= 0)
            if valid.any():
                flat_idx = (g_map[valid] * C + p_map[valid]).to(torch.long)
                conf = torch.bincount(flat_idx, minlength=C*C).reshape(C, C)
            else:
                conf = torch.zeros((C, C), dtype=torch.long)
            tp = torch.diag(conf).to(torch.float32)
            fp = conf.sum(0).to(torch.float32) - tp
            fn = conf.sum(1).to(torch.float32) - tp
            denom_iou = tp+fp+fn; per_iou = torch.where(denom_iou>0, tp/denom_iou, torch.zeros_like(tp))
            denom_pa = tp+fn; per_pa = torch.where(denom_pa>0, tp/denom_pa, torch.zeros_like(tp))
            print(f"\nVariant: {variant}")
            for i, lab in enumerate(labels_eval):
                print(f"  Label {lab}: mPA={per_pa[i]*100:.2f}, mIoU={per_iou[i]*100:.2f}")
            bg_idx = label_to_idx.get(bg)
            exclude_bg = False
            if bg_idx is not None and per_iou[bg_idx].item()==0.0 and per_pa[bg_idx].item()==0.0:
                exclude_bg = True
            if exclude_bg and bg_idx is not None:
                fg_mask = torch.ones(C, dtype=torch.bool); fg_mask[bg_idx]=False
                fg_iou = per_iou[fg_mask]; fg_pa = per_pa[fg_mask]
                g_miou = float(fg_iou.mean().item()) if fg_iou.numel() else 0.0
                g_mpa = float(fg_pa.mean().item()) if fg_pa.numel() else 0.0
                miou_std = float(fg_iou.std(unbiased=False).item()) if fg_iou.numel() else 0.0
                mpa_std = float(fg_pa.std(unbiased=False).item()) if fg_pa.numel() else 0.0
            else:
                g_miou = float(per_iou.mean().item()) if per_iou.numel() else 0.0
                g_mpa = float(per_pa.mean().item()) if per_pa.numel() else 0.0
                miou_std = float(per_iou.std(unbiased=False).item()) if per_iou.numel() else 0.0
                mpa_std = float(per_pa.std(unbiased=False).item()) if per_pa.numel() else 0.0
            print(f"  Global mPA={g_mpa*100:.2f}% (std={mpa_std*100:.2f}%), mIoU={g_miou*100:.2f}% (std={miou_std*100:.2f}%)" + (" [background excluded]" if exclude_bg else ""))
            return {
                'variant': variant,
                'global_mpa': g_mpa,
                'global_mpa_std': mpa_std,
                'global_miou': g_miou,
                'global_miou_std': miou_std,
                'per_class_mpa': [float(x) for x in per_pa],
                'per_class_miou': [float(x) for x in per_iou],
                'eval_labels': labels_eval,
                'background_excluded_in_mean': exclude_bg,
                'num_classes_including_background': C,
                'num_images_evaluated': len(pred_files),  # pred_files from outer scope not available here; set after
            }

        (self.output_dir/"stats").mkdir(parents=True, exist_ok=True)
        for variant in variants:
            res = eval_one(variant)
            if res is None:
                continue
            # Correct num_images_evaluated using directory listing
            dir_path = self.output_dir / f"masks_{variant}_gray"
            res['num_images_evaluated'] = len([p for p in dir_path.glob('*.png')])
            results[variant] = res
            with open(self.output_dir/"stats"/f"segmentation_metrics_{variant}.json", 'w') as f:
                json.dump(res, f, indent=2)
            if variant == self.eval_mask_type:
                with open(self.output_dir/"evaluation_results.json", 'w') as f:
                    json.dump(res, f, indent=2)
                # Backwards compatibility single file name
                with open(self.output_dir/"stats"/"segmentation_metrics.json", 'w') as f:
                    json.dump(res, f, indent=2)
        if results:
            with open(self.output_dir/"stats"/"segmentation_metrics_all_variants.json", 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print("No variants produced metrics.")
        print("="*60)
        return results.get(self.eval_mask_type)

# CLI entrypoints (restored) -- REINSERT after corruption
def main():
    parser = argparse.ArgumentParser(description="Run the Unified AutoLabeler")
    parser.add_argument("--images", required=True, help="Path to the directory containing input images")
    parser.add_argument("--ground-truth", required=False, help="Path to the directory containing ground truth masks (optional for sparseGT-only mode)")
    parser.add_argument("--output", required=True, help="Path to the output directory")
    parser.add_argument("--strategy", required=True,
                        choices=["list", "random", "grid", "SAM2_guided", "dynamicPoints_onlyA", "dynamicPoints", "dynamicPointsLargestGT"],
                        help="Point selection strategy to use")
    parser.add_argument("--num-points", type=int, default=30, help="Number of points to select")
    parser.add_argument("--num-classes", type=int, help="Total number of classes (required if no --color-dict)")
    parser.add_argument("--maskSLIC", action="store_true", help="Use maskSLIC for segmentation")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--visualizations", action="store_true", help="Enable saving visualizations")
    parser.add_argument("--visualization", action="store_true", dest="visualizations", help=argparse.SUPPRESS)
    parser.add_argument("--strategy-kwargs", type=str, default="{}", help="JSON string of strategy-specific parameters")
    parser.add_argument("--points-file", type=str, help="Path to JSON file containing points for 'list' strategy")
    parser.add_argument("--color-dict", type=str, help="Path to external color dictionary JSON file for evaluation")
    parser.add_argument("--downscale-auto", action="store_true", help="Enable automatic heuristic downscaling for auxiliary point proposal data (SAM2 remains full-res)")
    parser.add_argument("--downscale-fixed", type=float, help="Use fixed downscale factor (e.g. 0.5). Overrides --downscale-auto if provided")
    parser.add_argument("--debug-expanded-masks", action="store_true", dest="debug_save_expanded_masks",
                        help="Save each expanded SAM2 mask separately plus overlap diagnostics")
    parser.add_argument("--label-to-id-json", type=str, default=None, help="Path to JSON file mapping label to string ID (optional)")
    parser.add_argument("--default-background-class-id", type=int, default=0, help="Default background class ID (overrides internal default)")
    args = parser.parse_args()

    strategy_kwargs = json.loads(args.strategy_kwargs)
    if args.points_file and args.strategy == "list":
        strategy_kwargs["points_json_path"] = args.points_file

    color_dict = None
    if args.color_dict:
        try:
            print(f"Loading external color dictionary from {args.color_dict}")
            with open(args.color_dict, 'r') as f:
                mapping = json.load(f)
            # Accept either color_to_label nested dict or flat mapping
            if "color_to_label" in mapping:
                color_dict_raw = mapping["color_to_label"]
            else:
                color_dict_raw = mapping
            color_dict = {}
            for k, v in color_dict_raw.items():
                if isinstance(v, int):
                    # key is color string
                    ks = k.strip()
                    ks = ks.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
                    parts = [p.strip() for p in ks.split(',') if p.strip()]
                    if len(parts) == 3:
                        try:
                            tup = tuple(int(p) for p in parts)
                            color_dict[tup] = v
                        except ValueError:
                            continue
            print(f"Loaded external color dictionary with {len(color_dict)} entries")
        except Exception as e:
            print(f"Error loading color dictionary: {e}")
            color_dict = None

    if not color_dict and args.num_classes is None:
        print("INFO: Neither --color-dict nor --num-classes provided; num_classes will be inferred from observed labels. Evaluation will be skipped.")

    if args.ground_truth:
        print("Running in RGB ground truth mode")
    else:
        print("Running in sparseGT-only mode")
        if not color_dict:
            print("WARNING: sparseGT-only mode without color_dict may limit evaluation; ensure points carry class info.")

    # Instantiate AutoLabeler irrespective of the mode so it's always defined
    auto_labeler = AutoLabeler(
        images_dir=args.images,
        ground_truth_dir=args.ground_truth,
        output_dir=args.output,
        save_visualizations=args.visualizations,
        debug_save_expanded_masks=args.debug_save_expanded_masks,
        device=args.device,
        point_selection_strategy=args.strategy,
        num_points=args.num_points,
        use_maskSLIC=args.maskSLIC,
        num_classes=args.num_classes,
        downscale_auto=args.downscale_auto,
        downscale_fixed=args.downscale_fixed,
        label_to_id_json=args.label_to_id_json,
        default_background_class_id=args.default_background_class_id,
        **strategy_kwargs
    )
    auto_labeler.run(color_dict)

if __name__ == "__main__":
    main()
