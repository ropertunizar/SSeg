import json
import numpy as np
from abc import ABC, abstractmethod
import time
import cv2
import math

from typing import Optional
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class PointSelectionStrategy(ABC):
    """Base class for point selection strategies."""
    def __init__(self, seed: Optional[int] = None, **kwargs):
        """Initialize the strategy with optional parameters."""
        self.uses_contrastive_learning = False
        self._seed = seed
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    
    @abstractmethod
    def select_points(self, segmenter, image, expanded_masks=None):
        """
        Select points for expansion.
        
        Args:
            segmenter: The SAM segmenter
            image: The input image
            expanded_masks: List of already expanded masks
            
        Returns:
            tuple: (points, classes) where:
                - points: List of (y, x) coordinates
                - classes: List of class labels or None if no class info available
        """
        pass

    # Optional hook for strategies that support precomputation; safe no-op default
    def setup_simple(self, image, generated_masks):  # pragma: no cover - trivial
        return

    # Capability query: does this strategy provide per-point class labels (without needing an image)?
    # Default: False. Strategies that can report class availability (e.g., CSV list) should override.
    def has_class_info(self, image_name=None):  # pragma: no cover - simple default
        return False

class ListSelectionStrategy(PointSelectionStrategy):
    """Extremely simple CSV-based list strategy.

    Expected CSV columns (case-insensitive variants supported):
      - row / y
      - col / x
      - optional: image (if per-image annotation file; if absent, all rows apply to every image)
      - optional: class / label

    Behavior:
      * Reads the CSV once, caches a pandas DataFrame & detected column names.
      * On select_points(image_name), filters rows by image column if present.
      * Returns (points ndarray [N,2], classes list|None) with EXACT pixel coordinates (no scaling heuristics).
    """

    def __init__(self, points_path=None, annotations_file=None, points_file=None, points_json_path=None, image_name=None, **kwargs):
        super().__init__(**kwargs)
        # Canonical argument: points_path. Accept legacy aliases for backward compatibility.
        if points_path is None:
            points_path = annotations_file or points_file or points_json_path
        if points_path is None:
            try:
                print("[ListSelectionStrategy] WARNING: No points_path (or legacy annotations_file/points_file) provided. Class info unavailable.")
            except Exception:
                pass
        self.points_path = points_path
        # Maintain existing attribute name expected elsewhere
        self.annotations_file = points_path
        self.image_name = image_name
        self.uses_contrastive_learning = False
        self._df = None
        self._cols = None  # dict: image,row,col,label(optional)

    # -------- Internal helpers --------
    def _load_csv_once(self):
        if self._df is not None or self.annotations_file is None:
            return
        import os, pandas as pd
        if not os.path.exists(self.annotations_file):
            try:
                print(f"[ListSelectionStrategy] Annotations file does NOT exist: {self.annotations_file}")
            except Exception:
                pass
            self._df = None
            return
        # Basic file stats + first line preview for debugging
        try:
            fsize = os.path.getsize(self.annotations_file)
            with open(self.annotations_file, 'r', errors='ignore') as fh:
                first_two = [next(fh, '').rstrip('\n') for _ in range(2)]
        except Exception:
            pass
        # Let pandas auto-detect delimiter; simplest.
        try:
            self._df = pd.read_csv(self.annotations_file)
        except Exception:
            self._df = None
            return
        try:
            print(f"[ListSelectionStrategy] DataFrame shape after initial read: {None if self._df is None else self._df.shape}")
        except Exception:
            pass
        # If only one column parsed, attempt a fallback with python engine and potential alternative separators
        if self._df is not None and self._df.shape[1] == 1:
            single_col_name = str(self._df.columns[0])
            if ',' in single_col_name or ';' in single_col_name:
                try:
                    print("[ListSelectionStrategy] Single wide column detected; retrying with engine='python' and sep=None for sniffing")
                    self._df = pd.read_csv(self.annotations_file, engine='python', sep=None)
                    print(f"[ListSelectionStrategy] DataFrame shape after retry: {self._df.shape}")
                except Exception as e:
                    try:
                        print(f"[ListSelectionStrategy] Retry failed: {e}")
                    except Exception:
                        pass
        # Detect columns (robust normalization: strip, lower, remove spaces & underscores)
        raw_cols = list(self._df.columns)
        norm = {}
        for c in raw_cols:
            nc = str(c).strip()
            # Normalize aggressively: lowercase, remove spaces, tabs, underscores, hyphens
            norm_key = nc.lower().replace(' ', '').replace('\t','').replace('_','').replace('-', '')
            norm[nc] = norm_key
        # Candidate sets (will normalize similarly)
        col_map = {
            'image': ['image','imagename','filename','file','name'],
            'row':   ['row','y'],
            'col':   ['col','column','x'],
            'label': ['class','label','labelcode','label_code','label code','label-code','label_code',
                      'classid','class_id','class id','category']
        }
        found = {}
        for role, candidates in col_map.items():
            cand_norm = [c.lower().replace(' ', '').replace('_','').replace('-', '') for c in candidates]
            for original, nval in norm.items():
                if nval in cand_norm:
                    found[role] = original
                    break
            if role not in found:
                try:
                    print(f"[ListSelectionStrategy] Did not detect {role} column. Candidates tried (normalized): {cand_norm}")
                except Exception:
                    pass
        # Debug prints for user visibility
        try:
            print(f"[ListSelectionStrategy] Loaded CSV: {self.annotations_file}")
            print(f"[ListSelectionStrategy] Raw columns: {raw_cols}")
            print(f"[ListSelectionStrategy] Detected mapping: {found}")
            if 'label' not in found:
                print("[ListSelectionStrategy] No label column detected. Acceptable names include: class, label, label_code, label code, class_id, category")
        except Exception:
            pass
        # Require row & col; image optional; label optional
        if 'row' not in found or 'col' not in found:
            # Invalidate if essential columns missing
            try:
                print(f"[ListSelectionStrategy] Essential columns missing (row in found? {'row' in found}, col in found? {'col' in found}). Invalidating DataFrame.")
            except Exception:
                pass
            self._df = None
        self._cols = found

    def _filter_rows(self, image_name):
        if self._df is None:
            return np.empty((0,2), dtype=int), None
        df = self._df
        cols = self._cols or {}
        # If no image col, take all rows
        if 'image' not in cols:
            sub = df
        else:
            if image_name is None:
                return np.empty((0,2), dtype=int), None
            import os
            # Normalize input name and prepare candidate variants
            name_raw = str(image_name).strip()
            name_lower = name_raw.lower().replace('\\', '/').lstrip('./')
            basename = os.path.basename(name_lower)
            base_no_ext = os.path.splitext(basename)[0]
            candidates = {
                name_lower,
                basename,
                base_no_ext,
                './' + name_lower,
                name_lower.lstrip('./')
            }

            # Normalize CSV column values for robust comparison (forward slashes, lowercased)
            col_series = df[cols['image']].astype(str).str.replace('\\\\', '/').str.replace('\\', '/').str.lower().str.lstrip('./')

            # 1) fast isin check against prepared candidates
            try:
                mask = col_series.isin(candidates)
            except Exception:
                mask = col_series == name_lower

            # 2) fallback: endswith basename (handles absolute/full paths in CSV)
            if not mask.any():
                try:
                    mask = col_series.str.endswith(basename)
                except Exception:
                    mask = col_series == name_lower
            # 3) fallback: compare stems (handle .jpg vs .jpeg etc)
            if not mask.any():
                try:
                    from pathlib import PurePosixPath
                    csv_stems = col_series.apply(lambda s: PurePosixPath(s).stem.lower())
                    stem_eq = (csv_stems == base_no_ext)
                    if stem_eq.any():
                        mask = stem_eq
                except Exception:
                    # robust fallback for older pandas/numpy; build list and compare
                    try:
                        from pathlib import PurePosixPath
                        csv_list = col_series.tolist()
                        import numpy as _np
                        stem_equal_mask = _np.array([PurePosixPath(s).stem.lower() == base_no_ext for s in csv_list], dtype=bool)
                        if stem_equal_mask.any():
                            mask = stem_equal_mask
                    except Exception:
                        pass

            # 4) final fallback: if name had no extension, try appending common extensions
            if not mask.any() and '.' not in name_lower:
                for ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
                    try:
                        mask_ext = col_series == (name_lower + ext)
                    except Exception:
                        mask_ext = col_series == (name_lower + ext)
                    if mask_ext.any():
                        mask = mask_ext
                        break
            sub = df[mask]
            # Diagnostic: if no rows matched, print helpful context to debug mismatches
            if sub.empty:
                try:
                    uniq = col_series.unique()
                    sample_vals = [str(x) for x in (uniq[:10] if hasattr(uniq, '__getitem__') else list(uniq))]
                except Exception:
                    sample_vals = []
                try:
                    print("[ListSelectionStrategy][DEBUG] No rows matched for image_name=", image_name)
                    print("  candidates=", sorted(list(candidates)))
                    print("  basename=", basename, " base_no_ext=", base_no_ext)
                    print("  first 10 unique CSV image values=", sample_vals)
                    print("  any exact match?", int((col_series == name_lower).any()), " any basename match?", int((col_series == basename).any()))
                except Exception:
                    pass
        if sub.empty:
            return np.empty((0,2), dtype=int), None
        rcol = self._cols['row']
        ccol = self._cols['col']
        pts = sub[[rcol, ccol]].to_numpy(dtype=float)
        # Optional classes
        if 'label' in self._cols:
            classes = sub[self._cols['label']].tolist()
        else:
            classes = None
        return pts, classes

    # -------- Public API --------
    def select_points(self, segmenter, image, expanded_masks=None, image_name=None):
        self._load_csv_once()
        active_image = image_name or self.image_name
        pts, classes = self._filter_rows(active_image)
        if pts.size == 0:
            return np.empty((0,2), dtype=int), classes
        # If we have the image shape, clamp; otherwise just cast to int
        if image is not None:
            H, W = image.shape[:2]
            pts = np.clip(pts, [0,0], [H-1, W-1])

        return pts.astype(int), classes

    def has_class_info(self, image_name=None):
        """Return True if the loaded CSV has a detectable class/label column.

        Independent of whether a particular image has rows; this avoids false negatives during
        early configuration validation when the first image sampled has no annotated points.
        """
        self._load_csv_once()
        available = bool(self._cols and 'label' in self._cols)
        try:
            if available:
                print(f"[ListSelectionStrategy] has_class_info=True (mapping={self._cols})")
            else:
                print(f"[ListSelectionStrategy] has_class_info=False (mapping={self._cols})")
        except Exception:
            pass
        return available

class RandomSelectionStrategy(PointSelectionStrategy):
    """Uniform random point sampler ignoring masks (baseline)."""

    def __init__(self, num_points=25, random_seed=None, seed=None, **kwargs):
        effective_seed = seed if seed is not None else random_seed
        super().__init__(seed=effective_seed, **kwargs)
        self.num_points = int(num_points)
        self.uses_contrastive_learning = False

    def select_points(self, segmenter, image, expanded_masks=None, image_name=None):
        """Return N random (y,x) coordinates within image bounds.

        Returns:
            (points_ndarray, None) where points_ndarray has shape (N,2) int.
        """
        H, W = image.shape[:2]
        ys = self._rng.integers(0, H, size=self.num_points, dtype=int)
        xs = self._rng.integers(0, W, size=self.num_points, dtype=int)
        pts = np.stack([ys, xs], axis=1)  # (N,2)
        return pts, None
    
class GridSelectionStrategy(PointSelectionStrategy):
    """
    Selects points by dividing the image into a grid with equal number of rows and columns,
    as close as possible but not greater than num_points. Points are placed at the center of each cell.
    If num_points is not a perfect square, uses the nearest square below num_points (e.g. for 5, uses 2x2=4).
    """
    def __init__(self, num_points=25, **kwargs):
        super().__init__(**kwargs)
        self.num_points = num_points
        self.uses_contrastive_learning = False

    def select_points(self, segmenter, image, expanded_masks=None, image_name=None):
        import numpy as np
        H, W = image.shape[:2]
        # Find the largest n such that n*n <= num_points
        n = int(np.floor(np.sqrt(self.num_points)))
        n_rows = n_cols = n
        cell_h = H / n_rows
        cell_w = W / n_cols

        points = []
        for row in range(n_rows):
            for col in range(n_cols):
                y = int((row + 0.5) * cell_h)
                x = int((col + 0.5) * cell_w)
                y = min(y, H - 1)
                x = min(x, W - 1)
                points.append((y, x))

        return points, None  # Always n*n points, no class information for grid selection

class SAM2GuidedSelectionStrategy(PointSelectionStrategy):
    """
    Pick up to num_points mask centroids from SAM2 masks using a simple area/confidence
    scoring and greedy non‑overlap. Falls back to random uncovered pixels if not enough
    good masks exist.

    Scoring:
        score = area_weight * norm(area) + conf_weight * norm(confidence)

    Overlap control:
        A candidate mask is accepted if IoU(candidate, union(selected)) < overlap_thresh.
        (Fast approximation; avoids pairwise IoU loop.)
    """
    
    def __init__(self, num_points=25, area_weight=0.9, conf_weight=0.1, **kwargs):
        """
        Args:
            num_points (int): Max number of points to return.
            area_weight (float): Weight for normalized mask area (0..1).
            conf_weight (float): Weight for normalized SAM confidence (0..1).
        """
        super().__init__(**kwargs)
        self.num_points = int(num_points)
        self.overlap_thresh = 0.30
        self.area_weight = float(area_weight)
        self.conf_weight = float(conf_weight)
        self.min_area = 50  # pixels
    # Use the base class RNG

    
    def select_points(self, segmenter, image, expanded_masks=None, image_name=None):
        """Return up to num_points (y,x) points.

        segmenter.masks is expected to be a list of dicts with at least:
            'segmentation': (H,W) bool/uint8 array
            'score' or 'predicted_iou': confidence float

        expanded_masks: optional list of (mask, seg_id, label) tuples already chosen;
        candidate masks with >25% pixel overlap (w.r.t. candidate area) are skipped.
        """
        H, W = image.shape[:2]
        if not hasattr(segmenter, 'masks') or not segmenter.masks:
            return np.empty((0, 2), dtype=int)

        skip_masks = []
        if expanded_masks:
            for em, *_ in expanded_masks:
                skip_masks.append(em.astype(bool))

        cand_masks = []  # stacked later
        cand_areas = []
        cand_scores = []

        for md in segmenter.masks:
            m = md.get('segmentation')
            if m is None:
                continue
            m = m.astype(bool)
            area = int(m.sum())
            if area < self.min_area:
                continue

            # overlap suppression w/ already expanded
            reject = False
            if skip_masks:
                inter_any = 0
                for sm in skip_masks:
                    inter = np.logical_and(m, sm).sum()
                    if area > 0 and inter / area > 0.25:
                        reject = True
                        break
                    inter_any += inter
            if reject:
                continue

            conf = md.get('score', md.get('predicted_iou', 0.0))
            cand_masks.append(m)
            cand_areas.append(float(area))
            cand_scores.append(float(conf))

        if not cand_masks:
            return np.empty((0, 2), dtype=int)

        masks_arr = np.stack(cand_masks, axis=0)  # (K,H,W)
        areas = np.asarray(cand_areas, dtype=float)
        confs = np.asarray(cand_scores, dtype=float)

        # Normalize features
        def _norm(v):
            vmin, vmax = float(v.min()), float(v.max())
            if vmax - vmin < 1e-8:
                return np.full_like(v, 0.5)
            return (v - vmin) / (vmax - vmin)
        areas_n = _norm(areas)
        confs_n = _norm(confs)
        scores = self.area_weight * areas_n + self.conf_weight * confs_n
        order = np.argsort(-scores)

        occupied = np.zeros((H, W), dtype=bool)
        chosen_idx = []
        for k in order:
            if len(chosen_idx) >= self.num_points:
                break
            mk = masks_arr[k]
            inter = np.logical_and(mk, occupied).sum()
            union = np.logical_or(mk, occupied).sum()
            iou = 0.0 if union == 0 else inter / union
            if iou < self.overlap_thresh:
                chosen_idx.append(k)
                occupied |= mk

        pts = []
        for k in chosen_idx:
            mk = masks_arr[k]
            ys, xs = np.nonzero(mk)
            if ys.size == 0:
                continue
            pts.append((int(round(ys.mean())), int(round(xs.mean()))))

        # Fill with random uncovered pixels if short
        n_missing = self.num_points - len(pts)
        if n_missing > 0:
            free_y, free_x = np.where(~occupied)
            if free_y.size > 0:
                # Deterministic because self._rng is created after global seeding
                idxs = self._rng.choice(free_y.size, size=min(n_missing, free_y.size), replace=False)
                for i in idxs:
                    pts.append((int(free_y[i]), int(free_x[i])))

        if not pts:
            return np.empty((0, 2), dtype=int)
        return np.asarray(pts, dtype=int)
    

class DynamicPointsSelectionStrategy(PointSelectionStrategy):
    """
    Iterative point selection combining exploration, exploitation, and feature dissimilarity.
    Use `setup(image, generated_masks)` once per image, then call `get_next_point(last_mask, last_feature)` each iteration.
    Visualize acquisition both as pixel heatmap (`display_acquisition_heatmap`) and mask-centroid scores (`display_mask_score`).
    """
    def __init__(self, num_points=25, lambda_balance=0.5, heatmap_fraction=0.5):
        super().__init__()
        self.num_points = num_points
        self.lambda_balance = lambda_balance
        # placeholders for per-image data
        self.image_shape = None
        self.d_max = None  # max Euclidean distance in image
        self.mask_centroids = None  # array of (y,x) for each unlabeled mask
        self.mask_weights = None    # importance weight per mask
        self.selected_points = []   # list of (y,x) already chosen
        self.selected_count = 0
        self.expanded_masks = []  # list of already expanded masks
        self.heatmap_fraction = heatmap_fraction  # fraction of points to select from heat
        self._pix_y = None              # (N,) pixel y coords flattened
        self._pix_x = None              # (N,) pixel x coords flattened
        self._d_prev_pix = None         # (N,) running min distance to any selected point
        self._d_centroid_min = None     # (M,) running min distance from each centroid to any selected point
        self._K = None                  # (N,M) cached smoothing kernel 1/(d+eps)
        self._den_C = None              # (N,) cached denominator sum(K)
        self._base_centroid_term = None # (N,M) (1 - d_to_cent/d_max)
        # Visualization state (single persistent window)
        self._fig = None
        self._ax = None
        self._im = None
        self._scatter = None

    def _compute_centroid(self, mask):
        ys, xs = np.nonzero(mask)
        return np.array([ys.mean(), xs.mean()])
    
    def select_points(self, segmenter, image, expanded_masks=None):
        return super().select_points(segmenter, image, expanded_masks)

    def _snapped_point(self, mask, y, x):
        """If (y,x) is outside mask, snap to nearest inside-mask pixel."""
        H, W = mask.shape
        yI = np.clip(int(round(y)), 0, H-1)
        xI = np.clip(int(round(x)), 0, W-1)
        if mask[yI, xI]:
            return yI, xI
        # find all inside pixels and choose the closest
        ys, xs = np.nonzero(mask)
        if len(ys)==0:
            return None
        d2 = (ys - yI)**2 + (xs - xI)**2
        idx = np.argmin(d2)
        return int(ys[idx]), int(xs[idx])

    def setup_simple(self, image, generated_masks):
        """
        Fast setup: use only the centroid of each mask, no propagation or IoU computation.
        Args:
            image: HxW[xC] numpy array
            generated_masks: list of dicts with 'mask' (HxW bool) and 'score' float
        """
        H, W = image.shape[:2]
        self.image_shape = (H, W)
        self.d_max = np.hypot(W, H)

        reps, areas, scores = [], [], []
        for m in generated_masks:
            mask = m['segmentation']
            area = mask.sum()
            if area == 0:
                continue
            c = self._compute_centroid(mask)
            reps.append(c)
            areas.append(area)
            scores.append(m['predicted_iou'])

        if not reps:
            raise ValueError("No valid masks")

        self.mask_centroids = np.vstack(reps)
        area_weight = 1.0
        score_weight = 0.0
        w = (area_weight * np.array(areas)) + (score_weight * np.array(scores) * np.array(areas))
        self.mask_weights = w / w.sum()

        # Precompute pixel grid and distances for efficiency
        ys = np.arange(H)
        xs = np.arange(W)
        Y, X = np.meshgrid(ys, xs, indexing='ij')
        self.pix = np.stack([Y.ravel(), X.ravel()], axis=1)  # (H*W, 2)
        # Precompute distances pixel->centroids once
        self.d_to_cent = cdist(self.pix, self.mask_centroids)
        # Cache per-pixel coordinate vectors for fast incremental distance updates
        self._pix_y = self.pix[:, 0]
        self._pix_x = self.pix[:, 1]
        # Initialize running min distances (exploration / suppression)
        self._d_prev_pix = np.full(self.pix.shape[0], self.d_max, dtype=np.float32)
        self._d_centroid_min = np.full(self.mask_centroids.shape[0], np.inf, dtype=np.float32)
        # Lazily build kernel & base centroid term (avoid double memory if extremely large)
        # We build immediately for speed unless memory pressure is expected.
        eps = 1e-6
        self._K = (1.0 / (self.d_to_cent + eps)).astype(np.float32)
        self._den_C = (self._K.sum(axis=1) + 1e-12).astype(np.float32)
        self._base_centroid_term = (1.0 - (self.d_to_cent / self.d_max)).astype(np.float32)

        # reset selection
        self.selected_points = []
        self.selected_count = 0
        self.expanded_masks = []
        # Keep reference to generated masks for later random-phase exclusion
        self.generated_masks = generated_masks

    def compute_pixel_acquisition_map(self):
        """Compute acquisition map using cached distances (no cdist per call)."""
        H, W = self.image_shape
        N = self.pix.shape[0]

        # Exploration term uses cached running min distances
        d_prev = self._d_prev_pix if self._d_prev_pix is not None else np.full(N, self.d_max)
        E_map = np.clip(d_prev / self.d_max, 0.0, 1.0)

        # Fast path: pure exploration (lambda_balance == 0)
        if self.lambda_balance <= 0.0:
            return E_map.reshape(H, W)

        # If all centroids have already been sampled (rare) just return exploration
        if len(self.selected_points) >= len(self.mask_centroids):
            return E_map.reshape(H, W)

        # Weight suppression using cached min centroid distances
        w = self.mask_weights.copy()
        if self.selected_points and self._d_centroid_min is not None:
            sigma = getattr(self, 'suppression_sigma', 6.0)
            d_cent_sel = self._d_centroid_min
            atten = np.exp(- (d_cent_sel ** 2) / (2 * sigma ** 2))
            w *= (1.0 - atten)
        w = np.maximum(w, 0.0)
        sw = w.sum()
        if sw > 0:
            w /= sw
        else:
            w = self.mask_weights.copy()

        # Kernel & base term (already float32)
        if self._K is None or self._base_centroid_term is None:
            eps = 1e-6
            self._K = (1.0 / (self.d_to_cent + eps)).astype(np.float32)
            self._den_C = (self._K.sum(axis=1) + 1e-12).astype(np.float32)
            self._base_centroid_term = (1.0 - (self.d_to_cent / self.d_max)).astype(np.float32)

        # Exploitation term: (K * base_term) @ w / den_C
        # Instead of allocating large temp, multiply base term by w via dot product logic
        raw_weighted = (self._base_centroid_term * w[np.newaxis, :])  # (N,M)
        num_C = np.einsum('ij,ij->i', self._K, raw_weighted, optimize=True)
        O_map = num_C / self._den_C
        # Normalize exploitation to [0,1]
        cmin = float(O_map.min())
        cmax = float(O_map.max())
        if cmax > cmin:
            O_map = (O_map - cmin) / (cmax - cmin)
        else:
            O_map = np.zeros_like(O_map)

        # Fast path: pure exploitation (lambda_balance == 1)
        if self.lambda_balance >= 1.0:
            return O_map.reshape(H, W)

        A = ( self.lambda_balance * O_map + (1 - self.lambda_balance) * E_map)
        return A.reshape(H, W)

    def display_acquisition_heatmap(self, next_point=None):
        """
        Show the pixel-wise acquisition heatmap and optionally highlight the selected next point.
        Args:
            next_point: (y, x) tuple to mark in red
        """
        A_map = self.compute_pixel_acquisition_map()
        try:
            # Create persistent figure/axes once, or recreate if closed
            fig_missing = (self._fig is None) or (not plt.fignum_exists(self._fig.number))
            if fig_missing:
                self._fig, self._ax = plt.subplots(figsize=(8, 6))
                self._im = self._ax.imshow(A_map, origin='upper', cmap='viridis')
                self._ax.axis('off')
                try:
                    self._fig.colorbar(self._im, ax=self._ax, label='Acquisition')
                except Exception:
                    pass
                # Create an empty scatter artist we'll update
                self._scatter = self._ax.scatter([], [], s=100, facecolors='none', edgecolors='red')
                self._fig.canvas.manager.set_window_title('Acquisition Heatmap') if hasattr(self._fig.canvas.manager, 'set_window_title') else None
                plt.show(block=False)
            # Update image data
            if self._im is not None:
                self._im.set_data(A_map)
                # If A_map isn't normalized in [0,1], update clim; else it's harmless
                try:
                    self._im.set_clim(vmin=float(np.min(A_map)), vmax=float(np.max(A_map)))
                except Exception:
                    pass
            # Update scatter point
            if next_point is not None and self._scatter is not None:
                y, x = int(next_point[0]), int(next_point[1])
                self._scatter.set_offsets(np.array([[x, y]]))
            elif self._scatter is not None:
                # Clear scatter if no point provided
                self._scatter.set_offsets(np.empty((0, 2)))
            # Redraw
            self._fig.canvas.draw_idle()
            plt.pause(3)
        except Exception:
            # Silently ignore display errors (e.g., headless)
            pass

    def get_next_point(self, last_mask=None):
        """
        Choose the next sampling point according to:
        - for the first heatmap_fraction*num_points: highest-acquisition heatmap peaks
        - thereafter: random points outside both SAM-generated masks and already-expanded masks
        
        Args:
            last_mask:    bool mask of the last point’s propagated region
        Returns:
            (y, x) tuple for the next sample
        """
        # 1) record last iteration’s mask & feature
        if last_mask is not None:
            self.expanded_masks.append(last_mask)
        
        H, W = self.image_shape
        # Number of initial selections from the acquisition heatmap
        threshold_count = math.ceil(self.num_points * self.heatmap_fraction)
        
        if self.selected_count < threshold_count:
            # Heatmap phase - use acquisition map
            A_map = self.compute_pixel_acquisition_map()
            y, x = np.unravel_index(np.argmax(A_map), A_map.shape)
        else:
            # Random phase - exclude all existing masks
            sam_union = np.zeros((H, W), bool)
            for m in getattr(self, "generated_masks", []):
                sam_union |= m['segmentation'].astype(bool)
            exp_union = np.zeros((H, W), bool)
            for em in self.expanded_masks:
                exp_union |= em.astype(bool)

            # Mask out already-selected points
            candidate_mask = ~sam_union & ~exp_union
            for pt in self.selected_points:
                py, px = int(pt[0]), int(pt[1])
                candidate_mask[py, px] = False

            ys, xs = np.where(candidate_mask)
            if len(ys) > 0:
                idx = self._rng.integers(len(ys))
                y, x = int(ys[idx]), int(xs[idx])
            else:
                y, x = int(self._rng.integers(0, H)), int(self._rng.integers(0, W))
        
        # 2) update selection state (and incremental distance caches)
        selected = (int(y), int(x))
        self.selected_points.append(np.array(selected))

        # Update running min pixel distances (exploration cache)
        if self._d_prev_pix is not None:
            dy = self._pix_y - selected[0]
            dx = self._pix_x - selected[1]
            dist_new = np.sqrt(dy * dy + dx * dx).astype(np.float32)
            self._d_prev_pix = np.minimum(self._d_prev_pix, dist_new)

        # Update centroid min distances (suppression cache)
        if self._d_centroid_min is not None:
            dcy = self.mask_centroids[:, 0] - selected[0]
            dcx = self.mask_centroids[:, 1] - selected[1]
            dist_cent_new = np.sqrt(dcy * dcy + dcx * dcx).astype(np.float32)
            self._d_centroid_min = np.minimum(self._d_centroid_min, dist_cent_new)
        
        self.selected_count += 1

        # Display acquisition heatmap after each selection (non-blocking)
        # try:
        #     self.display_acquisition_heatmap(next_point=selected)
        # except Exception:
        #     pass

        return selected


class ToolSelectionStrategy(PointSelectionStrategy):
    """Interactive strategy that uses only actual user clicks (no learned preference map).

    Behavior:
      - For the initial fraction of points selects argmax of acquisition map (same as Dynamic).
      - Afterwards samples uniformly from pixels not covered by SAM-generated masks or expanded masks.
      - The acquisition map uses exploration + exploitation similar to DynamicPointsSelectionStrategy but
        does NOT multiply by any user preference map.
    """
    def __init__(self, num_points=None, lambda_balance=0.5, heatmap_fraction=0.5):
        super().__init__()
        self.lambda_balance = (1 - float(lambda_balance))
        # per-image placeholders
        self.image_shape = None
        self.d_max = None
        self.mask_centroids = None
        self.mask_weights = None
        self.selected_points = []
        self.suggested_points = []
        self.selected_count = 0
        self.expanded_masks = []
        self.heatmap_fraction = float(heatmap_fraction)
        # cached structures (filled in setup_simple)
        self.pix = None
        self._pix_y = None
        self._pix_x = None
        self._d_prev_pix = None
        self._d_centroid_min = None
        self._K = None
        self._den_C = None
        self._base_centroid_term = None
        self.generated_masks = None

    def _compute_centroid(self, mask):
        ys, xs = np.nonzero(mask)
        return np.array([ys.mean(), xs.mean()])

    def setup_simple(self, image, generated_masks):
        H, W = image.shape[:2]
        self.image_shape = (H, W)
        self.d_max = np.hypot(W, H)

        reps, areas, scores = [], [], []
        for m in generated_masks:
            mask = m['segmentation']
            area = int(mask.sum())
            if area == 0:
                continue
            reps.append(self._compute_centroid(mask))
            areas.append(area)
            scores.append(float(m.get('predicted_iou', 0.0)))

        if not reps:
            raise ValueError("No valid masks")

        self.mask_centroids = np.vstack(reps)
        # simple weighting by area (same as dynamic)
        area_weight = 1.0
        score_weight = 0.0
        w = (area_weight * np.array(areas)) + (score_weight * np.array(scores) * np.array(areas))
        self.mask_weights = (w / w.sum())

        ys = np.arange(H)
        xs = np.arange(W)
        Y, X = np.meshgrid(ys, xs, indexing='ij')
        self.pix = np.stack([Y.ravel(), X.ravel()], axis=1)
        self.d_to_cent = cdist(self.pix, self.mask_centroids)
        self._pix_y = self.pix[:, 0]
        self._pix_x = self.pix[:, 1]
        self._d_prev_pix = np.full(self.pix.shape[0], self.d_max, dtype=np.float32)
        self._d_centroid_min = np.full(self.mask_centroids.shape[0], np.inf, dtype=np.float32)
        eps = 1e-6
        self._K = (1.0 / (self.d_to_cent + eps)).astype(np.float32)
        self._den_C = (self._K.sum(axis=1) + 1e-12).astype(np.float32)
        self._base_centroid_term = (1.0 - (self.d_to_cent / self.d_max)).astype(np.float32)

        # reset selection state
        self.selected_points = []
        self.suggested_points = []
        self.selected_count = 0
        self.expanded_masks = []
        # keep reference for exclusion in random phase
        self.generated_masks = generated_masks

    def compute_pixel_acquisition_map(self):
        H, W = self.image_shape
        N = self.pix.shape[0]

        # Exploration term: distance from selected points
        d_prev = self._d_prev_pix if self._d_prev_pix is not None else np.full(N, self.d_max)
        E_map = np.clip(d_prev / self.d_max, 0.0, 1.0)

        if self.lambda_balance <= 0.0:
            return E_map.reshape(H, W)

        if len(self.selected_points) >= len(self.mask_centroids):
            return E_map.reshape(H, W)

        # --- Centroid weighting with Gaussian attenuation ---
        w = self.mask_weights.copy()
        if self.selected_points and self._d_centroid_min is not None:
            sigma = float(getattr(self, 'suppression_sigma', 15.0))
            d_cent_sel = self._d_centroid_min
            atten = np.exp(-(d_cent_sel ** 2) / (2.0 * sigma * sigma))
            w *= (1.0 - atten)
        w = np.maximum(w, 0.0)
        sw = w.sum()
        if sw > 0:
            w /= sw
        else:
            w = self.mask_weights.copy()

        # --- Object-level acquisition ---
        if self._K is None or self._base_centroid_term is None:
            eps = 1e-6
            self._K = (1.0 / (self.d_to_cent + eps)).astype(np.float32)
            self._den_C = (self._K.sum(axis=1) + 1e-12).astype(np.float32)
            self._base_centroid_term = (1.0 - (self.d_to_cent / self.d_max)).astype(np.float32)

        raw_weighted = (self._base_centroid_term * w[np.newaxis, :])
        num_C = np.einsum('ij,ij->i', self._K, raw_weighted, optimize=True)
        O_map = num_C / self._den_C
        cmin, cmax = float(O_map.min()), float(O_map.max())
        if cmax > cmin:
            O_map = (O_map - cmin) / (cmax - cmin)
        else:
            O_map = np.zeros_like(O_map)
        
        self._last_O_map = O_map.reshape(H, W) 

        # --- Blend exploration & exploitation ---
        A = (self.lambda_balance * O_map + (1 - self.lambda_balance) * E_map).reshape(H, W)

        # --- Hard suppression window around selected points ---
        if self.selected_points:
            sigma = float(getattr(self, 'suppression_sigma', 15.0))
            Y, X = np.ogrid[:H, :W]
            for py, px in self.selected_points:
                if 0 <= py < H and 0 <= px < W:
                    dist2 = (Y - py) ** 2 + (X - px) ** 2
                    suppression = 1.0 - np.exp(-dist2 / (2.0 * sigma * sigma))
                    A *= suppression

        return A


    def display_acquisition_heatmap(self, A_map, next_point=None):
        # Disabled: Do not call Matplotlib GUI from threads. Use main-thread slot in app.py instead.
        pass

    def display_acquisition_heatmap_main_thread(self, A_map, next_point=None):
        # Disabled: Use main-thread slot in app.py for Matplotlib display.
        pass

    def display_last_acquisition_map(self):
        # Disabled: Do not call Matplotlib GUI from threads.
        pass
    

    def select_points(self, segmenter, image, expanded_masks=None):
        pass

    def _get_random_point(self, H, W):
        sam_union = np.zeros((H, W), bool)
        for m in getattr(self, "generated_masks", []):
            sam_union |= m['segmentation'].astype(bool)
        exp_union = np.zeros((H, W), bool)
        for em in self.expanded_masks:
            exp_union |= em.astype(bool)

        candidate_mask = ~sam_union & ~exp_union
        for pt in self.selected_points:
            py, px = int(pt[0]), int(pt[1])
            if 0 <= py < H and 0 <= px < W:
                candidate_mask[py, px] = False

        ys, xs = np.where(candidate_mask)
        if len(ys) > 0:
            idx = self._rng.integers(len(ys))
            y, x = int(ys[idx]), int(xs[idx])
        else:
            y, x = int(self._rng.integers(0, H)), int(self._rng.integers(0, W))

        self._last_acquisition_map = None
        self._last_selected_point = None
        print(f"[DEBUG] RANDOM point selected: (y={y}, x={x})")
        return (y, x)


    def get_next_point(self, last_mask=None, actual_last_point=None):
        """
        Choose the next sampling point, learning from where the user actually clicked.
        """
        # Record last iteration's mask (unioned by caller)
        if last_mask is not None:
            try:
                self.expanded_masks.append(last_mask)
                self._last_mask_area = np.sum(last_mask)
                print(f"[DEBUG] Last mask area: {self._last_mask_area} pixels")
            except Exception:
                print("[DEBUG] Failed to record last mask area")
                pass
        H, W = self.image_shape

        # If caller provided the actual last point (user click), update caches
        if actual_last_point is not None:
            try:
                sel_pt = (int(actual_last_point[0]), int(actual_last_point[1]))
                self.selected_points.append(np.array(sel_pt))

                if self._d_prev_pix is not None:
                    dy = self._pix_y - sel_pt[0]
                    dx = self._pix_x - sel_pt[1]
                    dist_new = np.sqrt(dy * dy + dx * dx).astype(np.float32)
                    self._d_prev_pix = np.minimum(self._d_prev_pix, dist_new)

                if self._d_centroid_min is not None:
                    dcy = self.mask_centroids[:, 0] - sel_pt[0]
                    dcx = self.mask_centroids[:, 1] - sel_pt[1]
                    dist_cent_new = np.sqrt(dcy * dcy + dcx * dcx).astype(np.float32)
                    self._d_centroid_min = np.minimum(self._d_centroid_min, dist_cent_new)

                self.selected_count += 1
            except Exception:
                pass

        # --- Dynamic phase (heatmap) ---
        use_dynamic = not getattr(self, "_dynamic_done", False)
        if use_dynamic:
            A_map = self.compute_pixel_acquisition_map()
            y, x = np.unravel_index(np.argmax(A_map), A_map.shape)

            # Scores
            best_A = A_map[y, x]
            best_O = None
            if hasattr(self, "_last_O_map") and self._last_O_map is not None:
                best_O = float(self._last_O_map[y, x])

            # --- Plain image coverage of expanded masks ---
            coverage = 0.0
            try:
                if self.expanded_masks:
                    exp_union = np.logical_or.reduce([m.astype(bool) for m in self.expanded_masks])
                    coverage = exp_union.sum() / float(H * W)
            except Exception:
                pass

            print(f"[DEBUG] Coverage (image fraction): {coverage:.3f}")

            if best_O is not None:
                raw_ratio = best_O / (best_A + 1e-8)
                contrib_ratio = (self.lambda_balance * best_O) / (best_A + 1e-8)

                last_area = getattr(self, "_last_mask_area", 1.0)  # 1.0 = safe default
                last_area /= float(H * W)
                print(f"[DEBUG] O/A ratio: {raw_ratio:.3f} | contrib_ratio=(λ·O)/A: {contrib_ratio:.3f}")
                print(f"[DEBUG] Last mask area fraction: {last_area:.4f}")

                # --- Switch condition ---
                if coverage > 0 and (contrib_ratio < 0.3 or last_area < 0.001):
                    self._dynamic_done = True
                    print("[DEBUG] Switching to RANDOM phase (dynamic exhausted or last mask too small).")
                    suggested = self._get_random_point(H, W)
                else:
                    try:
                        # self.display_acquisition_heatmap(A_map, next_point=(y, x))
                        self._last_acquisition_map = np.array(A_map, copy=True)
                        self._last_selected_point = (int(y), int(x))
                    except Exception:
                        self._last_acquisition_map = None
                        self._last_selected_point = None
                    suggested = (int(y), int(x))

        # --- Random phase ---
        else:
            print("[DEBUG] RANDOM phase: selecting random point outside masks.")
            suggested = self._get_random_point(H, W)

        try:
            self.suggested_points.append(suggested)
            print(f"[DEBUG] Suggested point: (y={suggested[0]}, x={suggested[1]})")
        except Exception:
            print("[DEBUG] Failed to record suggested point")
            pass

        return suggested

    def record_actual_click(self, actual_point):
        """
        Record where the user actually clicked (called from the app after user interaction).
        Note: The actual recording happens in get_next_point when called next time.
        
        Args:
            actual_point: (y, x) where user actually clicked
        """
        pass


# Factory for creating point selection strategies
class PointSelectionFactory:
    """Factory for creating point selection strategies."""
    
    _strategies = {
        "list": ListSelectionStrategy,
        "random": RandomSelectionStrategy,
        "grid": GridSelectionStrategy,
        "SAM2_guided": SAM2GuidedSelectionStrategy,
        "dynamicPoints": DynamicPointsSelectionStrategy,
        "ToolSelectionStrategy": ToolSelectionStrategy
    }
    
    @classmethod
    def create_strategy(cls, strategy_name, **kwargs):
        """
        Create a point selection strategy.
        
        Args:
            strategy_name: Name of the strategy
            kwargs: Additional parameters for the strategy
            
        Returns:
            An instance of the selected strategy
        """
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Special handling: list strategy may receive legacy arg names
        if strategy_name == "list":
            # Normalize to new canonical argument points_path.
            if 'points_path' not in kwargs:
                legacy = kwargs.get('annotations_file') or kwargs.get('points_file') or kwargs.get('points_json_path')
                if legacy is not None:
                    kwargs['points_path'] = legacy
            try:
                print(f"[PointSelectionFactory] Creating 'list' strategy with points_path={kwargs.get('points_path')}")
            except Exception:
                pass
        return cls._strategies[strategy_name](**kwargs)
    
    @classmethod
    def strategy_uses_contrastive(cls, strategy_name):
        """
        Check if the given strategy uses contrastive learning.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            True if the strategy uses contrastive learning, False otherwise.
        """
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Check the `uses_contrastive_learning` attribute of the strategy class
        strategy_class = cls._strategies[strategy_name]
        return getattr(strategy_class, "uses_contrastive_learning", False)