import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
import cv2
import contextlib

# Add the current directory to the Python path to find the local segment_anything folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.measure import perimeter
from sklearn.cluster import KMeans


class Segmenter:
    def __init__(self, image=None, sam2_checkpoint_path=None, sam2_config_path=None, device="cuda"):
        """
        Initialize the Segmenter class with SAM2, optionally without an image.
        """
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.sam2_checkpoint_path = sam2_checkpoint_path
        self.sam2_config_path = sam2_config_path
        self.sam2_model = None
        self.predictor = None
        self.mask_generator = None
        # Always define masks-related attributes
        self.masks = []
        self.selected_masks = set()
        self.selected_points = []
        self.rejected_masks = set()

        # Initialize with the provided image if available
        if image is not None:
            self.just_set_image(image)

    def set_image(self, image):
        """
        Set a new image for the Segmenter and reinitialize related attributes.
        """
        assert image is not None, "An image must be provided."

        # Check if the image is already set
        if hasattr(self, 'image') and np.array_equal(self.image, image):
            return

        # Set the new image and reinitialize attributes
        self.image = image
        self.height = image.shape[0]
        self.width = image.shape[1]

        # Initialize SAM2 predictor if not already initialized
        # start_init = time.time()
        self._initialize_sam_model()
        # print(f"Initialized SAM2 model in {time.time() - start_init:.2f} seconds", flush=True)
        
        # start_set_image = time.time()
        self.predictor.set_image(self.image)
        # print(f"Set image in {time.time() - start_set_image:.2f} seconds", flush=True)
        
        # Generate masks for the new image
        # start_generate_masks = time.time()
        self.masks = self._generate_masks()
        # print(f"Generated {len(self.masks)} masks in {time.time() - start_generate_masks:.2f} seconds", flush=True)

        # Reset selection state
        self.selected_masks = set()
        self.selected_points = []
        self.rejected_masks = set()

        return self.masks, self.predictor.get_features()[0]
    
    def just_set_image(self, image):
        """
        Set a new image for the Segmenter without reinitializing the SAM2 model.
        This is useful when you want to change the image but keep the existing model.
        """
        assert image is not None, "An image must be provided."

        # Check if the image is already set
        if hasattr(self, 'image') and np.array_equal(self.image, image):
            return self.masks

        self._initialize_sam_model()

        # Set the new image and update dimensions
        self.image = image
        self.height = image.shape[0]
        self.width = image.shape[1]

        # Set the image in the predictor
        self.predictor.set_image(self.image)

        # Generate masks (mirrors set_image behavior)
        self.masks = self._generate_masks()

        # Reset selection state
        self.selected_masks = set()
        self.selected_points = []
        self.rejected_masks = set()

        return self.masks

    def _generate_masks(self):
        """
        Generate masks for the given image using SAM2.

        Returns:
            list: List of generated masks for the image, excluding non-informative masks.
        """
        self.sam2_model.to(self.device)
             
        # Generate masks
        masks = self.mask_generator.generate(self.image)[0]

        # Image area (width x height)
        image_area = self.image.shape[0] * self.image.shape[1]

        # Threshold to determine if a mask covers "too much" of the image as a percentage
        coverage_threshold = 0.95  # Adjust based on your needs (0.95 = 95%)

        # Filter out masks that are non-informative
        filtered_masks = [
            mask for mask in masks if (mask['area'] / image_area) < coverage_threshold
        ]

        # Predict expansion for each mask centroid and sort by predicted area
        mask_info = []
        for mask in filtered_masks:
            segmentation = mask['segmentation']
            indices = list(zip(*segmentation.nonzero()))
            if indices:
                # Calculate centroid in [y, x] format
                centroid = np.mean(indices, axis=0)
                # Convert to [x, y] format for prediction
                point = np.array([[centroid[1], centroid[0]]])
                labels = np.array([1])  # Positive point
                
                # Predict expansion for this centroid
                masks, scores, logits = self.predictor.predict(
                    point_coords=point,
                    point_labels=labels,
                    multimask_output=True,
                )
                
                if masks is not None:
                    # Get the best mask using weighted selection
                    best_mask_idx = self._weighted_mask_selection(masks, scores)
                    predicted_mask = masks[best_mask_idx]
                    predicted_area = predicted_mask.sum()
                    
                    mask_info.append({
                        'mask': mask,
                        'centroid': centroid,  # Keep in [y, x] format for display
                        'point': point[0],     # Store [x, y] format for prediction
                        'area': predicted_area
                    })
        
        # Sort masks by predicted area in descending order
        mask_info.sort(key=lambda x: x['area'], reverse=True)
        
        # Update filtered_masks with sorted order
        filtered_masks = [info['mask'] for info in mask_info]
            
        return filtered_masks

    def _compute_mask_metrics(self, mask, score):
        """
        Compute and normalize mask metrics: compactness, size penalty, and score.

        Args:
            mask (np.array): Binary mask for the segment.
            score (float): Score assigned by SAM for the mask.

        Returns:
            tuple: Normalized compactness, size penalty, and score.
        """
        # Mask metrics
        mask_area = mask.sum()  # Total pixels in the mask
        mask_perimeter = perimeter(mask)  # Perimeter of the mask

        # Compactness: Avoid divide-by-zero errors
        if mask_area > 0:
            # Ideal perimeter for a circle with the same area
            ideal_perimeter = 2 * np.sqrt(np.pi * mask_area)

            # Compactness: The ratio of the perimeter to the ideal perimeter (closer to 1 is more compact)
            if mask_perimeter > 0:
                raw_compactness = ideal_perimeter / mask_perimeter  # Inverse, so lower perimeter = higher compactness
            else:
                raw_compactness = 0  # Handle the case when mask_perimeter is 0
        else:
            raw_compactness = 0

        # Normalize compactness (keeping compactness between 0 and 1)
        compactness = min(raw_compactness, 1)  # Ensure compactness doesn't exceed 1

        # Normalized size penalty
        total_pixels = self.height * self.width
        normalized_area = mask_area / total_pixels  # Fraction of the image covered by the mask

        # Gentle penalty for very small masks (e.g., < 1% of image)
        if normalized_area < 0.001:  # Only apply penalty for masks smaller than 1% of the image
            small_mask_penalty = normalized_area ** 4  # Soft quadratic penalty
        else:
            small_mask_penalty = 0  # No penalty for larger masks

        # Large mask penalty
        large_mask_penalty = (normalized_area - 0.4) ** 4 if normalized_area > 0.5 else 0

        # Combine penalties gently
        size_penalty = normalized_area + small_mask_penalty + large_mask_penalty

        # Return normalized metrics
        return compactness, size_penalty, score

    def _weighted_mask_selection(self, masks, scores, weights=(1.0, 0.8, 1.4), point=None, label=None):
        best_score = -np.inf
        best_index = -1  # Initialize with an invalid index

        w_s, w_c, w_a = weights  # Weights for SAM Score, Compactness, and Size

        for i, mask in enumerate(masks):
            # Compute metrics
            compactness, size_penalty, sam_score = self._compute_mask_metrics(mask, scores[i])

            # Weighted score (nonlinear terms)
            weighted_score = (
                    w_s * sam_score +  # Higher SAM score is better
                    w_c * np.log(1 + compactness) -  # Higher compactness is better (log smoothing)
                    w_a * size_penalty  # Lower size penalty is better
            )

            # Select best mask
            if weighted_score > best_score:
                best_score = weighted_score
                best_index = i  # Store the index of the best mask

        return best_index

    def propagate_points(self, points, labels, update_expanded_mask=True):
        """
        Propagate points into a mask using SAM2

        Args:
            points: Point prompt coordinates
            labels: Point prompt labels. 1 if positive, 0 if negative.
            update_expanded_mask (bool): Whether to update the expanded_areas_mask. 
                                       Should be True only for actual point predictions, 
                                       False for dynamic expansion visualization.

        Returns:
            np.array: Mask propagated from points.
        """
        # Convert points and labels to the correct format
        # The predictor expects NumPy arrays, not PyTorch tensors
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        elif isinstance(points, list):
            points = np.array(points)

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        elif isinstance(labels, list):
            labels = np.array(labels)

        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        predicted_mask = masks[self._weighted_mask_selection(masks, scores)]
        
        return predicted_mask.astype(bool)
    
    def cleanup(self):
        """
        Clean up resources by moving models to CPU and clearing GPU memory.
        This should be called when switching to a new image.
        """
        if self.device == "cuda":
            # Clear GPU memory
            torch.cuda.empty_cache()

    def _initialize_sam_model(self):
        """Ensure the SAM2 model is loaded and initialized."""
        if self.sam2_model is None:
            # Create autocast context manager only when running on CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                self.autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                self.autocast_context.__enter__()
                # Enable TF32 on Ampere+ GPUs
                try:
                    if torch.cuda.get_device_properties(0).major >= 8:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                except Exception:
                    # If querying device properties fails, continue without TF32 tweaks
                    pass
            else:
                # CPU path: no-op context to keep code flow identical
                self.autocast_context = contextlib.nullcontext()
                self.autocast_context.__enter__()

            self.sam2_model = build_sam2(
                self.sam2_config_path, self.sam2_checkpoint_path, device=self.device, apply_postprocessing=False
            )

            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=self.sam2_model,
                points_per_side=28,          # fewer, more spread-out points
                points_per_batch=512,
                pred_iou_thresh=0.5,         # allow larger, lower-confidence masks
                stability_score_thresh=0.9,
                stability_score_offset=0.7,
                mask_threshold=0.25,         # lower cutoff to grow masks
                box_nms_thresh=0.45,         # merge overlapping boxes aggressively
                crop_n_layers=0,             # no small-object crops
                min_mask_region_area=1500,   # remove fragments <1300px on 512×512
                multimask_output=True,      # one mask per prompt
            )

            self.predictor = SAM2ImagePredictor(self.sam2_model)