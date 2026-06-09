import sys
import os
import cv2
import numpy as np
import time
import tempfile
import torch
import traceback
import warnings

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QMessageBox, QDialog, QProgressBar, QLineEdit, QListWidget,
    QListWidgetItem, QColorDialog, QGridLayout, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QGuiApplication
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer

from segmenter_sam2 import Segmenter
from point_selection_strategies import ToolSelectionStrategy
from plas.segmenter_plas import SuperpixelLabelExpander
from app_modules import LabelDialog, OverlapDialog

# Reduce noisy external warnings (safe to ignore)
# SIP/PyQt deprecation noise
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*sipPyTypeDict\(\).*")

# SAM2 optional extension warning (post-processing fallback)
warnings.filterwarnings("ignore", category=UserWarning, message=r".*cannot import name '_C' from 'sam2'.*")

# PyTorch SDPA backend: avoid flash/mem-efficient attempts and related warnings
try:
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

# Silence specific SDPA kernel selection chatter
warnings.filterwarnings("ignore", category=UserWarning, message=r".*Flash attention kernel not used.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*Memory Efficient attention has been runtime disabled.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*Memory efficient kernel not used because.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*CuDNN attention kernel not used.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*Expected query, key and value to all be of dtype.*scaled_dot_product_attention.*")

def load_stylesheet(file_path):
    """Load stylesheet from a file"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading stylesheet: {e}")
        return ""

# Subclass QLabel to capture mouse clicks on the image
class ClickableLabel(QLabel):
    clicked = pyqtSignal(object)
    right_clicked = pyqtSignal(object)  # New signal for right-click
    mouse_moved = pyqtSignal(object)  # New signal for mouse move

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.interactions_enabled = False

    def mousePressEvent(self, event):
        if self.interactions_enabled:
            if event.button() == Qt.MouseButton.LeftButton:
                self.clicked.emit(event.pos())
            elif event.button() == Qt.MouseButton.RightButton:
                self.right_clicked.emit(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (self.interactions_enabled):   
            self.mouse_moved.emit(event.pos())
        super().mouseMoveEvent(event)

# Thread to suggest the next point using the adaptive interactive selection strategy
class PointSuggestionThread(QThread):
    result_ready = pyqtSignal(tuple)
    heatmap_ready = pyqtSignal(object, object)  # (A_map, next_point)
    error_occurred = pyqtSignal(str)

    def __init__(self, strategy, segmenter, image, expanded_masks=None, last_mask=None, last_feature=None, actual_last_point=None, parent=None):
        super().__init__(parent)
        self.strategy = strategy
        self.segmenter = segmenter
        self.image = image
        self.expanded_masks = expanded_masks or []
        self.last_mask = last_mask
        self.last_feature = last_feature
        self.actual_last_point = actual_last_point

    def run(self):
        try:
            next_point = self.strategy.get_next_point(
                last_mask=self.last_mask,
                actual_last_point=self.actual_last_point
            )
            # If the strategy has a last acquisition map, emit it for main-thread display
            A_map = getattr(self.strategy, '_last_acquisition_map', None)
            if A_map is not None:
                self.heatmap_ready.emit(A_map, next_point)
            self.result_ready.emit(next_point)
        except Exception as e:
            self.error_occurred.emit(str(e))
            import traceback
            traceback.print_exc()

# Thread to expand the user-selected point into a mask using an existing segmenter
class MaskExpansionThread(QThread):
    result_ready = pyqtSignal(object)

    def __init__(self, segmenter, points, labels, box=None):
        super().__init__()
        self.segmenter = segmenter
        self.points = points
        self.labels = labels
        self.box = box

    def run(self):
        mask = self.segmenter.propagate_points(
            self.points, self.labels, update_expanded_mask=True, box=self.box,
        )
        self.result_ready.emit(mask)


class ImageViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Viewer")
        # Pick an initial size that's a sensible fraction of the screen, so
        # very tall/wide images get more room out of the box. Window stays
        # resizable; the image label expands with it.
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            init_w = max(900, min(int(avail.width() * 0.8), 1800))
            init_h = max(700, min(int(avail.height() * 0.9), 1400))
        else:
            init_w, init_h = 1400, 900
        self.resize(init_w, init_h)
        self.setMinimumSize(900, 700)

        # Image display — expanding so the canvas grows with the window.
        self.image_label = ClickableLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.clicked.connect(self.on_image_clicked)
        self.image_label.right_clicked.connect(self.on_image_right_clicked)
        self.image_label.interactions_enabled = False

        # Buttons
        stylesheet = load_stylesheet("app_modules/button_styles.qss")
        app.setStyleSheet(stylesheet)

        self.select_button = QPushButton("Select Folder", self)
        self.select_button.clicked.connect(self.select_folder)
        self.select_button.setFixedSize(150, 40)
        self.select_button.setProperty("class", "select-folder-button")
        self.select_button.enterEvent = lambda e: self.on_cursor_over_button()

        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_labeling)
        self.start_button.setFixedSize(150, 40)
        self.start_button.setEnabled(False)
        self.start_button.setProperty("class", "start-button")
        self.start_button.enterEvent = lambda e: self.on_cursor_over_button()

        # Navigation buttons
        self.prev_button = QPushButton("<", self)
        self.prev_button.clicked.connect(self.prev_image)
        self.prev_button.setFixedSize(40, 40)
        self.prev_button.setEnabled(False)
        self.prev_button.setProperty("class", "navigation-button")
        self.prev_button.enterEvent = lambda e: self.on_cursor_over_button()

        self.next_button = QPushButton(">", self)
        self.next_button.clicked.connect(self.next_image)
        self.next_button.setFixedSize(40, 40)
        self.next_button.setEnabled(False)
        self.next_button.setProperty("class", "navigation-button")
        self.next_button.enterEvent = lambda e: self.on_cursor_over_button()

        # New buttons for point selection
        self.switch_button = QPushButton("Negative", self)
        self.switch_button.clicked.connect(self.switch_point_type)
        self.switch_button.setFixedSize(150, 40)
        self.switch_button.setEnabled(False)
        self.switch_button.setProperty("class", "switch-button-negative")
        self.switch_button.enterEvent = lambda e: self.on_cursor_over_button()

        # BBox prompt button. Toggles a 2-click bbox-drawing mode (1st click =
        # first corner, 2nd click = opposite corner; live preview between).
        self.bbox_button = QPushButton("BBox", self)
        self.bbox_button.clicked.connect(self.toggle_bbox_mode)
        self.bbox_button.setFixedSize(80, 40)
        self.bbox_button.setEnabled(False)
        self.bbox_button.setProperty("class", "bbox-button-idle")
        self.bbox_button.enterEvent = lambda e: self.on_cursor_over_button()

        self.finish_button = QPushButton("✓", self)
        self.finish_button.clicked.connect(self.on_finish_button_clicked)
        self.finish_button.setFixedSize(40, 40)
        self.finish_button.setEnabled(False)
        self.finish_button.setProperty("class", "finish-button")
        self.finish_button.enterEvent = lambda e: self.on_cursor_over_button()

        # Toggle mask visibility button
        self.toggle_masks_button = QPushButton("👁️", self)
        self.toggle_masks_button.clicked.connect(self.toggle_masks_visibility)
        self.toggle_masks_button.setFixedSize(40, 40)
        self.toggle_masks_button.setEnabled(False)
        self.toggle_masks_button.setProperty("class", "toggle-masks-button")
        self.toggle_masks_button.enterEvent = lambda e: self.on_cursor_over_button()

        # Create a horizontal layout for the bottom buttons
        bottom_button_layout = QHBoxLayout()
        bottom_button_layout.addStretch()
        bottom_button_layout.addWidget(self.start_button)
        bottom_button_layout.addWidget(self.switch_button)
        bottom_button_layout.addWidget(self.bbox_button)
        bottom_button_layout.addWidget(self.finish_button)
        bottom_button_layout.addWidget(self.toggle_masks_button)
        bottom_button_layout.addStretch()

        # Create a container widget for the image and navigation buttons.
        # No fixed size — the grid stretches with the parent so the image
        # column gets all the extra space when the window is resized.
        image_container = QWidget()
        image_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        image_container_layout = QGridLayout(image_container)
        image_container_layout.setContentsMargins(0, 0, 0, 0)
        image_container_layout.setSpacing(0)
        # Only the middle column (with the image) gets stretch — the side
        # nav-button columns stay compact.
        image_container_layout.setColumnStretch(0, 0)
        image_container_layout.setColumnStretch(1, 1)
        image_container_layout.setColumnStretch(2, 0)
        image_container_layout.setRowStretch(0, 1)

        # Add image label to the center
        image_container_layout.addWidget(self.image_label, 0, 1)  # Changed to column 1

        # Add navigation buttons to corners
        image_container_layout.addWidget(self.prev_button, 0, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        image_container_layout.addWidget(self.next_button, 0, 2, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Main layout: image on left, bottom buttons below
        main_layout = QVBoxLayout()
        
        # Create a horizontal layout for the top row (select folder button)
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.select_button)
        top_layout.addStretch()
        
        main_layout.addLayout(top_layout)
        # stretch=1 so the image canvas takes all the vertical slack the
        # top/bottom button rows don't need. No AlignCenter — that would
        # collapse the widget to its sizeHint and leave huge empty space.
        main_layout.addWidget(image_container, 1)
        main_layout.addLayout(bottom_button_layout)
        
        self.setLayout(main_layout)

        # Variables
        self.image_list = []
        self.current_index = 0
        self.current_image = None
        self.overlay_image = None
        self.displayed_pixmap = None
        self.segmenter = None
        self.plas_segmenter = None  # Initialize PLAS segmenter
        self.labels = {}
        self.expanded_masks = []
        self.combined_mask_overlay = None
        
        self.image_label.mouse_moved.connect(self.on_mouse_moved)

        # New variables for multiple points
        self.positive_points = []
        self.negative_points = []
        self.current_point_type = "positive"  # or "negative"
        self.is_selecting_points = False

        # BBox prompt state. bbox_mode goes True when the user clicks the BBox
        # button; the next image-click sets `bbox_first_corner`, and the click
        # after that produces `current_bbox` (XYXY in image coords) and
        # auto-exits bbox_mode. `input_history` is a unified placement log so
        # Ctrl+Z can undo points and bboxes in actual placement order.
        self.bbox_mode = False
        self.bbox_first_corner = None
        self.current_bbox = None
        self.input_history = []  # entries: ("pos_point",), ("neg_point",), ("bbox", new_xyxy, prev_xyxy)

        # Initially hide the point selection buttons
        self.switch_button.hide()
        self.bbox_button.hide()
        self.finish_button.hide()
        self.toggle_masks_button.hide()

        # Initialize additional variables
        self.current_image_path = None
        self.current_label = None
        self.expanded_areas_mask = None
        self.current_mask = None
        self.is_expanding = False
        self.current_mask_area = 0
        self.total_image_area = 0
        self.coverage_ratio = 0
        self.last_actual_click = None  # Track the user's actual last click

        self.current_mode = "creation"  # or "selection"
        self.is_over_mask = False
        self.current_mask_index = None
        self.masks_visible = True
        
        # Initialize the point selection strategy
        self.point_strategy = None

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Images")
        if folder:
            self.image_list = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"))
            ]
            self.image_list.sort()
            self.current_index = 0
            
            # Reset all labeling-related state when selecting a new folder
            self.segmenter = None  # Reset segmenter for a new folder
            self.plas_segmenter = None  # Reset PLAS segmenter
            self.point_strategy = None  # Reset point strategy
            self.expanded_masks = []
            self.combined_mask_overlay = None
            self.suggested_point = None
            
            # Reset points and labeling state
            self.positive_points = []
            self.negative_points = []
            self.is_selecting_points = False
            self.last_actual_click = None  # Reset tracking
            
            # Reset UI to initial state
            self.image_label.interactions_enabled = False
            self.switch_button.hide()
            self.bbox_button.hide()
            self.finish_button.hide()
            self.toggle_masks_button.hide()
            self.toggle_masks_button.setEnabled(False)
            self.start_button.setEnabled(True)

            if self.image_list:
                # Load and display the first image of the new folder
                current_image_path = self.image_list[self.current_index]
                image = self._load_and_normalize_image(current_image_path)
                if image is None:
                    return
                self.current_image = image
                self.show_image()
                
                # Enable navigation buttons
                self.next_button.setEnabled(True)
                self.prev_button.setEnabled(True)

    def start_labeling(self):
        if not self.image_list or self.current_index >= len(self.image_list):
            return
        
        # Enable image interactions and show point selection buttons
        self.image_label.interactions_enabled = True
        self.switch_button.show()
        self.switch_button.setEnabled(False)  # Initially disabled
        self.bbox_button.show()
        self.bbox_button.setEnabled(True)  # bbox is usable from the first click
        self.finish_button.show()
        self.finish_button.setEnabled(False)  # Will be enabled after first positive point
        self.start_button.setEnabled(False)  # Disable Start button until Next Image is pressed
        self.is_selecting_points = True

        # Show the toggle masks button but disable it until first mask is created
        self.toggle_masks_button.show()
        self.toggle_masks_button.setEnabled(False)

        # Create a modal progress dialog
        wait_dialog = QDialog(self)
        wait_dialog.setWindowTitle("Generating Proposed Point")
        wait_dialog.setModal(True)
        wait_layout = QVBoxLayout()
        wait_label = QLabel("Please wait while the proposed point is being generated...")
        wait_layout.addWidget(wait_label)
        progress_bar = QProgressBar(wait_dialog)
        progress_bar.setRange(0, 0)  # Indeterminate mode
        wait_layout.addWidget(progress_bar)
        wait_dialog.setLayout(wait_layout)
        wait_dialog.resize(400, 150)
        wait_dialog.show()
        QApplication.processEvents()

        QTimer.singleShot(100, lambda: self.initialize_segmenter_and_start_thread(wait_dialog))
        wait_dialog.exec()


    import sys

    def print_memory_usage(self):
        """Print the memory usage of key objects in GB."""
        expanded_masks_size = sys.getsizeof(self.expanded_masks)
        combined_mask_overlay_size = sys.getsizeof(self.combined_mask_overlay) if self.combined_mask_overlay is not None else 0

        # Convert sizes to MB
        expanded_masks_mb = expanded_masks_size / (1024 ** 2)
        combined_mask_overlay_mb = combined_mask_overlay_size / (1024 ** 2)

        print(f"Memory Usage:")
        print(f"  - expanded_masks: {expanded_masks_mb:.4f} MB")
        print(f"  - combined_mask_overlay: {combined_mask_overlay_mb:.4f} MB")

    def initialize_segmenter_and_start_thread(self, wait_dialog):
        sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        # sam_checkpoint = "checkpoints/vit_b_coralscop.pth"
        sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        current_image_path = self.image_list[self.current_index]

        image = self._load_and_normalize_image(current_image_path)
        if image is None:
            return
        self.current_image = image

        if self.segmenter is None:
            # Segmenter(image=None, sam2_checkpoint_path=None, sam2_config_path=None, device='cuda')
            # The second arg in app previously pointed to a SAM1 checkpoint; Segmenter expects SAM2 checkpoint and config.
            # Pass correct positions: (image=None, sam2_checkpoint_path, sam2_config_path, device)
            self.segmenter = Segmenter(
                image=None,
                sam2_checkpoint_path=sam2_checkpoint,
                sam2_config_path=sam2_cfg,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            # Generate masks for the current image
            self.segmenter.set_image(self.current_image)
            
            # Initialize PLAS segmenter for superpixel-based expansion
            self.plas_segmenter = SuperpixelLabelExpander("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize the point selection strategy
            wait_dialog.setWindowTitle("Initializing Point Selection Strategy")
            wait_layout = wait_dialog.layout()
            wait_layout.itemAt(0).widget().setText("Setting up interactive point selection...")
            QApplication.processEvents()
            
            # Get generated masks and features for the strategy
            generated_masks = self.segmenter.masks
            print(f"Found {len(generated_masks)} potential objects")
            
            # Extract features for all masks
            features = []
            for mask_data in generated_masks:
                # Use a simple approach for features - could be improved
                mask = mask_data['segmentation']
                # Simple feature: area and position
                area = np.sum(mask)
                if area > 0:
                    y_indices, x_indices = np.where(mask)
                    centroid_y = np.mean(y_indices)
                    centroid_x = np.mean(x_indices)
                    # Create a simple feature vector
                    feature = np.array([area, centroid_y, centroid_x, mask_data.get('predicted_iou', 0.5)])
                    features.append(feature)
                else:
                    features.append(np.zeros(4))
            
            features = np.array(features)
            
            # Initialize the strategy
            self.point_strategy = ToolSelectionStrategy()
            # Setup expects only (image, generated_masks)
            self.point_strategy.setup_simple(self.current_image, generated_masks)
            
            # Store generated masks in the strategy for the random phase
            self.point_strategy.generated_masks = generated_masks
            
        # Get the first suggested point
        self.point_thread = PointSuggestionThread(
            self.point_strategy, 
            self.segmenter, 
            self.current_image,
            expanded_masks=self.expanded_masks,
            last_mask=None,
            last_feature=None,
            actual_last_point=None
        )
        self.point_thread.result_ready.connect(lambda next_point: self.on_point_suggestion_complete(next_point, wait_dialog))
        self.point_thread.error_occurred.connect(lambda error: self.on_point_suggestion_error(error, wait_dialog))
        self.point_thread.heatmap_ready.connect(lambda A_map, next_point: self.display_acquisition_heatmap_main_thread(A_map, next_point))
        self.point_thread.start()
    def display_acquisition_heatmap_main_thread(self, A_map, next_point):
        return
        # This slot runs in the main thread and is safe for Matplotlib GUI
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.imshow(A_map, origin='upper', cmap='viridis')
            if next_point is not None:
                y, x = next_point
                plt.scatter(x, y, s=100, facecolors='none', edgecolors='red', linewidth=2)
            plt.axis('off')
            plt.colorbar(label='Acquisition Score')
            plt.title('Adaptive Interactive Strategy - Acquisition Map')
            plt.show(block=False)
            plt.pause(0.001)
        except Exception as e:
            print(f"Error displaying heatmap: {e}")

    def on_point_suggestion_complete(self, next_point, wait_dialog):
        wait_dialog.accept()
        if next_point:
            self.suggested_point = next_point
        else:
            print("No valid point proposed.")
            self.suggested_point = None
        self.show_image(overlay_point=self.suggested_point)
    
    def on_point_suggestion_error(self, error_msg, wait_dialog):
        wait_dialog.accept()
        print(f"Error in point suggestion: {error_msg}")
        # Fall back to center of image
        if hasattr(self, 'current_image') and self.current_image is not None:
            h, w = self.current_image.shape[:2]
            self.suggested_point = (h // 2, w // 2)
        else:
            self.suggested_point = None
        self.show_image(overlay_point=self.suggested_point)

    # Max long-side (in pixels) we feed to SAM2. The auto mask generator
    # upsamples low-res mask logits to the image's full resolution; on a
    # 2048×12000 TIFF that allocation runs to ~140 GiB and OOMs even a 24 GB
    # GPU. 2048 keeps the upsample tractable and matches the upper bound
    # used in most SAM2 inference recipes.
    SAM2_MAX_SIDE = 2048

    def _load_and_normalize_image(self, path):
        """Read an image from disk in RGB uint8, downscaling to fit inside
        SAM2_MAX_SIDE on its longest side. Returns None on read failure.

        Handles:
          - >8-bit TIFFs (uint16 / float) — normalized to uint8.
          - single-channel TIFFs — broadcast to 3 channels.
          - multi-page TIFFs — only the first page is used (cv2 default).
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[load] cv2.imread returned None for: {path}")
            return None

        # Normalize bit depth to uint8 if needed
        if img.dtype != np.uint8:
            mn, mx = float(img.min()), float(img.max())
            if mx > mn:
                img = ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)

        # Channel handling: gray -> RGB; BGR -> RGB; BGRA -> RGB
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            print(f"[load] Unexpected image shape {img.shape} for {path}")
            return None

        # Downscale if longest side exceeds the cap
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side > self.SAM2_MAX_SIDE:
            scale = self.SAM2_MAX_SIDE / long_side
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            print(f"[load] Downscaling {os.path.basename(path)} {w}x{h} -> {new_w}x{new_h}")
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return img

    def _record_positive_point(self, point):
        """Append a positive point and remember it in the undo history."""
        self.positive_points.append(point)
        self.input_history.append(("pos_point",))

    def _record_negative_point(self, point):
        """Append a negative point and remember it in the undo history."""
        self.negative_points.append(point)
        self.input_history.append(("neg_point",))

    def on_image_clicked(self, pos):
        if self.current_image is None or self.displayed_pixmap is None:
            return

        # Convert click position to image coordinates
        current_point = self.get_image_coordinates(pos)
        if current_point is None:
            return

        # BBox-drawing takes precedence over normal point placement when
        # active. Click 1 = first corner, click 2 = opposite corner.
        if self.bbox_mode and self.current_mode != "selection":
            if self.bbox_first_corner is None:
                self.bbox_first_corner = current_point
                print(f"BBox first corner: {current_point}")
                self.update_preview_with_points()
            else:
                x1 = min(self.bbox_first_corner[0], current_point[0])
                y1 = min(self.bbox_first_corner[1], current_point[1])
                x2 = max(self.bbox_first_corner[0], current_point[0])
                y2 = max(self.bbox_first_corner[1], current_point[1])
                # Reject degenerate boxes (single point / line) — treat as miss
                if x2 - x1 < 2 or y2 - y1 < 2:
                    print("BBox too small, ignoring; pick a wider opposite corner")
                    return
                prev = self.current_bbox
                self.current_bbox = (x1, y1, x2, y2)
                self.input_history.append(("bbox", self.current_bbox, prev))
                self.bbox_first_corner = None
                # Auto-exit bbox mode and restore button styling
                self.bbox_mode = False
                self.bbox_button.setText("BBox")
                self.bbox_button.setProperty("class", "bbox-button-idle")
                self.bbox_button.style().unpolish(self.bbox_button)
                self.bbox_button.style().polish(self.bbox_button)
                self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
                self.finish_button.setEnabled(True)
                print(f"BBox set: {self.current_bbox}")
                self.update_preview_with_points()
            return

        # Check which mode we're in
        if hasattr(self, 'current_mode'):
            # Don't do anything on clicks in visualization mode
            if self.current_mode == "visualization":
                return
            elif self.current_mode == "selection":
                mask_index = self.get_mask_at_position(current_point)
                if mask_index is not None:
                    # Show context menu for the selected mask
                    self.current_mask_index = mask_index
                    self.show_mask_context_menu(pos, mask_index)
                    return
            elif self.current_mode == "creation":
                # In creation mode, always allow point placement
                # Check if we're clicking over an existing mask for potential merging
                mask_index = self.get_mask_at_position(current_point)
                
                # Check overlap first (for positive points only) — if the click
                # lands on an existing mask, the overlap dialog takes over and
                # we shouldn't pre-record the point.
                overlap_info = None
                if self.current_point_type == "positive":
                    overlap_info = self.check_point_overlap(current_point)

                if overlap_info:
                    self.handle_point_overlap(current_point, overlap_info)
                else:
                    if self.current_point_type == "positive":
                        self._record_positive_point(current_point)
                        print(f"Positive point added at: {current_point}")
                        self.finish_button.setEnabled(True)
                        self.switch_button.setEnabled(True)
                    else:
                        self._record_negative_point(current_point)
                        print(f"Negative point added at: {current_point}")
                    # Track actual user click for adaptive strategy
                    self.last_actual_click = current_point
                    self.update_preview_with_points()
        else:
            # Fallback if mode is not set
            if self.current_point_type == "positive":
                self._record_positive_point(current_point)
                print(f"Positive point added at: {current_point}")
                self.finish_button.setEnabled(True)
                self.switch_button.setEnabled(True)
            else:
                self._record_negative_point(current_point)
                print(f"Negative point added at: {current_point}")
            # Track actual user click for adaptive strategy
            self.last_actual_click = current_point
            # Update preview with the new point
            self.update_preview_with_points()

    def on_image_right_clicked(self, pos):
        """Handle right-click on image - show context menu for masks"""
        if self.current_image is None or self.displayed_pixmap is None:
            return

        # Convert click position to image coordinates
        current_point = self.get_image_coordinates(pos)
        if current_point is None:
            return
        
        # Only show context menu in creation mode and if we're over a mask
        if (hasattr(self, 'current_mode') and self.current_mode == "creation" and 
            self.masks_visible):
            mask_index = self.get_mask_at_position(current_point)
            if mask_index is not None:
                # Show context menu for the selected mask
                self.current_mask_index = mask_index
                self.show_mask_context_menu(pos, mask_index)

    def update_preview_with_points(self):
        """Update display with current points, bbox and preview mask"""
        # Start with original image and add cached overlay if masks are visible
        overlay_image = self.current_image.copy()
        if self.masks_visible and self.combined_mask_overlay is not None:
            overlay_image = cv2.addWeighted(overlay_image, 1.0, self.combined_mask_overlay, 0.8, 0)

        # Draw all current points
        for point in self.positive_points:
            cv2.circle(overlay_image, point, 4, (0, 255, 0), -1)  # Green for positive
        for point in self.negative_points:
            cv2.circle(overlay_image, point, 4, (255, 0, 0), -1)  # Red for negative

        # Draw the committed bbox, if any (cyan).
        if self.current_bbox is not None:
            x1, y1, x2, y2 = self.current_bbox
            cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Show dynamic expansion if we have any prompts at all (points or box)
        if self.positive_points or self.negative_points or self.current_bbox is not None:
            points = np.array(self.positive_points + self.negative_points) if (self.positive_points or self.negative_points) else None
            labels = np.array([1] * len(self.positive_points) + [0] * len(self.negative_points)) if (self.positive_points or self.negative_points) else None
            preview_mask = self.segmenter.propagate_points(
                points, labels, update_expanded_mask=False, box=self.current_bbox,
            )

            if preview_mask is not None:
                colored_preview = np.zeros_like(overlay_image)
                colored_preview[preview_mask > 0] = (128, 128, 128)  # Gray for preview
                overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_preview, 0.4, 0)

        # If we're between the two bbox clicks, show the first corner as a
        # crosshair to remind the user where the drag started.
        if self.bbox_mode and self.bbox_first_corner is not None:
            cx, cy = self.bbox_first_corner
            cv2.line(overlay_image, (cx - 8, cy), (cx + 8, cy), (0, 255, 255), 2)
            cv2.line(overlay_image, (cx, cy - 8), (cx, cy + 8), (0, 255, 255), 2)

        # Show suggested point cross when updating preview with points
        if hasattr(self, 'suggested_point') and self.suggested_point:
            row, col = self.suggested_point
            cv2.line(overlay_image, (col, row - 6), (col, row + 6), (255, 255, 0), 2)
            cv2.line(overlay_image, (col - 6, row), (col + 6, row), (255, 255, 0), 2)

        self.update_display(overlay_image)

    def delete_mask(self, mask_index):
        """Delete a mask and update the display"""
        if not (0 <= mask_index < len(self.expanded_masks)):
            return
            
        mask, label, _ = self.expanded_masks[mask_index]
        
        # Remove from expanded masks
        self.expanded_masks.pop(mask_index)
        
        # Regenerate the combined mask overlay
        self.regenerate_combined_mask_overlay()
        
        # Reset selection state
        self.current_mask_index = None
        self.current_mode = "creation"
        self.is_over_mask = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        
        # Print informative message
        print(f"Deleted mask with label '{label}'")
        
        # Update the display
        self.update_display_with_current_state()

    def change_mask_label(self, mask_index):
        """Change the label of a mask"""
        if not (0 <= mask_index < len(self.expanded_masks)):
            return
            
        mask, old_label, _ = self.expanded_masks[mask_index]
        
        # Show label dialog to pick a new label
        dialog = LabelDialog(self.labels, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_label = dialog.selected_label
            if new_label is None:
                new_label = dialog.new_label_edit.text()
                if new_label and dialog.chosen_color:
                    self.labels[new_label] = dialog.chosen_color
            
            if new_label and new_label in self.labels:
                color = self.labels[new_label]
                
                # Update the mask in the list
                self.expanded_masks[mask_index] = (mask, new_label, color)
                
                # Update the combined overlay
                self.regenerate_combined_mask_overlay()
                
                print(f"Changed mask label from '{old_label}' to '{new_label}'")
                
                # Update the display
                self.update_display_with_current_state()

    def regenerate_combined_mask_overlay(self):
        """Regenerate the combined mask overlay from all masks"""
        if not self.expanded_masks:
            self.combined_mask_overlay = None
            return
    
        self.combined_mask_overlay = np.zeros_like(self.current_image)
        
        for mask, _, color in self.expanded_masks:
            colored_mask = np.zeros_like(self.current_image)
            colored_mask[mask > 0] = [color.red(), color.green(), color.blue()]
            self.combined_mask_overlay[mask > 0] = colored_mask[mask > 0]

    def show_mask_context_menu(self, pos, mask_index):
        """Show context menu for the selected mask"""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction  # Import QAction from QtGui instead of QtWidgets
        
        if mask_index is None or mask_index >= len(self.expanded_masks):
            return
        
        mask, label, color = self.expanded_masks[mask_index]
        
        # Create context menu
        context_menu = QMenu(self)
        
        # Add actions
        delete_action = QAction(f"Delete '{label}' mask", self)
        change_label_action = QAction(f"Change label (current: '{label}')", self)
        
        # Add actions to menu
        context_menu.addAction(delete_action)
        context_menu.addAction(change_label_action)
        
        # Connect actions to handlers
        delete_action.triggered.connect(lambda: self.delete_mask(mask_index))
        change_label_action.triggered.connect(lambda: self.change_mask_label(mask_index))
        
        # Show context menu at cursor position
        context_menu.exec(self.mapToGlobal(pos))

    def on_mask_expansion_complete(self, mask, stored_positive_points):
        if mask is None:
            print("No mask generated.")
            self.is_expanding = False  # Reset expansion state
            self.finish_button.setEnabled(True)  # Re-enable the finish button
            return

        dialog = LabelDialog(self.labels, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            label = dialog.selected_label
            if label is None:
                label = dialog.new_label_edit.text()
                if label and dialog.chosen_color:
                    self.labels[label] = dialog.chosen_color
            color = self.labels.get(label)
            if color:
                # Update current_mask and current_label for future reference
                self.current_mask = mask
                self.current_label = label
                
                # Keep storing in expanded_masks for future reference
                self.expanded_masks.append((mask, label, color))

                # Enable the toggle masks button after first mask is created
                if not self.toggle_masks_button.isEnabled():
                    self.toggle_masks_button.setEnabled(True)
                
                # Create overlay image BEFORE using it
                overlay_image = self.current_image.copy()
                
                # Update the combined overlay for efficient display
                if self.combined_mask_overlay is None:
                    self.combined_mask_overlay = np.zeros_like(self.current_image)
                
                colored_mask = np.zeros_like(overlay_image)
                colored_mask[mask > 0] = [color.red(), color.green(), color.blue()]
                self.combined_mask_overlay[mask > 0] = colored_mask[mask > 0]
                
                # Update display with the new combined overlay and suggested point
                overlay_image = cv2.addWeighted(self.current_image, 1.0, self.combined_mask_overlay, 0.8, 0)
                
                self.update_display(overlay_image)
                
                # Calculate and display coverage statistics
                self.current_mask_area = np.sum(mask)
                self.total_image_area = self.current_image.shape[0] * self.current_image.shape[1]
                self.coverage_ratio = self.current_mask_area / self.total_image_area
                
                print(f"Added mask with label '{label}' - Coverage: {self.coverage_ratio:.2%}")
                
        # Reset expansion state regardless of dialog result
        self.is_expanding = False
        
        # Re-enable the finish button for the next mask
        self.finish_button.setEnabled(True)
        
        # Instead of automatically starting a new labeling cycle, just prepare for the next one
        # This way the user can place points for the next mask at their own pace
        self.switch_button.setEnabled(False)
        if self.current_point_type == "negative":
            self.switch_point_type()  # Reset to positive if it was negative

        # Get next suggested point using the point selection strategy
        if hasattr(self, 'point_strategy') and self.point_strategy is not None:
            # Update the strategy with the last created mask
            if mask is not None:
                # Extract simple features for the created mask
                area = np.sum(mask)
                if area > 0:
                    y_indices, x_indices = np.where(mask)
                    centroid_y = np.mean(y_indices)
                    centroid_x = np.mean(x_indices)
                    # Create a simple feature vector
                    last_feature = np.array([area, centroid_y, centroid_x, 0.5])
                    
                    # Start a new thread to get the next suggested point
                    # Convert user click from (x, y) to (y, x) format for the strategy
                    actual_click_yx = None
                    if self.last_actual_click:
                        x, y = self.last_actual_click
                        actual_click_yx = (y, x)  # Convert (x, y) to (y, x)
                    
                    self.point_thread = PointSuggestionThread(
                        self.point_strategy, 
                        self.segmenter, 
                        self.current_image,
                        expanded_masks=self.expanded_masks,
                        last_mask=mask,
                        last_feature=last_feature,
                        actual_last_point=actual_click_yx
                    )
                    self.point_thread.result_ready.connect(lambda next_point: self.on_next_point_ready(next_point))
                    self.point_thread.error_occurred.connect(lambda error: self.on_next_point_error(error))
                    self.point_thread.heatmap_ready.connect(lambda A_map, next_point: self.display_acquisition_heatmap_main_thread(A_map, next_point))
                    self.point_thread.start()
                    
                    # Reset the actual click tracking after using it
                    self.last_actual_click = None

    def on_next_point_ready(self, next_point):
        """Handle when the next suggested point is ready"""
        if next_point:
            self.suggested_point = next_point
        else:
            print("No valid next point proposed.")
            self.suggested_point = None
        # If the strategy stored an acquisition map, display it on the main thread for debugging
        try:
            if hasattr(self, 'point_strategy') and self.point_strategy is not None:
                strat = self.point_strategy
                A_map = getattr(strat, '_last_acquisition_map', None)
                last_pt = getattr(strat, '_last_selected_point', None)
                # if A_map is not None and hasattr(strat, 'display_acquisition_heatmap_main_thread'):
                #     try:
                #         strat.display_acquisition_heatmap_main_thread(A_map, next_point if next_point is not None else last_pt)
                #     except Exception:
                #         pass
        except Exception:
            pass

        # Update the display with the new suggested point
        self.update_display_with_current_state()

    def on_next_point_error(self, error_msg):
        """Handle errors when getting next point"""
        print(f"Error getting next point: {error_msg}")
        # Fall back to center of image or keep current point
        if not hasattr(self, 'suggested_point') or self.suggested_point is None:
            if hasattr(self, 'current_image') and self.current_image is not None:
                h, w = self.current_image.shape[:2]
                self.suggested_point = (h // 2, w // 2)
            else:
                self.suggested_point = None

    def toggle_masks_visibility(self):
        """Toggle the visibility of all masks"""
        self.masks_visible = not self.masks_visible
        
        # Update button appearance and mode
        if self.masks_visible:
            self.toggle_masks_button.setText("👁️❌")
            # Return to creation mode when showing masks
            self.current_mode = "creation" 
        else:
            # When hiding masks, switch to visualization mode
            self.toggle_masks_button.setText("👁️")
            self.current_mode = "visualization"
            self.is_over_mask = False
            self.current_mask_index = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        # Update the display
        if self.current_image is not None:
            self.update_display_with_current_state()

    def get_mask_at_position(self, pos):
        """Determine which mask (if any) is at the given position"""
        if not self.expanded_masks or not self.masks_visible:
            return None
        
        x, y = pos
        
        # Check each mask in reverse order (newest first)
        for i in range(len(self.expanded_masks) - 1, -1, -1):
            mask, _, _ = self.expanded_masks[i]
            
            # Check if the position is within this mask
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                return i
        
        return None

    def check_point_overlap(self, point):
        """Check if a point overlaps with any existing mask and return mask info"""
        if not self.expanded_masks:
            return None
            
        x, y = point
        
        # Check each mask
        for i, (mask, label, color) in enumerate(self.expanded_masks):
            # Check if the point is within this mask
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                return {
                    'mask_index': i,
                    'mask': mask,
                    'label': label,
                    'color': color
                }
        
        return None

    def handle_point_overlap(self, clicked_point, overlap_info):
        """Handle when user clicks on a point that overlaps with existing masks"""
        overlapping_mask = overlap_info['mask']
        overlapping_label = overlap_info['label']
        
        # Show dialog to user
        dialog = OverlapDialog(self.suggested_point, overlapping_label, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.result_choice == "same_object":
                self.handle_same_object_merge(clicked_point, overlap_info)
            elif dialog.result_choice == "different_object":
                self.handle_different_object_overlap(clicked_point, overlap_info)
        # If canceled, do nothing
    
    def handle_same_object_merge(self, clicked_point, overlap_info):
        """Handle when user says the point belongs to the same object - union masks"""
        # Add the point and create mask as normal
        self.positive_points.append(clicked_point)
        self.last_actual_click = clicked_point
        
        # Generate the new mask
        points = np.array(self.positive_points + self.negative_points)
        labels = np.array([1] * len(self.positive_points) + [0] * len(self.negative_points))
        new_mask = self.segmenter.propagate_points(points, labels, update_expanded_mask=False)
        
        if new_mask is not None:
            # Union with the existing mask
            existing_mask = overlap_info['mask']
            existing_label = overlap_info['label']
            existing_color = overlap_info['color']
            mask_index = overlap_info['mask_index']
            
            # Create union of masks
            union_mask = np.logical_or(existing_mask, new_mask).astype(np.uint8)

            # Replace only the selected mask (mask_index) with the union, keep other masks of the same label
            if 0 <= mask_index < len(self.expanded_masks):
                self.expanded_masks[mask_index] = (union_mask, existing_label, existing_color)
            else:
                # Fallback: if index is invalid, just append
                self.expanded_masks.append((union_mask, existing_label, existing_color))
            
            # Regenerate combined overlay
            self.regenerate_combined_mask_overlay()
            
            print(f"Merged masks for label '{existing_label}'")

            # Inform the point strategy about the updated expanded mask.
            try:
                if hasattr(self, 'point_strategy') and self.point_strategy is not None:
                    # Update expanded_masks in the strategy if present
                    if hasattr(self.point_strategy, 'expanded_masks'):
                        if len(self.point_strategy.expanded_masks) > mask_index:
                            self.point_strategy.expanded_masks[mask_index] = union_mask
                        else:
                            self.point_strategy.expanded_masks.append(union_mask)
            except Exception:
                pass
            
            # Clear points and reset UI
            self.positive_points = []
            self.negative_points = []
            self.finish_button.setEnabled(True)
            self.switch_button.setEnabled(False)
            if self.current_point_type == "negative":
                self.switch_point_type()
            
            # Start next point suggestion
            self.start_next_point_suggestion(union_mask)
    
    def handle_different_object_overlap(self, clicked_point, overlap_info):
        """Handle when user says it's a different object - resolve overlap"""
        # Add the point and create mask as normal
        self.positive_points.append(clicked_point)
        self.last_actual_click = clicked_point
        
        # Generate the new mask
        points = np.array(self.positive_points + self.negative_points)
        labels = np.array([1] * len(self.positive_points) + [0] * len(self.negative_points))
        new_mask = self.segmenter.propagate_points(points, labels, update_expanded_mask=False)
        
        if new_mask is not None:
            # Get label for the new mask
            dialog = LabelDialog(self.labels, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                new_label = dialog.selected_label
                if new_label is None:
                    new_label = dialog.new_label_edit.text()
                    if new_label and dialog.chosen_color:
                        self.labels[new_label] = dialog.chosen_color
                
                new_color = self.labels.get(new_label)
                if new_color:
                    # Use the merge_overlapping_masks function to resolve overlap
                    self.resolve_mask_overlap(new_mask, new_label, new_color, overlap_info)
    
    def resolve_mask_overlap(self, new_mask, new_label, new_color, overlap_info):
        """Resolve overlapping masks by keeping non-overlapping parts and handling overlap properly"""
        # Get the overlapping mask
        existing_mask = overlap_info['mask']
        existing_label = overlap_info['label']
        existing_color = overlap_info['color']
        overlapping_index = overlap_info['mask_index']
        
        # Find the actual overlap area
        overlap_area = np.logical_and(new_mask, existing_mask)
        overlap_pixels = np.sum(overlap_area)
        
        if overlap_pixels == 0:
            # No actual overlap, just add the new mask
            self.expanded_masks.append((new_mask.astype(np.uint8), new_label, new_color))
        else:
            # There is actual overlap - resolve it
            # Remove overlap from existing mask (existing mask keeps non-overlapping parts)
            existing_mask_cleaned = np.logical_and(existing_mask, ~overlap_area).astype(np.uint8)
            
            # Remove overlap from new mask (new mask keeps non-overlapping parts)  
            new_mask_cleaned = np.logical_and(new_mask, ~overlap_area).astype(np.uint8)
            
            # Assign the overlap area to the new mask (since user clicked there)
            overlap_mask = overlap_area.astype(np.uint8)
            new_mask_with_overlap = np.logical_or(new_mask_cleaned, overlap_mask).astype(np.uint8)
            
            # Update the existing mask (remove the overlapping index and add cleaned version)
            self.expanded_masks[overlapping_index] = (existing_mask_cleaned, existing_label, existing_color)
            
            # Add the new mask with the overlap area
            self.expanded_masks.append((new_mask_with_overlap, new_label, new_color))
        
        # Regenerate combined overlay
        self.regenerate_combined_mask_overlay()
        
        # Clear points and reset UI
        self.positive_points = []
        self.negative_points = []
        self.finish_button.setEnabled(True)
        self.switch_button.setEnabled(False)
        if self.current_point_type == "negative":
            self.switch_point_type()
        
        # Start next point suggestion with the new mask
        if self.expanded_masks:
            last_mask = self.expanded_masks[-1][0]  # Use the last created mask
            self.start_next_point_suggestion(last_mask)
        
        # Force display update to show the merged masks
        self.update_display_with_current_state()
    
    def start_next_point_suggestion(self, last_mask):
        """Start the next point suggestion thread"""
        if hasattr(self, 'point_strategy') and self.point_strategy is not None:
            # Extract features for the mask
            area = np.sum(last_mask)
            if area > 0:
                y_coords, x_coords = np.where(last_mask)
                centroid_y, centroid_x = np.mean(y_coords), np.mean(x_coords)
                last_feature = np.array([area, centroid_y, centroid_x])
            else:
                last_feature = np.array([0, 0, 0])
            
            # Convert user click from (x, y) to (y, x) format for the strategy
            actual_click_yx = None
            if self.last_actual_click:
                x, y = self.last_actual_click
                actual_click_yx = (y, x)
            
            # Start new thread
            self.point_thread = PointSuggestionThread(
                self.point_strategy, 
                self.segmenter, 
                self.current_image,
                expanded_masks=self.expanded_masks,
                last_mask=last_mask,
                last_feature=last_feature,
                actual_last_point=actual_click_yx
            )
            self.point_thread.result_ready.connect(lambda next_point: self.on_next_point_ready(next_point))
            self.point_thread.error_occurred.connect(lambda error: self.on_next_point_error(error))
            self.point_thread.heatmap_ready.connect(lambda A_map, next_point: self.display_acquisition_heatmap_main_thread(A_map, next_point))
            self.point_thread.start()
            
            # Reset the actual click tracking after using it
            self.last_actual_click = None
        
        # Update display
        self.update_display_with_current_state()

    def update_display_with_current_state(self):
        """Update the display with current masks and points"""
        # Start with the original image
        overlay_image = self.current_image.copy()
        
        # Add all expanded masks if visible
        if self.masks_visible and self.expanded_masks:
            # Generate a fresh overlay
            combined_overlay = np.zeros_like(self.current_image)
            for i, (mask, _, color) in enumerate(self.expanded_masks):
                # Highlight selected mask with brighter color
                alpha = 0.8 if i == self.current_mask_index else 0.6
                colored_mask = np.zeros_like(overlay_image)
                colored_mask[mask > 0] = [color.red(), color.green(), color.blue()]
                combined_overlay[mask > 0] = colored_mask[mask > 0]
            
            overlay_image = cv2.addWeighted(overlay_image, 1.0, combined_overlay, 0.8, 0)
        
        # Draw points in creation mode
        if self.current_mode == "creation":
            # Draw all current points
            for point in self.positive_points:
                cv2.circle(overlay_image, point, 4, (0, 255, 0), -1)  # Green for positive
            for point in self.negative_points:
                cv2.circle(overlay_image, point, 4, (255, 0, 0), -1)  # Red for negative
            
            # Always show suggested point (arrow/cross) during labeling
            if hasattr(self, 'suggested_point') and self.suggested_point:
                row, col = self.suggested_point
                cv2.line(overlay_image, (col, row - 6), (col, row + 6), (255, 255, 0), 2)
                cv2.line(overlay_image, (col - 6, row), (col + 6, row), (255, 255, 0), 2)
        
        # Update the display
        self.update_display(overlay_image)

    def on_mouse_moved(self, pos):
        """Handle mouse movement - determine mode and update preview"""
        # Skip if not in interactive mode
        if not self.is_selecting_points or self.current_image is None:
            return
        
        # Get image coordinates from screen coordinates
        image_pos = self.get_image_coordinates(pos)
        
        # Special handling for cursor outside the image area: freeze at the
        # current prompt set (points + bbox). Delegate to the unified
        # preview path so points + bbox stay in sync everywhere.
        if image_pos is None:
            if self.positive_points or self.negative_points or self.current_bbox is not None:
                self.update_preview_with_points()
            return
        
        # If we reach here, the cursor is inside the image
        # Check which mode we're in
        if self.current_mode == "creation":
            # Call the dynamic expansion function with the current position
            self.dynamic_expand(pos)
        elif self.current_mode == "selection":
            # Handle hover over masks
            mask_index = self.get_mask_at_position(image_pos)
            if mask_index is not None:
                if not self.is_over_mask or self.current_mask_index != mask_index:
                    self.is_over_mask = True
                    self.current_mask_index = mask_index
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                    # Update display to highlight the mask
                    self.update_display_with_current_state()
            else:
                if self.is_over_mask:
                    self.is_over_mask = False
                    self.current_mask_index = None
                    self.setCursor(Qt.CursorShape.ArrowCursor)
                    # Update display to remove highlight
                    self.update_display_with_current_state()

    def on_cursor_over_button(self):
        """Called when cursor enters any button.

        Freezes the preview at "what the current prompts would produce" —
        i.e. points + bbox together — so leaving the canvas to click Finish
        doesn't visually wipe the bbox or revert the SAM2 preview to
        points-only.
        """
        if not self.is_selecting_points or self.current_image is None or self.segmenter is None:
            return

        # In visualization mode (masks hidden), keep behavior simple: base
        # image plus the suggested-point cross.
        if self.current_mode == "visualization" or not self.masks_visible:
            overlay_image = self.current_image.copy()
            if hasattr(self, 'suggested_point') and self.suggested_point and self.current_mode == "creation":
                row, col = self.suggested_point
                cv2.line(overlay_image, (col, row - 6), (col, row + 6), (255, 255, 0), 2)
                cv2.line(overlay_image, (col - 6, row), (col + 6, row), (255, 255, 0), 2)
            self.update_display(overlay_image)
            return

        # Delegate to the unified preview path — it already draws committed
        # points + committed bbox + a SAM2 preview that conditions on both.
        self.update_preview_with_points()

    def dynamic_expand(self, pos):
        """Handle dynamic expansion with cursor position"""
        # Skip if not in creation mode or not selecting points
        if not self.is_selecting_points or self.current_image is None or self.displayed_pixmap is None:
            return

        # Get image coordinates for the cursor position
        cursor_point = self.get_image_coordinates(pos)
        if cursor_point is None:
            return

        # BBox armed but no first corner yet: don't run the dynamic
        # point-expansion at all (it's distracting and irrelevant — the next
        # click sets a corner, not a point). Just keep the static overlay
        # with already-committed prompts visible.
        if self.bbox_mode and self.bbox_first_corner is None:
            self.update_preview_with_points()
            return

        # Mid-bbox-draw: render a live rectangle from the first corner to the
        # cursor, plus a SAM2 preview using points + that temporary box.
        if self.bbox_mode and self.bbox_first_corner is not None:
            self._render_bbox_draw_preview(cursor_point)
            return

        # First check if we're over an existing mask
        mask_index = self.get_mask_at_position(cursor_point)

        # Update cursor and mask selection state
        if mask_index is not None and self.masks_visible:
            # If we just entered a mask, switch to pointing hand cursor
            if not self.is_over_mask or self.current_mask_index != mask_index:
                self.is_over_mask = True
                self.current_mask_index = mask_index
                self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            # If we just left a mask, reset cursor and selection state
            if self.is_over_mask:
                self.is_over_mask = False
                self.current_mask_index = None
                self.setCursor(Qt.CursorShape.ArrowCursor)

        # Start with original image and add cached overlay if it exists
        overlay_image = self.current_image.copy()
        if self.combined_mask_overlay is not None:
            overlay_image = cv2.addWeighted(overlay_image, 1.0, self.combined_mask_overlay, 0.8, 0)

        # Draw all current points
        for point in self.positive_points:
            cv2.circle(overlay_image, point, 4, (0, 255, 0), -1)  # Green for positive
        for point in self.negative_points:
            cv2.circle(overlay_image, point, 4, (255, 0, 0), -1)  # Red for negative

        # Draw the committed bbox if any
        if self.current_bbox is not None:
            x1, y1, x2, y2 = self.current_bbox
            cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Create points array with all current points plus cursor
        points = np.array(self.positive_points + self.negative_points + [cursor_point])
        cursor_label = 0 if self.current_point_type == "negative" else 1
        labels = np.array([1] * len(self.positive_points) + [0] * len(self.negative_points) + [cursor_label])
        preview_mask = self.segmenter.propagate_points(
            points, labels, update_expanded_mask=False, box=self.current_bbox,
        )

        # Add the preview mask if it exists
        if preview_mask is not None:
            colored_preview = np.zeros_like(overlay_image)
            colored_preview[preview_mask > 0] = (128, 128, 128)  # Gray instead of green
            overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_preview, 0.4, 0)

        # Draw cursor point (normal mode - just the colored point)
        cursor_color = (0, 255, 0) if self.current_point_type == "positive" else (255, 0, 0)
        cv2.circle(overlay_image, cursor_point, 4, cursor_color, -1)

        # Always show suggested point arrow/cross during labeling
        if hasattr(self, 'suggested_point') and self.suggested_point:
            row, col = self.suggested_point
            cv2.line(overlay_image, (col, row - 6), (col, row + 6), (255, 255, 0), 2)
            cv2.line(overlay_image, (col - 6, row), (col + 6, row), (255, 255, 0), 2)

        self.update_display(overlay_image)

    def _render_bbox_draw_preview(self, cursor_point):
        """Render the live rectangle while the user is between the 1st and
        2nd bbox click. Includes a SAM2 preview using the committed points
        plus the in-progress box, so the user can judge the box before they
        commit the 2nd click.
        """
        overlay_image = self.current_image.copy()
        if self.combined_mask_overlay is not None:
            overlay_image = cv2.addWeighted(overlay_image, 1.0, self.combined_mask_overlay, 0.8, 0)

        # Existing points (committed)
        for point in self.positive_points:
            cv2.circle(overlay_image, point, 4, (0, 255, 0), -1)
        for point in self.negative_points:
            cv2.circle(overlay_image, point, 4, (255, 0, 0), -1)

        # In-progress rectangle from first_corner to cursor
        x1 = min(self.bbox_first_corner[0], cursor_point[0])
        y1 = min(self.bbox_first_corner[1], cursor_point[1])
        x2 = max(self.bbox_first_corner[0], cursor_point[0])
        y2 = max(self.bbox_first_corner[1], cursor_point[1])
        cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Preview SAM2 with the temp box (skip if degenerate to avoid noise)
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            points = np.array(self.positive_points + self.negative_points) if (self.positive_points or self.negative_points) else None
            labels = np.array([1] * len(self.positive_points) + [0] * len(self.negative_points)) if (self.positive_points or self.negative_points) else None
            preview_mask = self.segmenter.propagate_points(
                points, labels, update_expanded_mask=False, box=(x1, y1, x2, y2),
            )
            if preview_mask is not None:
                colored_preview = np.zeros_like(overlay_image)
                colored_preview[preview_mask > 0] = (128, 128, 128)
                overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_preview, 0.4, 0)

        # Crosshair at the first corner (so the anchor stays visible)
        cx, cy = self.bbox_first_corner
        cv2.line(overlay_image, (cx - 8, cy), (cx + 8, cy), (0, 255, 255), 2)
        cv2.line(overlay_image, (cx, cy - 8), (cx, cy + 8), (0, 255, 255), 2)

        self.update_display(overlay_image)

    def dynamic_expand_with_negative(self, pos):
        if self.current_image is None or self.displayed_pixmap is None:
            return

        cursor_point = self.get_image_coordinates(pos)
        if cursor_point is None:
            return

        # Two point expansion: fixed positive and moving negative
        points = np.array([self.positive_points[0], cursor_point])
        # Create appropriate labels (1 for positive points, 0 for negative points)
        labels = np.array([1, 0])
        
        preview_mask = self.segmenter.propagate_points(points, labels, update_expanded_mask=False)
        
        # Start with original image and add cached overlay if it exists
        overlay_image = self.current_image.copy()
        if self.combined_mask_overlay is not None:
            overlay_image = cv2.addWeighted(overlay_image, 1.0, self.combined_mask_overlay, 0.8, 0)
        
        # Add the preview mask
        if preview_mask is not None:
            colored_preview = np.zeros_like(overlay_image)
            colored_preview[preview_mask > 0] = (128, 128, 128)  # Gray instead of green
            overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_preview, 0.4, 0)
        
        # Draw the points - moved after all overlays
        pos_x, pos_y = self.positive_points[0]
        cv2.circle(overlay_image, (pos_x, pos_y), 4, (0, 255, 0), -1)  # Green for positive
        cv2.circle(overlay_image, cursor_point, 4, (255, 0, 0), -1)  # Red for negative

        # Show suggested point cross during negative point expansion too
        if hasattr(self, 'suggested_point') and self.suggested_point:
            row, col = self.suggested_point
            cv2.line(overlay_image, (col, row - 6), (col, row + 6), (255, 255, 0), 2)
            cv2.line(overlay_image, (col - 6, row), (col + 6, row), (255, 255, 0), 2)

        self.update_display(overlay_image)

    def update_display(self, overlay_image):
        """Helper method to update the display with a new image"""
        height, width, channel = overlay_image.shape
        bytes_per_line = 3 * width
        qimage = QImage(overlay_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.displayed_pixmap = scaled_pixmap
        self.image_label.setPixmap(scaled_pixmap)

    def get_image_coordinates(self, pos):
        """Helper function to convert screen coordinates to image coordinates"""
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        pixmap_width = self.displayed_pixmap.width()
        pixmap_height = self.displayed_pixmap.height()

        offset_x = (label_width - pixmap_width) / 2
        offset_y = (label_height - pixmap_height) / 2

        if not (offset_x <= pos.x() <= offset_x + pixmap_width and 
                offset_y <= pos.y() <= offset_y + pixmap_height):
            return None

        original_h, original_w, _ = self.current_image.shape
        ratio_x = original_w / pixmap_width
        ratio_y = original_h / pixmap_height

        orig_x = int((pos.x() - offset_x) * ratio_x)
        orig_y = int((pos.y() - offset_y) * ratio_y)
        
        return (orig_x, orig_y)

    def show_image(self, overlay_point=None):
        if not self.image_list or self.current_index >= len(self.image_list):
            print("No image available to display.")
            return

        # Start with the original image
        if self.current_image is None:
            current_image_path = self.image_list[self.current_index]
            image = self._load_and_normalize_image(current_image_path)
            if image is None:
                return
            self.current_image = image

        # Create a fresh overlay with the original image
        image = self.current_image.copy()

        # Add all expanded masks
        if self.expanded_masks:
            for mask, label, color in self.expanded_masks:
                colored_mask = np.zeros_like(image)
                colored_mask[mask > 0] = [color.red(), color.green(), color.blue()]
                image = cv2.addWeighted(image, 1.0, colored_mask, 0.8, 0)

        # Store the current state of the overlay
        self.overlay_image = image.copy()

        # Add the suggested point if it exists
        if overlay_point:
            row, col = overlay_point
            line_length = 6
            line_color = (255, 255, 0)
            line_thickness = 2
            cv2.line(image, (col, row - line_length), (col, row + line_length), line_color, line_thickness)
            cv2.line(image, (col - line_length, row), (col + line_length, row), line_color, line_thickness)

        # Update the display
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.displayed_pixmap = scaled_pixmap
        self.image_label.setPixmap(scaled_pixmap)

    def next_image(self):
        if not self.image_list or self.current_index >= len(self.image_list):
            QMessageBox.information(self, "Finished", "You have finished labeling all the images.")
            self.reset_viewer()
            return

        # If we have expanded masks, ask what to save
        if self.expanded_masks:
            save_dialog = SaveDialog(self)
            if save_dialog.exec() == QDialog.DialogCode.Accepted:
                if save_dialog.selected_option:
                    self.save_image(self.image_list[self.current_index], save_dialog.selected_option)
            # Continue to next image regardless of save choice

        # Disable image interactions until Start button is pressed again
        self.image_label.interactions_enabled = False
        
        # Reset all the labeling-related variables
        self.segmenter = None
        self.plas_segmenter = None  # Reset PLAS segmenter
        self.expanded_masks = []
        self.overlay_image = None
        self.suggested_point = None
        self.combined_mask_overlay = None  # Clear the combined mask overlay when moving to next image
        
        # Hide the point selection buttons and enable Start button
        self.switch_button.hide()
        self.bbox_button.hide()
        self.finish_button.hide()
        self.toggle_masks_button.hide()
        self.toggle_masks_button.setEnabled(False)
        self.start_button.setEnabled(True)

        # Move to next image
        self.current_index += 1
        
        if self.current_index < len(self.image_list):
            # Reset point strategy for new image
            self.point_strategy = None
            # Load and display the new image
            current_image_path = self.image_list[self.current_index]
            image = self._load_and_normalize_image(current_image_path)
            if image is None:
                return
            self.current_image = image
            self.show_image()
        else:
            QMessageBox.information(self, "Finished", "You have finished labeling all the images.")
            self.reset_viewer()

    def save_image(self, image_path, save_type="combined"):
        if not self.expanded_masks:  # If no masks were created, don't save anything
            return
            
        # Create a background image with custom color
        height, width, _ = self.current_image.shape
        # background_color = [63, 69, 131]  # Custom RGB background color
        background_color = [0, 0, 0]  # White background
        
        # Create the SAM2 segmentation mask
        sam2_mask_image = np.full((height, width, 3), fill_value=background_color, dtype=np.uint8)  # Custom background
        
        # Add all expanded masks with their respective colors
        for mask, label, color in self.expanded_masks:
            colored_mask = np.zeros_like(sam2_mask_image)
            colored_mask[mask > 0] = [color.red(), color.green(), color.blue()]
            # Use direct assignment instead of addWeighted since we want solid colors
            sam2_mask_image[mask > 0] = colored_mask[mask > 0]
        
        # Prepare session subfolder based on date and hour
        from datetime import datetime
        if not hasattr(self, 'session_save_folder'):
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_save_folder = os.path.join(os.getcwd(), "app_annotations", session_id)
            os.makedirs(self.session_save_folder, exist_ok=True)
        save_folder = self.session_save_folder
        image_name = os.path.basename(image_path)
        base_name, ext = os.path.splitext(image_name)
        
        if save_type in ["sam2_only", "both"]:
            # Save SAM2-only mask
            sam2_path = os.path.join(save_folder, f"{base_name}_sam2_mask{ext}")
            sam2_mask_bgr = cv2.cvtColor(sam2_mask_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(sam2_path, sam2_mask_bgr)
            print(f"Saved SAM2-only segmentation to: {sam2_path}")
        
        if save_type in ["combined", "both"]:
            # Create combined SAM2+PLAS mask
            try:
                # Prepare points and labels for PLAS expansion
                points_to_process = []
                labels_to_process = []
                int_labels_to_rgb = {}
                
                for i, (mask, label, color) in enumerate(self.expanded_masks):
                    # Sample some points from each mask for PLAS expansion
                    mask_indices = np.argwhere(mask > 0)
                    if len(mask_indices) > 0:
                        # Sample up to 5 points per mask
                        sample_size = min(5, len(mask_indices))
                        sampled_indices = np.random.choice(len(mask_indices), sample_size, replace=False)
                        sampled_points = mask_indices[sampled_indices]
                        
                        # Convert from (row, col) to (col, row) format for PLAS
                        sampled_points_xy = [(point[1], point[0]) for point in sampled_points]
                        
                        points_to_process.extend(sampled_points_xy)
                        labels_to_process.extend([i + 1] * len(sampled_points))  # Start labels from 1
                        int_labels_to_rgb[i + 1] = [color.red(), color.green(), color.blue()]
                
                if points_to_process and hasattr(self, 'plas_segmenter'):
                    # Get SAM2 features if available
                    features_sam2 = getattr(self.segmenter, 'features_sam2', None)
                    
                    # PLAS expansion
                    plas_mask = self.plas_segmenter.expand_labels(
                        self.current_image, 
                        points_to_process, 
                        labels_to_process, 
                        features_sam2=features_sam2
                    )
                    
                    # Create PLAS mask RGB
                    plas_mask_rgb = np.full((height, width, 3), fill_value=background_color, dtype=np.uint8)
                    for label, color in int_labels_to_rgb.items():
                        plas_mask_rgb[plas_mask == label] = color
                    
                    # Combine SAM2 and PLAS: where SAM2 is unlabeled (black), use PLAS
                    combined_mask_rgb = sam2_mask_image.copy()
                    unlabeled_pixels = np.all(sam2_mask_image == background_color, axis=2)
                    combined_mask_rgb[unlabeled_pixels] = plas_mask_rgb[unlabeled_pixels]
                    
                    # Save combined mask
                    combined_path = os.path.join(save_folder, f"{base_name}_combined_mask{ext}")
                    combined_mask_bgr = cv2.cvtColor(combined_mask_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(combined_path, combined_mask_bgr)
                    print(f"Saved combined SAM2+PLAS segmentation to: {combined_path}")
                else:
                    print("PLAS segmenter not available, saving SAM2-only instead")
                    fallback_path = os.path.join(save_folder, f"{base_name}_combined_mask{ext}")
                    sam2_mask_bgr = cv2.cvtColor(sam2_mask_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(fallback_path, sam2_mask_bgr)
                    print(f"Saved SAM2-only (fallback) segmentation to: {fallback_path}")
                    
            except Exception as e:
                print(f"Error during PLAS combination: {e}")
                # Fallback to SAM2 only
                fallback_path = os.path.join(save_folder, f"{base_name}_combined_mask{ext}")
                sam2_mask_bgr = cv2.cvtColor(sam2_mask_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(fallback_path, sam2_mask_bgr)
                print(f"Saved SAM2-only (error fallback) segmentation to: {fallback_path}")

    def reset_viewer(self):
        self.image_label.clear()
        self.image_list = []
        self.current_index = 0
        self.segmenter = None
        self.plas_segmenter = None  # Reset PLAS segmenter
        self.point_strategy = None  # Reset point strategy
        self.start_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.prev_button.setEnabled(False)

    def resizeEvent(self, event):
        """Redraw the current overlay at the new label size so the image
        scales smoothly while the user drags the window. Pixmap scaling
        already respects aspect ratio in update_display/show_image."""
        super().resizeEvent(event)
        # Only redraw if we actually have an image loaded; resize fires once
        # during construction before anything is set up.
        if getattr(self, "current_image", None) is not None:
            try:
                self.update_display_with_current_state()
            except Exception:
                # Don't crash the GUI if a redraw happens mid-state-transition
                pass

    def closeEvent(self, event):
        if self.image_list and 0 <= self.current_index < len(self.image_list) and self.expanded_masks:
            # Only show save dialog if there are masks to save
            save_dialog = SaveDialog(self)
            if save_dialog.exec() == QDialog.DialogCode.Accepted:
                if save_dialog.selected_option:
                    self.save_image(self.image_list[self.current_index], save_dialog.selected_option)
                event.accept()
            else:
                # User cancelled, ask if they want to exit without saving
                confirmation = QMessageBox(self)
                confirmation.setWindowTitle("Exit without saving")
                confirmation.setText("Are you sure you want to exit without saving?")
                confirmation.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                confirmation.setDefaultButton(QMessageBox.StandardButton.No)
                
                if confirmation.exec() == QMessageBox.StandardButton.Yes:
                    event.accept()
                else:
                    event.ignore()
        else:
            event.accept()

    def switch_point_type(self):
        """Switch between positive and negative point selection"""
        if self.current_point_type == "positive":
            self.current_point_type = "negative"
            self.switch_button.setText("Positive")
            self.switch_button.setProperty("class", "switch-button-positive")
        else:
            self.current_point_type = "positive"
            self.switch_button.setText("Negative")
            self.switch_button.setProperty("class", "switch-button-negative")

        # Force style refresh
        self.switch_button.style().unpolish(self.switch_button)
        self.switch_button.style().polish(self.switch_button)

    def toggle_bbox_mode(self):
        """Enter / leave bbox-drawing mode.

        Active state: next image-click sets the first corner; the click after
        that sets the opposite corner, stores the bbox, and auto-exits the
        mode. Mouse-move between the two clicks renders a live rectangle.
        """
        self.bbox_mode = not self.bbox_mode
        if not self.bbox_mode:
            # Cancel any in-progress placement
            self.bbox_first_corner = None
        # Visual cue: append a filled-square glyph when active
        self.bbox_button.setText("BBox ◼" if self.bbox_mode else "BBox")
        self.bbox_button.setProperty(
            "class",
            "bbox-button-active" if self.bbox_mode else "bbox-button-idle",
        )
        self.bbox_button.style().unpolish(self.bbox_button)
        self.bbox_button.style().polish(self.bbox_button)
        # Cursor: cross-hair while armed, arrow otherwise.
        if self.bbox_mode:
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
        # Refresh preview so the cursor visual updates immediately
        if self.is_selecting_points:
            self.update_preview_with_points()

    def on_finish_button_clicked(self):
        """Handle finish button click - expand mask and update UI"""
        if not self.is_expanding and (self.positive_points or self.current_bbox is not None):
            # Stop any ongoing expansion
            if hasattr(self, 'expansion_thread') and self.expansion_thread and self.expansion_thread.isRunning():
                self.expansion_thread.terminate()
                self.expansion_thread.wait()

            # Combine positive and negative points (may be empty when only a
            # bbox is used).
            if self.positive_points or self.negative_points:
                points = np.array(self.positive_points + self.negative_points)
                labels = np.array([1] * len(self.positive_points) + [0] * len(self.negative_points))
            else:
                points, labels = None, None

            # Store positive points for reference (overlap/strategy logic uses them).
            stored_positive_points = self.positive_points.copy()
            box_for_run = self.current_bbox

            # Create expansion thread with correct points and labels format
            self.expansion_thread = MaskExpansionThread(
                self.segmenter,
                points,
                labels,
                box=box_for_run,
            )

            # Connect to result_ready signal instead of finished signal
            self.expansion_thread.result_ready.connect(
                lambda mask: self.on_mask_expansion_complete(mask, stored_positive_points)
            )

            # Start expansion
            self.is_expanding = True
            self.expansion_thread.start()

            # Update UI
            self.finish_button.setEnabled(False)

            # Clear prompts and history for next mask. Reset bbox state too so
            # the next cycle starts clean.
            self.positive_points = []
            self.negative_points = []
            self.current_bbox = None
            self.bbox_first_corner = None
            self.bbox_mode = False
            self.bbox_button.setText("BBox")
            self.bbox_button.setProperty("class", "bbox-button-idle")
            self.bbox_button.style().unpolish(self.bbox_button)
            self.bbox_button.style().polish(self.bbox_button)
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
            self.input_history = []
            # Note: Don't reset last_actual_click here - it's needed for the next thread

    def prev_image(self):
        """Move to the previous image"""
        if not self.image_list or self.current_index <= 0:
            return

        # If we have expanded masks, ask what to save
        if self.expanded_masks:
            save_dialog = SaveDialog(self)
            if save_dialog.exec() == QDialog.DialogCode.Accepted:
                if save_dialog.selected_option:
                    self.save_image(self.image_list[self.current_index], save_dialog.selected_option)
            # Continue to previous image regardless of save choice

        # Disable image interactions until Start button is pressed again
        self.image_label.interactions_enabled = False
        
        # Reset all the labeling-related variables
        self.segmenter = None
        self.plas_segmenter = None  # Reset PLAS segmenter
        self.expanded_masks = []
        self.overlay_image = None
        self.suggested_point = None
        self.combined_mask_overlay = None  # Clear the combined mask overlay when moving to next image
        
        # Hide the point selection buttons and enable Start button
        self.switch_button.hide()
        self.bbox_button.hide()
        self.finish_button.hide()
        self.toggle_masks_button.hide()
        self.start_button.setEnabled(True)

        # Move to previous image
        self.current_index -= 1

        # Load and display the new image
        current_image_path = self.image_list[self.current_index]
        image = self._load_and_normalize_image(current_image_path)
        if image is None:
            return
        self.current_image = image
        self.show_image()

    # Add this method to the ImageViewer class
    def keyPressEvent(self, event):
        """Handle key press events"""
        from PyQt6.QtGui import QKeySequence
        from PyQt6.QtCore import Qt
        
        # Check for Ctrl+Z
        if event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if self.current_mode == "creation" and self.is_selecting_points:
                # If we're mid-bbox-draw (first corner placed but not the
                # second), cancel that draw first — it has no history entry
                # yet so this only resets local state.
                if self.bbox_mode and self.bbox_first_corner is not None:
                    print("Cancelled in-progress bbox")
                    self.bbox_first_corner = None
                    self.update_preview_with_points()
                elif self.input_history:
                    entry = self.input_history.pop()
                    kind = entry[0]
                    if kind == "pos_point" and self.positive_points:
                        removed = self.positive_points.pop()
                        print(f"Removed positive point at: {removed}")
                        if not self.positive_points and self.current_bbox is None:
                            self.finish_button.setEnabled(False)
                    elif kind == "neg_point" and self.negative_points:
                        removed = self.negative_points.pop()
                        print(f"Removed negative point at: {removed}")
                    elif kind == "bbox":
                        _, _, prev_bbox = entry
                        print(f"Removed bbox {self.current_bbox}; restored {prev_bbox}")
                        self.current_bbox = prev_bbox
                        if (
                            not self.positive_points
                            and self.current_bbox is None
                        ):
                            self.finish_button.setEnabled(False)
                    # Update the display
                    self.update_preview_with_points()
        
        # Call the parent class handler
        super().keyPressEvent(event)

    def is_mask_overlapping(self, new_mask):
        """Check if a mask has significant overlap with any existing mask."""
        if not self.expanded_masks:
            return False

        # Create a combined binary mask of all existing masks
        if not hasattr(self, 'combined_binary_mask') or self.combined_binary_mask is None:
            self.combined_binary_mask = np.zeros_like(new_mask, dtype=bool)
            for mask, _, _ in self.expanded_masks:
                self.combined_binary_mask = np.logical_or(self.combined_binary_mask, mask)

        # Check overlap with the combined mask
        overlap = np.logical_and(self.combined_binary_mask, new_mask)
        overlap_area = np.sum(overlap)
        new_mask_area = np.sum(new_mask)
        return overlap_area / new_mask_area > 0.25

class SaveDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Segmentation")
        self.setModal(True)
        self.selected_option = None
        
        # Load the stylesheet for buttons
        stylesheet = load_stylesheet("app_modules/button_styles.qss")
        self.setStyleSheet(stylesheet)
        
        layout = QVBoxLayout()
        
        # Title label
        title_label = QLabel("Choose what to save:")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Buttons with proper styling
        self.sam2_only_button = QPushButton("SAM2 Only")
        self.sam2_only_button.clicked.connect(self.save_sam2_only)
        self.sam2_only_button.setProperty("class", "start-button")  # Blue style
        layout.addWidget(self.sam2_only_button)
        
        self.combined_button = QPushButton("SAM2 + PLAS Combined")
        self.combined_button.clicked.connect(self.save_combined)
        self.combined_button.setProperty("class", "start-button")  # Blue style
        layout.addWidget(self.combined_button)
        
        self.both_button = QPushButton("Both (SAM2 Only + Combined)")
        self.both_button.clicked.connect(self.save_both)
        self.both_button.setProperty("class", "start-button")  # Blue style
        layout.addWidget(self.both_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setProperty("class", "select-folder-button")  # Neutral style
        layout.addWidget(self.cancel_button)
        
        self.setLayout(layout)
        self.resize(300, 200)
    
    def save_sam2_only(self):
        self.selected_option = "sam2_only"
        self.accept()
    
    def save_combined(self):
        self.selected_option = "combined"
        self.accept()
    
    def save_both(self):
        self.selected_option = "both"
        self.accept()


# Run the application
app = QApplication(sys.argv)
viewer = ImageViewer()
viewer.show()
sys.exit(app.exec())
