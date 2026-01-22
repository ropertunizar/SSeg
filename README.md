<!--
README Template for a Research / Benchmark / Code Release Repository
Replace placeholders like <PROJECT_NAME>, <PAPER_TITLE>, <YEAR>, <HOMEPAGE_URL>, etc.
Remove any sections you do not need.
-->

<h1 align="center">SSeg: Active Sparse Point-Label Augmentation for Sementic Segmentation</h1>
<h3 align="center">CV4EO Workshop - WACV 2026</h3>

<div align="center">
  <a href="https://github.com/cborjamoreno" target="_blank">Cesar Borja</a>,
  <a href="https://github.com/cplou99" target="_blank">Carlos Plou</a>,
  <a >Ruben Martinez-Cantin</a>,
  <a href="https://github.com/anacmurillo" target="_blank">Ana C. Murillo</a>
</div>

<div align="center">
  DIIS - Instituto de Investigación en Ingeniería de Aragón (I3A), Universidad de Zaragoza, Spain
</div>


<div align="center">
  <a href="https://sites.google.com/unizar.es/sseg/home"><strong>🌍 Homepage</strong></a> |
  <a href="https://arxiv.org/abs/2510.10163"><strong>📝 Paper</strong></a> |
</div>

---

## 🔔 News
- 🆕 01/2026: Code released.
- 🥳 01/2026: Paper accepted at Computer Vision for Earth Observation Workshop Series - WACV 2026.

---

## 📖 Description
SSeg is a framework for efficient multi-class image segmentation that expands sparse point annotations into full masks using a hybrid SAM2–superpixel strategy with active point sampling

![SSeg Example](assets/teaser.png)

![SSeg Tool](assets/tool.mp4)

## 🛠️ Requirements

### Environment Setup

```bash
conda create -n spuw python=3.10
conda activate spuw
pip install -r requirements.txt
```

### Download Model Checkpoints

```bash
cd checkpoints/
./download_ckpts.sh
```

### CUDA & Driver Recommendations

- This machine uses CUDA 11.5.
- For best performance, use the same CUDA.
- If CUDA is not available, the code will automatically run on CPU.

## 🚀 Usage

You can run the framework in two modes:

### Option A: Interactive GUI
Best for testing single images and visualizing results in real-time.

- **Interactive GUI:**
  ```bash
  python app.py

### Option B: Benchmark/Batch Mode
Best for running experiments on full datasets and generating metrics.
  ```
- **Benchmark/Batch Mode:**
  ```bash
  python run.py
  ```
  (Requires ground truth images for point-label assignment)

### Configuring Batch Experiments

To configure batch experiments, you do not use command-line arguments. Instead, you modify the configuration list directly inside run.py.

### How to Add Experiments
Open run.py and scroll to the experiments list (around line 220). Add a dictionary for each experiment you want to queue:

```python
experiments = [
    {
        "name": "experiment_name",      # Folder name for results
        "strategy": "dynamicPoints",    # Sampling strategy
        "num_points": 25,               # Point budget
        "images": "path/to/images",     # Input directory
        "ground_truth": "path/to/gt",   # GT directory (required for dynamic strategies)
        
        # Advanced Parameters
        "lambda_balance": 0.5,          # (Dynamic only) Balance exploration/exploitation
        "heatmap_fraction": 0.5,        # (Dynamic only) % of points from uncertainty map
        "maskSLIC": True,               # Enable superpixel refinement
        "visualizations": True          # Save debug images
    }
]
```

### Parameter Reference

These keys can be used inside the experiment dictionary in `run.py`:

**Required**
- `name`: Identifier for the experiment (creates output subfolder).
- `strategy`: Active sampling logic: `random`, `grid`, `list`, `dynamicPoints`, `SAM2_guided`.
- `images`: Path to the directory containing input images.

**Optional (General)**
- `ground_truth`: Path to ground truth masks. Required if using `dynamic` strategies or for evaluation.
- `num_points`: Total point budget per image (default: 30).
- `color_dict`: Path to JSON file mapping colors to class IDs (essential for RGB GTs).
- `default-background-class-id`: Integer ID to use for the background class (default: 0).
- `maskSLIC`: Set to `True` to enable MaskSLIC superpixel refinement.
- `visualizations`: Set to `True` to save overlay images for debugging.

**Optional (Strategy Specific)**
- `lambda_balance`: (Dynamic only) Float [0-1]. Higher values favor exploitation (coverage).
- `heatmap_fraction`: (Dynamic only) Float [0-1]. Ratio of points sampled from uncertainty heatmap.
- `strategy_kwargs`: A dictionary of extra parameters specific to the chosen strategy (e.g., `{"propagation_overlap_policy": "last"}`).

**Debugging**
- `debug_expanded_masks`: Set to `True` to save individual SAM2 expansion steps.


## 📜 License
 Discuss with your supervisor the license that you want to set and include the licenses of any previous repo in which your code was inspired.
 
## 📝 Citation
```bibtex
@article{borja2025sparseuwseg,
  title={SparseUWSeg: Active Sparse Point-Label Augmentation for Underwater Semantic Segmentation},
  author={Borja, C{\'e}sar and Plou, Carlos and Martinez-Cant{\'\i}n, Rub{\'e}n and Murillo, Ana C},
  journal={arXiv preprint arXiv:2510.10163},
  year={2025}
}
```

## 🙏 Acknowledgements
This work was supported by a DGA scholarship and by DGA project T45\_23R, and grants AIA2025-163563-C31, PID2024-159284NB-I00, PID2021-125514NB-I00 and PID2024-158322OB-I00 funded by MCIN/AEI/10.13039/501100011033 and ERDF.

