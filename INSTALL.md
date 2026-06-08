# Installing SSeg

SSeg requires a working PyTorch build that matches your CUDA driver. Pinning
a specific `torch==X.Y+cuZZZ` in an env file breaks for anyone whose machine
has a different CUDA — that's why this repo installs torch **separately**
from everything else.

There are two paths:

* **A. Automated** — run `./setup.sh`. Detects your CUDA, picks the right
  PyTorch wheel, installs everything. Recommended unless you have unusual
  setup.
* **B. Manual** — same steps, run by hand. Use this if `setup.sh` fails or
  you want to understand what's happening.

Either way you end up with one conda env (`spuw` by default), the right
PyTorch wheel, the rest of the deps from `requirements.txt`, and the SAM2
checkpoint.

---

## Prerequisites

| Requirement | Tested with                              |
|-------------|------------------------------------------|
| OS          | Linux (Ubuntu 22.04)                     |
| GPU driver  | NVIDIA driver ≥ 525                      |
| CUDA driver | ≥ 11.8 (check with `nvidia-smi`)         |
| Python      | 3.10                                     |
| conda       | miniconda or anaconda                    |
| Disk        | ~5 GB (env + 900 MB SAM2 checkpoint)     |

CPU-only is supported but slow — every `run.py` invocation falls back
automatically if CUDA is unavailable.

Check your CUDA driver version:

```bash
nvidia-smi | grep "CUDA Version"
#   CUDA Version: 12.6   <- pick the torch wheel that matches THIS
```

The number above is the **driver's** CUDA version, not the toolkit's
(`nvcc --version` may differ — that's fine). PyTorch wheels are forward-
compatible: a `cu121` wheel runs on a 12.6 driver.

---

## A. Automated install

```bash
git clone <repo-url> SSeg
cd SSeg
bash setup.sh              # creates conda env "spuw", installs everything
conda activate spuw
bash checkpoints/download_ckpts.sh    # downloads SAM2 checkpoint (~900 MB)
python run.py              # runs the demo experiment
```

`setup.sh` accepts:

```bash
bash setup.sh --env-name myenv     # override env name (default: spuw)
bash setup.sh --cuda 11.8          # force CUDA variant (default: auto)
bash setup.sh --cpu                # CPU-only install
```

---

## B. Manual install

### 1. Create a clean conda env with Python 3.10

```bash
conda create -n spuw python=3.10 -y
conda activate spuw
export PYTHONNOUSERSITE=1   # don't pull packages from ~/.local — see troubleshooting
```

### 2. Install PyTorch matching your CUDA

Pick the row that matches your `nvidia-smi` CUDA version (use the **closest
lower or equal** CUDA wheel — PyTorch is forward-compatible):

| Your CUDA  | Install command                                                                           |
|------------|-------------------------------------------------------------------------------------------|
| 12.4+      | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`        |
| 12.1–12.3  | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`        |
| 11.8–12.0  | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`        |
| CPU only   | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`          |

Verify:

```bash
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
# 2.X.Y+cuYYY  cuda: True
```

If `cuda: False`, the wheel doesn't match your driver — pick a different row.

### 3. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

This installs: `opencv-python`, `scikit-learn`, `numpy`, `matplotlib`, `tqdm`,
`pandas`, `scikit-image`, `iopath`, `hydra-core`, `torchmetrics`, `Pillow`,
`scipy`, and `gdown`.

(`PyQt6` is needed **only for `app.py`** — the interactive GUI tool. If
you're running batch experiments via `run.py`, skip it. If you do want the
GUI: `pip install PyQt6`.)

### 4. Download model checkpoints

```bash
cd checkpoints/
bash download_ckpts.sh
cd ..
```

This pulls `sam2.1_hiera_large.pt` (~900 MB) and a standardization weights
file from Google Drive. Requires `wget` or `curl`.

### 5. Smoke test

```bash
python run.py
```

The default `run.py` runs a small demo experiment from `demo/`. It should:

1. Print `Found N images` (6 for the demo).
2. Initialize SAM2.
3. Show a tqdm progress bar.
4. Write results to `experiments/demo_YYYYMMDD_HHMMSS/`.
5. End with a results table including `mPA` and `mIoU` columns.

If you see that, the install is good.

---

## Troubleshooting

### `torch` installed from `~/.local/` instead of the conda env
Pip honors the user-site packages (`~/.local/lib/python3.10/site-packages`)
by default. If torch is already installed there from a previous project,
`pip install torch ...` will say "Requirement already satisfied" and skip
the install, leaving your env with a CUDA variant that doesn't match what
you asked for. `setup.sh` works around this by exporting
`PYTHONNOUSERSITE=1` and passing `--no-user` to pip. If installing
manually, do the same:

```bash
export PYTHONNOUSERSITE=1
pip install --no-user torch torchvision --index-url <pytorch-index>
pip install --no-user -r requirements.txt
```

### `ImportError: numpy.core.multiarray failed to import`
You have `numpy >= 2.0` but a binary package (often `opencv-python` or
`pandas`) was built against `numpy < 2`. Either pin
`pip install "numpy<2"` or upgrade the offending package.

### `torch.cuda.is_available()` is False after install
Wrong PyTorch wheel for your driver. Re-run step 2 with a different CUDA
variant (try one notch lower — e.g. cu121 if cu124 didn't work).

### `RuntimeError: No CUDA GPUs are available` during run
Either you're on a CPU-only box (pass `--device cpu` to `run.py`), or the
torch install isn't actually CUDA-enabled (see above).

### `download_ckpts.sh` fails on the Google Drive file
`pip install gdown` then re-run. Google Drive's anonymous download flow
sometimes requires the `gdown` helper.

### Demo runs but `mIoU` is 0 or nan
Usually a color-dict mismatch with the GT masks. Make sure
`demo/color_dict.json` corresponds to the colors in `demo/labels/*.png`.

### `ModuleNotFoundError: No module named 'sam2'`
You're running from the wrong directory. `cd` into the SSeg repo root
before invoking `python run.py` (or `app.py`); the local `sam2/` package
is on `sys.path` only when CWD is the repo root.

### Hydra / config errors on SAM2 init
A common cause is an incomplete checkpoint download. Verify the file size:

```bash
ls -lh checkpoints/sam2.1_hiera_large.pt   # should be ~900 MB
```

If it's much smaller, re-run `download_ckpts.sh`.

---

## Why we don't ship a conda env file

The original `labelex.yml` in this codebase pinned `pytorch=1.12=py310_cu113`
— that env became unusable as soon as the code switched to SAM2 (which needs
torch ≥ 1.13) or when reproduced on a machine with a different CUDA.
Decoupling torch from the rest of the requirements keeps the project portable
across machines. If you really want a lockfile for **your specific machine**,
generate it after install:

```bash
conda env export --no-builds > env-$(hostname).yml
```
