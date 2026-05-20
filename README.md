# HD Map-Aided Localization with Learned EKF Covariance

## Purpose

This project studies ground-vehicle localization by combining:

- an HD map prior,
- classical sensor fusion with an Extended Kalman Filter (EKF), and
- a learned measurement covariance matrix $R$.

The core idea is to improve localization robustness and trajectory accuracy by injecting learned uncertainty estimates into EKF updates while constraining the solution with map context.

## Evaluation Scope

Experiments are evaluated on the KITTI odometry benchmark using:

- sequence `09`
- sequence `10`

Saved outputs for these sequences are stored under `results/09/` and `results/10/`.

## Repository Structure (Tracked/Included)

The structure below focuses on source, configs, and project assets that are intended to be part of the repository.

```text
.
├── README.md
├── requirements.txt
├── configs/
├── notebooks/
│   ├── dead_reckoning.ipynb
│   ├── dead_reckoning+predicted_R.ipynb
│   ├── learned_map_aided_localization.ipynb
│   ├── result_and_analysis.ipynb
│   └── visualization.ipynb
├── parameters/
│   └── best_imu_cov_net.pt
├── results/
│   ├── 09/
│   └── 10/
├── src/
│   ├── filter/
│   │   └── ekf.py
│   ├── model/
│   │   └── imu_noise_net.py
│   └── utils/
│       ├── dataset.py
│       ├── filter_utils.py
│       ├── map_utils.py
│       ├── misc.py
│       ├── ransac_icp.py
│       └── transformation_utils.py
└── data/
	└── KITTI/
```

## Excluded by `.gitignore`

The following paths/patterns are currently ignored and should not be treated as versioned project documentation targets:

- virtual environments and caches: `venv/`, `__pycache__/`, `*.egg-info/`
- build artifacts: `dist/`, `build/`
- editor/system files: `.vscode/`, `.idea/`, `.DS_Store`, swap files
- logs and temp files: `*.log`, `*.tmp`, `*.temp`
- temporary and experiment folders: `temp/`, `_experiments/`
- large raw dataset folders:
  - `data/KITTI/sync/2011_09_30`
  - `data/KITTI/unsync/2011_09_30`
- map raw files: `*.graphml`

Note: If any ignored files are already tracked historically, Git may still show them in the working tree. The list above reflects intended ignore policy.

## Setup Requirements

### 1. System

- macOS/Linux (tested on Unix-like workflow)
- Python `3.9+` recommended

### 2. Python Environment

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Dependencies

Install Python packages:

```bash
pip install -r requirements.txt
```

### 4. Data

Prepare KITTI data under `data/KITTI/` with the expected directory layout used by notebooks/utilities.

## Starter / Quick Start

1. Activate your environment:

	```bash
	source venv/bin/activate
	```

2. Launch Jupyter:

	```bash
	jupyter notebook
	```

3. Run notebooks in this order (recommended):

- `notebooks/dead_reckoning.ipynb`
- `notebooks/dead_reckoning+predicted_R.ipynb`
- `notebooks/learned_map_aided_localization.ipynb`
- `notebooks/result_and_analysis.ipynb`

4. Review outputs in `results/09/` and `results/10/`.

## Implementation Overview

- `src/filter/ekf.py`: EKF prediction/update logic for pose estimation.
- `src/model/imu_noise_net.py`: learned model used to estimate covariance-related quantities.
- `src/utils/`: data loading, transformations, map utilities, and supporting algorithms.

## Reproducibility Notes

- Keep `requirements.txt` pinned for consistent runs.
- Ensure KITTI calibration and timestamp files remain consistent across experiments.
- Store sequence-specific outputs separately (`09`, `10`) to simplify comparison.
