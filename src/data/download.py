"""
download.py

Script for downloading and versioning raw datasets from Kaggle for the NeurIPS_Polymer_2025 project.
This script is intended to be run via the central CLI:
    uv run run.py script=download

Main steps:
1. Checks if the raw data directory already contains data and a version file.
2. If not present, downloads the dataset from Kaggle using the competition name specified in the config.
3. Unzips all downloaded files and removes the zip archives.
4. Computes a hash of the folder contents for versioning and logs it to MLflow.

Configuration is passed via Hydra/omegaconf DictConfig, with relevant paths and dataset name under cfg.data.
"""

import subprocess
import zipfile
import hashlib
import mlflow
from pathlib import Path
from omegaconf import DictConfig

def compute_folder_hash(folder: Path) -> str:
    """
    Compute a SHA256 hash of all files in a folder for dataset versioning.
    Used to uniquely identify the dataset state.
    """
    sha256 = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
    return sha256.hexdigest()

def main(cfg: DictConfig):
    """
    Main entry point for dataset download and versioning.
    Steps:
    - Ensure raw and processed directories exist.
    - If raw data and version file exist, log and exit.
    - Otherwise, download dataset from Kaggle, unzip, compute version, and log to MLflow.
    """
    # Extract relevant config section
    cfg = cfg.data
    raw_dir = Path(cfg.raw_dir)
    processed_dir = Path(cfg.processed_dir)
    version_file = Path(cfg.version_file)

    # Ensure folders exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Set MLflow experiment for tracking
    mlflow.set_experiment('data_ingestion')

    # Step 1 — Check if dataset already exists & version is recorded
    if raw_dir.exists() and any(raw_dir.iterdir()):
        print(f"✅ Raw data already present in {raw_dir}")
        if version_file.exists():
            # Log existing version
            version = version_file.read_text().strip()
            print(f"📦 Current dataset version: {version}")
            mlflow.log_param("data_version", version)
        else:
            # Compute and record version if missing
            print("⚠️ Version file missing, computing new version...")
            version = compute_folder_hash(raw_dir)
            version_file.write_text(version)
            mlflow.log_param("data_version", version)
        return

    # Step 2 — Download dataset from Kaggle
    print(f"⬇️ Downloading dataset {cfg.dataset} from Kaggle...")
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", cfg.dataset, "-p", str(raw_dir)],
        check=True
    )

    # Step 3 — Unzip all files in raw_dir and remove zip archives
    for file in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(raw_dir)
        file.unlink()  # remove zip

    # Step 4 — Compute and save version hash
    version = compute_folder_hash(raw_dir)
    version_file.write_text(version)
    mlflow.log_param("data_version", version)

    print(f"✅ Dataset ready at {raw_dir}")
    print(f"📦 Dataset version: {version}")

if __name__ == "__main__":
    main()
