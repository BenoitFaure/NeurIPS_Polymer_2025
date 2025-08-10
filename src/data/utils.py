import hashlib
from pathlib import Path

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