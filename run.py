"""
run.py

Central CLI entry point for the NeurIPS_Polymer_2025 project.
Allows running various project scripts (data processing, training, etc.) via a
`script` config argument.

Usage:
    python run.py script=download [other hydra overrides]

Example:
    python run.py script=download --help
"""

import importlib
import sys

import hydra
from omegaconf import DictConfig

# List your scripts here as (module_path, description)
scripts = [
    ("src.data.download", "Download and preprocess dataset"),
    ("src.data.preprocess", "Process raw dataset"),
    # add more scripts here
]

def run_script(module_path: str, cfg: DictConfig):
    """
    Dynamically import the module by dotted path and run its `main()` function.
    """
    module = importlib.import_module(module_path)
    if not hasattr(module, "main"):
        print(f"Module {module_path} does not have a main() function.")
        sys.exit(1)
    module.main(cfg)

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Check script argument presence
    if not hasattr(cfg, "script") or not cfg.script:
        print("Error: You must specify which script to run using `script=SCRIPT_NAME`.")
        print("Available scripts:")
        for mod_path, desc in scripts:
            print(f"  - {mod_path.split('.')[-1]}: {desc}")
        sys.exit(1)

    script_name = cfg.script
    # Find matching module_path for the script name
    for module_path, _ in scripts:
        if module_path.split('.')[-1] == script_name:
            run_script(module_path, cfg)
            break
    else:
        print(f"Unknown script '{script_name}'. Available scripts:")
        for mod_path, desc in scripts:
            print(f"  - {mod_path.split('.')[-1]}: {desc}")
        sys.exit(1)

if __name__ == "__main__":
    main()