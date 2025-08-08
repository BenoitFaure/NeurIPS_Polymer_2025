def main():
    print("Hello from neurips-polymer-2025!")

    # Test PyTorch installation and CUDA availability
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch is not installed.")

    # Test PyTorch Geometric installation
    try:
        import torch_geometric
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        print("PyTorch Geometric is not installed.")


if __name__ == "__main__":
    main()
