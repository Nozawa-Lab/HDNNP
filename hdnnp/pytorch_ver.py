try:
    import torch_scatter
    print(f"torch_scatter version: {torch_scatter.__version__}")
    print("torch_scatter is available.")
except ImportError:
    print("torch_scatter is NOT installed.")
except Exception as e:
    print(f"An error occurred: {e}")