import tensorflow as tf
import h5py

MODEL_PATH = "D:/cloud/handwrittendigitrecognition.keras"

# Try to open the model file and inspect its structure
with h5py.File(MODEL_PATH, 'r') as f:
    print("Model structure:")
    def print_structure(name, obj):
        print(f"Name: {name}")
        if isinstance(obj, h5py.Dataset):
            print(f"  Type: Dataset")
            print(f"  Shape: {obj.shape}")
            print(f"  Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"  Type: Group")
    
    f.visititems(print_structure)
