import os

# Correct path (using forward slashes or raw string)
file_path = "p/ساخت مدل/train_model.py"  # or r"p\ساخت مدل\train_model.py"

if os.path.exists(file_path):
    print("File found!")
    # Proceed with your code
else:
    print("File not found. Check the path!")