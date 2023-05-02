import os
import numpy as np

# Define the directory path to search for files
directory = './accuracy/cross/'

# Recursively search through directory and subdirectories
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.npy'):  # check if file is a numpy file
            # if "lenet" in file:
            filepath = os.path.join(root, file)
            try:
                arr = np.load(filepath)  # load the numpy array
                print(f'File: {filepath}\nArray values: {arr}\n')
            except Exception as e:
                print(f'Error loading numpy array from {filepath}: {e}')
