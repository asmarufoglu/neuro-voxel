import os 
import nibabel as nib
import numpy as np

DATA_PATH = r"C:\Users\semih\Desktop\d1\spatial-comp-lab\neuro-voxel\data\sample_patient"
FILE_NAME = "BraTS20_Training_001_flair.nii"
FILE_PATH = os.path.join(DATA_PATH, FILE_NAME)


def check_nifti_file():
    print(f"----NIfTI File Analyse: {FILE_NAME} ----")

    if not os.path.exists(FILE_PATH):
        print(f"ERROR: No File! Path: {FILE_PATH}")
        print("Ensure that the data folder contains a patient record. ")
        return

    try:
        img = nib.load(FILE_PATH)
        print("File successfully uploaded!")
    except Exception as e:
        print(f"Error when loading data!: {e}")
        return
        
    data = img.get_fdata()

    print(f"\n Data Type (dtype): {data.dtype}")
    print(f"\n Shapes: {data.shape}")
    
    print(f"\n Affine Matrix:\n{img.affine}")

    header = img.header
    zooms = header.get_zooms()
    print(f"\n Voxel Spacing: {zooms}")
    print(f"   -> X axis: {zooms[0]:.2f} mm")
    print(f"   -> Y axis: {zooms[1]:.2f} mm")
    print(f"   -> Y axis: {zooms[2]:.2f} mm")

    print("\n Testing successfull !! ")

if __name__ == "__main__":
    check_nifti_file()