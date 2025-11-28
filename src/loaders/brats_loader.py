import os
import glob
import nibabel as nib
import numpy as np
from src.core.structure import PatientVolume

class BraTSLoader:
    def __init__(self, root_dir: str):
        
        self.root_dir = root_dir
    
    def load_patient(self, patient_id: str) -> PatientVolume:

        patient_path = os.path.join(self.root_dir, patient_id)

        if not os.path.exists(patient_path):
            raise FileNotFoundError(f"No patient folder: {patient_path}")
        
        print(f"Loading: {patient_id}")

        modalities = {}
        mask = None
        affine = None
        spacing = None

        suffixes = {
            't1': '*_t1.nii*',
            't1ce': '*_t1ce.nii*',
            't2': '*_t2.nii*',
            'flair': '*_flair.nii*'
        }

        for mod_name, suffix in suffixes.items():
            search_pattern = os.path.join(patient_path, suffix)
            found_files = glob.glob(search_pattern)

            if found_files:
                
                try:
                    file_path = found_files[0]
                    img = nib.load(file_path)
                    data = img.get_fdata().astype(np.float32)

                    modalities[mod_name] = data
            
                    if affine is None:
                        affine = img.affine
                        spacing = img.header.get_zooms()
                
                except Exception as e:
                    print(f"Error loading {mod_name}: {e}")
            else:
                print("f Attention: {mod_name} modality not founds.")

        mask_pattern = os.path.join(patient_path, "*_seg.nii")
        mask_files = glob.glob(mask_pattern)
        if mask_files:
            try:
                mask_img = nib.load(mask_files[0])
                mask = mask_img.get_fdata().astype(np.uint8)
                print(f"Mask loaded.")
            except Exception as e:
                print(f"Error loading mask: {e}")
        else:
            print(f"Mask not found.")
            
        return PatientVolume(
            id=patient_id,
            modalities=modalities,
            mask=mask,
            affine=affine,
            spacing=spacing
        )
