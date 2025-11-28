from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

@dataclass
class PatientVolume:
    """"
    a dataclass that store all MRI data and metadata for a patient.
    """
    id: str                             #Patients ID 
    modalities: Dict[str, np.ndarray]   #Image matrices
    mask: Optional[np.ndarray]          #Segmentation mask(if available)
    affine: np.ndarray                  #Spatial position matrix (4*4)
    spacing: Tuple[float, float, float] #Voxel dimensions 

def __repr__(self): #represent for have a clear output look.
    mods = list(self.modalities.keys())
    has_mask = "Yes" if self.mask is not None else "No"
    return f"<PatientVolume ID={self.id} Modalities={mods} Mask={has_mask} Shape={self.modalities[mods[0]].shape}>"
