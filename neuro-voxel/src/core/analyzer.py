import numpy as np
import pyvista as pv
from src.core.structure import PatientVolume

class VolumeAnalyzer:
    def calculate_volume(self, patient: PatientVolume, label_idx: int) -> float:
        """
        It calculates the volume of a specific label (tumour piece) in cm³
        Formula: Voxel number * Voxel volume / 1000
        """
        if patient.mask is None:
            return 0.0
        
        voxel_count = np.sum(patient.mask == label_idx)

        one_voxel_vol = (patient.spacing[0] *
                         patient.spacing[1] *
                         patient.spacing[2])
        
        total_vol = voxel_count * one_voxel_vol

        return total_vol / 1000.0
    
    def get_mesh_from_mask(self, patient: PatientVolume, label_idx: int):
        """
        It produces a 3d mesh -surface- from the mask using the marching cubes or contour algorithms.
        """
        if patient.mask is None:
            return None
        
        binary_mask = np.where(patient.mask == label_idx, 1, 0)

        grid = pv.wrap(binary_mask)

        grid.spacing = patient.spacing

        try:
            mesh = grid.contour(isosurfaces=[0.5])
            mesh = mesh.smooth(n_iter=100)
            
            return mesh
        except Exception as e:
            print(f"Mash could not be created (Label {label_idx} may be missing): {e}")
            return None

    def get_brain_mesh_from_t1(self, patient: PatientVolume) -> pv.PolyData | None:
        """
        Generates a mesh of the brain surface using the T1 modality.
        This creates a 'glass brain' effect for visualization.
        """
        if 't1' not in patient.modalities:
            return None
        
        t1_data = patient.modalities['t1']
        
        # Simple thresholding: Assume brain tissue > 0 (background is usually 0)
        # We assume background is black (0).
        grid = pv.wrap(t1_data)
        grid.spacing = patient.spacing
        
        try:
            # Contour at a low value (e.g., 10) to capture the outer skull/brain surface
            mesh = grid.contour(isosurfaces=[10])
            mesh = mesh.smooth(n_iter=50)
            return mesh
        except Exception as e:
            print(f"Error generating brain surface: {e}")
            return None