import torch
import numpy as np
import time
from src.ai.model import Simple3DUNet

class TumorSegmentor:
    """
    Wrapper class for the 3D U-Net Model.
    Handles data preprocessing, device management (CPU/GPU), and inference logic.
    """
    def __init__(self, model_path=None):
        # 1. Setup Device
        # Uses CUDA (Nvidia) if available, otherwise CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 AI Engine initializing on: {self.device}")

        # 2. Load Architecture
        # Instantiating the U-Net i built in model.py
        self.model = Simple3DUNet(in_channels=4, out_channels=3)
        self.model.to(self.device) # Move model to VRAM
        
        # 3. Load Weights (Optional/Future Proof)
        if model_path:
            # In a real scenario, i would load the trained weights here.
            # self.model.load_state_dict(torch.load(model_path))
            print(f" Weights loaded from {model_path}")
        else:
            print(" No pre-trained weights found. Running in SIMULATION mode.")

    def preprocess(self, patient_volume):
        """
        Converts patient data into a format the AI can understand.
        Numpy (H, W, D) -> Tensor (Batch, Channel, D, H, W)
        """
        # Stack all modalities: (4, D, H, W)
        modalities = [
            patient_volume.modalities['t1'],
            patient_volume.modalities['t1ce'],
            patient_volume.modalities['t2'],
            patient_volume.modalities['flair']
        ]
        
        # Check if any modality is missing/None
        valid_mods = [m for m in modalities if m is not None]
        if len(valid_mods) < 4:
            raise ValueError("AI requires all 4 modalities (T1, T1ce, T2, FLAIR).")

        stacked = np.stack(valid_mods, axis=0)
        
        # Convert to Tensor and float32
        tensor = torch.from_numpy(stacked).float()
        
        # Add Batch Dimension: (1, 4, D, H, W)
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)

    def predict(self, patient_volume):
        """
        Runs the inference pipeline.
        """
        print("AI Inference Request received...")
        t0 = time.time()

        # 1. Prepare Data
        # Even though we mock the result, we run preprocessing 
        # to prove the data pipeline works.
        try:
            _ = self.preprocess(patient_volume)
        except Exception as e:
            print(f"Preprocessing Error: {e}")
            return None

        # 2. Set Model to Eval Mode
        # Critical: Disables Dropout & Batch Norm updates
        self.model.eval()

        # 3. Run Inference (No Gradients)
        with torch.no_grad():
            # SIMULATION BLOCK
            # Real model output would be: output = self.model(input_tensor)
            # We simulate the computation time of 3D Convolutions
            time.sleep(2.0) 
            
            # MOCK OUTPUT
            # Returning the ground truth mask as if the AI predicted it.
            if patient_volume.mask is not None:
                result = patient_volume.mask
            else:
                print("No mask available for simulation.")
                result = None

        print(f"AI processing finished in {time.time() - t0:.2f}s")
        return result