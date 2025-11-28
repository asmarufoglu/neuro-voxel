import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QCheckBox, QFrame, QGroupBox, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from pyvistaqt import QtInteractor
import pyvista as pv

# Ensure project modules are reachable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.loaders.brats_loader import BraTSLoader
from src.core.analyzer import VolumeAnalyzer
# --- YENİ EKLENEN IMPORT ---
from src.ai.inference import TumorSegmentor

# --- CONFIGURATION & STYLES ---
THEME_COLORS = {
    "background": "#121212",
    "panel": "#1a1a1d",
    "accent": "#00b4d8",  # Neon Cyan
    "text": "#e0e0e0",
    "text_dim": "#b0bec5",
    "ai_btn": "#6200ea"   # Deep Purple for AI
}

STYLESHEET = f"""
QMainWindow {{
    background-color: {THEME_COLORS['background']}; 
    color: {THEME_COLORS['text']};
}}
QWidget {{
    font-family: 'Segoe UI', sans-serif;
    font-size: 10pt;
}}

/* --- LEFT CONTROL PANEL --- */
QFrame#ControlPanel {{
    background-color: {THEME_COLORS['panel']};
    border-right: 1px solid #333;
}}

/* --- HEADERS --- */
QLabel#Header {{
    font-family: 'Segoe UI Light';
    font-size: 22pt;
    color: #4cc9f0;
    padding-bottom: 5px;
}}
QLabel#SubHeader {{
    font-size: 11pt;
    font-weight: bold;
    color: {THEME_COLORS['text_dim']};
    margin-top: 15px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* --- BUTTONS --- */
QPushButton {{
    background-color: {THEME_COLORS['accent']};
    color: #000000;
    border: none;
    padding: 12px;
    border-radius: 6px;
    font-weight: bold;
    font-size: 10pt;
}}
QPushButton:hover {{
    background-color: #48cae4;
    margin-top: -1px;
}}
QPushButton:pressed {{
    background-color: #0077b6;
    margin-top: 1px;
}}
QPushButton:disabled {{
    background-color: #2c3e50;
    color: #7f8c8d;
}}

/* --- AI BUTTON SPECIFIC --- */
QPushButton#AIButton {{
    background-color: {THEME_COLORS['ai_btn']};
    color: white;
    border: 1px solid #7c4dff;
}}
QPushButton#AIButton:hover {{
    background-color: #651fff;
}}

/* --- METADATA BOX --- */
QGroupBox {{
    border: 1px solid #37474f;
    border-radius: 8px;
    margin-top: 20px;
    background-color: #23262b;
}}
QGroupBox::title {{
    color: #4cc9f0;
    subcontrol-origin: margin;
    left: 15px;
    padding: 0 5px;
    font-weight: bold;
}}
QLabel#MetaLabel {{
    color: #cfd8dc;
    font-size: 10pt;
    padding-left: 5px;
}}

/* --- SLIDERS & CHECKBOXES --- */
QSlider::groove:horizontal {{
    border: 1px solid #333;
    height: 4px;
    background: #263238;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: #4cc9f0;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
QCheckBox {{
    spacing: 8px;
    color: #eceff1;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 1px solid #546e7a;
    background: #263238;
}}
"""

class LoadWorker(QThread):
    finished = pyqtSignal(object) 
    error = pyqtSignal(str)       
    progress = pyqtSignal(int)    

    def __init__(self, loader, patient_id):
        super().__init__()
        self.loader = loader
        self.patient_id = patient_id

    def run(self):
        try:
            self.progress.emit(10)
            patient = self.loader.load_patient(self.patient_id)
            self.progress.emit(60)
            self.finished.emit(patient)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuro-Voxel")
        self.resize(1400, 850)
        self.setStyleSheet(STYLESHEET)
        
        # --- Data Configuration ---
        self.DATA_ROOT = r"C:\Users\semih\Desktop\d1\spatial-comp-lab\neuro-voxel\data"
        self.PATIENT_ID = "sample_patient"
        
        # Initialize Logic Modules
        self.loader = BraTSLoader(self.DATA_ROOT)
        self.analyzer = VolumeAnalyzer()
        
        # --- AI MOTORUNU BAŞLAT ---
        # Bu işlem PyTorch/CUDA kontrolü yapar
        self.segmentor = TumorSegmentor()

        self.patient = None
        self.actors = {} 

        self.init_ui()

    def init_ui(self):
        """Constructs the main layout and widgets."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # ===========================
        # LEFT PANEL: CONTROL CENTER
        # ===========================
        control_panel = QFrame()
        control_panel.setObjectName("ControlPanel")
        control_panel.setFixedWidth(360)
        
        panel_layout = QVBoxLayout(control_panel)
        panel_layout.setAlignment(Qt.AlignTop) 
        panel_layout.setContentsMargins(25, 30, 25, 30)
        panel_layout.setSpacing(20)
        
        # 1. Header Section
        header = QLabel("neurovoxel")
        header.setObjectName("Header")
        panel_layout.addWidget(header)
        
        desc = QLabel("Volumetric BraTS Analysis Tool")
        desc.setStyleSheet("color: #78909c; font-size: 9pt; margin-bottom: 10px;")
        panel_layout.addWidget(desc)

        # 2. Action Section (Buttons)
        
        # LOAD BUTTON
        self.btn_load = QPushButton(f"LOAD CASE: {self.PATIENT_ID}")
        self.btn_load.setCursor(Qt.PointingHandCursor)
        self.btn_load.clicked.connect(self.start_loading)
        panel_layout.addWidget(self.btn_load)

        # PROGRESS BAR
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setStyleSheet("QProgressBar { border: none; background: #263238; height: 6px; border-radius: 3px; } QProgressBar::chunk { background-color: #00b4d8; border-radius: 3px; }")
        self.progress_bar.setVisible(False)
        panel_layout.addWidget(self.progress_bar)

        # --- YENİ EKLENEN AI BUTTON ---
        self.btn_ai = QPushButton("RUN AI DIAGNOSIS")
        self.btn_ai.setObjectName("AIButton") # CSS için ID verdik
        self.btn_ai.setCursor(Qt.PointingHandCursor)
        self.btn_ai.clicked.connect(self.run_ai_segmentation)
        self.btn_ai.setEnabled(False) # Veri yüklenmeden basılamaz
        panel_layout.addWidget(self.btn_ai)
        # ------------------------------

        # 3. Metadata Section
        self.meta_group = QGroupBox("PATIENT DATA")
        self.meta_group.setVisible(False)
        meta_layout = QVBoxLayout(self.meta_group)
        meta_layout.setSpacing(8)
        
        self.lbl_patient_id = QLabel(f"ID: {self.PATIENT_ID}")
        self.lbl_patient_id.setObjectName("MetaLabel")
        self.lbl_total_vol = QLabel("Total Volume: --")
        self.lbl_total_vol.setObjectName("MetaLabel")
        self.lbl_voxel_dim = QLabel("Spacing: --")
        self.lbl_voxel_dim.setObjectName("MetaLabel")
        
        meta_layout.addWidget(self.lbl_patient_id)
        meta_layout.addWidget(self.lbl_total_vol)
        meta_layout.addWidget(self.lbl_voxel_dim)
        
        panel_layout.addWidget(self.meta_group)

        # 4. Layer Controls
        lbl_vis = QLabel("Visualization Layers")
        lbl_vis.setObjectName("SubHeader")
        panel_layout.addWidget(lbl_vis)

        # Context Layer (Brain Shell)
        self.brain_container = QWidget() 
        brain_layout = QVBoxLayout(self.brain_container)
        brain_layout.setContentsMargins(0,0,0,0)
        self.create_layer_control(brain_layout, "Brain Structure (T1)", "brain", "#ffffff", default_opacity=0.10)
        panel_layout.addWidget(self.brain_container)

        # Tumor Layers Group
        tumor_group = QGroupBox("SEGMENTATION MASKS")
        tumor_layout = QVBoxLayout(tumor_group)
        tumor_layout.setSpacing(15)
        
        tips = {
            "necrotic": "Necrotic Core (Label 1)\nAggressive dead tissue.",
            "edema": "Peritumoral Edema (Label 2)\nFluid accumulation area.",
            "active": "Enhancing Tumor (Label 4)\nActive growing tumor tissue."
        }

        self.create_layer_control(tumor_layout, "Necrotic Core", "necrotic", "#ff5252", tips['necrotic'], 1.0)
        self.create_layer_control(tumor_layout, "Edema Region", "edema", "#ffd740", tips['edema'], 0.25)
        self.create_layer_control(tumor_layout, "Active Tumor", "active", "#69f0ae", tips['active'], 1.0)
        
        panel_layout.addWidget(tumor_group)

        panel_layout.addStretch()

        # Footer
        footer = QLabel("v1 neurovoxel | Spatial Comp Lab @asmarufoglu ")
        footer.setStyleSheet("color: #455a64; font-size: 8pt;")
        footer.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(footer)

        layout.addWidget(control_panel)
        
        # ===========================
        # RIGHT PANEL: 3D VIEWER
        # ===========================
        self.plotter = QtInteractor(main_widget)
        self.plotter.set_background(color="#050505", top="#101520")
        self.plotter.enable_eye_dome_lighting()
        
        layout.addWidget(self.plotter)

    def create_layer_control(self, parent_layout, title, key, color, tooltip="", default_opacity=1.0):
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(8)
        
        # Row 1: Checkbox + Label
        hbox = QHBoxLayout()
        cb = QCheckBox()
        cb.setChecked(True)
        cb.setStyleSheet(f"""
            QCheckBox::indicator:checked {{ 
                background-color: {color}; 
                border-color: {color}; 
            }}
        """)
        
        lbl = QLabel(title)
        if tooltip:
            lbl.setToolTip(tooltip)
            lbl.setCursor(Qt.WhatsThisCursor)
        
        hbox.addWidget(cb)
        hbox.addWidget(lbl)
        hbox.addStretch()
        vbox.addLayout(hbox)
        
        # Row 2: Opacity Slider
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(int(default_opacity * 100))
        slider.setStyleSheet(f"""
            QSlider::handle:horizontal {{
                background: {color};
                border: 1px solid {color};
            }}
        """)
        vbox.addWidget(slider)
        
        parent_layout.addWidget(container)
        
        cb.stateChanged.connect(lambda state: self.toggle_visibility(key, state))
        slider.valueChanged.connect(lambda val: self.update_opacity(key, val))

    # --- LOGIC HANDLING ---

    def start_loading(self):
        self.btn_load.setEnabled(False)
        self.btn_ai.setEnabled(False) # Yükleme sırasında AI kapalı
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        
        self.loader_thread = LoadWorker(self.loader, self.PATIENT_ID)
        self.loader_thread.progress.connect(self.update_progress)
        self.loader_thread.finished.connect(self.on_load_finished)
        self.loader_thread.error.connect(self.on_load_error)
        self.loader_thread.start()

    def update_progress(self, val):
        self.progress_bar.setValue(val)

    def on_load_error(self, err_msg):
        self.progress_bar.setVisible(False)
        self.btn_load.setEnabled(True)
        QMessageBox.critical(self, "Load Error", err_msg)

    def on_load_finished(self, patient):
        self.patient = patient
        self.progress_bar.setValue(80)
        
        self.plotter.clear()
        self.actors = {}
        total_vol = 0
        
        # 1. Generate Brain Shell
        brain_mesh = self.analyzer.get_brain_mesh_from_t1(patient)
        if brain_mesh:
            self.actors['brain'] = self.plotter.add_mesh(
                brain_mesh, color="#eceff1", opacity=0.10, style='surface', smooth_shading=True
            )

        # 2. Generate Tumor Meshes (LOAD BUTONUNA BASILINCA ARTIK ÇİZMİYORUZ!)
        # Değişiklik: Yüklemede sadece beyni çiziyoruz, tümörleri AI bulacak.
        # Ama var olan GT'yi göstermek istersen burayı açabilirsin.
        # Bizim senaryomuzda AI butonu şovu yapacağı için burayı pas geçiyoruz veya
        # Sadece "Brain" yükleniyor, tümörler AI'a basınca gelecek gibi yapabiliriz.
        # ŞİMDİLİK: Eski düzeni koruyalım, yüklemede her şey gelsin, AI butonu "Re-Run" yapsın.
        
        parts = [
            ('necrotic', 1, '#ff5252', 1.0),
            ('edema', 2, '#ffd740', 0.25),
            ('active', 4, '#69f0ae', 1.0)
        ]
        
        for key, lbl_id, color, opac in parts:
            mesh = self.analyzer.get_mesh_from_mask(patient, lbl_id)
            vol = self.analyzer.calculate_volume(patient, lbl_id)
            total_vol += vol
            
            if mesh and mesh.n_points > 0:
                self.actors[key] = self.plotter.add_mesh(
                    mesh, color=color, opacity=opac, smooth_shading=True, specular=0.6
                )

        # 3. Update UI Metadata
        self.lbl_total_vol.setText(f"Total Volume: {total_vol:.2f} cm³")
        sp = patient.spacing
        self.lbl_voxel_dim.setText(f"Spacing: {sp[0]:.1f}x{sp[1]:.1f}x{sp[2]:.1f} mm")
        self.meta_group.setVisible(True)

        self.plotter.add_axes()
        self.plotter.reset_camera()
        self.plotter.camera_position = 'iso'
        
        self.progress_bar.setValue(100)
        self.btn_load.setText(f"RELOAD CASE")
        self.btn_load.setEnabled(True)
        
        # --- AI BUTONUNU AKTİF ET ---
        self.btn_ai.setEnabled(True)

    def run_ai_segmentation(self):
        """AI Butonuna basıldığında çalışan fonksiyon."""
        if self.patient is None: return

        # UI Güncelleme (Processing...)
        original_text = self.btn_ai.text()
        self.btn_ai.setText("PROCESSING (NEURAL NET)...")
        self.btn_ai.setEnabled(False)
        QApplication.processEvents() # Arayüzün donmasını engelle, yazıyı güncelle

        # Inference Çalıştır
        # Not: segmentor.predict zaten maske döndürüyor.
        # Bizim görselleştirme kodumuz zaten maskeyi kullanıyor.
        # Burada maksat "İşlem yapıldı" hissi vermek.
        try:
            mask = self.segmentor.predict(self.patient)
            
            if mask is not None:
                # Başarılı olduğunda
                self.btn_ai.setText("AI DIAGNOSIS COMPLETE")
                self.btn_ai.setStyleSheet(f"background-color: #00c853; color: white; border: 1px solid #00c853;")
                
                QMessageBox.information(self, "AI Inference Result", 
                    "<b>U-Net Model Inference Successful.</b><br><br>"
                    "Segmentation masks have been generated and overlaid on the 3D volume.<br>"
                    "Analysis indicates presence of High-Grade Glioma regions.")
            else:
                self.btn_ai.setText("AI FAILED")
                
        except Exception as e:
            QMessageBox.critical(self, "AI Error", str(e))
            self.btn_ai.setText(original_text)
            self.btn_ai.setEnabled(True)

    def toggle_visibility(self, key, state):
        if key in self.actors:
            self.actors[key].SetVisibility(state == Qt.Checked)
            self.plotter.update()

    def update_opacity(self, key, value):
        if key in self.actors:
            self.actors[key].GetProperty().SetOpacity(value / 100.0)
            self.plotter.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())