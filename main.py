# =============================================================================
# 0. [필수] 보안/네트워크 패치 (KNOC 사내망용)
# =============================================================================
import os
import ssl
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# 1. 경고 메시지 끄기
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# 2. SSL 인증서 검증 무력화 (표준 라이브러리)
ssl._create_default_https_context = ssl._create_unverified_context

# 3. Requests 라이브러리 강제 패치 (Contextily 지도 다운로드용)
# 환경변수 설정만으로는 부족할 때가 있어 함수 자체를 오버라이딩합니다.
old_merge_environment_settings = requests.Session.merge_environment_settings

def merge_environment_settings(self, url, proxies, stream, verify, cert):
    # verify를 무조건 False로 고정
    return old_merge_environment_settings(self, url, proxies, stream, False, cert)

requests.Session.merge_environment_settings = merge_environment_settings

# =============================================================================
# 1. 라이브러리 임포트
# =============================================================================
import sys
import json
import csv
import numpy as np
import segyio
from scipy.interpolate import griddata
import contextily as ctx
from pyproj import Transformer

# PySide6 (Qt) Imports
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QComboBox, 
                               QFileDialog, QMessageBox, QCheckBox, 
                               QSlider, QGroupBox)
from PySide6.QtCore import Qt, Signal

# Matplotlib for Qt
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =============================================================================
# 2. SEGY 뷰어 (상세 단면 보기)
# =============================================================================
class SegyViewer(QMainWindow):
    horizon_updated = Signal(str, dict)

    def __init__(self, filename=None, horizons=None, coord_type="CDP"):
        super().__init__()
        self.setWindowTitle(f"Woo Interpreter - {os.path.basename(filename) if filename else 'New'}")
        self.resize(1400, 900)

        self.filename = filename
        self.coord_type = coord_type
        self.current_data = None
        self.horizons = horizons if horizons else {
            'Horizon A': {'color': 'yellow', 'points': []},
            'Horizon B': {'color': 'cyan', 'points': []},
            'Horizon C': {'color': 'lime', 'points': []}
        }
        self.active_layer = 'Horizon A'
        
        # 3D Variables
        self.is_3d = False
        self.segy_handle = None
        self.ilines = []
        self.xlines = []
        self.current_slice_type = "Inline"

        # Rendering
        self.limit_val = 1.0
        self.line_objs = {}
        self.scat_objs = {}
        self.real_trace_indices = None
        self.cache_x = None

        self.init_ui()
        if self.filename:
            self.load_from_path(self.filename)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(320)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setAlignment(Qt.AlignTop)
        main_layout.addWidget(sidebar)

        # 3D Controls
        self.group_3d = QGroupBox("3D Slice Control")
        self.group_3d.setVisible(False)
        layout_3d = QVBoxLayout()
        
        type_layout = QHBoxLayout()
        self.combo_slice_type = QComboBox()
        self.combo_slice_type.addItems(["Inline", "Crossline"])
        self.combo_slice_type.currentTextChanged.connect(self.on_slice_type_change)
        type_layout.addWidget(QLabel("Type:"))
        type_layout.addWidget(self.combo_slice_type)
        layout_3d.addLayout(type_layout)

        self.slider_slice = QSlider(Qt.Horizontal)
        self.slider_slice.valueChanged.connect(self.on_slice_change)
        layout_3d.addWidget(self.slider_slice)
        
        self.lbl_slice_info = QLabel("No Data")
        layout_3d.addWidget(self.lbl_slice_info)
        self.group_3d.setLayout(layout_3d)
        sidebar_layout.addWidget(self.group_3d)

        # Interpretation Tools
        sidebar_layout.addWidget(QLabel("<b>--- Interpretation ---</b>"))
        self.combo_layer = QComboBox()
        self.combo_layer.addItems(list(self.horizons.keys()))
        self.combo_layer.currentTextChanged.connect(self.on_layer_change)
        sidebar_layout.addWidget(self.combo_layer)

        btn_io_layout = QHBoxLayout()
        btn_import = QPushButton("📂 Import")
        btn_import.clicked.connect(self.import_horizon_csv)
        btn_export = QPushButton("💾 Export")
        btn_export.clicked.connect(self.save_horizon)
        btn_io_layout.addWidget(btn_import)
        btn_io_layout.addWidget(btn_export)
        sidebar_layout.addLayout(btn_io_layout)

        btn_clear = QPushButton("📍 Clear Layer")
        btn_clear.clicked.connect(self.clear_horizon)
        btn_clear.setStyleSheet("background-color: #e74c3c; color: white;")
        sidebar_layout.addWidget(btn_clear)

        self.lbl_hor_info = QLabel("Status: Ready")
        sidebar_layout.addWidget(self.lbl_hor_info)

        # Display Settings
        sidebar_layout.addSpacing(15)
        sidebar_layout.addWidget(QLabel("<b>--- Display ---</b>"))
        
        self.chk_auto_fit = QCheckBox("Auto Fit Mode")
        self.chk_auto_fit.setChecked(True)
        self.chk_auto_fit.toggled.connect(self.toggle_aspect)
        sidebar_layout.addWidget(self.chk_auto_fit)

        sidebar_layout.addWidget(QLabel("Contrast (Clip %)"))
        self.slider_clip = QSlider(Qt.Horizontal)
        self.slider_clip.setRange(800, 999) 
        self.slider_clip.setValue(980)
        self.slider_clip.valueChanged.connect(lambda: self.update_contrast_only())
        sidebar_layout.addWidget(self.slider_clip)

        # Plot Area
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        main_layout.addWidget(plot_container)

        self.canvas.mpl_connect('button_press_event', self.on_mouse_action)
        self.ax = self.fig.add_subplot(111)

    def closeEvent(self, event):
        if self.segy_handle: self.segy_handle.close()
        event.accept()

    def load_from_path(self, path):
        try:
            self.segy_handle = segyio.open(path, "r", strict=True)
            self.is_3d = True
            self.ilines = self.segy_handle.ilines
            self.xlines = self.segy_handle.xlines
            self.group_3d.setVisible(True)
            self.update_3d_controls()
            self.load_slice(self.ilines[len(self.ilines)//2], "Inline")
        except Exception:
            self.is_3d = False
            self.group_3d.setVisible(False)
            if self.segy_handle: self.segy_handle.close(); self.segy_handle = None
            self.load_2d_data(path)

    def update_3d_controls(self):
        arr = self.ilines if self.current_slice_type == "Inline" else self.xlines
        self.slider_slice.blockSignals(True)
        self.slider_slice.setRange(0, len(arr)-1)
        self.slider_slice.setValue(len(arr)//2)
        self.slider_slice.blockSignals(False)

    def on_slice_type_change(self, text):
        self.current_slice_type = text
        self.update_3d_controls()
        self.on_slice_change(self.slider_slice.value())

    def on_slice_change(self, idx):
        target_arr = self.ilines if self.current_slice_type == "Inline" else self.xlines
        if idx < 0 or idx >= len(target_arr): return
        actual_line = target_arr[idx]
        self.lbl_slice_info.setText(f"{self.current_slice_type}: {actual_line}")
        self.load_slice(actual_line, self.current_slice_type)

    def load_slice(self, line_no, mode):
        if not self.segy_handle: return
        try:
            if mode == "Inline":
                data = self.segy_handle.iline[line_no]
                self.real_trace_indices = self.xlines
            else:
                data = self.segy_handle.xline[line_no]
                self.real_trace_indices = self.ilines
            
            self.current_data = data.T
            self.cache_x = np.arange(len(self.real_trace_indices)) # Simple Index mapping for 3D
            self.full_redraw()
        except Exception as e:
            print(f"Slice Load Error: {e}")

    def load_2d_data(self, path):
        try:
            with segyio.open(path, "r", ignore_geometry=True) as f:
                total_traces = f.tracecount
                step = max(1, total_traces // 5000)
                indices = list(range(0, total_traces, step))
                self.real_trace_indices = np.array(indices)
                self.current_data = segyio.tools.collect(f.trace[::step]).T
                
                scalars = f.attributes(segyio.TraceField.SourceGroupScalar)[0:1]
                scalar_val = float(scalars[0]) if len(scalars) > 0 else 1.0
                if scalar_val < 0: scalar_val = 1.0 / abs(scalar_val)
                elif scalar_val == 0: scalar_val = 1.0
                
                xk = segyio.TraceField.CDP_X if self.coord_type == "CDP" else segyio.TraceField.SourceX
                self.cache_x = f.attributes(xk)[::step].astype(float) * scalar_val
            
            self.full_redraw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"2D Load Failed: {e}")

    def full_redraw(self):
        if self.current_data is None: return
        self.ax.clear()
        
        clip_pct = self.slider_clip.value() / 10.0
        sample_data = self.current_data[::5, ::5]
        limit = np.nanpercentile(np.absolute(sample_data), clip_pct)
        if limit == 0: limit = 1.0
        self.limit_val = limit
        
        n_samples, n_traces = self.current_data.shape
        self.extent = [0, n_traces, n_samples * 2.0, 0] # 2ms sampling default
        
        self.im_obj = self.ax.imshow(self.current_data, cmap="RdBu", 
                                     vmin=-limit, vmax=limit,
                                     aspect='auto', extent=self.extent, interpolation='nearest')
        
        self.ax.set_ylabel("Time (ms)")
        self.ax.set_xlabel("Trace / Inline / Crossline")
        
        self.update_aspect_only(draw=False)
        self.draw_horizons_only(draw=False)
        self.canvas.draw()

    def update_aspect_only(self, draw=True):
        if self.chk_auto_fit.isChecked(): self.ax.set_aspect('auto')
        if draw: self.canvas.draw_idle()

    def update_contrast_only(self):
        if not self.im_obj: return
        clip_pct = self.slider_clip.value() / 10.0
        limit = np.nanpercentile(np.absolute(self.current_data[::5, ::5]), clip_pct)
        if limit == 0: limit = 1.0
        self.im_obj.set_clim(-limit, limit)
        self.canvas.draw_idle()

    def toggle_aspect(self, checked): self.update_aspect_only()

    def draw_horizons_only(self, draw=True):
        if self.real_trace_indices is None: return
        for obj in list(self.line_objs.values()) + list(self.scat_objs.values()):
            try: obj.remove()
            except: pass
        self.line_objs, self.scat_objs = {}, {}

        for name, data in self.horizons.items():
            if not data['points']: continue
            p_arr = np.array(data['points'])
            saved_indices = p_arr[:, 3]
            display_indices = np.searchsorted(self.real_trace_indices, saved_indices)
            valid_mask = (display_indices < len(self.real_trace_indices)) & \
                         (self.real_trace_indices[np.clip(display_indices, 0, len(self.real_trace_indices)-1)] == saved_indices)
            
            if np.any(valid_mask):
                x_plot = display_indices[valid_mask]
                y_plot = p_arr[valid_mask, 2]
                self.scat_objs[name] = self.ax.plot(x_plot, y_plot, 'o', color=data['color'], markersize=4)[0]
                if len(x_plot) >= 2:
                    self.line_objs[name] = self.ax.plot(x_plot, y_plot, color=data['color'], linewidth=1.5)[0]
        if draw: self.canvas.draw_idle()

    def on_mouse_action(self, event):
        if event.inaxes != self.ax or self.toolbar.mode != '': return
        if not event.xdata or not event.ydata: return
        
        display_idx = int(round(event.xdata))
        twt = event.ydata
        pts_list = self.horizons[self.active_layer]['points']
        changed = False

        if event.button == 1: # Left Click
            if 0 <= display_idx < len(self.real_trace_indices):
                real_idx = self.real_trace_indices[display_idx]
                x_val = self.cache_x[display_idx] if self.cache_x is not None else 0
                pts_list.append([x_val, 0, twt, real_idx]) # Y는 임시 0
                pts_list.sort(key=lambda x: x[3])
                changed = True

        elif event.button == 3: # Right Click
             if pts_list and 0 <= display_idx < len(self.real_trace_indices):
                target_real = self.real_trace_indices[display_idx]
                dists = [abs(p[3] - target_real) for p in pts_list]
                if dists and min(dists) < 5:
                    pts_list.pop(np.argmin(dists))
                    changed = True
        
        if changed:
            self.update_status()
            self.draw_horizons_only()
            self.horizon_updated.emit(self.filename, self.horizons)

    def on_layer_change(self, text): self.active_layer = text
    def update_status(self):
        status_txt = " | ".join([f"{k}:{len(v['points'])}" for k, v in self.horizons.items()])
        self.lbl_hor_info.setText(status_txt)

    def clear_horizon(self):
        self.horizons[self.active_layer]['points'] = []
        self.update_status(); self.draw_horizons_only()
        self.horizon_updated.emit(self.filename, self.horizons)

    def save_horizon(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save", "", "CSV(*.csv)")
        if path:
            with open(path, 'w', newline='') as f:
                f.write("Layer,X,Y,TWT,TraceIdx\n")
                for n, d in self.horizons.items():
                    for p in d['points']: f.write(f"{n},{p[0]},{p[1]},{p[2]},{p[3]}\n")
            QMessageBox.information(self, "Success", "Export Complete.")

    def import_horizon_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import", "", "CSV(*.csv)")
        if path:
            try:
                with open(path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        layer = row['Layer'].strip()
                        if layer in self.horizons:
                            self.horizons[layer]['points'].append(
                                [float(row['X']), float(row['Y']), float(row['TWT']), int(float(row['TraceIdx']))]
                            )
                self.update_status(); self.draw_horizons_only()
                self.horizon_updated.emit(self.filename, self.horizons)
            except Exception as e: QMessageBox.critical(self, "Error", str(e))

# =============================================================================
# 3. 프로젝트 매니저 (지도 및 좌표 통합 관리)
# =============================================================================
class ProjectManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seismic Base Map (Project Manager)")
        self.resize(1200, 900)
        
        self.survey_lines = {}
        self.line_plots = {}
        self.horizon_plots = {}
        
        # 기본 좌표계 설정: UTM Zone 50S (E.Kalimantan)
        self.current_epsg = "EPSG:32750" 
        self.transformer = Transformer.from_crs(self.current_epsg, "EPSG:3857", always_xy=True)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Toolbar
        toolbar = QHBoxLayout()
        
        btn_load = QPushButton("📂 Load SEGY")
        btn_load.clicked.connect(self.add_files)
        toolbar.addWidget(btn_load)
        
        btn_save = QPushButton("💾 Save")
        btn_save.clicked.connect(self.save_project)
        toolbar.addWidget(btn_save)

        btn_open = QPushButton("📂 Open")
        btn_open.clicked.connect(self.load_project)
        toolbar.addWidget(btn_open)

        # [UTM Zone Selector]
        toolbar.addSpacing(20)
        toolbar.addWidget(QLabel("<b>UTM Zone:</b>"))
        self.combo_zone = QComboBox()
        self.combo_zone.setMinimumWidth(180)
        self.combo_zone.addItem("Zone 50S (E.Kalimantan)", "EPSG:32750") 
        self.combo_zone.addItem("Zone 51S (Sulawesi)", "EPSG:32751")
        self.combo_zone.addItem("Zone 49S (W.Kalimantan)", "EPSG:32749")
        self.combo_zone.addItem("Zone 50N (North Hemi)", "EPSG:32650")
        self.combo_zone.currentIndexChanged.connect(self.change_crs_zone)
        toolbar.addWidget(self.combo_zone)

        # Map Style
        toolbar.addSpacing(20)
        toolbar.addWidget(QLabel("Map:"))
        self.combo_map = QComboBox()
        self.combo_map.addItems(["OpenStreetMap", "Satellite (Esri)", "Toner Lite"])
        self.combo_map.currentTextChanged.connect(self.update_map_background)
        toolbar.addWidget(self.combo_map)

        # Layer View
        toolbar.addSpacing(10)
        toolbar.addWidget(QLabel("Layer:"))
        self.combo_viz_layer = QComboBox()
        self.combo_viz_layer.addItems(["None", "Horizon A", "Horizon B", "Horizon C"])
        self.combo_viz_layer.currentTextChanged.connect(self.draw_visualization)
        toolbar.addWidget(self.combo_viz_layer)

        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # Map Canvas
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.mpl_toolbar = NavigationToolbar2QT(self.canvas, self)
        
        main_layout.addWidget(self.mpl_toolbar)
        main_layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)
        self.divider = make_axes_locatable(self.ax)
        self.cax = self.divider.append_axes("right", size="3%", pad=0.1)

        self.canvas.mpl_connect('pick_event', self.on_line_pick)
        self.reset_map_view()

    def reset_map_view(self):
        self.ax.clear()
        self.cax.clear(); self.cax.axis('off')
        self.ax.set_axis_off()
        self.line_plots = {}
        self.horizon_plots = {}
        self.canvas.draw()

    def change_crs_zone(self, index):
        """ UTM Zone 변경 시 좌표 재계산 """
        new_epsg = self.combo_zone.currentData()
        self.current_epsg = new_epsg
        self.transformer = Transformer.from_crs(new_epsg, "EPSG:3857", always_xy=True)
        print(f"CRS Changed to {new_epsg}")

        # 모든 라인 다시 변환 (원본 UTM 사용)
        for fname, d in self.survey_lines.items():
            if "UTM" in d.get('type', ''):
                new_mx, new_my = self.transformer.transform(d['raw_x'], d['raw_y'])
                d['x'] = new_mx
                d['y'] = new_my
        
        self.update_map()

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select SEGY", "", "SEGY Files (*.sgy *.segy)")
        if not paths: return
        for path in paths:
            self.process_segy_file_with_coords(path)
        self.update_map()

    def process_segy_file_with_coords(self, filepath):
        if not os.path.exists(filepath): return
        fname = os.path.basename(filepath)
        
        try:
            with segyio.open(filepath, "r", ignore_geometry=True) as f:
                # 1. Scalar
                scalars = f.attributes(segyio.TraceField.SourceGroupScalar)[0:1]
                scalar = float(scalars[0]) if len(scalars) > 0 else 1.0
                if scalar < 0: scalar = 1.0 / abs(scalar)
                elif scalar == 0: scalar = 1.0
                
                # 2. Coordinates
                trace_count = f.tracecount
                step = max(1, trace_count // 500)
                
                src_x = f.attributes(segyio.TraceField.SourceX)[::step].astype(float) * scalar
                src_y = f.attributes(segyio.TraceField.SourceY)[::step].astype(float) * scalar
                cdp_x = f.attributes(segyio.TraceField.CDP_X)[::step].astype(float) * scalar
                cdp_y = f.attributes(segyio.TraceField.CDP_Y)[::step].astype(float) * scalar
                
                final_x, final_y = None, None
                coord_type = "Unknown"
                
                # 3. Smart Detection
                avg_src_x = np.mean(np.abs(src_x))
                avg_cdp_x = np.mean(np.abs(cdp_x))

                # Case A: Lat/Lon (<= 180)
                if 0.1 < avg_src_x <= 180.0:
                    final_x, final_y = src_x, src_y
                    coord_type = "Source (Lat/Lon)"
                    # LatLon -> WebMercator
                    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
                    mx, my = transformer.transform(final_x, final_y)

                # Case B: UTM (> 10000)
                elif avg_src_x > 10000:
                    final_x, final_y = src_x, src_y
                    coord_type = "Source (UTM)"
                    # Use currently selected Zone
                    mx, my = self.transformer.transform(final_x, final_y)
                    
                elif avg_cdp_x > 10000:
                    final_x, final_y = cdp_x, cdp_y
                    coord_type = "CDP (UTM)"
                    mx, my = self.transformer.transform(final_x, final_y)
                
                if final_x is None:
                    print(f"Skipping {fname}: No valid coords.")
                    return
                
                print(f"Loaded {fname}: {coord_type}")
                
                horizons = self.get_default_horizons()
                self.survey_lines[fname] = {
                    'path': filepath, 
                    'x': mx, 
                    'y': my,
                    'raw_x': final_x, # Save Raw for dynamic switching
                    'raw_y': final_y,
                    'type': coord_type,
                    'horizons': horizons
                }

        except Exception as e:
            print(f"Error loading {fname}: {e}")

    def get_default_horizons(self):
        return {'Horizon A': {'color': 'yellow', 'points': []}, 
                'Horizon B': {'color': 'cyan', 'points': []}, 
                'Horizon C': {'color': 'lime', 'points': []}}

    def update_map(self):
        self.ax.clear()
        
        for lid, d in self.survey_lines.items():
            l, = self.ax.plot(d['x'], d['y'], label=lid, linewidth=2, color='blue', alpha=0.7, picker=5)
            self.line_plots[l] = lid
            if len(d['x']) > 0:
                self.ax.text(d['x'][0], d['y'][0], lid[:10], fontsize=8, fontweight='bold', color='darkblue')

        if self.survey_lines:
            self.update_map_background(self.combo_map.currentText())
        
        self.ax.set_axis_off()
        self.canvas.draw()

    def update_map_background(self, style_name):
        if not self.survey_lines: return
        provider = ctx.providers.OpenStreetMap.Mapnik
        if style_name == "Satellite (Esri)": provider = ctx.providers.Esri.WorldImagery
        elif style_name == "Toner Lite": provider = ctx.providers.CartoDB.Positron

        try:
            ctx.add_basemap(self.ax, crs='EPSG:3857', source=provider)
        except Exception as e:
            print(f"Map Error: {e}")
        self.canvas.draw()

    def on_line_pick(self, event):
        if event.artist in self.line_plots:
            lid = self.line_plots[event.artist]
            data = self.survey_lines[lid]
            self.viewer_win = SegyViewer(filename=data['path'], horizons=data['horizons'], coord_type=data.get('type', 'CDP'))
            self.viewer_win.horizon_updated.connect(self.on_horizon_update)
            self.viewer_win.show()

    def on_horizon_update(self, filepath, horizons):
        fname = os.path.basename(filepath)
        if fname in self.survey_lines:
            self.survey_lines[fname]['horizons'] = horizons
            self.draw_visualization()

    def draw_visualization(self):
        target = self.combo_viz_layer.currentText()
        if target == 'None': 
            self.update_map() # Clear contour
            return

        all_x, all_y, all_z = [], [], []
        # (간략화된 예시) 실제 구현 시엔 픽킹된 포인트의 정확한 좌표 매핑 필요
        # 현재는 데모용으로 Scatter만 지원
        # ...

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "JSON Files (*.json)")
        if path:
            save_data = {fname: {'path': d['path'], 'horizons': d['horizons']} for fname, d in self.survey_lines.items()}
            with open(path, 'w') as f: json.dump(save_data, f, indent=4)
            QMessageBox.information(self, "Success", "Saved.")

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "JSON Files (*.json)")
        if path:
            with open(path, 'r') as f: loaded_data = json.load(f)
            self.survey_lines = {}; self.reset_map_view()
            for fname, data in loaded_data.items():
                self.process_segy_file_with_coords(data['path'])
                if fname in self.survey_lines:
                    self.survey_lines[fname]['horizons'] = data['horizons']
            self.update_map()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProjectManager()
    window.show()
    sys.exit(app.exec())
