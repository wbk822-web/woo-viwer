import sys, os, json
import numpy as np
import segyio

# --- Speed Optimization ---
try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# --- GIS Support ---
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
# --------------------------

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QFileDialog, QSlider, QCheckBox, QComboBox, 
                               QMessageBox, QDoubleSpinBox, QTabWidget, QSpinBox, QListWidget, 
                               QListWidgetItem, QAbstractItemView, QDialog, QFormLayout, 
                               QDialogButtonBox, QGroupBox, QPlainTextEdit, QTableWidget, 
                               QTableWidgetItem, QHeaderView, QRadioButton, QButtonGroup, QStatusBar, QSplitter)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, StrMethodFormatter

# =============================================================================
# 1. Header Viewer
# =============================================================================
class SegyHeaderViewer(QDialog):
    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Header Inspector - {os.path.basename(filename)}")
        self.resize(900, 600)
        l = QVBoxLayout(self); self.tabs = QTabWidget(); l.addWidget(self.tabs)
        self.txt = QPlainTextEdit(); self.txt.setReadOnly(True); self.txt.setFont(QFont("Consolas", 10))
        self.tabs.addTab(self.txt, "Text Header")
        self.tbl = QTableWidget(); self.tbl.setColumnCount(3); self.tbl.setHorizontalHeaderLabels(["Byte","Desc","Value"])
        self.tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tabs.addTab(self.tbl, "Binary Header")
        self.load(filename)
        l.addWidget(QPushButton("Close", clicked=self.accept))

    def load(self, fn):
        try:
            with segyio.open(fn, ignore_geometry=True) as f:
                try: t = segyio.tools.wrap(f.text[0]) if hasattr(segyio.tools,'wrap') else f.text[0].decode('ascii','ignore')
                except: t = "Decode Error"
                self.txt.setPlainText(str(t))
                bh = f.bin; self.tbl.setRowCount(len(bh))
                smap = {3201:"Job ID", 3205:"Line", 3213:"Trc/Ens", 3217:"Interval", 3221:"Samples", 3225:"Format", 3255:"Meas Sys"}
                for k,v in sorted(bh.items(), key=lambda x: int(x[0])):
                    r = self.tbl.rowCount(); self.tbl.insertRow(r)
                    self.tbl.setItem(r,0,QTableWidgetItem(str(k)))
                    self.tbl.setItem(r,1,QTableWidgetItem(smap.get(k,f"Byte {k}")))
                    self.tbl.setItem(r,2,QTableWidgetItem(str(v)))
        except Exception as e: self.txt.setPlainText(f"Err: {e}")

# =============================================================================
# 2. Import Wizard
# =============================================================================
class SegyHeaderDialog(QDialog):
    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.setWindowTitle(f"Import Settings - {os.path.basename(filename)}")
        self.resize(500, 700)
        l = QVBoxLayout(self)
        h = QHBoxLayout(); h.addWidget(QLabel(f"<b>{os.path.basename(filename)}</b>"))
        h.addWidget(QPushButton("Check Headers", clicked=lambda: SegyHeaderViewer(filename, self).exec()))
        l.addLayout(h)
        
        g1 = QGroupBox("1. CRS"); f1 = QFormLayout(g1)
        self.cb_crs = QComboBox(); self.cb_crs.setEditable(True)
        self.cb_crs.addItems(["WGS 84 / UTM zone 52S", "WGS 84 / UTM zone 53S", "WGS 84 / UTM zone 52N", "Unknown"])
        f1.addRow("System:", self.cb_crs); l.addWidget(g1)
        
        g2 = QGroupBox("2. Bytes"); f2 = QFormLayout(g2)
        self.sb_cdp = QSpinBox(); self.sb_cdp.setRange(1,240); self.sb_cdp.setValue(21)
        self.sb_x = QSpinBox(); self.sb_x.setRange(1,240); self.sb_x.setValue(181)
        self.sb_y = QSpinBox(); self.sb_y.setRange(1,240); self.sb_y.setValue(185)
        f2.addRow("CDP:", self.sb_cdp); f2.addRow("X:", self.sb_x); f2.addRow("Y:", self.sb_y)
        l.addWidget(g2)
        
        g3 = QGroupBox("3. Scalar"); v3 = QVBoxLayout(g3)
        self.rb_h = QRadioButton("Header (Byte 71)"); self.rb_h.setChecked(True)
        self.sb_sc = QSpinBox(); self.sb_sc.setRange(1,240); self.sb_sc.setValue(71)
        self.rb_m = QRadioButton("Manual"); self.rb_n = QRadioButton("None")
        self.db_mx = QDoubleSpinBox(); self.db_mx.setRange(1e-6, 1e6); self.db_mx.setValue(1.0); self.db_mx.setDecimals(6)
        self.db_my = QDoubleSpinBox(); self.db_my.setRange(1e-6, 1e6); self.db_my.setValue(1.0); self.db_my.setDecimals(6)
        bg = QButtonGroup(self); bg.addButton(self.rb_h); bg.addButton(self.rb_m); bg.addButton(self.rb_n)
        h1 = QHBoxLayout(); h1.addWidget(self.rb_h); h1.addWidget(QLabel("Byte:")); h1.addWidget(self.sb_sc)
        h2 = QHBoxLayout(); h2.addWidget(self.rb_m); h2.addWidget(QLabel("X*:")); h2.addWidget(self.db_mx); h2.addWidget(QLabel("Y*:")); h2.addWidget(self.db_my)
        v3.addLayout(h1); v3.addLayout(h2); v3.addWidget(self.rb_n); l.addWidget(g3)
        
        g4 = QGroupBox("4. Data"); f4 = QFormLayout(g4)
        self.sb_sr = QDoubleSpinBox(); self.sb_sr.setRange(0.01, 1000); self.sb_sr.setValue(4.0)
        f4.addRow("SR (ms):", self.sb_sr); l.addWidget(g4)
        
        self.chk = QCheckBox("Apply to all"); l.addWidget(self.chk)
        bb = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); l.addWidget(bb)
        self.detect(filename)

    def detect(self, fn):
        try:
            with segyio.open(fn, ignore_geometry=True) as f:
                iv = f.bin[segyio.BinField.Interval]; 
                if iv>0: self.sb_sr.setValue(iv/1000.0)
        except: pass

    def get_data(self):
        m = 'h' if self.rb_h.isChecked() else ('m' if self.rb_m.isChecked() else 'n')
        return {
            'crs':self.cb_crs.currentText(), 'cdp_b':self.sb_cdp.value(), 'x_b':self.sb_x.value(), 'y_b':self.sb_y.value(),
            'sc_m':m, 'sc_b':self.sb_sc.value(), 'mx':self.db_mx.value(), 'my':self.db_my.value(),
            'sr':self.sb_sr.value(), 'all':self.chk.isChecked()
        }

# =============================================================================
# 3. Data Object
# =============================================================================
class SeismicObject:
    def __init__(self, fn, data, coords, cdps, sets):
        self.filename = fn; self.name = os.path.basename(fn)
        self.raw_data = data; self.real_coords = coords; self.cdps = cdps; self.settings = sets
        self.trace_count = len(coords)
        self.idx_coords = np.column_stack((np.arange(self.trace_count), np.zeros(self.trace_count)))
        self.crs_name = sets.get('crs','Unknown')
        self.horizons = {k:{'color':c,'points':[]} for k,c in [('H_A','yellow'),('H_B','cyan'),('H_C','lime')]}
        self.shift_ms = 0; self.is_flipped = False; self.contrast = 980
        self.composite_data = None; self.intersections = []

# =============================================================================
# 4. Separate Section Window (Pop-up)
# =============================================================================
class SeismicSectionWindow(QMainWindow):
    request_file_change = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seismic Viewer")
        self.resize(1000, 600)
        cen = QWidget(); self.setCentralWidget(cen); lay = QVBoxLayout(cen)
        
        # Navigation
        h_nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀"); self.btn_prev.clicked.connect(self.go_prev)
        self.btn_next = QPushButton("▶"); self.btn_next.clicked.connect(self.go_next)
        self.combo_files = QComboBox()
        self.combo_files.currentIndexChanged.connect(self.on_combo_changed)
        h_nav.addWidget(self.btn_prev); h_nav.addWidget(self.combo_files, 1); h_nav.addWidget(self.btn_next)
        lay.addLayout(h_nav)
        
        self.fig = Figure(facecolor='#F0F0F0'); self.cv = FigureCanvasQTAgg(self.fig)
        lay.addWidget(NavigationToolbar2QT(self.cv, cen)); lay.addWidget(self.cv)
        self.ax = self.fig.add_subplot(111); self.ax.set_facecolor('black')
        
        self.cv.mpl_connect('button_press_event', self.on_click)
        self.cv.mpl_connect('scroll_event', self.on_scroll)
        self.cv.mpl_connect('motion_notify_event', self.on_move)
        
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.current_obj = None; self.active_hor = "H_A"
        self.cross_v = None; self.cross_h = None; self.show_cdp = False; self.internal_change = False

    def update_file_list(self, names, current_idx):
        self.internal_change = True
        self.combo_files.clear(); self.combo_files.addItems(names)
        if 0 <= current_idx < len(names): self.combo_files.setCurrentIndex(current_idx)
        self.internal_change = False
        self.btn_prev.setEnabled(current_idx > 0); self.btn_next.setEnabled(current_idx < len(names) - 1)

    def on_combo_changed(self, idx):
        if not self.internal_change and idx >= 0: self.request_file_change.emit(idx)
    def go_prev(self):
        idx = self.combo_files.currentIndex()
        if idx > 0: self.combo_files.setCurrentIndex(idx - 1)
    def go_next(self):
        idx = self.combo_files.currentIndex()
        if idx < self.combo_files.count() - 1: self.combo_files.setCurrentIndex(idx + 1)
    def set_active_horizon(self, key): self.active_hor = key

    # --- [최적화 적용된 draw 함수] ---
    def draw(self, obj):
        self.current_obj = obj
        if not obj: self.ax.clear(); self.cv.draw(); return
        
        full_data = obj.composite_data if obj.composite_data is not None else obj.raw_data
        
        # [최적화 1] 다운샘플링 (Decimation)
        MAX_RES = 2000 
        h, w = full_data.shape
        dy = max(1, h // MAX_RES)
        dx = max(1, w // MAX_RES)
        d = full_data[::dy, ::dx]
        
        sr = obj.settings['sr']
        shift_samples = int(obj.shift_ms / sr)
        
        # Shift 처리 (다운샘플링된 비율 고려)
        if shift_samples != 0:
            s_shift = int(shift_samples / dy)
            d = np.roll(d, s_shift, 0)
            if s_shift > 0: d[:s_shift, :] = 0
            
        if obj.is_flipped: d = np.fliplr(d)
        
        self.ax.clear(); self.cross_v = None; self.cross_h = None
        
        abs_data = np.abs(d)
        lim = 1.0 if np.max(abs_data)==0 else np.nanpercentile(abs_data, obj.contrast/10.0)
        if lim==0: lim=1.0
        
        max_time = h * sr
        
        # [최적화 2] imshow extent 사용
        self.ax.imshow(d, cmap='RdBu', aspect='auto', vmin=-lim, vmax=lim, 
                       extent=[0, w, max_time, 0], interpolation='nearest')
                       
        self.ax.set_title(f"{obj.name}\nCRS: {obj.crs_name}"); self.ax.set_ylabel("TWT (ms)")
        
        if "Composite" in obj.name and hasattr(obj, 'intersections'):
            for idx in obj.intersections:
                px = (w-1)-idx if obj.is_flipped else idx
                self.ax.axvline(x=px, color='black', lw=1.5)

        if self.show_cdp and hasattr(obj, 'cdps') and "Composite" not in obj.name:
            def format_cdp(x, p):
                idx = int(round(x))
                if 0 <= idx < w:
                    real_idx = (w-1)-idx if obj.is_flipped else idx
                    return str(obj.cdps[real_idx])
                return ""
            self.ax.xaxis.set_major_formatter(FuncFormatter(format_cdp))
            self.ax.set_xlabel("CDP")
        else:
            self.ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p: str(int(x))))
            self.ax.set_xlabel("Trace Index")
            
        for k, v in obj.horizons.items():
            if not v['points']: continue
            p = np.array(v['points'])
            x = (w-1)-p[:,0] if obj.is_flipped else p[:,0]
            self.ax.plot(x, p[:,1]+obj.shift_ms, 'o-', c=v['color'], ms=4)
            
        self.cv.draw()

    # --- [들여쓰기 수정된 on_scroll 함수] ---
    def on_scroll(self, e):
        if e.inaxes!=self.ax: return
        sc = 1.2 if e.button=='down' else 1/1.2; xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
        w, h = xl[1]-xl[0], yl[1]-yl[0]
        rx = (xl[1]-e.xdata)/w; ry = (yl[1]-e.ydata)/h
        self.ax.set_xlim([e.xdata-w*sc*(1-rx), e.xdata+w*sc*rx])
        self.ax.set_ylim([e.ydata-h*sc*(1-ry), e.ydata+h*sc*ry]); self.cv.draw_idle()

    def on_click(self, e):
        if not self.current_obj or e.inaxes!=self.ax: return
        o = self.current_obj; nt = (o.composite_data if o.composite_data is not None else o.raw_data).shape[1]
        ix = int(round(e.xdata)); rix = (nt-1)-ix if o.is_flipped else ix
        if not (0<=rix<nt): return
        pts = o.horizons[self.active_hor]['points']
        if e.button==1: pts[:]=[p for p in pts if p[0]!=rix]; pts.append([rix, e.ydata-o.shift_ms]); pts.sort(key=lambda x:x[0])
        elif e.button==3 and pts: pts.pop(np.argmin([abs(p[0]-rix) for p in pts]))
        self.draw(o)

    def on_move(self, e):
        if not e.inaxes or not self.current_obj: return
        if not self.cross_v:
            self.cross_v = self.ax.axvline(x=e.xdata, color='red', lw=0.5, ls='--')
            self.cross_h = self.ax.axhline(y=e.ydata, color='red', lw=0.5, ls='--')
        else: self.cross_v.set_xdata([e.xdata]); self.cross_h.set_ydata([e.ydata])
        self.cv.draw_idle()
        o = self.current_obj; d = o.composite_data if o.composite_data is not None else o.raw_data
        ix = int(round(e.xdata)); rix = (d.shape[1]-1)-ix if o.is_flipped else ix
        msg = f"Trace: {ix} | Time: {e.ydata:.1f}ms"
        if 0<=rix<d.shape[1]:
            s_idx = int((e.ydata-o.shift_ms)/o.settings['sr'])
            if 0<=s_idx<d.shape[0]: msg += f" | Amp: {d[s_idx, rix]:.2f}"
            if hasattr(o,'cdps') and len(o.cdps)>rix: msg += f" | CDP: {o.cdps[rix]}"
            if self.parent() and hasattr(self.parent(), 'update_map_cursor'):
                self.parent().update_map_cursor(rix, o)
        self.status.showMessage(msg)

# =============================================================================
# 5. Main Window
# =============================================================================
class SegyViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 Seismic Viewer + GIS")
        self.resize(1600, 900)
        self.seismic_objects = {}; self.current_obj = None; self.waypoints = []
        self.shapefile_layers = [] 
        
        self.map_marker = None; self.snap_marker = None; self.snap_coord = None
        self.map_kdtree = None; self.map_coords_cache = None; self.map_index_lookup = []
        
        self.extra_windows = [] 
        
        self.win_section = SeismicSectionWindow(self)
        self.win_section.request_file_change.connect(self.on_section_file_change)
        self.win_section.show()
        
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.init_ui()

    def init_ui(self):
        main = QWidget(); self.setCentralWidget(main); layout = QHBoxLayout(main)
        
        sidebar = QWidget(); sidebar.setFixedWidth(320); sl = QVBoxLayout(sidebar); layout.addWidget(sidebar)
        sl.addWidget(QPushButton("📂 Load SEGY", clicked=self.load_segy))
        
        btn_shp = QPushButton("🌍 Load Shapefile", clicked=self.load_shapefile)
        if not HAS_GEOPANDAS: 
            btn_shp.setEnabled(False)
            btn_shp.setToolTip("pip install geopandas required")
        sl.addWidget(btn_shp)
        
        sl.addWidget(QLabel("<b>Loaded Lines:</b>"))
        self.lst = QListWidget(); self.lst.setSelectionMode(QAbstractItemView.SingleSelection)
        self.lst.itemSelectionChanged.connect(self.sel_item)
        self.lst.itemChanged.connect(self.chk_item); sl.addWidget(self.lst)
        
        sl.addWidget(QPushButton("➕ New Viewer (Multi-Window)", clicked=self.spawn_new_viewer))
        
        h_vis = QHBoxLayout()
        h_vis.addWidget(QPushButton("👁️ Select Only", clicked=self.show_only_selected))
        h_vis.addWidget(QPushButton("✅ All On", clicked=self.show_all_files))
        h_vis.addWidget(QPushButton("⬜ All Off", clicked=self.hide_all_files))
        sl.addLayout(h_vis)

        h_del = QHBoxLayout()
        h_del.addWidget(QPushButton("🗑️ Remove", clicked=self.remove_item))
        h_del.addWidget(QPushButton("🧹 Clear All", clicked=self.clear_all_files))
        sl.addLayout(h_del)
        
        hp = QHBoxLayout(); hp.addWidget(QPushButton("Save", clicked=self.save_p)); hp.addWidget(QPushButton("Load", clicked=self.load_p))
        sl.addLayout(hp); sl.addSpacing(10)
        
        gc = QGroupBox("Display"); gl = QVBoxLayout(gc)
        gl.addWidget(QLabel("Contrast:")); self.sl_c = QSlider(Qt.Horizontal); self.sl_c.setRange(800,999); self.sl_c.setValue(980)
        self.sl_c.valueChanged.connect(self.upd_view); gl.addWidget(self.sl_c)
        
        hk = QHBoxLayout()
        self.ck_f = QCheckBox("Flip L/R"); self.ck_f.toggled.connect(self.upd_view); hk.addWidget(self.ck_f)
        self.ck_c = QCheckBox("Show CDP"); self.ck_c.toggled.connect(self.upd_view); hk.addWidget(self.ck_c)
        gl.addLayout(hk)
        
        self.cb_h = QComboBox(); self.cb_h.addItems(['H_A','H_B','H_C'])
        self.cb_h.currentTextChanged.connect(lambda t: self.win_section.set_active_horizon(t))
        self.win_section.set_active_horizon('H_A')
        gl.addWidget(self.cb_h); gl.addWidget(QPushButton("Clear Horizon", clicked=self.clr_hor))
        
        hs = QHBoxLayout(); hs.addWidget(QLabel("Shift:")); self.sb_s = QSpinBox(); self.sb_s.setRange(-5000,5000); self.sb_s.setSingleStep(4)
        self.sb_s.valueChanged.connect(self.upd_view); hs.addWidget(self.sb_s)
        gl.addLayout(hs); sl.addWidget(gc); self.grp_ctrl = gc; gc.setEnabled(False)
        sl.addStretch()
        
        map_area = QWidget(); lm = QVBoxLayout(map_area); layout.addWidget(map_area, stretch=1)
        h_mc = QHBoxLayout()
        
        self.rb_sel = QRadioButton("Select Line"); self.rb_sel.setChecked(True)
        self.rb_draw = QRadioButton("Draw Path")
        bg_map = QButtonGroup(self); bg_map.addButton(self.rb_sel); bg_map.addButton(self.rb_draw)
        h_mc.addWidget(self.rb_sel); h_mc.addWidget(self.rb_draw)
        
        h_mc.addWidget(QPushButton("✂️ Extract Composite", clicked=self.create_composite))
        h_mc.addWidget(QPushButton("❌ Clear Path", clicked=self.clr_path))
        self.ck_fix = QCheckBox("Force Index"); self.ck_fix.toggled.connect(self.draw_map); h_mc.addWidget(self.ck_fix)
        
        h_mc.addStretch(); lm.addLayout(h_mc)
        
        self.fig_m = Figure(); self.cv_m = FigureCanvasQTAgg(self.fig_m)
        lm.addWidget(NavigationToolbar2QT(self.cv_m, map_area)); lm.addWidget(self.cv_m)
        self.ax_m = self.fig_m.add_subplot(111); 
        self.cv_m.mpl_connect('button_press_event', self.on_map_click)
        self.cv_m.mpl_connect('motion_notify_event', self.on_map_hover)

    def load_shapefile(self):
        if not HAS_GEOPANDAS: return
        fn, _ = QFileDialog.getOpenFileName(self, "Open Shapefile", "", "Shapefile (*.shp)")
        if fn:
            try:
                gdf = gpd.read_file(fn)
                import random
                color = "#%06x" % random.randint(0, 0xFFFFFF)
                self.shapefile_layers.append({'name': os.path.basename(fn), 'data': gdf, 'color': color})
                self.status.showMessage(f"Loaded Shapefile: {os.path.basename(fn)}", 3000)
                self.draw_map()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load SHP: {str(e)}")

    def spawn_new_viewer(self):
        new_win = SeismicSectionWindow(None)
        new_win.setAttribute(Qt.WA_DeleteOnClose)
        if self.current_obj:
            new_win.draw(self.current_obj)
            names = [self.lst.item(i).text() for i in range(self.lst.count())]
            curr = self.lst.currentRow()
            new_win.update_file_list(names, curr)
        new_win.show()
        self.extra_windows.append(new_win) 

    def sync_file_list(self):
        names = [self.lst.item(i).text() for i in range(self.lst.count())]
        curr = self.lst.currentRow()
        self.win_section.update_file_list(names, curr)

    def on_section_file_change(self, idx):
        if 0 <= idx < self.lst.count(): self.lst.setCurrentRow(idx)

    def on_map_click(self, e):
        if e.inaxes!=self.ax_m: return
        
        if self.rb_sel.isChecked():
            if not self.map_kdtree: return
            dist, idx = self.map_kdtree.query([e.xdata, e.ydata])
            xlim = self.ax_m.get_xlim(); threshold = (xlim[1] - xlim[0]) * 0.05
            if dist < threshold:
                file_key = self.map_index_lookup[idx]
                for r in range(self.lst.count()):
                    if self.lst.item(r).data(Qt.UserRole) == file_key:
                        self.lst.setCurrentRow(r)
                        break
        else:
            if e.button==1: 
                final_x, final_y = self.snap_coord if self.snap_coord else (e.xdata, e.ydata)
                self.waypoints.append([final_x, final_y])
            elif e.button==3 and self.waypoints: self.waypoints.pop()
            self.draw_map()

    # --- [최적화 적용된 draw_map 함수] ---
    def draw_map(self):
        self.ax_m.clear(); fix=self.ck_fix.isChecked(); yoff=0; self.map_marker=None; self.snap_marker=None
        self.map_coords_cache = []; all_coords_list = []; self.map_index_lookup = []
        
        if not fix and HAS_GEOPANDAS: 
            for layer in self.shapefile_layers:
                try:
                    # [최적화 3] rasterized=True 사용
                    layer['data'].plot(ax=self.ax_m, color=layer['color'], edgecolor='black', alpha=0.3, linewidth=1, rasterized=True)
                except Exception: pass

        for i in range(self.lst.count()):
            it = self.lst.item(i)
            if it.checkState()==Qt.Checked:
                key = it.data(Qt.UserRole)
                o = self.seismic_objects[key]
                if "Composite" in o.name: continue
                c = o.idx_coords.copy() if fix else o.real_coords
                if fix: c[:,1]+=yoff; yoff+=50
                
                # [최적화 4] Scatter 대신 Plot 사용 + 다운샘플링
                step = max(1, len(c)//3000)
                self.ax_m.plot(c[::step,0], c[::step,1], '-', lw=1, alpha=0.8, label=o.name)
                
                all_coords_list.append(c)
                self.map_index_lookup.extend([key] * len(c))
                
        if all_coords_list and HAS_SCIPY:
            self.map_coords_cache = np.vstack(all_coords_list)
            self.map_kdtree = cKDTree(self.map_coords_cache)
        else: self.map_kdtree = None

        if self.waypoints: 
            wp=np.array(self.waypoints)
            self.ax_m.plot(wp[:,0], wp[:,1], 'r-o', lw=2)
        
        self.ax_m.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        self.ax_m.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        
        self.fig_m.tight_layout()
        self.cv_m.draw()

    def load_segy(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Open", "", "SEGY (*.sgy *.segy)")
        if not fs: return
        sets = None
        for fn in fs:
            if fn in self.seismic_objects: continue
            if not sets:
                d = SegyHeaderDialog(fn, self)
                if d.exec()!=QDialog.Accepted: continue
                t = d.get_data(); 
                if t['all']: sets = t
            self.read_file(fn, sets if sets else t)
        self.draw_map()

    def read_file(self, fn, s):
        try:
            with segyio.open(fn, ignore_geometry=True) as f:
                d = f.trace.raw[:].T
                rx = f.attributes(s['x_b'])[:]; ry = f.attributes(s['y_b'])[:]
                try: cdps = f.attributes(s['cdp_b'])[:]
                except: cdps = np.arange(len(rx))+1
                if s['sc_m']=='n': rc=np.column_stack((rx,ry))
                elif s['sc_m']=='m': rc=np.column_stack((rx*s['mx'], ry*s['my']))
                else:
                    sc=f.attributes(s['sc_b'])[:]; sc=np.where(sc==0,1,sc)
                    xf=rx.astype(float); yf=ry.astype(float)
                    m=sc>0; xf[m]*=sc[m]; yf[m]*=sc[m]
                    d_=sc<0; dv=np.abs(sc[d_]); xf[d_]/=dv; yf[d_]/=dv
                    rc=np.column_stack((xf,yf))
                o = SeismicObject(fn, d, rc, cdps, s)
                self.seismic_objects[fn] = o
                it = QListWidgetItem(f"[{o.crs_name[:8]}] {o.name}")
                it.setData(Qt.UserRole, fn); it.setCheckState(Qt.Checked)
                self.lst.addItem(it)
        except Exception as e: QMessageBox.critical(self, "Err", str(e))
        self.sync_file_list()

    def create_composite(self):
        if len(self.waypoints)<2: return
        if not HAS_SCIPY: QMessageBox.warning(self,"Warning","Install scipy"); return
        use_idx=self.ck_fix.isChecked(); yoff=0; coords=[]; traces=[]; 
        max_ns = 0
        for i in range(self.lst.count()):
            it = self.lst.item(i)
            if it.checkState()==Qt.Checked:
                o = self.seismic_objects[it.data(Qt.UserRole)]
                if "Composite" in o.name: continue
                if o.raw_data.shape[0] > max_ns: max_ns = o.raw_data.shape[0]
                c = o.idx_coords.copy() if use_idx else o.real_coords
                if use_idx: c[:,1]+=yoff; yoff+=50
                coords.append(c); traces.append((o, len(c)))
        
        if not coords: return
        tree = cKDTree(np.vstack(coords))
        pts=[]; intersects=[]; cur=0
        for i in range(len(self.waypoints)-1):
            p1,p2 = np.array(self.waypoints[i]), np.array(self.waypoints[i+1])
            dist = np.linalg.norm(p2-p1); steps = int(max(dist/25.0, 5)) 
            seg = [p1+(p2-p1)*f for f in np.linspace(0,1,steps)]
            pts.extend(seg); cur+=len(seg); intersects.append(cur-1)
        
        dists, idxs = tree.query(np.array(pts))
        counts=[t[1] for t in traces]; cum=np.cumsum(counts); objs=[t[0] for t in traces]
        final=[]
        MAX_DIST = 2500.0
        gap_count = 0
        
        for i, gi in enumerate(idxs):
            if dists[i] > MAX_DIST: 
                final.append(np.zeros(max_ns)); gap_count += 1
            else:
                fi = np.searchsorted(cum, gi, side='right')
                li = gi if fi==0 else gi-cum[fi-1]
                raw = objs[fi].raw_data[:, li]
                if len(raw) == max_ns: final.append(raw)
                elif len(raw) < max_ns:
                    padded = np.zeros(max_ns); padded[:len(raw)] = raw; final.append(padded)
                else: final.append(raw[:max_ns])
            
        name = f"Composite_{len(self.seismic_objects)}"; s = objs[0].settings.copy(); s['crs']="Composite"
        co = SeismicObject(name, np.column_stack(final), pts, np.arange(len(pts)), s)
        co.intersections = intersects
        self.seismic_objects[name] = co
        it=QListWidgetItem(f"✂️ {name}"); it.setData(Qt.UserRole, name); it.setCheckState(Qt.Checked)
        self.lst.addItem(it); self.lst.setCurrentItem(it)
        self.sync_file_list()
        self.status.showMessage(f"Composite Created with {gap_count} gaps.", 5000)

    def sel_item(self):
        s = self.lst.selectedItems()
        if not s: return
        self.current_obj = self.seismic_objects[s[0].data(Qt.UserRole)]
        self.grp_ctrl.setEnabled(True)
        o = self.current_obj
        self.sl_c.blockSignals(True); self.sl_c.setValue(o.contrast); self.sl_c.blockSignals(False)
        self.ck_f.blockSignals(True); self.ck_f.setChecked(o.is_flipped); self.ck_f.blockSignals(False)
        self.sb_s.blockSignals(True); self.sb_s.setValue(o.shift_ms); self.sb_s.blockSignals(False)
        self.win_section.draw(o)
        self.sync_file_list()

    def chk_item(self, item): self.draw_map()
    def upd_view(self):
        if self.current_obj:
            o = self.current_obj; o.contrast=self.sl_c.value(); o.is_flipped=self.ck_f.isChecked(); o.shift_ms=self.sb_s.value()
            self.win_section.show_cdp=self.ck_c.isChecked()
            self.win_section.draw(o)
    def clr_hor(self):
        if self.current_obj: self.current_obj.horizons[self.cb_h.currentText()]['points']=[]; self.win_section.draw(self.current_obj)
    def remove_item(self):
        sel = self.lst.selectedItems()
        if not sel: return
        key = sel[0].data(Qt.UserRole); del self.seismic_objects[key]
        self.lst.takeItem(self.lst.row(sel[0]))
        self.current_obj = None; self.grp_ctrl.setEnabled(False); self.win_section.draw(None); self.draw_map()
        self.sync_file_list()
    def clear_all_files(self):
        if QMessageBox.question(self, "Clear", "Remove all?", QMessageBox.Yes|QMessageBox.No) == QMessageBox.Yes:
            self.seismic_objects.clear(); self.lst.clear(); self.current_obj=None; self.waypoints=[]; self.shapefile_layers=[]
            self.grp_ctrl.setEnabled(False); self.win_section.draw(None); self.draw_map()
            self.sync_file_list()
    def update_map_cursor(self, trace_idx, obj):
        use_idx = self.ck_fix.isChecked()
        if hasattr(obj, 'real_coords') and len(obj.real_coords) > trace_idx:
            if use_idx: cx, cy = obj.idx_coords[trace_idx]
            else: cx, cy = obj.real_coords[trace_idx]
            if not self.map_marker: self.map_marker, = self.ax_m.plot([cx], [cy], 'rX', markersize=12, markeredgewidth=2)
            else: self.map_marker.set_data([cx], [cy])
            self.cv_m.draw_idle()
    def clr_path(self): self.waypoints=[]; self.draw_map()
    def show_only_selected(self):
        sel = self.lst.selectedItems(); 
        if not sel: return
        target = sel[0].data(Qt.UserRole)
        for i in range(self.lst.count()):
            it = self.lst.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) == target else Qt.Unchecked)
        self.draw_map()
    def show_all_files(self):
        for i in range(self.lst.count()): self.lst.item(i).setCheckState(Qt.Checked)
        self.draw_map()
    def hide_all_files(self):
        for i in range(self.lst.count()): self.lst.item(i).setCheckState(Qt.Unchecked)
        self.draw_map()
    def on_map_hover(self, event):
        if not event.inaxes or not self.map_kdtree: 
            if self.snap_marker: self.snap_marker.set_data([], []); self.cv_m.draw_idle()
            self.snap_coord = None; return
        dist, idx = self.map_kdtree.query([event.xdata, event.ydata])
        xlim = self.ax_m.get_xlim(); threshold = (xlim[1] - xlim[0]) * 0.02 
        if dist < threshold:
            cx, cy = self.map_coords_cache[idx]; self.snap_coord = (cx, cy)
            if not self.snap_marker: self.snap_marker, = self.ax_m.plot([cx], [cy], 'go', markersize=8, markeredgecolor='black', alpha=0.7)
            else: self.snap_marker.set_data([cx], [cy])
            self.cv_m.draw_idle()
        else:
            self.snap_coord = None
            if self.snap_marker: self.snap_marker.set_data([], []); self.cv_m.draw_idle()
    def save_p(self):
        fn,_ = QFileDialog.getSaveFileName(self,"Save","","JSON (*.json)")
        if fn:
            d={'pts':self.waypoints,'fs':[]}
            for f,o in self.seismic_objects.items():
                if "Composite" not in f: d['fs'].append({'fn':f,'st':o.settings,'vp':{'c':o.contrast,'f':o.is_flipped,'s':o.shift_ms},'hz':o.horizons})
            with open(fn,'w') as f: json.dump(d,f)
            QMessageBox.information(self,"Saved","Done")
    def load_p(self):
        fn,_ = QFileDialog.getOpenFileName(self,"Load","","JSON (*.json)")
        if fn:
            with open(fn,'r') as f: d=json.load(f)
            self.waypoints=d['pts']; self.seismic_objects={}; self.lst.clear(); self.shapefile_layers=[]
            for e in d['fs']:
                if os.path.exists(e['fn']):
                    self.read_file(e['fn'], e['st'])
                    if e['fn'] in self.seismic_objects: 
                        o=self.seismic_objects[e['fn']]
                        o.contrast=e['vp']['c']; o.is_flipped=e['vp']['f']; o.shift_ms=e['vp']['s']; o.horizons=e['hz']
            self.draw_map()
            self.sync_file_list()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SegyViewer()
    window.show()
    sys.exit(app.exec())
