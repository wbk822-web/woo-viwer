import sys
import lasio
import pandas as pd
import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor, QBrush, QPen, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QFileDialog, QMessageBox, QLineEdit,
    QCheckBox, QLabel, QListWidgetItem, QFrame, QColorDialog,
    QComboBox, QTabWidget, QInputDialog, 
    QTreeWidget, QTreeWidgetItem, QScrollArea,
    QGroupBox, QSplitter, QFormLayout
)
from PySide6.QtCore import Qt

# 커브별 기본 색상 리스트
CURVE_COLORS = ['blue', 'red', 'green', 'cyan', 'magenta', 'orange', 'black'] 

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("LAS 분석기 (Final Fix - Ghosting & Tooltip)")
        self.setGeometry(100, 100, 1400, 900)
        
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.main_splitter)

        # --- 왼쪽 패널 ---
        self.tab_widget = QTabWidget()
        
        self.track_tab = QWidget()
        self.setup_tracks_tab_layout() 
        self.tab_widget.addTab(self.track_tab, "Tracks")
        
        self.top_tab = QWidget()
        self.setup_top_tab_layout() 
        self.tab_widget.addTab(self.top_tab, "Tops")

        self.calc_tab = QWidget()
        self.setup_calc_tab_layout()
        self.tab_widget.addTab(self.calc_tab, "Interpretation")
        
        self.main_splitter.addWidget(self.tab_widget)
        
        # --- 오른쪽 플롯 영역 ---
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('k') 
        self.main_splitter.addWidget(self.plot_widget)
        
        self.main_splitter.setSizes([450, 950]) 
        
        # --- 데이터 ---
        self.las_data = None
        self.data_df = None 
        self.all_curve_names = [] 
        self.tracks_model = {} 
        self.plot_tracks = {} 
        self.well_tops = {} 
        
        # [중요] Scale 2 ViewBox들을 추적하여 삭제하기 위한 리스트
        self.secondary_views = [] 

        # --- 마우스 이벤트 ---
        self.mouse_proxy_move = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved, 
            rateLimit=60, 
            slot=self.mouse_moved_across_plots
        )
        self.mouse_proxy_click = pg.SignalProxy(
            self.plot_widget.scene().sigMouseClicked,
            slot=self.on_plot_clicked 
        )

    # -----------------------------------------------------------------
    # UI Setup
    # -----------------------------------------------------------------
    def setup_tracks_tab_layout(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); self.tracks_layout = QVBoxLayout(content)

        self.load_btn = QPushButton("1. LAS 파일 열기"); self.load_btn.clicked.connect(self.load_las_file)
        self.tracks_layout.addWidget(self.load_btn); self.tracks_layout.addWidget(self.create_separator())

        grp_track = QGroupBox("2. 트랙 관리")
        ly_track = QVBoxLayout(grp_track)
        self.track_list = QTreeWidget(); self.track_list.setHeaderHidden(True)
        self.track_list.currentItemChanged.connect(self.on_track_selection_changed)
        ly_track.addWidget(self.track_list)
        ly_btn = QHBoxLayout()
        btn_add = QPushButton("[+] 트랙 추가"); btn_add.clicked.connect(self.on_add_track)
        btn_del = QPushButton("[-] 트랙 삭제"); btn_del.clicked.connect(self.on_delete_track)
        ly_btn.addWidget(btn_add); ly_btn.addWidget(btn_del); ly_track.addLayout(ly_btn)
        self.tracks_layout.addWidget(grp_track)

        grp_assign = QGroupBox("3. 커브 할당 & 축 이동")
        ly_assign = QVBoxLayout(grp_assign)
        
        ly_lists = QHBoxLayout()
        v1 = QVBoxLayout(); v1.addWidget(QLabel("Available:")); self.list_avail = QListWidget(); self.list_avail.setSelectionMode(QListWidget.ExtendedSelection); v1.addWidget(self.list_avail); ly_lists.addLayout(v1)
        v_btns = QVBoxLayout(); v_btns.addStretch()
        btn_to_right = QPushButton(">>"); btn_to_right.clicked.connect(self.on_assign_curve)
        btn_to_left = QPushButton("<<"); btn_to_left.clicked.connect(self.on_unassign_curve)
        v_btns.addWidget(btn_to_right); v_btns.addWidget(btn_to_left); v_btns.addStretch(); ly_lists.addLayout(v_btns)
        v2 = QVBoxLayout(); v2.addWidget(QLabel("Assigned [Scale No.]:")); self.list_assigned = QListWidget(); v2.addWidget(self.list_assigned); ly_lists.addLayout(v2)
        ly_assign.addLayout(ly_lists)

        ly_props = QHBoxLayout()
        btn_col = QPushButton("색상 변경"); btn_col.clicked.connect(self.on_change_curve_color)
        self.btn_axis = QPushButton("▼ 축 이동 (Scale 1 ↔ 2)")
        self.btn_axis.setStyleSheet("font-weight: bold; color: #0055ff;")
        self.btn_axis.clicked.connect(self.on_toggle_curve_axis)
        ly_props.addWidget(btn_col); ly_props.addWidget(self.btn_axis)
        ly_assign.addLayout(ly_props)
        self.tracks_layout.addWidget(grp_assign)

        self.grp_settings = QGroupBox("4. 트랙 설정 (Dual Scale)"); ly_set = QVBoxLayout(self.grp_settings)
        
        g1 = QGroupBox("Scale 1 (Top Axis)"); ly_g1 = QVBoxLayout(g1)
        self.lbl_linked1 = QLabel("연동된 커브: -"); self.lbl_linked1.setStyleSheet("color: blue; font-weight: bold;")
        ly_g1.addWidget(self.lbl_linked1)
        l1 = QHBoxLayout()
        self.min1 = QLineEdit("0"); self.max1 = QLineEdit("100"); self.log1 = QCheckBox("Log")
        l1.addWidget(QLabel("Min:")); l1.addWidget(self.min1); l1.addWidget(QLabel("Max:")); l1.addWidget(self.max1); l1.addWidget(self.log1)
        ly_g1.addLayout(l1); ly_set.addWidget(g1)

        g2 = QGroupBox("Scale 2 (Bottom Axis - Dashed)"); ly_g2 = QVBoxLayout(g2)
        self.lbl_linked2 = QLabel("연동된 커브: -"); self.lbl_linked2.setStyleSheet("color: red; font-weight: bold;")
        ly_g2.addWidget(self.lbl_linked2)
        l2 = QHBoxLayout()
        self.min2 = QLineEdit("0"); self.max2 = QLineEdit("100"); self.log2 = QCheckBox("Log")
        l2.addWidget(QLabel("Min:")); l2.addWidget(self.min2); l2.addWidget(QLabel("Max:")); l2.addWidget(self.max2); l2.addWidget(self.log2)
        ly_g2.addLayout(l2); ly_set.addWidget(g2)

        g_fill = QGroupBox("Fill Settings"); ly_fill = QVBoxLayout(g_fill)
        h_fill = QHBoxLayout()
        self.chk_fill = QCheckBox("Enable"); self.cmb_fill_type = QComboBox(); self.cmb_fill_type.addItems(["Baseline", "Curve-Curve"])
        self.cmb_fill_type.currentTextChanged.connect(self.on_fill_type_changed)
        h_fill.addWidget(self.chk_fill); h_fill.addWidget(self.cmb_fill_type); ly_fill.addLayout(h_fill)
        h_fill2 = QHBoxLayout()
        self.lbl_ref = QLabel("Ref:"); self.txt_ref = QLineEdit("0.0")
        self.lbl_target = QLabel("To:"); self.cmb_target = QComboBox()
        self.btn_fill_col = QPushButton("Color"); self.btn_fill_col.clicked.connect(self.open_color_picker)
        self.lbl_fill_prev = QLabel(); self.lbl_fill_prev.setFixedSize(20,20); self.cur_fill_col = QColor("yellow"); self.update_fill_prev()
        h_fill2.addWidget(self.lbl_ref); h_fill2.addWidget(self.txt_ref); h_fill2.addWidget(self.lbl_target); h_fill2.addWidget(self.cmb_target); h_fill2.addWidget(self.btn_fill_col); h_fill2.addWidget(self.lbl_fill_prev)
        ly_fill.addLayout(h_fill2)
        ly_set.addWidget(g_fill)

        btn_apply = QPushButton("설정 적용 (Apply Plots)"); btn_apply.setStyleSheet("background-color: #dddddd; font-weight: bold; padding: 5px;")
        btn_apply.clicked.connect(self.on_apply_settings)
        ly_set.addWidget(btn_apply)

        self.tracks_layout.addWidget(self.grp_settings)
        self.tracks_layout.addStretch()
        scroll.setWidget(content); self.tracks_layout.setContentsMargins(0,0,0,0)
        self.track_tab.setLayout(QVBoxLayout()); self.track_tab.layout().addWidget(scroll)
        self.grp_settings.setEnabled(False); self.on_fill_type_changed("Baseline")

    def setup_top_tab_layout(self):
        ly = QVBoxLayout(self.top_tab); ly.addWidget(QLabel("Well Top 목록"))
        self.tree_tops = QTreeWidget(); self.tree_tops.setColumnCount(2); self.tree_tops.setHeaderLabels(["Name", "Depth"])
        self.tree_tops.itemChanged.connect(self.on_top_changed)
        ly.addWidget(self.tree_tops)
        btn_del = QPushButton("선택 삭제"); btn_del.clicked.connect(self.del_top); ly.addWidget(btn_del)

    def setup_calc_tab_layout(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True); content = QWidget(); ly = QVBoxLayout(content)
        grp_archie = QGroupBox("Archie Water Saturation"); ly_archie = QVBoxLayout(grp_archie)
        f_c = QFormLayout(); self.cmb_rt = QComboBox(); self.cmb_phi = QComboBox()
        f_c.addRow("Rt:", self.cmb_rt); f_c.addRow("Phi:", self.cmb_phi)
        self.chk_phi_perc = QCheckBox("Phi is %"); ly_archie.addLayout(f_c); ly_archie.addWidget(self.chk_phi_perc)
        l_p = QHBoxLayout(); self.txt_a = QLineEdit("1"); self.txt_m = QLineEdit("2"); self.txt_n = QLineEdit("2"); self.txt_rw = QLineEdit("0.1")
        for l, w in [("a",self.txt_a),("m",self.txt_m),("n",self.txt_n),("Rw",self.txt_rw)]: l_p.addWidget(QLabel(l)); l_p.addWidget(w)
        ly_archie.addLayout(l_p); btn = QPushButton("Calculate Sw"); btn.clicked.connect(self.run_archie_calc); ly_archie.addWidget(btn)
        
        grp_gen = QGroupBox("General Formula"); ly_gen = QVBoxLayout(grp_gen)
        f_g = QFormLayout(); self.txt_new_name = QLineEdit("NewC"); self.txt_formula = QLineEdit()
        f_g.addRow("Name:", self.txt_new_name); f_g.addRow("Formula:", self.txt_formula)
        btn_g = QPushButton("Run Formula"); btn_g.clicked.connect(self.run_general_calc)
        ly_gen.addLayout(f_g); ly_gen.addWidget(btn_g)
        ly.addWidget(grp_archie); ly.addWidget(grp_gen); ly.addStretch(); scroll.setWidget(content)
        self.calc_tab.setLayout(QVBoxLayout()); self.calc_tab.layout().addWidget(scroll)

    def create_separator(self):
        l = QFrame(); l.setFrameShape(QFrame.HLine); l.setFrameShadow(QFrame.Sunken); return l

    # -----------------------------------------------------------------
    # Logic
    # -----------------------------------------------------------------
    def load_las_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open LAS", "", "LAS Files (*.las)")
        if not path: return
        try:
            las = lasio.read(path)
            self.las_data = las
            self.data_df = las.df().reset_index().set_index(las.curves[0].mnemonic)
            
            # [수정] 0 이하의 값 NaN 처리 (왼쪽 수직선 제거)
            self.data_df[self.data_df <= 0] = np.nan

            self.all_curve_names = [c for c in self.data_df.columns if c != las.curves[0].mnemonic]
            self.tracks_model.clear(); self.well_tops.clear(); self.track_list.clear(); self.grp_settings.setEnabled(False)
            self.list_assigned.clear(); self.refresh_ui_lists()
        except Exception as e: QMessageBox.critical(self, "오류", str(e))

    def refresh_ui_lists(self):
        cur_track = self.track_list.currentItem(); assigned_in_track = []
        if cur_track:
             t_name = cur_track.data(0, Qt.UserRole)
             if t_name in self.tracks_model: assigned_in_track = list(self.tracks_model[t_name]["curves"].keys())
        self.list_avail.clear(); self.list_avail.addItems([c for c in self.all_curve_names if c not in assigned_in_track])
        self.cmb_target.clear(); self.cmb_target.addItems(assigned_in_track)
        self.cmb_rt.clear(); self.cmb_rt.addItems(self.all_curve_names)
        self.cmb_phi.clear(); self.cmb_phi.addItems(self.all_curve_names)
        self.refresh_linked_curves_label()

    def refresh_linked_curves_label(self):
        cur = self.track_list.currentItem()
        if not cur:
            self.lbl_linked1.setText("연동된 커브: -"); self.lbl_linked2.setText("연동된 커브: -"); return
        t_name = cur.data(0, Qt.UserRole)
        curves = self.tracks_model.get(t_name, {}).get("curves", {})
        c1 = [k for k, v in curves.items() if v.get("axis", 1) == 1]
        c2 = [k for k, v in curves.items() if v.get("axis", 1) == 2]
        self.lbl_linked1.setText(f"연동된 커브 (Top): {', '.join(c1) if c1 else '없음'}")
        self.lbl_linked2.setText(f"연동된 커브 (Bot): {', '.join(c2) if c2 else '없음'}")

    def on_add_track(self):
        cnt=len(self.tracks_model)+1; name=f"Track-{cnt}"
        while name in self.tracks_model: cnt+=1; name=f"Track-{cnt}"
        self.tracks_model[name] = {"r1": {"min":0, "max":100, "log":False}, "r2": {"min":0, "max":100, "log":False}, "curves": {}, "fill": {"en":False, "type":"Baseline", "lev":0.0, "tgt":"", "col":"#FFFF00"}}
        item = QTreeWidgetItem([name]); item.setData(0, Qt.UserRole, name)
        self.track_list.addTopLevelItem(item); self.track_list.setCurrentItem(item); self.update_plots()

    def on_delete_track(self):
        cur=self.track_list.currentItem()
        if cur: del self.tracks_model[cur.data(0, Qt.UserRole)]; self.track_list.takeTopLevelItem(self.track_list.indexOfTopLevelItem(cur)); self.update_plots()

    def on_track_selection_changed(self, cur, prev):
        if not cur: self.grp_settings.setEnabled(False); self.refresh_ui_lists(); self.list_assigned.clear(); return
        self.grp_settings.setEnabled(True); name=cur.data(0, Qt.UserRole); data=self.tracks_model[name]
        self.min1.setText(str(data["r1"]["min"])); self.max1.setText(str(data["r1"]["max"])); self.log1.setChecked(data["r1"]["log"])
        self.min2.setText(str(data["r2"]["min"])); self.max2.setText(str(data["r2"]["max"])); self.log2.setChecked(data["r2"]["log"])
        fs=data["fill"]; self.chk_fill.setChecked(fs["en"]); self.cmb_fill_type.setCurrentText(fs["type"])
        self.on_fill_type_changed(fs["type"]); self.txt_ref.setText(str(fs["lev"])); self.cur_fill_col=QColor(fs["col"]); self.update_fill_prev()
        self.cmb_target.setCurrentText(fs["tgt"])
        self.list_assigned.clear()
        for c, p in data["curves"].items():
            it = QListWidgetItem(f"[Scale {p.get('axis',1)}] {c}")
            it.setForeground(QBrush(QColor(p["color"])))
            it.setData(Qt.UserRole, c)
            self.list_assigned.addItem(it)
        self.refresh_ui_lists()

    def on_assign_curve(self):
        cur=self.track_list.currentItem(); sels=self.list_avail.selectedItems()
        if not cur or not sels: return
        td=self.tracks_model[cur.data(0, Qt.UserRole)]
        if len(td["curves"])+len(sels)>7: return
        for i, s in enumerate(sels):
            cn=s.text(); idx=(len(td["curves"])+i)%len(CURVE_COLORS)
            td["curves"][cn]={"color":CURVE_COLORS[idx], "axis":1}
        self.on_track_selection_changed(cur,None); self.update_plots()

    def on_unassign_curve(self):
        cur=self.track_list.currentItem(); sel=self.list_assigned.currentItem()
        if cur and sel: del self.tracks_model[cur.data(0, Qt.UserRole)]["curves"][sel.data(Qt.UserRole)]; self.on_track_selection_changed(cur,None); self.update_plots()

    def on_change_curve_color(self):
        cur_t=self.track_list.currentItem(); cur_c=self.list_assigned.currentItem()
        if cur_t and cur_c:
            t_nm=cur_t.data(0, Qt.UserRole); c_nm=cur_c.data(Qt.UserRole)
            col=QColorDialog.getColor(QColor(self.tracks_model[t_nm]["curves"][c_nm]["color"]), self)
            if col.isValid(): self.tracks_model[t_nm]["curves"][c_nm]["color"]=col.name(); self.on_track_selection_changed(cur_t,None); self.update_plots()

    def on_toggle_curve_axis(self):
        cur_t=self.track_list.currentItem(); cur_c=self.list_assigned.currentItem()
        if not cur_t or not cur_c: QMessageBox.warning(self, "알림", "커브를 선택하세요."); return
        t_nm=cur_t.data(0, Qt.UserRole); c_nm=cur_c.data(Qt.UserRole)
        p = self.tracks_model[t_nm]["curves"][c_nm]
        p["axis"] = 2 if p.get("axis", 1) == 1 else 1
        self.on_track_selection_changed(cur_t,None); self.update_plots()

    def on_apply_settings(self):
        cur=self.track_list.currentItem()
        if cur:
            nm=cur.data(0, Qt.UserRole)
            try:
                self.tracks_model[nm]["r1"]={"min":float(self.min1.text()), "max":float(self.max1.text()), "log":self.log1.isChecked()}
                self.tracks_model[nm]["r2"]={"min":float(self.min2.text()), "max":float(self.max2.text()), "log":self.log2.isChecked()}
                self.tracks_model[nm]["fill"]={"en":self.chk_fill.isChecked(), "type":self.cmb_fill_type.currentText(), "lev":float(self.txt_ref.text()), "tgt":self.cmb_target.currentText(), "col":self.cur_fill_col.name()}
                self.update_plots()
            except: pass

    def on_fill_type_changed(self, txt):
        if txt == "Baseline": self.lbl_ref.show(); self.txt_ref.show(); self.lbl_target.hide(); self.cmb_target.hide()
        else: self.lbl_ref.hide(); self.txt_ref.hide(); self.lbl_target.show(); self.cmb_target.show()
    def update_fill_prev(self): self.lbl_fill_prev.setStyleSheet(f"background:{self.cur_fill_col.name()}; border:1px solid #333")
    def open_color_picker(self):
        c=QColorDialog.getColor(self.cur_fill_col, self)
        if c.isValid(): self.cur_fill_col=c; self.update_fill_prev()
    def add_curve_to_data(self, n, d): self.data_df[n]=d; self.all_curve_names.append(n); self.refresh_ui_lists()
    def run_archie_calc(self):
        try:
            a=float(self.txt_a.text()); m=float(self.txt_m.text()); n=float(self.txt_n.text()); rw=float(self.txt_rw.text())
            phi=self.data_df[self.cmb_phi.currentText()].copy(); rt=self.data_df[self.cmb_rt.currentText()]
            if self.chk_phi_perc.isChecked(): phi=phi/100.0
            sw=((a*rw)/(phi**m*rt))**(1.0/n); self.add_curve_to_data("Sw", sw.clip(0,1))
        except: pass
    def run_general_calc(self):
        try: self.add_curve_to_data(self.txt_new_name.text(), self.data_df.eval(self.txt_formula.text()))
        except: pass
    def on_top_changed(self, i, c): self.update_plots() 
    def del_top(self): self.update_plots() 
    def on_plot_clicked(self, evt): 
        pos=evt[0].scenePos()
        for p in self.plot_tracks.values():
            if p.vb.sceneBoundingRect().contains(pos):
                t, ok=QInputDialog.getText(self, "New Top", "Name:")
                if ok and t: self.well_tops[t]=p.vb.mapSceneToView(pos).y(); self.update_plots()

    # -----------------------------------------------------------------
    # Plotting & Hovering (Fix Ghosting & Tooltip)
    # -----------------------------------------------------------------
    def mouse_moved_across_plots(self, evt):
        pos = evt[0] 
        for track_name, plot_item in self.plot_tracks.items():
            if plot_item.vb.sceneBoundingRect().contains(pos):
                mouse_point = plot_item.vb.mapSceneToView(pos)
                cursor_depth = mouse_point.y()
                
                if hasattr(plot_item, 'crosshairs'):
                    v_line, h_line, label_item = plot_item.crosshairs
                    v_line.setPos(mouse_point.x()); h_line.setPos(cursor_depth)
                    
                    try:
                        idx_loc = self.data_df.index.get_indexer([cursor_depth], method='nearest')[0]
                        actual_depth = self.data_df.index[idx_loc]
                        
                        tooltip_html = f"<div style='background-color:rgba(0,0,0,0.7); padding:3px;'>"
                        tooltip_html += f"<span style='color: white; font-weight: bold;'>Depth: {actual_depth:.2f}</span><br>"
                        
                        if track_name in self.tracks_model:
                            curves = self.tracks_model[track_name]["curves"]
                            for c_name, props in curves.items():
                                if c_name in self.data_df.columns:
                                    val = self.data_df[c_name].iloc[idx_loc]
                                    col = props['color']
                                    ax = props.get("axis", 1)
                                    tooltip_html += f"<span style='color: {col};'>[S{ax}] {c_name}: {val:.4f}</span><br>"
                        tooltip_html += "</div>"
                        
                        label_item.setHtml(tooltip_html)
                        label_item.setPos(mouse_point.x(), cursor_depth) 
                        v_line.show(); h_line.show(); label_item.show()
                    except: pass
            else:
                if hasattr(plot_item, 'crosshairs'):
                    v_line, h_line, label_item = plot_item.crosshairs
                    v_line.hide(); h_line.hide(); label_item.hide()

    def update_plots(self):
        # [중요] 기존 플롯 클리어
        self.plot_widget.clear(); self.plot_tracks.clear()
        
        # [핵심] 유령 ViewBox 제거: 기존에 생성된 Scale 2 뷰박스들을 scene에서 강제 삭제
        if hasattr(self, 'secondary_views'):
            for v in self.secondary_views:
                if v.scene(): v.scene().removeItem(v)
            self.secondary_views.clear()
        else:
            self.secondary_views = []

        if self.data_df is None: return
        depths = self.data_df.index.values
        min_depth = self.data_df.index.min(); max_depth = self.data_df.index.max()
        first_p = None; c_idx = 0
        
        for name, data in self.tracks_model.items():
            # 1. Main Plot (Scale 1)
            p1 = self.plot_widget.addPlot(row=0, col=c_idx); p1.showAxis('top', True); p1.showAxis('bottom', False)
            p1.setLabel('top', name); p1.invertY(True)
            p1.showGrid(x=True, y=True, alpha=0.3)
            p1.setYRange(min_depth, max_depth) # Depth Range Fix
            
            r1 = data["r1"]
            try: p1.setXRange(r1["min"], r1["max"]); p1.setLogMode(r1["log"], False)
            except: pass
            
            if not first_p: first_p = p1; p1.setLabel('left', 'Depth')
            else: p1.setYLink(first_p); p1.showAxis('left', False)

            v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=1, style=Qt.DashLine))
            h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('w', width=1, style=Qt.DashLine))
            label_item = pg.TextItem(anchor=(0, 1)); label_item.setZValue(100) 
            p1.addItem(v_line); p1.addItem(h_line); p1.addItem(label_item)
            v_line.hide(); h_line.hide(); label_item.hide()
            p1.crosshairs = (v_line, h_line, label_item)
            
            curves1 = {k:v for k,v in data["curves"].items() if v.get("axis", 1)==1}
            curves2 = {k:v for k,v in data["curves"].items() if v.get("axis", 1)==2}
            plot_items = {}

            # Draw Scale 1
            for c, p in curves1.items():
                if c in self.data_df:
                    item = p1.plot(self.data_df[c].values, depths, pen=pg.mkPen(p["color"], width=2))
                    plot_items[c] = item
            
            # Draw Scale 2 (Overlay)
            if curves2:
                v2 = pg.ViewBox()
                # [핵심] 리스트에 추가해서 나중에 지울 수 있게 함
                self.secondary_views.append(v2)
                p1.scene().addItem(v2)
                p1.getAxis('bottom').linkToView(v2)
                v2.setYLink(p1) # Y축만 링크
                
                p1.showAxis('bottom', True); p1.getAxis('bottom').setLabel(f"{name} (Scale 2)")
                r2 = data["r2"]
                try: v2.setXRange(r2["min"], r2["max"]) 
                except: pass
                
                def update_v2_geometry():
                    v2.setGeometry(p1.vb.sceneBoundingRect())
                    v2.setYRange(p1.vb.viewRange()[1][0], p1.vb.viewRange()[1][1], padding=0)
                p1.vb.sigResized.connect(update_v2_geometry)
                update_v2_geometry() 
                
                for c, p in curves2.items():
                    if c in self.data_df:
                        pen = pg.mkPen(p["color"], width=2, style=Qt.DashLine)
                        item = pg.PlotCurveItem(self.data_df[c].values, depths, pen=pen)
                        v2.addItem(item)

            for t, d in self.well_tops.items(): p1.addItem(pg.InfiniteLine(pos=d, angle=0, pen='r'))

            fs = data["fill"]
            if fs["en"] and curves1:
                try:
                    c1 = list(curves1.keys())[0]; i1 = plot_items.get(c1)
                    if i1:
                        b = pg.mkBrush(fs["col"])
                        if fs["type"]=="Baseline": i1.setFillLevel(fs["lev"]); i1.setFillBrush(b)
                        elif fs["type"]=="Curve-Curve":
                            tgt=fs["tgt"]; i2=plot_items.get(tgt)
                            if i2 and c1!=tgt: p1.addItem(pg.FillBetweenItem(i1, i2, brush=b))
                except: pass

            self.plot_tracks[name] = p1; c_idx += 1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())
