import sys
import lasio
import pandas as pd
import pyqtgraph as pg
from scipy.signal import savgol_filter 

try:
    from sklearn.ensemble import RandomForestRegressor
    import joblib
except ImportError:
    print("오류: 'scikit-learn' 또는 'joblib' 라이브러리가 없습니다.")
    print("터미널에서 'pip install scikit-learn joblib'을 실행하세요.")
    sys.exit()

from PySide6.QtGui import QColor, QBrush, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QFileDialog, QMessageBox, QLineEdit,
    QCheckBox, QLabel, QListWidgetItem, QFrame, QColorDialog,
    QComboBox, QTabWidget, QInputDialog, 
    QTreeWidget, QTreeWidgetItem, QScrollArea,
    QGroupBox, QSplitter 
)
from PySide6.QtCore import Qt, QSize

# 커브별 기본 색상 리스트
CURVE_COLORS = ['b', 'r', 'g', 'c', 'm', 'y', 'k'] # (b)lue, (r)ed, (g)reen, (c)yan, (m)agenta, (y)ellow, (b)lack

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Woo 테크로그")
        self.setGeometry(200, 200, 1200, 800)
        
        # 1. QSplitter (창 너비 조절) 도입
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.main_splitter)

        # --- 2. 왼쪽: 탭 위젯 ---
        self.tab_widget = QTabWidget()
        
        self.track_tab = QWidget()
        self.setup_tracks_tab_layout() # (수정) '색상 변경' 버튼 추가됨
        self.tab_widget.addTab(self.track_tab, "Tracks")
        
        self.top_tab = QWidget()
        self.setup_top_tab_layout() 
        self.tab_widget.addTab(self.top_tab, "Tops")
        
        self.ml_tab = QWidget()
        self.setup_ml_tab_layout()
        self.tab_widget.addTab(self.ml_tab, "AI / ML")
        
        self.main_splitter.addWidget(self.tab_widget) # Splitter에 왼쪽 탭 추가
        
        # --- 3. 오른쪽: 플롯 영역 ---
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.main_splitter.addWidget(self.plot_widget) # Splitter에 오른쪽 플롯 추가
        
        self.main_splitter.setSizes([300, 900]) # 초기 크기 (이전 250에서 300으로 늘림)
        
        # --- 4. 클래스 변수 ---
        self.las_data = None
        self.data_df = None 
        self.all_curve_names = [] 
        self.tracks_model = {} # (수정) 데이터 모델 구조 변경됨
        self.plot_tracks = {} 
        self.plot_data_items = {}
        self.well_tops = {} 
        self.current_ml_model = None 

        # --- 5. 마우스 이벤트 프록시 ---
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
    # ##### (수정) 'Tracks' 탭 - '색상 변경' 버튼 추가 #####
    # -----------------------------------------------------------------
    def setup_tracks_tab_layout(self):
        """'Tracks' 탭의 UI 레이아웃을 '트랙 중심'으로 새로 설계합니다."""
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content_widget = QWidget()
        self.tracks_layout = QVBoxLayout(scroll_content_widget)

        # --- 0. LAS 파일 로드 ---
        self.load_button = QPushButton("1. LAS 파일 열기")
        self.load_button.clicked.connect(self.load_las_file)
        self.tracks_layout.addWidget(self.load_button)
        self.tracks_layout.addWidget(self.create_separator())

        # --- 1. 트랙 관리 패널 ---
        track_mgmt_group = QGroupBox("2. 트랙 관리")
        track_mgmt_layout = QVBoxLayout(track_mgmt_group)
        self.track_list_widget = QTreeWidget()
        self.track_list_widget.setHeaderHidden(True)
        self.track_list_widget.currentItemChanged.connect(self.on_track_selection_changed)
        track_mgmt_layout.addWidget(self.track_list_widget)
        track_btn_layout = QHBoxLayout()
        self.add_track_btn = QPushButton("[+ 새 트랙]")
        self.add_track_btn.clicked.connect(self.on_add_track)
        self.del_track_btn = QPushButton("[- 트랙 삭제]")
        self.del_track_btn.clicked.connect(self.on_delete_track)
        track_btn_layout.addWidget(self.add_track_btn)
        track_btn_layout.addWidget(self.del_track_btn)
        track_mgmt_layout.addLayout(track_btn_layout)
        self.tracks_layout.addWidget(track_mgmt_group)

        # --- 2. 커브 할당 패널 ---
        curve_assign_group = QGroupBox("3. 커브 할당")
        curve_assign_layout = QHBoxLayout(curve_assign_group)
        available_layout = QVBoxLayout()
        available_layout.addWidget(QLabel("Available Curves:"))
        self.available_curves_list = QListWidget()
        self.available_curves_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        available_layout.addWidget(self.available_curves_list)
        curve_assign_layout.addLayout(available_layout)
        assign_btn_layout = QVBoxLayout()
        assign_btn_layout.addStretch()
        self.assign_curve_btn = QPushButton(">>")
        self.assign_curve_btn.clicked.connect(self.on_assign_curve)
        self.unassign_curve_btn = QPushButton("<<")
        self.unassign_curve_btn.clicked.connect(self.on_unassign_curve)
        assign_btn_layout.addWidget(self.assign_curve_btn)
        assign_btn_layout.addWidget(self.unassign_curve_btn)
        assign_btn_layout.addStretch()
        curve_assign_layout.addLayout(assign_btn_layout)
        assigned_layout = QVBoxLayout()
        assigned_layout.addWidget(QLabel("Assigned Curves (Max 7):"))
        self.assigned_curves_list = QListWidget()
        self.assigned_curves_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        assigned_layout.addWidget(self.assigned_curves_list)
        
        # (신규) 개별 커브 색상 변경 버튼
        self.change_color_btn = QPushButton("선택 커브 색상 변경...")
        self.change_color_btn.clicked.connect(self.on_change_curve_color)
        assigned_layout.addWidget(self.change_color_btn)
        
        curve_assign_layout.addLayout(assigned_layout)
        self.tracks_layout.addWidget(curve_assign_group)

        # --- 3. 트랙 설정 패널 ---
        self.track_settings_group = QGroupBox("4. 트랙 설정 (선택된 트랙)")
        track_settings_layout = QVBoxLayout(self.track_settings_group)
        track_settings_layout.addWidget(QLabel("값 범위 (Min):"))
        self.track_min = QLineEdit("0")
        track_settings_layout.addWidget(self.track_min)
        track_settings_layout.addWidget(QLabel("값 범위 (Max):"))
        self.track_max = QLineEdit("100") 
        track_settings_layout.addWidget(self.track_max)
        self.track_log = QCheckBox("로그(Log) 스케일") 
        track_settings_layout.addWidget(self.track_log)
        track_settings_layout.addWidget(self.create_separator())
        self.fill_check = QCheckBox("색상 채우기 (Fill)")
        track_settings_layout.addWidget(self.fill_check)
        self.fill_type_combo = QComboBox()
        self.fill_type_combo.addItems(["기준값 (Baseline)", "커브 간 (Curve-Curve)"])
        self.fill_type_combo.currentTextChanged.connect(self.on_fill_type_changed)
        track_settings_layout.addWidget(self.fill_type_combo)
        self.fill_ref_label = QLabel("기준값 (Fill Level):")
        self.fill_ref_input = QLineEdit("0.0")
        track_settings_layout.addWidget(self.fill_ref_label)
        track_settings_layout.addWidget(self.fill_ref_input)
        self.fill_target_label = QLabel("대상 커브 (트랙 내):")
        self.fill_target_combo = QComboBox()
        track_settings_layout.addWidget(self.fill_target_label)
        track_settings_layout.addWidget(self.fill_target_combo)
        color_layout = QHBoxLayout()
        self.fill_color_button = QPushButton("채우기 색상 선택...")
        self.fill_color_button.clicked.connect(self.open_color_picker)
        color_layout.addWidget(self.fill_color_button)
        self.fill_color_preview = QLabel()
        self.fill_color_preview.setFixedSize(30, 30)
        self.current_fill_color = QColor('#FFFF00')
        self.update_color_preview()
        color_layout.addWidget(self.fill_color_preview)
        track_settings_layout.addLayout(color_layout)
        self.apply_button = QPushButton("트랙 설정 적용")
        self.apply_button.clicked.connect(self.on_apply_track_settings) 
        track_settings_layout.addWidget(self.apply_button)
        self.tracks_layout.addWidget(self.track_settings_group)
        self.tracks_layout.addStretch()
        scroll_area.setWidget(scroll_content_widget)
        main_track_tab_layout = QVBoxLayout(self.track_tab)
        main_track_tab_layout.addWidget(scroll_area)
        main_track_tab_layout.setContentsMargins(0,0,0,0)
        self.track_settings_group.setEnabled(False)
        self.on_fill_type_changed("기준값 (Baseline)")

    # --- (이하 탭들은 14단계와 동일) ---
    def setup_top_tab_layout(self):
        self.top_layout = QVBoxLayout(self.top_tab)
        self.top_layout.addWidget(QLabel("Well Top 목록 (차트 클릭: 추가 / 더블클릭: 수정)"))
        self.top_tree_widget = QTreeWidget()
        self.top_tree_widget.setColumnCount(2)
        self.top_tree_widget.setHeaderLabels(["Well Top 이름", "Depth"])
        self.top_tree_widget.setColumnWidth(0, 150)
        self.top_tree_widget.itemChanged.connect(self.on_top_item_changed)
        self.top_layout.addWidget(self.top_tree_widget)
        self.delete_top_button = QPushButton("선택한 Top 삭제")
        self.delete_top_button.clicked.connect(self.delete_selected_top)
        self.top_layout.addWidget(self.delete_top_button)
        self.top_layout.addStretch()
    def setup_ml_tab_layout(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content_widget = QWidget()
        self.ml_layout = QVBoxLayout(scroll_content_widget)
        self.ml_layout.addWidget(QLabel("--- 1. 피처 선택 ---"))
        self.ml_layout.addWidget(QLabel("Input 커브 (X): (다중 선택 가능)"))
        self.ml_input_list = QListWidget()
        self.ml_input_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection) 
        self.ml_input_list.setMinimumHeight(100) 
        self.ml_layout.addWidget(self.ml_input_list)
        self.ml_layout.addWidget(QLabel("Target 커브 (Y): (예측할 커브)"))
        self.ml_target_combo = QComboBox()
        self.ml_layout.addWidget(self.ml_target_combo)
        self.ml_layout.addWidget(self.create_separator())
        self.ml_layout.addWidget(QLabel("--- 2. 계산 & 스무딩 ---"))
        self.ml_layout.addWidget(QLabel("수식 계산 (예: (GR-20)/100)"))
        self.ml_formula_input = QLineEdit()
        self.ml_layout.addWidget(self.ml_formula_input)
        self.ml_layout.addWidget(QLabel("새 커브 이름:"))
        self.ml_formula_name_input = QLineEdit("VSHALE")
        self.ml_layout.addWidget(self.ml_formula_name_input)
        self.ml_calculate_button = QPushButton("계산 실행")
        self.ml_calculate_button.clicked.connect(self.calculate_curve)
        self.ml_layout.addWidget(self.ml_calculate_button)
        self.ml_layout.addWidget(self.create_separator())
        self.ml_layout.addWidget(QLabel("스무딩 (대상 커브는 Target(Y) 사용)"))
        self.ml_smooth_algo_combo = QComboBox()
        self.ml_smooth_algo_combo.addItems(["Moving Average", "Median Filter", "Savitzky-Golay"])
        self.ml_layout.addWidget(self.ml_smooth_algo_combo)
        self.ml_layout.addWidget(QLabel("Window (홀수, 예: 5, 7..):"))
        self.ml_smooth_window_input = QLineEdit("5")
        self.ml_layout.addWidget(self.ml_smooth_window_input)
        self.ml_smooth_run_button = QPushButton("스무딩 실행")
        self.ml_smooth_run_button.clicked.connect(self.run_smoothing)
        self.ml_layout.addWidget(self.ml_smooth_run_button)
        self.ml_layout.addWidget(self.create_separator())
        self.ml_layout.addWidget(QLabel("--- 3. 모델 훈련 ---"))
        self.ml_train_button = QPushButton("모델 훈련 실행")
        self.ml_train_button.clicked.connect(self.run_model_training)
        self.ml_layout.addWidget(self.ml_train_button)
        self.ml_save_button = QPushButton("훈련된 모델 저장...")
        self.ml_save_button.clicked.connect(self.save_model)
        self.ml_layout.addWidget(self.ml_save_button)
        self.ml_layout.addWidget(self.create_separator())
        self.ml_layout.addWidget(QLabel("--- 4. 예측 실행 ---"))
        self.ml_load_button = QPushButton("외부 모델 로드...")
        self.ml_load_button.clicked.connect(self.load_model)
        self.ml_layout.addWidget(self.ml_load_button)
        self.ml_status_label = QLabel("모델 상태: 훈련/로드되지 않음")
        self.ml_status_label.setStyleSheet("color: gray;")
        self.ml_layout.addWidget(self.ml_status_label)
        self.ml_layout.addWidget(QLabel("새 커브 이름 (Output):"))
        self.ml_new_name_input = QLineEdit("DT_pred")
        self.ml_layout.addWidget(self.ml_new_name_input)
        self.ml_fill_gaps_check = QCheckBox("Target 커브의 결측치만 채우기")
        self.ml_fill_gaps_check.setChecked(True)
        self.ml_layout.addWidget(self.ml_fill_gaps_check)
        self.ml_predict_button = QPushButton("예측 실행")
        self.ml_predict_button.clicked.connect(self.run_prediction)
        self.ml_layout.addWidget(self.ml_predict_button)
        self.ml_layout.addStretch()
        scroll_area.setWidget(scroll_content_widget)
        main_ml_tab_layout = QVBoxLayout(self.ml_tab)
        main_ml_tab_layout.addWidget(scroll_area)
        main_ml_tab_layout.setContentsMargins(0,0,0,0)
        
    def create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    # -----------------------------------------------------------------
    # ##### (수정) 'Tracks' 탭 로직 - 데이터 모델 변경 #####
    # -----------------------------------------------------------------
    def on_add_track(self):
        track_count = len(self.tracks_model) + 1
        new_track_name = f"Track-{track_count}"
        while new_track_name in self.tracks_model:
            track_count += 1; new_track_name = f"Track-{track_count}"
        
        # (수정) curves가 딕셔너리{}로 변경됨
        self.tracks_model[new_track_name] = {
            "settings": {"min": 0, "max": 100, "log": False},
            "curves": {}, # 리스트[]가 아닌 딕셔너리
            "fill": {"enabled": False, "type": "baseline", "level": 0.0, "target": "", "color": "#FFFF00"}
        }
        item = QTreeWidgetItem([new_track_name]); item.setData(0, Qt.UserRole, new_track_name)
        self.track_list_widget.addTopLevelItem(item); self.track_list_widget.setCurrentItem(item); self.update_plots()
    
    def on_delete_track(self):
        current_item = self.track_list_widget.currentItem()
        if not current_item: QMessageBox.warning(self, "오류", "삭제할 트랙을 선택하세요."); return
        track_name = current_item.data(0, Qt.UserRole)
        if track_name in self.tracks_model: del self.tracks_model[track_name]
        self.track_list_widget.takeTopLevelItem(self.track_list_widget.indexOfTopLevelItem(current_item)); self.update_plots()

    def on_track_selection_changed(self, current_item, previous_item):
        if not current_item:
            self.track_settings_group.setEnabled(False); self.available_curves_list.clear(); self.assigned_curves_list.clear(); return
            
        self.track_settings_group.setEnabled(True); track_name = current_item.data(0, Qt.UserRole); 
        if track_name not in self.tracks_model: return # (방어 코드)
        track_data = self.tracks_model[track_name]
        
        settings = track_data["settings"]; self.track_min.setText(str(settings["min"])); self.track_max.setText(str(settings["max"])); self.track_log.setChecked(settings["log"])
        fill_settings = track_data["fill"]; self.fill_check.setChecked(fill_settings["enabled"])
        fill_type_str = "기준값 (Baseline)" if fill_settings["type"] == "baseline" else "커브 간 (Curve-Curve)"; self.fill_type_combo.setCurrentText(fill_type_str)
        self.on_fill_type_changed(fill_type_str); self.fill_ref_input.setText(str(fill_settings["level"])); self.current_fill_color = QColor(fill_settings["color"]); self.update_color_preview()

        # (수정) curves가 딕셔너리이므로 .keys()로 이름 목록을 가져옴
        assigned_curves_names = list(track_data["curves"].keys())
        self.assigned_curves_list.clear(); self.assigned_curves_list.addItems(assigned_curves_names)
        
        self.fill_target_combo.clear(); self.fill_target_combo.addItems(assigned_curves_names); self.fill_target_combo.setCurrentText(fill_settings["target"])
        
        self.available_curves_list.clear(); available = [name for name in self.all_curve_names if name not in assigned_curves_names]; self.available_curves_list.addItems(available)

    def on_apply_track_settings(self):
        current_item = self.track_list_widget.currentItem()
        if not current_item: QMessageBox.warning(self, "오류", "설정을 적용할 트랙을 선택하세요."); return
        track_name = current_item.data(0, Qt.UserRole)
        try:
            self.tracks_model[track_name]["settings"] = {"min": float(self.track_min.text()), "max": float(self.track_max.text()), "log": self.track_log.isChecked()}
            fill_type_str = self.fill_type_combo.currentText()
            self.tracks_model[track_name]["fill"] = {
                "enabled": self.fill_check.isChecked(), "type": "baseline" if fill_type_str == "기준값 (Baseline)" else "curve",
                "level": float(self.fill_ref_input.text()), "target": self.fill_target_combo.currentText(), "color": self.current_fill_color.name()
            }
            self.update_plots()
        except ValueError: QMessageBox.critical(self, "오류", "Min/Max/기준값은 숫자여야 합니다.")

    def on_assign_curve(self):
        current_track_item = self.track_list_widget.currentItem(); selected_curves = self.available_curves_list.selectedItems()
        if not current_track_item: QMessageBox.warning(self, "오류", "커브를 할당할 트랙을 먼저 선택하세요."); return
        if not selected_curves: QMessageBox.warning(self, "오류", "왼쪽에서 할당할 커브를 선택하세요."); return
        track_name = current_track_item.data(0, Qt.UserRole); track_data = self.tracks_model[track_name]
        
        current_curve_count = len(track_data["curves"])
        if current_curve_count + len(selected_curves) > 7:
            QMessageBox.warning(self, "제한", "한 트랙에는 최대 7개의 커브만 할당할 수 있습니다."); return
            
        for i, item in enumerate(selected_curves):
            curve_name = item.text()
            # (수정) 딕셔너리에 '기본 색상'과 함께 추가
            if curve_name not in track_data["curves"]:
                new_color_index = (current_curve_count + i) % len(CURVE_COLORS)
                track_data["curves"][curve_name] = {"color": CURVE_COLORS[new_color_index]}
                
        self.on_track_selection_changed(current_track_item, None); self.update_plots()

    def on_unassign_curve(self):
        current_track_item = self.track_list_widget.currentItem(); selected_curves = self.assigned_curves_list.selectedItems()
        if not current_track_item or not selected_curves: return
        track_name = current_track_item.data(0, Qt.UserRole); track_data = self.tracks_model[track_name]
        
        for item in selected_curves:
            curve_name = item.text()
            # (수정) 딕셔너리에서 Key-Value 쌍 삭제
            if curve_name in track_data["curves"]:
                del track_data["curves"][curve_name]
                
        self.on_track_selection_changed(current_track_item, None); self.update_plots()

    # -----------------------------------------------------------------
    # ##### (신규) 개별 커브 색상 변경 함수 #####
    # -----------------------------------------------------------------
    def on_change_curve_color(self):
        """'선택 커브 색상 변경...' 버튼 클릭 시"""
        current_track_item = self.track_list_widget.currentItem()
        selected_curve_item = self.assigned_curves_list.currentItem()
        
        if not current_track_item:
            QMessageBox.warning(self, "오류", "트랙을 먼저 선택하세요."); return
        if not selected_curve_item:
            QMessageBox.warning(self, "오류", "색상을 변경할 커브를 'Assigned Curves' 리스트에서 선택하세요."); return

        track_name = current_track_item.data(0, Qt.UserRole)
        curve_name = selected_curve_item.text()
        
        if track_name not in self.tracks_model or curve_name not in self.tracks_model[track_name]["curves"]:
            return # 데이터 모델 불일치 (방어 코드)
            
        # 1. 현재 색상 가져오기
        current_color_hex = self.tracks_model[track_name]["curves"][curve_name]["color"]
        current_color = QColor(current_color_hex)
        
        # 2. 색상 선택 팝업
        new_color = QColorDialog.getColor(current_color, self, f"{curve_name} 색상 선택")
        
        if new_color.isValid():
            # 3. 데이터 모델에 새 색상 저장
            self.tracks_model[track_name]["curves"][curve_name]["color"] = new_color.name()
            # 4. 플롯 즉시 갱신
            self.update_plots()

    def on_fill_type_changed(self, text):
        if text == "기준값 (Baseline)":
            self.fill_ref_label.show(); self.fill_ref_input.show(); self.fill_target_label.hide(); self.fill_target_combo.hide()
        elif text == "커브 간 (Curve-Curve)":
            self.fill_ref_label.hide(); self.fill_ref_input.hide(); self.fill_target_label.show(); self.fill_target_combo.show()
    def update_color_preview(self):
        self.fill_color_preview.setStyleSheet(f"background-color: {self.current_fill_color.name()}; border: 1px solid black;")
    def open_color_picker(self):
        color = QColorDialog.getColor(self.current_fill_color, self, "채우기 색상 선택")
        if color.isValid(): self.current_fill_color = color; self.update_color_preview()
    def refresh_all_curve_lists(self):
        current_track_item = self.track_list_widget.currentItem()
        if current_track_item: self.on_track_selection_changed(current_track_item, None)
        else: self.available_curves_list.clear(); self.available_curves_list.addItems(self.all_curve_names)
        self.ml_input_list.clear(); self.ml_target_combo.clear()
        self.ml_input_list.addItems(self.all_curve_names); self.ml_target_combo.addItems(self.all_curve_names)
        
    def load_las_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "LAS 파일 선택", "", "LAS Files (*.las)")
        if not file_path: return
        try:
            self.las_data = lasio.read(file_path)
            # (수정) 일부 LAS 파일은 인덱스 설정이 꼬이는 경우가 있어, reset_index 후 첫 번째 커브(DEPT)로 강제 재설정
            self.data_df = self.las_data.df().reset_index().set_index(self.las_data.curves[0].mnemonic)
            
            self.all_curve_names.clear(); self.tracks_model.clear(); self.well_tops.clear(); self.current_ml_model = None
            self.track_list_widget.clear(); self.track_settings_group.setEnabled(False); self.assigned_curves_list.clear()
            self.update_top_list_widget()
            self.ml_status_label.setText("모델 상태: 훈련/로드되지 않음"); self.ml_status_label.setStyleSheet("color: gray;")
            self.ml_input_list.clear(); self.ml_target_combo.clear()
            
            depth_col = self.las_data.curves[0].mnemonic
            # (수정) df.columns를 순회하여 실제 존재하는 모든 데이터 커브를 가져옴
            for curve_name in self.data_df.columns: 
                if curve_name == depth_col: continue 
                self.all_curve_names.append(curve_name)
            
            self.refresh_all_curve_lists() 
        except Exception as e:
            QMessageBox.critical(self, "오류", f"LAS 파일 로드 실패:\n{e}")

    def add_new_curve_to_ui(self, name, data):
        """새 커브를 마스터 리스트에 추가하고 모든 UI를 갱신합니다."""
        if name in self.all_curve_names:
            QMessageBox.warning(self, "경고", f"'{name}' 커브가 이미 존재하여 덮어썼습니다."); 
        else:
            self.all_curve_names.append(name)
        self.refresh_all_curve_lists()
    
    def on_plot_clicked(self, event):
        if self.data_df is None: return
        click_event = event[0]; clicked_plot = None
        for plot_item in self.plot_tracks.values():
            if plot_item.vb.sceneBoundingRect().contains(click_event.scenePos()): clicked_plot = plot_item; break
        if clicked_plot is not None:
            mouse_point = clicked_plot.vb.mapSceneToView(click_event.scenePos()); clicked_depth = mouse_point.y()
            top_name, ok = QInputDialog.getText(self, "Well Top 생성", "새 Top 이름:")
            if ok and top_name:
                if top_name in self.well_tops: QMessageBox.warning(self, "오류", "이미 존재하는 Top 이름입니다."); return
                self.well_tops[top_name] = clicked_depth; print(f"Well Top 추가: {top_name} @ {clicked_depth}"); self.update_top_list_widget(); self.update_plots()
    def update_top_list_widget(self):
        self.top_tree_widget.blockSignals(True); self.top_tree_widget.clear(); sorted_tops = sorted(self.well_tops.items(), key=lambda item: item[1]); items_to_add = []
        for name, depth in sorted_tops: item = QTreeWidgetItem([name, f"{depth:.4f}"]); item.setFlags(item.flags() | Qt.ItemIsEditable); item.setData(0, Qt.UserRole, name); items_to_add.append(item)
        self.top_tree_widget.addTopLevelItems(items_to_add); self.top_tree_widget.blockSignals(False)
    def delete_selected_top(self):
        current_item = self.top_tree_widget.currentItem()
        if current_item is None: QMessageBox.warning(self, "오류", "삭제할 Top을 리스트에서 선택하세요."); return
        top_name = current_item.data(0, Qt.UserRole);
        if top_name in self.well_tops: del self.well_tops[top_name]; print(f"Well Top 삭제: {top_name}"); self.update_top_list_widget(); self.update_plots()
    def on_top_item_changed(self, item: QTreeWidgetItem, column: int):
        old_name = item.data(0, Qt.UserRole)
        if old_name not in self.well_tops: return
        old_depth = self.well_tops[old_name]
        if column == 0:
            new_name = item.text(0)
            if new_name != old_name and new_name in self.well_tops:
                QMessageBox.warning(self, "오류", "이미 존재하는 이름입니다."); self.top_tree_widget.blockSignals(True); item.setText(0, old_name); self.top_tree_widget.blockSignals(False); return
            del self.well_tops[old_name]; self.well_tops[new_name] = old_depth; item.setData(0, Qt.UserRole, new_name)
        elif column == 1:
            try: new_depth = float(item.text(1)); self.well_tops[old_name] = new_depth
            except ValueError:
                QMessageBox.warning(self, "오류", "깊이는 숫자여야 합니다."); self.top_tree_widget.blockSignals(True); item.setText(1, f"{old_depth:.4f}"); self.top_tree_widget.blockSignals(False); return
        print(f"Well Top 수정: {self.well_tops}"); self.update_top_list_widget(); self.update_plots()
    def calculate_curve(self):
        if self.data_df is None: return
        new_name = self.ml_formula_name_input.text().strip()
        formula = self.ml_formula_input.text().strip()
        if not new_name or not formula: QMessageBox.warning(self, "오류", "수식과 새 커브 이름을 입력하세요."); return
        try:
            new_curve_data = self.data_df.eval(formula); self.data_df[new_name] = new_curve_data
            self.add_new_curve_to_ui(new_name, new_curve_data); QMessageBox.information(self, "성공", f"'{new_name}' 커브가 계산되었습니다.")
        except Exception as e: QMessageBox.critical(self, "계산 오류", f"수식 계산 중 오류 발생:\n{e}")
    def run_smoothing(self):
        if self.data_df is None: return
        target_curve = self.ml_target_combo.currentText()
        algo = self.ml_smooth_algo_combo.currentText()
        new_name = f"{target_curve}_{algo.split(' ')[0].lower()}"
        try: window = int(self.ml_smooth_window_input.text());
        except ValueError: QMessageBox.warning(self, "오류", "Window는 숫자여야 합니다."); return
        if window % 2 == 0: window += 1; self.ml_smooth_window_input.setText(str(window));
        if not target_curve: QMessageBox.warning(self, "오류", "스무딩할 Target(Y) 커브를 선택하세요."); return
        source_data = self.data_df[target_curve]; new_data = None
        try:
            if algo == "Moving Average": new_data = source_data.rolling(window=window, center=True).mean()
            elif algo == "Median Filter": new_data = source_data.rolling(window=window, center=True).median()
            elif algo == "Savitzky-Golay":
                polyorder = 2 # (임시) 하드코딩
                if window <= polyorder: QMessageBox.warning(self, "오류", "Window는 Polyorder(2)보다 커야 합니다."); return
                valid_data = source_data.dropna()
                if len(valid_data) < window: raise Exception("데이터가 Window 크기보다 적어 Sav-Gol을 적용할 수 없습니다.")
                smoothed_valid_data = savgol_filter(valid_data, window_length=window, polyorder=polyorder)
                new_data = pd.Series(smoothed_valid_data, index=valid_data.index); new_data = new_data.reindex(source_data.index)
            self.data_df[new_name] = new_data; self.add_new_curve_to_ui(new_name, new_data)
            QMessageBox.information(self, "성공", f"'{new_name}' 커브가 성공적으로 생성되었습니다.")
        except Exception as e: QMessageBox.critical(self, "스무딩 오류", f"오류 발생:\n{e}")
    def run_model_training(self):
        if self.data_df is None: QMessageBox.warning(self, "오류", "먼저 LAS 파일을 로드하세요."); return
        target_name = self.ml_target_combo.currentText(); input_items = self.ml_input_list.selectedItems(); input_names = [item.text() for item in input_items]
        if not target_name or not input_names: QMessageBox.warning(self, "오류", "Input 커브(X)와 Target 커브(Y)를 모두 선택하세요."); return
        if target_name in input_names: QMessageBox.warning(self, "오류", "Input 커브가 Target 커브를 포함할 수 없습니다."); return
        try:
            features = input_names + [target_name]; train_df = self.data_df[features].dropna()
            if train_df.empty: QMessageBox.warning(self, "오류", "선택된 커브 조합에 훈련할 데이터가 없습니다 (모든 행에 NaN 포함)."); return
            X_train = train_df[input_names]; y_train = train_df[target_name]; QMessageBox.information(self, "훈련 시작", f"{len(train_df)}개의 데이터로 훈련을 시작합니다...")
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1); model.fit(X_train, y_train); self.current_ml_model = model
            self.ml_status_label.setText(f"모델 훈련됨 (Target: {target_name})"); self.ml_status_label.setStyleSheet("color: green;"); QMessageBox.information(self, "훈련 완료", "모델 훈련이 완료되었습니다.")
        except Exception as e: QMessageBox.critical(self, "훈련 오류", f"훈련 중 오류 발생:\n{e}")
    def save_model(self):
        if self.current_ml_model is None: QMessageBox.warning(self, "오류", "먼저 모델을 훈련시키거나 로드하세요."); return
        file_path, _ = QFileDialog.getSaveFileName(self, "모델 저장", "", "Joblib Models (*.joblib)");
        if not file_path: return
        try: joblib.dump(self.current_ml_model, file_path); QMessageBox.information(self, "성공", f"모델이 {file_path}에 저장되었습니다.")
        except Exception as e: QMessageBox.critical(self, "저장 오류", f"모델 저장 중 오류 발생:\n{e}")
    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "모델 로드", "", "Joblib Models (*.joblib)");
        if not file_path: return
        try:
            self.current_ml_model = joblib.load(file_path)
            try: features = self.current_ml_model.feature_names_in_; status_text = f"모델 로드됨. (Features: {', '.join(features)})"
            except AttributeError: status_text = "모델 로드됨. (피처 이름 확인 불가)"
            self.ml_status_label.setText(status_text); self.ml_status_label.setStyleSheet("color: blue;"); QMessageBox.information(self, "성공", "모델을 성공적으로 로드했습니다.")
        except Exception as e: QMessageBox.critical(self, "로드 오류", f"모델 로드 중 오류 발생:\n{e}")
    def run_prediction(self):
        if self.current_ml_model is None: QMessageBox.warning(self, "오류", "먼저 모델을 훈련시키거나 로드하세요."); return
        if self.data_df is None: QMessageBox.warning(self, "오류", "데이터가 없습니다."); return
        new_name = self.ml_new_name_input.text().strip(); fill_gaps_only = self.ml_fill_gaps_check.isChecked()
        if not new_name: QMessageBox.warning(self, "오류", "새 커브 이름을 입력하세요."); return
        if new_name in self.all_curve_names: QMessageBox.warning(self, "오류", "이미 존재하는 커브 이름입니다."); return
        try:
            try: input_names = self.current_ml_model.feature_names_in_
            except AttributeError: QMessageBox.critical(self, "오류", "로드된 모델에서 피처 이름을 찾을 수 없습니다."); return
            predict_df = self.data_df[input_names].dropna()
            if predict_df.empty: QMessageBox.warning(self, "오류", "Input 커브에 예측할 데이터가 없습니다 (모든 행에 NaN 포함)."); return
            X_predict = predict_df; y_pred = self.current_ml_model.predict(X_predict); pred_series = pd.Series(y_pred, index=predict_df.index, name=new_name)
            if fill_gaps_only:
                target_name = self.ml_target_combo.currentText()
                if not target_name: QMessageBox.warning(self, "오류", "'결측치만 채우기'를 하려면 Target 커브(Y)를 선택해야 합니다."); return
                final_curve = self.data_df[target_name].copy(); final_curve.fillna(pred_series, inplace=True); self.data_df[new_name] = final_curve
            else: self.data_df[new_name] = pred_series 
            self.add_new_curve_to_ui(new_name, self.data_df[new_name])
            QMessageBox.information(self, "예측 완료", f"'{new_name}' 커브가 생성되었습니다.")
        except Exception as e: QMessageBox.critical(self, "예측 오류", f"예측 중 오류 발생:\n{e}")

    # --- (수정) 플로팅 및 호버링 - 개별 색상 적용 ---
    def mouse_moved_across_plots(self, event):
        pos = event[0] 
        for track_name, plot_item in self.plot_tracks.items():
            if plot_item.vb.sceneBoundingRect().contains(pos):
                mouse_point = plot_item.vb.mapSceneToView(pos)
                data_x = mouse_point.x(); data_y = mouse_point.y()
                crosshair_items = plot_item.crosshairs 
                crosshair_items[0].setPos(data_x); crosshair_items[1].setPos(data_y)
                tooltip_text = f"Track: {track_name}\nDepth: {data_y:.2f}\n"
                try:
                    nearest_index = self.data_df.index.get_indexer([data_y], method='nearest')[0]
                    actual_depth = self.data_df.index[nearest_index]
                    # (수정) 딕셔너리 순회
                    for curve_name, curve_props in self.tracks_model[track_name]["curves"].items():
                        color = curve_props['color'] # 저장된 색상 사용
                        value = self.data_df.loc[actual_depth, curve_name]
                        tooltip_text += f"<span style='color: {color}'>{curve_name}</span>: {value:.2f}\n"
                except Exception as e:
                    tooltip_text += f"Value(X): {data_x:.2f}"
                crosshair_items[2].setText(tooltip_text.strip())
                crosshair_items[2].setPos(data_x, data_y)
                crosshair_items[0].show(); crosshair_items[1].show(); crosshair_items[2].show()
            else:
                crosshair_items = plot_item.crosshairs
                crosshair_items[0].hide(); crosshair_items[1].hide(); crosshair_items[2].hide()
    
    def update_plots(self):
        self.plot_widget.clear(); self.plot_tracks.clear(); self.plot_data_items.clear()
        if self.data_df is None: return
        depth_index_np = self.data_df.index.values 
        first_plot = None; plot_col_index = 0
        
        for track_name, track_data in self.tracks_model.items():
            plot_item = self.plot_widget.addPlot(row=0, col=plot_col_index)
            plot_item.showAxis('bottom', False); plot_item.showAxis('top', True); plot_item.setLabel('top', track_name); plot_item.invertY(True) 
            settings = track_data["settings"]
            try: plot_item.setXRange(settings["min"], settings["max"]); plot_item.setLogMode(x=settings["log"], y=False)
            except Exception as e: print(f"스케일 설정 오류 ({track_name}): {e}")
            if first_plot is None: first_plot = plot_item; plot_item.setLabel('left', 'Depth') 
            else: plot_item.setYLink(first_plot) 
            v_line = pg.InfiniteLine(angle=90, movable=False, pen=(0,0,0,100)); h_line = pg.InfiniteLine(angle=0, movable=False, pen=(0,0,0,100)); tooltip = pg.TextItem(text="", color=(0,0,0), anchor=(-0.1, 1.1), border='w', fill=(255,255,255,150))
            plot_item.addItem(v_line, ignoreBounds=True); plot_item.addItem(h_line, ignoreBounds=True); plot_item.addItem(tooltip, ignoreBounds=True)
            v_line.hide(); h_line.hide(); tooltip.hide(); plot_item.crosshairs = (v_line, h_line, tooltip)
            for top_name, top_depth in self.well_tops.items():
                top_line = pg.InfiniteLine(pos=top_depth, angle=0, movable=False, pen='r', label=top_name)
                top_label = pg.InfLineLabel(top_line, text=top_name, position=0.05, anchor=(0, 0), color='r')
                plot_item.addItem(top_line, ignoreBounds=True); plot_item.addItem(top_label, ignoreBounds=True)
            
            track_curves = {}
            # (수정) 딕셔너리 순회
            for curve_name, curve_props in track_data["curves"].items():
                if curve_name not in self.data_df.columns: continue
                curve_data_np = self.data_df[curve_name].values
                color = curve_props['color'] # (수정) 저장된 색상 사용
                plot_data_item = plot_item.plot(x=curve_data_np, y=depth_index_np, pen=color, name=curve_name); track_curves[curve_name] = plot_data_item
            
            fill_settings = track_data["fill"]
            if fill_settings["enabled"]:
                try:
                    fill_brush = pg.mkBrush(fill_settings["color"])
                    if fill_settings["type"] == "baseline":
                        if track_data["curves"]:
                            first_curve_name = list(track_data["curves"].keys())[0] # (수정) 딕셔너리의 첫 번째 Key
                            first_curve_item = track_curves[first_curve_name]
                            first_curve_item.setFillBrush(fill_brush); first_curve_item.setFillLevel(fill_settings["level"])
                    elif fill_settings["type"] == "curve":
                        target_curve_name = fill_settings["target"]
                        if len(track_data["curves"]) >= 2 and target_curve_name in track_curves:
                            first_curve_name = list(track_data["curves"].keys())[0] # (수정) 딕셔너리의 첫 번째 Key
                            curve1_item = track_curves[first_curve_name]; curve2_item = track_curves[target_curve_name]
                            if first_curve_name == target_curve_name: QMessageBox.warning(self, "Fill 오류", f"'{track_name}'에서 커브 1과 대상 커브가 같습니다.")
                            else: fill_item = pg.FillBetweenItem(curve1_item, curve2_item, brush=fill_brush); plot_item.addItem(fill_item)
                except Exception as e: print(f"Fill 생성 오류 ({track_name}): {e}")
            self.plot_tracks[track_name] = plot_item
            plot_col_index += 1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
