import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import numpy as np
import segyio
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
import os
import csv

# -----------------------------------------------------------
# 1. 커스텀 툴바 (하단 메시지 숨김)
# -----------------------------------------------------------
class CustomToolbar(NavigationToolbar2Tk):
    def set_message(self, s): pass

# -----------------------------------------------------------
# 2. SEGY 뷰어 (고속 렌더링 + 최적화 버전 유지)
# -----------------------------------------------------------
class SegyViewer:
    def __init__(self, root, filename=None, on_update_callback=None, on_cursor_callback=None, coord_type="CDP"):
        self.root = root
        self.root.title(f"Woo Interpreter (Standard) - {filename.split('/')[-1] if filename else 'New'}")
        self.root.geometry("1400x900")

        self.filename = filename
        self.current_data = None
        self.extent = None
        self.on_update_callback = on_update_callback
        self.on_cursor_callback = on_cursor_callback
        self.coord_type = coord_type
        
        # 렌더링 최적화 객체
        self.im_obj = None     
        self.line_objs = {}    
        self.scat_objs = {}    
        self.limit_val = 1.0
        
        # 데이터 변수
        self.cache_x = None
        self.cache_y = None
        self.real_trace_indices = None
        self.var_auto_aspect = tk.BooleanVar(value=True)

        self.horizons = {
            'Horizon A': {'color': 'yellow', 'points': []},
            'Horizon B': {'color': 'cyan', 'points': []},
            'Horizon C': {'color': 'lime', 'points': []}
        }
        self.active_layer = 'Horizon A'

        # UI 구성
        self.side_bar = tk.Frame(root, width=320, bg="#f0f0f0", padx=10, pady=10)
        self.side_bar.pack(side=tk.LEFT, fill=tk.Y)
        self.side_bar.pack_propagate(False)
        self.main_frame = tk.Frame(root, bg="white")
        self.main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_ui()
        self.setup_plot()

        if self.filename:
            self.load_from_path(self.filename)

    def setup_ui(self):
        # 분석 도구
        tk.Label(self.side_bar, text="--- Analysis Tools ---", font=('Arial', 10, 'bold'), bg="#f0f0f0").pack(pady=(5,5))
        tk.Button(self.side_bar, text="🔍 Trace 헤더 보기", command=self.show_headers, bg="#3498db", fg="white").pack(fill=tk.X, pady=2)
        
        # 해석 도구
        tk.Label(self.side_bar, text="--- Interpretation ---", font=('Arial', 10, 'bold'), bg="#f0f0f0").pack(pady=(15,5))
        self.layer_selector = ttk.Combobox(self.side_bar, values=list(self.horizons.keys()), state="readonly")
        self.layer_selector.current(0)
        self.layer_selector.pack(fill=tk.X, pady=5)
        self.layer_selector.bind("<<ComboboxSelected>>", self.on_layer_change)
        
        btn_frame = tk.Frame(self.side_bar, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="📂 Import", command=self.import_horizon_csv, bg="#95a5a6", fg="white", width=15).pack(side=tk.LEFT, padx=1)
        tk.Button(btn_frame, text="💾 Export", command=self.save_horizon, bg="#2c3e50", fg="white", width=15).pack(side=tk.LEFT, padx=1)

        tk.Button(self.side_bar, text="📍 현재 층 초기화", command=self.clear_horizon, bg="#e74c3c", fg="white").pack(fill=tk.X, pady=2)
        
        self.hor_info = tk.Label(self.side_bar, text="A:0 | B:0 | C:0", bg="#f0f0f0", font=('Arial', 9))
        self.hor_info.pack(pady=5)
        tk.Label(self.side_bar, text="* 좌클릭: 픽킹 / 우클릭: 삭제", font=('Arial', 8), fg="#555").pack(pady=5)
        
        # 디스플레이 설정
        tk.Label(self.side_bar, text="--- Display Settings ---", font=('Arial', 10, 'bold'), bg="#f0f0f0").pack(pady=(15,5))
        
        self.chk_auto = tk.Checkbutton(self.side_bar, text="Auto Fit Mode", variable=self.var_auto_aspect, command=self.toggle_aspect, bg="#f0f0f0")
        self.chk_auto.pack(anchor="w")

        w_frame = tk.Frame(self.side_bar, bg="#f0f0f0")
        w_frame.pack(fill=tk.X)
        tk.Label(w_frame, text="W:", bg="#f0f0f0", width=3).pack(side=tk.LEFT)
        self.scale_w = tk.Scale(w_frame, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, command=lambda v: self.update_aspect_only(), label="Width Scale")
        self.scale_w.set(1.0); self.scale_w.pack(side=tk.LEFT, fill=tk.X, expand=True)

        h_frame = tk.Frame(self.side_bar, bg="#f0f0f0")
        h_frame.pack(fill=tk.X)
        tk.Label(h_frame, text="H:", bg="#f0f0f0", width=3).pack(side=tk.LEFT)
        self.scale_h = tk.Scale(h_frame, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, command=lambda v: self.update_aspect_only(), label="Height Scale")
        self.scale_h.set(1.0); self.scale_h.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(self.side_bar, text="Sampling Rate (ms)", bg="#f0f0f0").pack(anchor="w", pady=(10,0))
        sr_frame = tk.Frame(self.side_bar, bg="#f0f0f0")
        sr_frame.pack(fill=tk.X)
        self.sr_in = tk.Entry(sr_frame, width=15); self.sr_in.insert(0, "2.0"); self.sr_in.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.sr_in.bind('<Return>', lambda event: self.full_redraw())
        tk.Button(sr_frame, text="적용", command=self.full_redraw, bg="#bdc3c7", width=6).pack(side=tk.LEFT, padx=2)
        
        tk.Label(self.side_bar, text="Contrast (Clip %)", font=('Arial', 9, 'bold'), bg="#f0f0f0", fg="red").pack(pady=(15,0))
        self.clip = tk.Scale(self.side_bar, from_=80, to=99.9, orient=tk.HORIZONTAL, resolution=0.1, command=lambda v: self.update_contrast_only())
        self.clip.set(98); self.clip.pack(fill=tk.X)

    def toggle_aspect(self):
        if self.var_auto_aspect.get():
            self.scale_w.config(state="disabled", fg="gray")
            self.scale_h.config(state="disabled", fg="gray")
        else:
            self.scale_w.config(state="normal", fg="black")
            self.scale_h.config(state="normal", fg="black")
        self.update_aspect_only()

    def setup_plot(self):
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = CustomToolbar(self.canvas, self.main_frame)
        self.toolbar.update()
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_action)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def load_from_path(self, path):
        self.filename = path
        try:
            with segyio.open(self.filename, "r", ignore_geometry=True) as f:
                total_traces = f.tracecount
                MAX_DISPLAY_TRACES = 5000 
                step = max(1, total_traces // MAX_DISPLAY_TRACES)
                
                indices = list(range(0, total_traces, step))
                self.real_trace_indices = np.array(indices)
                
                self.current_data = segyio.tools.collect(f.trace[::step]).T
                
                sr = segyio.tools.dt(f)/1000
                self.sr_in.delete(0, tk.END); self.sr_in.insert(0, str(sr))
                n_samples, n_traces = self.current_data.shape
                self.extent = [0, n_traces, n_samples * sr, 0]

                scalars = f.attributes(segyio.TraceField.SourceGroupScalar)[0:1] 
                scalar_val = float(scalars[0]) if len(scalars) > 0 else 1.0
                if scalar_val == 0: scalar_val = 1.0
                elif scalar_val < 0: scalar_val = 1.0 / abs(scalar_val)

                if self.coord_type == "CDP":
                    xk, yk = segyio.TraceField.CDP_X, segyio.TraceField.CDP_Y
                else:
                    xk, yk = segyio.TraceField.SourceX, segyio.TraceField.SourceY
                
                self.cache_x = f.attributes(xk)[::step].astype(float) * scalar_val
                self.cache_y = f.attributes(yk)[::step].astype(float) * scalar_val

            self.full_redraw()
            title_text = f"Viewer - {self.filename.split('/')[-1]}"
            if step > 1: title_text += f" (Resampled 1/{step})"
            self.root.title(title_text)
        except Exception as e:
            messagebox.showerror("Error", f"Load Failed: {e}")

    def load_horizons_data(self, horizons_data):
        if horizons_data: self.horizons = horizons_data
        default_structure = {
            'Horizon A': {'color': 'yellow', 'points': []},
            'Horizon B': {'color': 'cyan', 'points': []},
            'Horizon C': {'color': 'lime', 'points': []}
        }
        for key, default_val in default_structure.items():
            if key not in self.horizons: self.horizons[key] = default_val
        self.update_status()
        self.draw_horizons_only()

    def full_redraw(self):
        if self.current_data is None: return
        try: sr = float(self.sr_in.get())
        except: sr = 2.0
        n_samples, n_traces = self.current_data.shape
        self.extent = [0, n_traces, n_samples * sr, 0]
        
        self.ax.clear()
        self.im_obj = None 
        self.line_objs = {}
        self.scat_objs = {}

        self.update_contrast_only(draw=False) 
        self.im_obj = self.ax.imshow(self.current_data, cmap="RdBu", 
                                     vmin=-self.limit_val, vmax=self.limit_val, 
                                     aspect='auto', extent=self.extent, interpolation='nearest')
        self.ax.set_ylabel("Time (ms)")
        self.ax.set_xlabel("Trace Number (Sampled)")
        self.update_aspect_only(draw=False)
        self.draw_horizons_only(draw=False)
        self.canvas.draw()

    def update_contrast_only(self, draw=True):
        if self.current_data is None: return
        clip_pct = float(self.clip.get())
        sample_data = self.current_data[::10, ::10]
        limit = np.nanpercentile(np.absolute(sample_data), clip_pct)
        if limit == 0: limit = 1.0
        self.limit_val = limit
        if self.im_obj:
            self.im_obj.set_clim(-limit, limit)
            if draw: self.canvas.draw_idle()

    def update_aspect_only(self, draw=True):
        if self.var_auto_aspect.get():
            self.ax.set_aspect('auto')
        else:
            w_scale = self.scale_w.get()
            h_scale = self.scale_h.get()
            if w_scale == 0: w_scale = 1.0
            self.ax.set_aspect(h_scale / w_scale)
        if draw: self.canvas.draw_idle()

    def draw_horizons_only(self, draw=True):
        if self.real_trace_indices is None: return
        for name in list(self.line_objs.keys()):
            try: self.line_objs[name].remove()
            except: pass
        for name in list(self.scat_objs.keys()):
            try: self.scat_objs[name].remove()
            except: pass
        self.line_objs.clear()
        self.scat_objs.clear()

        for name, data in self.horizons.items():
            if not data['points']: continue
            p_arr = np.array(data['points'])
            saved_real_indices = p_arr[:, 3]
            display_indices = np.searchsorted(self.real_trace_indices, saved_real_indices)
            valid_mask = (display_indices < len(self.real_trace_indices)) & \
                         (self.real_trace_indices[np.clip(display_indices, 0, len(self.real_trace_indices)-1)] == saved_real_indices)
            if np.any(valid_mask):
                x_plot = display_indices[valid_mask]
                y_plot = p_arr[valid_mask, 2]
                scat = self.ax.plot(x_plot, y_plot, 'o', color=data['color'], markersize=4)[0]
                self.scat_objs[name] = scat
                if len(x_plot) >= 2:
                    line = self.ax.plot(x_plot, y_plot, color=data['color'], linewidth=1.5)[0]
                    self.line_objs[name] = line
        if draw: self.canvas.draw_idle()

    def on_mouse_move(self, event):
        if event.inaxes != self.ax or self.cache_x is None: return
        trace_idx = int(round(event.xdata))
        if 0 <= trace_idx < len(self.cache_x):
            if self.on_cursor_callback:
                self.on_cursor_callback(self.cache_x[trace_idx], self.cache_y[trace_idx])

    def on_mouse_action(self, event):
        if event.inaxes != self.ax or not self.filename or self.toolbar.mode != '': return
        display_idx = int(round(event.xdata))
        twt = event.ydata
        pts_list = self.horizons[self.active_layer]['points']
        changed = False

        if event.button == 1: # 좌클릭
            if self.cache_x is not None and 0 <= display_idx < len(self.cache_x):
                real_idx = self.real_trace_indices[display_idx]
                pts_list.append([self.cache_x[display_idx], self.cache_y[display_idx], twt, real_idx])
                pts_list.sort(key=lambda x: x[3])
                changed = True
        elif event.button == 3: # 우클릭
            if pts_list and self.real_trace_indices is not None and 0 <= display_idx < len(self.real_trace_indices):
                target_real = self.real_trace_indices[display_idx]
                dists = [abs(p[3] - target_real) for p in pts_list]
                if dists and min(dists) < 500: 
                    pts_list.pop(np.argmin(dists))
                    changed = True
        if changed:
            self.update_status()
            self.draw_horizons_only()
            if self.on_update_callback: self.on_update_callback(self.filename, self.horizons)

    def on_scroll(self, event):
        if event.inaxes != self.ax: return
        scale = 1/1.25 if event.button == 'up' else 1.25
        cur_xlim, cur_ylim = self.ax.get_xlim(), self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_w, new_h = (cur_xlim[1]-cur_xlim[0])*scale, (cur_ylim[1]-cur_ylim[0])*scale
        rel_x, rel_y = (cur_xlim[1]-xdata)/(cur_xlim[1]-cur_xlim[0]), (cur_ylim[1]-ydata)/(cur_ylim[1]-cur_ylim[0])
        self.ax.set_xlim([xdata-new_w*(1-rel_x), xdata+new_w*rel_x])
        self.ax.set_ylim([ydata-new_h*(1-rel_y), ydata+new_h*rel_y])
        self.canvas.draw_idle()

    def on_layer_change(self, event): self.active_layer = self.layer_selector.get()
    def update_status(self): self.hor_info.config(text="Pts: " + " | ".join([f"{k[8]}:{len(v['points'])}" for k, v in self.horizons.items()]))

    def show_headers(self):
        if not self.filename: return
        win = tk.Toplevel(self.root); win.title("Trace Header")
        txt = scrolledtext.ScrolledText(win, width=60); txt.pack(fill=tk.BOTH, expand=True)
        try:
            with segyio.open(self.filename, "r", ignore_geometry=True) as f:
                for k, v in f.header[0].items(): txt.insert(tk.END, f"{str(k):<25} | {v}\n")
        except: pass
        txt.configure(state='disabled')

    def clear_horizon(self):
        self.horizons[self.active_layer]['points'] = []
        self.update_status(); self.draw_horizons_only()
        if self.on_update_callback: self.on_update_callback(self.filename, self.horizons)

    def save_horizon(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            with open(path, 'w', newline='') as f:
                f.write("Layer,X,Y,TWT,TraceIdx\n")
                for n, d in self.horizons.items():
                    for p in d['points']: f.write(f"{n},{p[0]},{p[1]},{p[2]},{p[3]}\n")
            messagebox.showinfo("Saved", "Export Complete.")

    def import_horizon_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path: return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                loaded_pts = {'Horizon A': [], 'Horizon B': [], 'Horizon C': []}
                for row in reader:
                    layer = row['Layer'].strip()
                    if layer in loaded_pts:
                        loaded_pts[layer].append([float(row['X']), float(row['Y']), float(row['TWT']), int(float(row['TraceIdx']))])
                for name, pts in loaded_pts.items():
                    if pts: self.horizons[name]['points'] = sorted(pts, key=lambda k: k[3])
                self.update_status(); self.draw_horizons_only()
                if self.on_update_callback: self.on_update_callback(self.filename, self.horizons)
                messagebox.showinfo("Import", "Horizon Loaded Successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import CSV: {e}")

# -----------------------------------------------------------
# 3. 프로젝트 매니저 (기본 Cubic Interpolation + 기능 유지)
# -----------------------------------------------------------
class ProjectManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Seismic Project Manager (Standard Mapping)")
        self.root.geometry("1200x900")
        self.survey_lines = {}
        self.line_plots = {}
        self.horizon_plots = {}
        self.cbar = None 
        self.cursor_marker = None 

        # 상단 툴바
        top_frame = tk.Frame(root, height=70, bg="#ecf0f1", bd=1, relief=tk.RAISED)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        btn_frame = tk.Frame(top_frame, bg="#ecf0f1")
        btn_frame.pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(btn_frame, text="📂 Load", command=self.add_files, bg="#2980b9", fg="white", width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="💾 Save", command=self.save_project, bg="#27ae60", fg="white", width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="📂 Open", command=self.load_project, bg="#f39c12", fg="white", width=8).pack(side=tk.LEFT, padx=2)
        
        sett_frame = tk.Frame(top_frame, bg="#ecf0f1")
        sett_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(sett_frame, text="Layer:", bg="#ecf0f1").pack(side=tk.LEFT)
        self.horizon_selector = ttk.Combobox(sett_frame, values=['None', 'Horizon A', 'Horizon B', 'Horizon C'], state="readonly", width=10)
        self.horizon_selector.current(0); self.horizon_selector.pack(side=tk.LEFT, padx=2)
        self.horizon_selector.bind("<<ComboboxSelected>>", self.on_viz_change)

        tk.Label(sett_frame, text="Mode:", bg="#ecf0f1").pack(side=tk.LEFT, padx=(5,0))
        self.view_mode = ttk.Combobox(sett_frame, values=['Scatter Points', 'Contour Map'], state="readonly", width=12)
        self.view_mode.current(1); self.view_mode.pack(side=tk.LEFT, padx=2)
        self.view_mode.bind("<<ComboboxSelected>>", self.on_viz_change)

        # 맵핑 설정 (Color Bar Min/Max만 유지, Radius 삭제)
        map_sett_frame = tk.LabelFrame(top_frame, text="Color Settings", bg="#ecf0f1", padx=5, pady=2)
        map_sett_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        tk.Label(map_sett_frame, text="Min:", bg="#ecf0f1").pack(side=tk.LEFT)
        self.ent_vmin = tk.Entry(map_sett_frame, width=5); self.ent_vmin.pack(side=tk.LEFT, padx=2)
        tk.Label(map_sett_frame, text="Max:", bg="#ecf0f1").pack(side=tk.LEFT)
        self.ent_vmax = tk.Entry(map_sett_frame, width=5); self.ent_vmax.pack(side=tk.LEFT, padx=2)
        tk.Button(map_sett_frame, text="Apply", command=self.draw_visualization, bg="#34495e", fg="white", width=6).pack(side=tk.LEFT, padx=5)

        self.status_lbl = tk.Label(root, text="Ready.", bd=1, relief=tk.SUNKEN, anchor=tk.W); self.status_lbl.pack(side=tk.BOTTOM, fill=tk.X)
        self.map_frame = tk.Frame(root, bg="white"); self.map_frame.pack(fill=tk.BOTH, expand=True)
        self.setup_initial_canvas()

    def setup_initial_canvas(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.divider = make_axes_locatable(self.ax)
        self.cax = self.divider.append_axes("right", size="3%", pad=0.1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self.map_frame).update()
        self.fig.canvas.mpl_connect('pick_event', self.on_line_pick)
        self.reset_map_view() 

    def reset_map_view(self):
        self.ax.clear(); self.cax.clear(); self.cax.axis('off')
        self.ax.set_title("Seismic Base Map")
        self.ax.set_xlabel("East (X)"); self.ax.set_ylabel("North (Y)")
        self.ax.set_aspect('auto'); self.ax.ticklabel_format(useOffset=False, style='plain')
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.cursor_marker, = self.ax.plot([], [], 'r+', ms=15, mew=2, zorder=10, label='Cursor')
        self.line_plots = {}; self.horizon_plots = {}

    def update_cursor_position(self, x, y):
        if self.cursor_marker:
            self.cursor_marker.set_data([x], [y])
            self.canvas.draw_idle()

    def process_segy_file(self, filepath, existing_horizons=None):
        try:
            if not os.path.exists(filepath): return None
            fname = os.path.basename(filepath)
            with segyio.open(filepath, "r", ignore_geometry=True) as f:
                scalars = f.attributes(segyio.TraceField.SourceGroupScalar)[0:1]
                scalar = float(scalars[0]) if len(scalars)>0 else 1.0
                if scalar==0: scalar=1.0
                elif scalar<0: scalar=1.0/abs(scalar)
                
                raw_x = f.attributes(segyio.TraceField.CDP_X)[0:100]
                raw_y = f.attributes(segyio.TraceField.CDP_Y)[0:100]
                coord_type = "CDP"
                if np.all(raw_x==0) and np.all(raw_y==0):
                    coord_type = "Source"
                    xk, yk = segyio.TraceField.SourceX, segyio.TraceField.SourceY
                else:
                    xk, yk = segyio.TraceField.CDP_X, segyio.TraceField.CDP_Y
                
                trace_count = f.tracecount
                step = max(1, trace_count // 1000)
                x = f.attributes(xk)[::step].astype(float) * scalar
                y = f.attributes(yk)[::step].astype(float) * scalar
                
                if existing_horizons: horizons = existing_horizons
                else: horizons = {'Horizon A': {'color': 'yellow', 'points': []}, 'Horizon B': {'color': 'cyan', 'points': []}, 'Horizon C': {'color': 'lime', 'points': []}}
                
                self.survey_lines[fname] = {'path': filepath, 'x': x, 'y': y, 'type': coord_type, 'horizons': horizons}
                return fname
        except Exception: return None

    def add_files(self):
        files = filedialog.askopenfilenames(filetypes=[("SEGY", "*.sgy *.segy")])
        if not files: return
        self.status_lbl.config(text="Loading headers..."); self.root.update()
        count = 0
        for filepath in files:
            if self.process_segy_file(filepath): count += 1
        self.update_map()
        self.status_lbl.config(text=f"{count} files loaded.")

    def save_project(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Project", "*.json")])
        if not path: return
        save_data = {fname: {'path': data['path'], 'horizons': data['horizons']} for fname, data in self.survey_lines.items()}
        try:
            with open(path, 'w', encoding='utf-8') as f: json.dump(save_data, f, indent=4)
            messagebox.showinfo("Success", "Project Saved.")
        except Exception as e: messagebox.showerror("Error", str(e))

    def load_project(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Project", "*.json")])
        if not path: return
        project_dir = os.path.dirname(path)
        try:
            with open(path, 'r', encoding='utf-8') as f: loaded_data = json.load(f)
            self.survey_lines = {}; self.reset_map_view()
            count = 0
            self.status_lbl.config(text="Restoring..."); self.root.update()
            for fname, data in loaded_data.items():
                file_path = data['path']
                if not os.path.exists(file_path):
                    alt = os.path.join(project_dir, os.path.basename(file_path))
                    if os.path.exists(alt): file_path = alt
                if self.process_segy_file(file_path, existing_horizons=data['horizons']): count += 1
            self.update_map(); self.draw_visualization()
            self.status_lbl.config(text=f"Restored: {count}")
            messagebox.showinfo("Success", "Loaded.")
        except Exception as e: messagebox.showerror("Error", str(e))

    def update_map(self):
        if not self.survey_lines: return
        colors = plt.cm.nipy_spectral(np.linspace(0,1,len(self.survey_lines)))
        idx = 0
        for lid, d in self.survey_lines.items():
            found = False
            for artist, name in self.line_plots.items():
                if name == lid: found = True; break
            if found: continue
            l, = self.ax.plot(d['x'], d['y'], label=lid, linewidth=2, color=colors[idx%len(colors)], picker=5)
            if len(d['x'])>0: self.ax.text(d['x'][0], d['y'][0], lid[:10], fontsize=8, color=colors[idx%len(colors)], fontweight='bold')
            self.line_plots[l] = lid
            idx += 1
        self.ax.legend(loc='upper right', fontsize='x-small')
        self.ax.relim(); self.ax.autoscale_view(); self.canvas.draw()

    def on_line_pick(self, event):
        if event.artist in self.line_plots:
            lid = self.line_plots[event.artist]
            data = self.survey_lines[lid]
            new_win = tk.Toplevel(self.root)
            viewer = SegyViewer(new_win, filename=data['path'], on_update_callback=self.on_horizon_update, on_cursor_callback=self.update_cursor_position, coord_type=data['type'])
            viewer.load_horizons_data(data['horizons'])

    def on_horizon_update(self, filepath, horizons):
        fname = os.path.basename(filepath)
        if fname in self.survey_lines:
            self.survey_lines[fname]['horizons'] = horizons
            self.draw_visualization()

    def on_viz_change(self, event): self.draw_visualization()

    # ------------------------------------------------------------------
    # [핵심] 맵핑 로직 완전 초기화 (Standard Cubic Interpolation)
    # ------------------------------------------------------------------
    def draw_visualization(self):
        for s in self.horizon_plots.values():
            try: s.remove()
            except: pass
            if hasattr(s, 'collections'):
                for c in s.collections: c.remove()
        self.horizon_plots.clear()
        self.cax.clear(); self.cax.axis('off')

        target = self.horizon_selector.get(); mode = self.view_mode.get()
        if target == 'None': self.canvas.draw(); return
        
        all_x, all_y, all_z = [], [], []
        for lid, d in self.survey_lines.items():
            if target in d['horizons']:
                pts = d['horizons'][target]['points']
                if pts:
                    p = np.array(pts)
                    all_x.extend(p[:,0]); all_y.extend(p[:,1]); all_z.extend(p[:,2])
        
        if not all_x: self.canvas.draw(); return
        all_x = np.array(all_x); all_y = np.array(all_y); all_z = np.array(all_z)

        # 사용자 설정
        try: vmin = float(self.ent_vmin.get())
        except: vmin = None
        try: vmax = float(self.ent_vmax.get())
        except: vmax = None

        mappable = None
        
        if mode == 'Scatter Points':
            sc = self.ax.scatter(all_x, all_y, c=all_z, cmap='viridis_r', s=30, 
                                 edgecolors='k', linewidth=0.5, zorder=5, vmin=vmin, vmax=vmax)
            self.horizon_plots['sc'] = sc; mappable = sc
            
        elif mode == 'Contour Map':
            if len(all_x) < 4: return
            
            res_x, res_y = 200, 200 
            xi = np.linspace(min(all_x), max(all_x), res_x)
            yi = np.linspace(min(all_y), max(all_y), res_y)
            Xi, Yi = np.meshgrid(xi, yi)
            
            try:
                # [초기화] 복잡한 로직 제거 -> 표준 Cubic 사용
                Zi = griddata((all_x, all_y), all_z, (Xi, Yi), method='cubic')

                levels = np.linspace(vmin if vmin else np.nanmin(Zi), 
                                     vmax if vmax else np.nanmax(Zi), 20)
                cf = self.ax.contourf(Xi, Yi, Zi, levels=levels, cmap='viridis_r', alpha=0.7, zorder=4, extend='both')
                self.horizon_plots['cf'] = cf; mappable = cf
                cl = self.ax.contour(Xi, Yi, Zi, levels=levels, colors='k', linewidths=0.4, zorder=5)
                self.horizon_plots['cl'] = cl
                
            except Exception as e: print(f"Contour Error: {e}")

        if mappable:
            self.cax.axis('on')
            plt.colorbar(mappable, cax=self.cax, label='Time (ms)')
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    manager = ProjectManager(root)
    root.mainloop()
