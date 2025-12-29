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
# 1. SEGY 뷰어 (UI 개선: 헤더 입력 삭제, SR 버튼 추가)
# -----------------------------------------------------------
class CustomToolbar(NavigationToolbar2Tk):
    def set_message(self, s): pass

class SegyViewer:
    def __init__(self, root, filename=None, on_update_callback=None, coord_type="CDP"):
        self.root = root
        self.root.title(f"Woo Interpreter - {filename.split('/')[-1] if filename else 'New'}")
        self.root.geometry("1400x900")

        self.filename = filename
        self.current_data = None
        self.extent = None
        self.on_update_callback = on_update_callback
        self.coord_type = coord_type
        self.clip_val = 98.0
        
        # Horizon 데이터
        self.horizons = {
            'Horizon A': {'color': 'yellow', 'points': []},
            'Horizon B': {'color': 'cyan', 'points': []},
            'Horizon C': {'color': 'lime', 'points': []}
        }
        self.active_layer = 'Horizon A'

        # 레이아웃
        self.side_bar = tk.Frame(root, width=300, bg="#f0f0f0", padx=10, pady=10)
        self.side_bar.pack(side=tk.LEFT, fill=tk.Y)
        self.side_bar.pack_propagate(False)
        self.main_frame = tk.Frame(root, bg="white")
        self.main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_ui()
        self.setup_plot()

        if self.filename:
            self.load_from_path(self.filename)

    def setup_ui(self):
        # 1. 분석 도구
        tk.Label(self.side_bar, text="--- Analysis Tools ---", font=('Arial', 10, 'bold'), bg="#f0f0f0").pack(pady=(5,5))
        tk.Button(self.side_bar, text="🔍 Trace 헤더 보기", command=self.show_headers, bg="#3498db", fg="white").pack(fill=tk.X, pady=2)
        
        # 2. 해석 도구
        tk.Label(self.side_bar, text="--- Interpretation ---", font=('Arial', 10, 'bold'), bg="#f0f0f0").pack(pady=(15,5))
        self.layer_selector = ttk.Combobox(self.side_bar, values=list(self.horizons.keys()), state="readonly")
        self.layer_selector.current(0)
        self.layer_selector.pack(fill=tk.X, pady=5)
        self.layer_selector.bind("<<ComboboxSelected>>", self.on_layer_change)
        
        # Import / Export
        btn_frame = tk.Frame(self.side_bar, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X, pady=2)
        tk.Button(btn_frame, text="📂 Import CSV", command=self.import_horizon_csv, bg="#95a5a6", fg="white", width=15).pack(side=tk.LEFT, padx=1)
        tk.Button(btn_frame, text="💾 Export CSV", command=self.save_horizon, bg="#2c3e50", fg="white", width=15).pack(side=tk.LEFT, padx=1)

        tk.Button(self.side_bar, text="📍 현재 층 초기화", command=self.clear_horizon, bg="#e74c3c", fg="white").pack(fill=tk.X, pady=2)
        
        self.hor_info = tk.Label(self.side_bar, text="A:0 | B:0 | C:0", bg="#f0f0f0", font=('Arial', 9))
        self.hor_info.pack(pady=5)
        tk.Label(self.side_bar, text="* 좌클릭: 픽킹 / 우클릭: 삭제", font=('Arial', 8), fg="#555").pack(pady=5)

        # 3. 디스플레이 설정 (헤더 입력창 삭제됨)
        tk.Label(self.side_bar, text="--- Display Settings ---", font=('Arial', 10, 'bold'), bg="#f0f0f0").pack(pady=(15,5))
        
        # [수정] Sampling Rate 입력 + 적용 버튼
        tk.Label(self.side_bar, text="Sampling Rate (ms)", bg="#f0f0f0").pack(anchor="w")
        sr_frame = tk.Frame(self.side_bar, bg="#f0f0f0")
        sr_frame.pack(fill=tk.X)
        
        self.sr_in = tk.Entry(sr_frame, width=15)
        self.sr_in.insert(0, "2.0")
        self.sr_in.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.sr_in.bind('<Return>', lambda event: self.refresh_plot())
        
        # [신규] 적용 버튼 추가
        tk.Button(sr_frame, text="적용", command=self.refresh_plot, bg="#bdc3c7", width=6).pack(side=tk.LEFT, padx=2)
        
        tk.Label(self.side_bar, text="Contrast (Clip %)", font=('Arial', 9, 'bold'), bg="#f0f0f0", fg="red").pack(pady=(15,0))
        self.clip = tk.Scale(self.side_bar, from_=80, to=99.9, orient=tk.HORIZONTAL, resolution=0.1, command=lambda x: self.refresh_plot())
        self.clip.set(98); self.clip.pack(fill=tk.X)

    def setup_plot(self):
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = CustomToolbar(self.canvas, self.main_frame)
        self.toolbar.update()
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_action)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def load_from_path(self, path):
        self.filename = path
        try:
            with segyio.open(self.filename, "r", ignore_geometry=True) as f:
                sr = segyio.tools.dt(f)/1000
                self.sr_in.delete(0, tk.END); self.sr_in.insert(0, str(sr))
                self.current_data = segyio.tools.collect(f.trace[:]).T
                n_samples, n_traces = self.current_data.shape
                self.extent = [0, n_traces, n_samples * sr, 0]
            self.ax.set_xlim(self.extent[0], self.extent[1])
            self.ax.set_ylim(self.extent[2], self.extent[3])
            self.refresh_plot()
            self.root.title(f"Viewer - {self.filename.split('/')[-1]}")
        except Exception as e:
            messagebox.showerror("Error", f"Load Failed: {e}")

    def load_horizons_data(self, horizons_data):
        self.horizons = horizons_data
        self.update_status()
        self.refresh_plot()

    def on_mouse_action(self, event):
        if event.inaxes != self.ax or not self.filename or self.toolbar.mode != '': return
        trace_idx = int(round(event.xdata))
        twt = event.ydata
        pts_list = self.horizons[self.active_layer]['points']
        changed = False

        if event.button == 1: # 좌클릭
            try:
                with segyio.open(self.filename, "r", ignore_geometry=True) as f:
                    if 0 <= trace_idx < f.tracecount:
                        # [수정] 헤더 이름 입력창 대신 coord_type으로 자동 판단
                        if self.coord_type == "CDP":
                            xk = segyio.TraceField.CDP_X
                            yk = segyio.TraceField.CDP_Y
                        else:
                            xk = segyio.TraceField.SourceX
                            yk = segyio.TraceField.SourceY

                        scalars = f.attributes(segyio.TraceField.SourceGroupScalar)[:]
                        scalar_val = float(scalars[0]) if len(scalars) > 0 else 1.0
                        if scalar_val == 0: scalar_val = 1.0
                        elif scalar_val < 0: scalar_val = 1.0 / abs(scalar_val)
                        
                        raw_x = float(f.header[trace_idx][xk])
                        raw_y = float(f.header[trace_idx][yk])
                        pts_list.append([raw_x * scalar_val, raw_y * scalar_val, twt, trace_idx])
                        pts_list.sort(key=lambda x: x[3])
                        changed = True
            except: pass
        elif event.button == 3: # 우클릭
            if pts_list:
                dists = [abs(p[3] - trace_idx) for p in pts_list]
                if min(dists) < 50: pts_list.pop(np.argmin(dists)); changed = True

        if changed:
            self.update_status(); self.refresh_plot()
            if self.on_update_callback: self.on_update_callback(self.filename, self.horizons)

    def refresh_plot(self):
        if self.current_data is None: return
        try: sr = float(self.sr_in.get())
        except: sr = 2.0
        n_samples, n_traces = self.current_data.shape
        self.extent = [0, n_traces, n_samples * sr, 0]
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        
        self.clip_val = float(self.clip.get())
        limit = np.nanpercentile(np.absolute(self.current_data), self.clip_val)
        if limit == 0: limit = 1.0
        
        self.ax.imshow(self.current_data, cmap="RdBu", vmin=-limit, vmax=limit, aspect='auto', extent=self.extent)
        for name, data in self.horizons.items():
            p_arr = np.array(data['points'])
            if len(p_arr) > 0:
                self.ax.plot(p_arr[:, 3], p_arr[:, 2], 'o', color=data['color'], markersize=4)
                if len(p_arr) >= 2:
                    self.ax.plot(p_arr[:, 3], p_arr[:, 2], color=data['color'], linewidth=1.5)
        self.ax.set_ylabel("Time (ms)")
        self.ax.set_xlabel("Trace Number")
        if xlim != (0.0, 1.0): self.ax.set_xlim(xlim); self.ax.set_ylim(ylim)
        self.canvas.draw()

    def on_layer_change(self, event): self.active_layer = self.layer_selector.get()
    
    def on_scroll(self, event):
        if event.inaxes != self.ax: return
        base_scale = 1.25; scale_factor = 1/base_scale if event.button == 'up' else base_scale
        cur_xlim, cur_ylim = self.ax.get_xlim(), self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_w = (cur_xlim[1]-cur_xlim[0])*scale_factor; new_h = (cur_ylim[1]-cur_ylim[0])*scale_factor
        rel_x = (cur_xlim[1]-xdata)/(cur_xlim[1]-cur_xlim[0]); rel_y = (cur_ylim[1]-ydata)/(cur_ylim[1]-cur_ylim[0])
        self.ax.set_xlim([xdata-new_w*(1-rel_x), xdata+new_w*rel_x])
        self.ax.set_ylim([ydata-new_h*(1-rel_y), ydata+new_h*rel_y])
        self.canvas.draw_idle()

    def show_headers(self):
        if not self.filename: return
        win = tk.Toplevel(self.root); win.title("Trace Header")
        txt = scrolledtext.ScrolledText(win, width=60); txt.pack(fill=tk.BOTH, expand=True)
        with segyio.open(self.filename, "r", ignore_geometry=True) as f:
            for k, v in f.header[0].items(): txt.insert(tk.END, f"{str(k):<25} | {v}\n")
        txt.configure(state='disabled')

    def update_status(self):
        info = "Pts: " + " | ".join([f"{k[8]}:{len(v['points'])}" for k, v in self.horizons.items()])
        self.hor_info.config(text=info)

    def clear_horizon(self):
        self.horizons[self.active_layer]['points'] = []
        self.update_status(); self.refresh_plot()
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
                        x, y = float(row['X']), float(row['Y'])
                        twt = float(row['TWT'])
                        tidx = int(float(row['TraceIdx']))
                        loaded_pts[layer].append([x, y, twt, tidx])
                for name, pts in loaded_pts.items():
                    if pts: self.horizons[name]['points'] = sorted(pts, key=lambda k: k[3])
                self.update_status(); self.refresh_plot()
                if self.on_update_callback: self.on_update_callback(self.filename, self.horizons)
                messagebox.showinfo("Import", "Horizon Loaded Successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import CSV: {e}")


# -----------------------------------------------------------
# 2. 프로젝트 매니저
# -----------------------------------------------------------
class ProjectManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Seismic Project Manager (Final)")
        self.root.geometry("1100x850")
        self.survey_lines = {}
        self.line_plots = {}
        self.horizon_plots = {}
        self.cbar = None 

        top_frame = tk.Frame(root, height=60, bg="#ecf0f1", bd=1, relief=tk.RAISED)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Button(top_frame, text="📂 Load SEGYs", command=self.add_files, 
                  bg="#2980b9", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5, pady=10)
        
        tk.Button(top_frame, text="💾 Save Project", command=self.save_project, 
                  bg="#27ae60", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="📂 Open Project", command=self.load_project, 
                  bg="#f39c12", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        tk.Label(top_frame, text="| Horizon:", bg="#ecf0f1").pack(side=tk.LEFT, padx=(10,5))
        self.horizon_selector = ttk.Combobox(top_frame, values=['None', 'Horizon A', 'Horizon B', 'Horizon C'], state="readonly", width=10)
        self.horizon_selector.current(0); self.horizon_selector.pack(side=tk.LEFT)
        self.horizon_selector.bind("<<ComboboxSelected>>", self.on_viz_change)

        tk.Label(top_frame, text="View:", bg="#ecf0f1").pack(side=tk.LEFT, padx=(10,5))
        self.view_mode = ttk.Combobox(top_frame, values=['Scatter Points', 'Contour Map'], state="readonly", width=12)
        self.view_mode.current(0); self.view_mode.pack(side=tk.LEFT)
        self.view_mode.bind("<<ComboboxSelected>>", self.on_viz_change)

        self.status_lbl = tk.Label(root, text="Ready.", bd=1, relief=tk.SUNKEN, anchor=tk.W); self.status_lbl.pack(side=tk.BOTTOM, fill=tk.X)
        self.map_frame = tk.Frame(root, bg="white"); self.map_frame.pack(fill=tk.BOTH, expand=True)
        self.setup_basemap()

    def setup_basemap(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_title("Seismic Base Map")
        self.ax.set_xlabel("East (X)"); self.ax.set_ylabel("North (Y)")
        self.ax.set_aspect('auto') 
        self.ax.ticklabel_format(useOffset=False, style='plain')
        self.ax.grid(True, linestyle='--', alpha=0.5)

        divider = make_axes_locatable(self.ax)
        self.cax = divider.append_axes("right", size="3%", pad=0.1)
        self.cax.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self.map_frame).update()
        self.fig.canvas.mpl_connect('pick_event', self.on_line_pick)

    def process_segy_file(self, filepath, existing_horizons=None):
        try:
            fname = filepath.split('/')[-1]
            if not os.path.exists(filepath): return None

            with segyio.open(filepath, "r", ignore_geometry=True) as f:
                scalars = f.attributes(segyio.TraceField.SourceGroupScalar)[:]; scalar = float(scalars[0]) if len(scalars)>0 else 1.0
                if scalar==0: scalar=1.0
                elif scalar<0: scalar=1.0/abs(scalar)
                
                raw_x = f.attributes(segyio.TraceField.CDP_X)[:]; raw_y = f.attributes(segyio.TraceField.CDP_Y)[:]
                coord_type = "CDP"
                if np.all(raw_x==0) and np.all(raw_y==0):
                    raw_x = f.attributes(segyio.TraceField.SourceX)[:]; raw_y = f.attributes(segyio.TraceField.SourceY)[:]
                    coord_type = "Source"
                if np.all(raw_x==0): return None
                
                x = raw_x.astype(float)*scalar; y = raw_y.astype(float)*scalar; step=max(1, len(x)//1000)
                horizons = existing_horizons if existing_horizons else {}
                self.survey_lines[fname] = {'path': filepath, 'x': x[::step], 'y': y[::step], 'type': coord_type, 'horizons': horizons}
                return fname
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None

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
        save_data = {}
        for fname, data in self.survey_lines.items():
            save_data[fname] = {'path': data['path'], 'horizons': data['horizons']}
        try:
            with open(path, 'w', encoding='utf-8') as f: json.dump(save_data, f, indent=4)
            messagebox.showinfo("Success", "Project saved successfully!")
        except Exception as e: messagebox.showerror("Error", f"Failed to save project: {e}")

    def load_project(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Project", "*.json")])
        if not path: return
        try:
            with open(path, 'r', encoding='utf-8') as f: loaded_data = json.load(f)
            self.survey_lines = {}; self.line_plots = {}; self.horizon_plots = {}
            self.ax.clear(); self.setup_basemap()
            count = 0
            self.status_lbl.config(text="Restoring project..."); self.root.update()
            for fname, data in loaded_data.items():
                if self.process_segy_file(data['path'], existing_horizons=data['horizons']): count += 1
            self.update_map(); self.draw_visualization()
            self.status_lbl.config(text=f"Project restored: {count} lines.")
            messagebox.showinfo("Success", "Project loaded successfully!")
        except Exception as e: messagebox.showerror("Error", f"Failed to load project: {e}")

    def update_map(self):
        if not self.survey_lines: return
        colors = plt.cm.nipy_spectral(np.linspace(0,1,len(self.survey_lines)))
        idx = 0
        for lid, d in self.survey_lines.items():
            if lid in self.line_plots: idx+=1; continue
            l, = self.ax.plot(d['x'], d['y'], label=lid, linewidth=2, color=colors[idx%len(colors)], picker=5)
            if len(d['x'])>0: self.ax.text(d['x'][0], d['y'][0], lid[:10], fontsize=8, color=colors[idx%len(colors)], fontweight='bold')
            self.line_plots[l] = lid; idx += 1
        self.ax.legend(loc='upper right', fontsize='x-small')
        self.ax.relim(); self.ax.autoscale_view(); self.canvas.draw()

    def on_line_pick(self, event):
        if event.artist in self.line_plots:
            lid = self.line_plots[event.artist]
            data = self.survey_lines[lid]
            self.status_lbl.config(text=f"Opening {lid}")
            new_win = tk.Toplevel(self.root)
            viewer = SegyViewer(new_win, filename=data['path'], on_update_callback=self.on_horizon_update, coord_type=data['type'])
            viewer.load_horizons_data(data['horizons'])

    def on_horizon_update(self, filepath, horizons):
        fname = filepath.split('/')[-1]
        if fname in self.survey_lines:
            self.survey_lines[fname]['horizons'] = horizons
            self.draw_visualization()

    def on_viz_change(self, event): self.draw_visualization()

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

        mappable = None
        if mode == 'Scatter Points':
            sc = self.ax.scatter(all_x, all_y, c=all_z, cmap='viridis_r', s=30, edgecolors='k', linewidth=0.5, zorder=5)
            self.horizon_plots['sc'] = sc; mappable = sc
        elif mode == 'Contour Map':
            if len(all_x) < 4: return
            xi = np.linspace(min(all_x), max(all_x), 200)
            yi = np.linspace(min(all_y), max(all_y), 200)
            Xi, Yi = np.meshgrid(xi, yi)
            try:
                Zi = griddata((all_x, all_y), all_z, (Xi, Yi), method='cubic')
                cf = self.ax.contourf(Xi, Yi, Zi, levels=15, cmap='viridis_r', alpha=0.7, zorder=4)
                self.horizon_plots['cf'] = cf; mappable = cf
                self.horizon_plots['cl'] = self.ax.contour(Xi, Yi, Zi, levels=15, colors='k', linewidths=0.5, zorder=5)
            except: pass

        if mappable:
            self.cax.axis('on')
            plt.colorbar(mappable, cax=self.cax, label='Time (ms)')
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    manager = ProjectManager(root)
    root.mainloop()
