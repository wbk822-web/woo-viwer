import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import numpy as np
import segyio
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# 툴바의 실시간 좌표 표시(레이아웃 흔들림 원인)를 방지하기 위한 커스텀 클래스
class CustomToolbar(NavigationToolbar2Tk):
    def set_message(self, s): pass

class SegyViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Woo Interpreter 20251226")
        self.root.geometry("1400x950")

        self.filename = ""
        self.current_data = None
        self.extent = None
        
        # Horizon 데이터 구조
        self.horizons = {
            'Horizon A': {'color': 'yellow', 'points': []},
            'Horizon B': {'color': 'cyan', 'points': []},
            'Horizon C': {'color': 'lime', 'points': []}
        }
        self.active_layer = 'Horizon A'

        # --- 레이아웃 설정 ---
        # 좌측 사이드바 (컨트롤 패널)
        self.side_bar = tk.Frame(root, width=300, bg="#f0f0f0", padx=10, pady=10)
        self.side_bar.pack(side=tk.LEFT, fill=tk.Y)
        self.side_bar.pack_propagate(False)

        # 우측 메인 영역 (단면도)
        self.main_frame = tk.Frame(root, bg="white")
        self.main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_ui()
        self.setup_plot()

    def setup_ui(self):
        """사이드바 UI 구성"""
        # 1. 파일 섹션
        tk.Label(self.side_bar, text="--- File & Analysis ---", font=('Arial', 10, 'bold'), bg="#f0f0f0").pack(pady=(5,5))
        tk.Button(self.side_bar, text="📂 SEGY 파일 열기", command=self.open_file, height=2, bg="#dcdcdc").pack(fill=tk.X, pady=2)
        tk.Button(self.side_bar, text="🔍 Trace 헤더 보기", command=self.show_headers, bg="#3498db", fg="white").pack(fill=tk.X, pady=2)
        tk.Button(self.side_bar, text="🗺️ Base Map (TWT Gradient)", command=self.show_basemap, bg="#e67e22", fg="white").pack(fill=tk.X, pady=5)

        # 2. 해석 섹션
        tk.Label(self.side_bar, text="--- Interpretation ---", font=('Arial', 10, 'bold'), bg="#f0f0f0").pack(pady=(15,5))
        self.layer_selector = ttk.Combobox(self.side_bar, values=list(self.horizons.keys()), state="readonly")
        self.layer_selector.current(0)
        self.layer_selector.pack(fill=tk.X, pady=5)
        self.layer_selector.bind("<<ComboboxSelected>>", self.on_layer_change)

        tk.Button(self.side_bar, text="📍 현재 층 포인트 초기화", command=self.clear_horizon, bg="#e74c3c", fg="white").pack(fill=tk.X, pady=2)
        tk.Button(self.side_bar, text="💾 해석 결과 저장 (CSV)", command=self.save_horizon, bg="#2c3e50", fg="white").pack(fill=tk.X, pady=2)
        
        self.hor_info = tk.Label(self.side_bar, text="A:0 | B:0 | C:0", bg="#f0f0f0", font=('Arial', 9))
        self.hor_info.pack(pady=5)
        
        guide = "* 휠: 확대/축소\n* 좌클릭: 해석 추가\n* 우클릭: 해석 삭제"
        tk.Label(self.side_bar, text=guide, font=('Arial', 8), fg="#555", justify=tk.LEFT).pack(pady=5)

        # 3. 설정 섹션
        tk.Label(self.side_bar, text="--- Display Settings ---", font=('Arial', 10, 'bold'), bg="#f0f0f0").pack(pady=(15,5))
        
        tk.Label(self.side_bar, text="X/Y Header Names", bg="#f0f0f0", font=('Arial', 8)).pack(anchor="w")
        h_frame = tk.Frame(self.side_bar, bg="#f0f0f0")
        h_frame.pack(fill=tk.X)
        self.x_hdr = tk.Entry(h_frame, width=10); self.x_hdr.insert(0, "SourceX"); self.x_hdr.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.y_hdr = tk.Entry(h_frame, width=10); self.y_hdr.insert(0, "SourceY"); self.y_hdr.pack(side=tk.LEFT, expand=True, fill=tk.X)

        tk.Label(self.side_bar, text="Sampling Rate (ms)", bg="#f0f0f0", font=('Arial', 8)).pack(anchor="w")
        self.sr_in = tk.Entry(self.side_bar); self.sr_in.insert(0, "2.0"); self.sr_in.pack(fill=tk.X)

        tk.Label(self.side_bar, text="Contrast (Clip %)", font=('Arial', 9, 'bold'), bg="#f0f0f0", fg="red").pack(pady=(15,0))
        self.clip = tk.Scale(self.side_bar, from_=80, to=99.9, orient=tk.HORIZONTAL, resolution=0.1, command=lambda x: self.refresh_plot())
        self.clip.set(98)
        self.clip.pack(fill=tk.X)

    def setup_plot(self):
        """메인 단면도 플롯 영역 설정"""
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1) # 여백 고정
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.toolbar = CustomToolbar(self.canvas, self.main_frame)
        self.toolbar.update()

        # 이벤트 바인딩
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_action)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    # --- 이벤트 핸들러 ---
    def on_layer_change(self, event):
        self.active_layer = self.layer_selector.get()

    def on_scroll(self, event):
        """마우스 휠 줌 기능"""
        if event.inaxes != self.ax: return
        base_scale = 1.25
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale
        
        cur_xlim, cur_ylim = self.ax.get_xlim(), self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        rel_x = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rel_y = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([xdata - new_width * (1 - rel_x), xdata + new_width * rel_x])
        self.ax.set_ylim([ydata - new_height * (1 - rel_y), ydata + new_height * rel_y])
        self.canvas.draw_idle()

    def on_mouse_action(self, event):
        """좌클릭(추가), 우클릭(삭제)"""
        if event.inaxes != self.ax or not self.filename or self.toolbar.mode != '': return
        
        trace_idx = int(round(event.xdata))
        twt = event.ydata
        pts_list = self.horizons[self.active_layer]['points']

        if event.button == 1: # 좌클릭
            try:
                with segyio.open(self.filename, "r", ignore_geometry=True) as f:
                    if 0 <= trace_idx < f.tracecount:
                        xk = getattr(segyio.TraceField, self.x_hdr.get())
                        yk = getattr(segyio.TraceField, self.y_hdr.get())
                        pts_list.append([f.header[trace_idx][xk], f.header[trace_idx][yk], twt, trace_idx])
                        pts_list.sort(key=lambda x: x[3])
            except: pass
        elif event.button == 3: # 우클릭
            if pts_list:
                dists = [abs(p[3] - trace_idx) for p in pts_list]
                if min(dists) < 50: pts_list.pop(np.argmin(dists))

        self.update_status(); self.refresh_plot()

    # --- 주요 기능 메서드 ---
    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("SEGY files", "*.sgy *.segy"), ("All", "*.*")])
        if file_path:
            self.filename = file_path
            with segyio.open(self.filename, "r", ignore_geometry=True) as f:
                sr = segyio.tools.dt(f)/1000
                self.sr_in.delete(0, tk.END); self.sr_in.insert(0, str(sr))
                # 초기 로드 범위 자동 설정 (전체 데이터)
                self.extent = [0, f.tracecount, len(f.samples)*sr, 0]
                self.current_data = segyio.tools.collect(f.trace[:]).T
            
            self.ax.set_xlim(self.extent[0], self.extent[1])
            self.ax.set_ylim(self.extent[2], self.extent[3])
            self.refresh_plot()

    def refresh_plot(self):
        if self.current_data is None: return
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        
        # Clip 기반 진폭 계산
        limit = np.nanpercentile(np.absolute(self.current_data), self.clip.get())
        if limit == 0: limit = 1.0
        
        self.ax.imshow(self.current_data, cmap="RdBu", vmin=-limit, vmax=limit, aspect='auto', extent=self.extent)
        
        # Horizon 그리기 (직선 보간)
        for name, data in self.horizons.items():
            p_arr = np.array(data['points'])
            if len(p_arr) > 0:
                self.ax.plot(p_arr[:, 3], p_arr[:, 2], 'o', color=data['color'], markersize=4)
                if len(p_arr) >= 2:
                    self.ax.plot(p_arr[:, 3], p_arr[:, 2], color=data['color'], linewidth=1.5)
        
        self.ax.set_ylabel("Time (ms)"); self.ax.set_xlabel("Trace Number")
        if xlim != (0.0, 1.0): # 줌 상태 유지
            self.ax.set_xlim(xlim); self.ax.set_ylim(ylim)
        self.canvas.draw()

    def show_basemap(self):
        """TWT 그레디언트가 적용된 베이스맵"""
        if not self.filename: return
        map_win = tk.Toplevel(self.root)
        map_win.title("Structural Base Map (Color=TWT)")
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # 배경 Survey 라인
        with segyio.open(self.filename, "r", ignore_geometry=True) as f:
            xk, yk = getattr(segyio.TraceField, self.x_hdr.get()), getattr(segyio.TraceField, self.y_hdr.get())
            step = max(1, f.tracecount // 2000) # 최대 2000개 점만 표시
            x_bg = [f.header[i][xk] for i in range(0, f.tracecount, step)]
            y_bg = [f.header[i][yk] for i in range(0, f.tracecount, step)]
            ax.plot(x_bg, y_bg, 'k.', alpha=0.1, markersize=1)

        # 해석된 포인트 그리기 (TWT 그레디언트)
        found_pts = False
        for n, d in self.horizons.items():
            p = np.array(d['points'])
            if len(p) > 0:
                sc = ax.scatter(p[:, 0], p[:, 1], c=p[:, 2], cmap='viridis_r', s=35, edgecolors='none', zorder=5, label=n)
                if n == self.active_layer: # 현재 층 기준 컬러바 표시
                    plt.colorbar(sc, ax=ax, label='Time (ms)')
                found_pts = True
        
        ax.set_title("Base Map View"); ax.legend(); ax.grid(True, alpha=0.2)
        cvs = FigureCanvasTkAgg(fig, master=map_win); cvs.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(cvs, map_win).update(); cvs.draw()

    def show_headers(self):
        if not self.filename: return
        win = tk.Toplevel(self.root); win.title("Trace 0 Header Details")
        txt = scrolledtext.ScrolledText(win, width=70, height=40)
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        with segyio.open(self.filename, "r", ignore_geometry=True) as f:
            for k, v in f.header[0].items(): txt.insert(tk.END, f"{str(k):<25} | {v}\n")
        txt.configure(state='disabled')

    def update_status(self):
        info = "포인트: " + " | ".join([f"{k[8]}:{len(v['points'])}" for k, v in self.horizons.items()])
        self.hor_info.config(text=info)

    def clear_horizon(self):
        self.horizons[self.active_layer]['points'] = []; self.update_status(); self.refresh_plot()

    def save_horizon(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            with open(path, 'w') as f:
                f.write("Layer,X,Y,TWT,TraceIdx\n")
                for n, d in self.horizons.items():
                    for p in d['points']: f.write(f"{n},{p[0]},{p[1]},{p[2]},{p[3]}\n")
            messagebox.showinfo("완료", "데이터가 CSV로 저장되었습니다.")

if __name__ == "__main__":
    root = tk.Tk()
    viewer = SegyViewer(root)
    root.mainloop()
