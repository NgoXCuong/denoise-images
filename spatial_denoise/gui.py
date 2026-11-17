"""Giao diện Tkinter; mọi cập nhật widget đều thực hiện trên luồng chính."""

import queue
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image

from .core import RunConfig, RunResult, run_pipeline, sobel
from .imaging import demo_image, gradient_display, load_image
from .viewer import ImageCard, SharedView


NAMES = {"original": "Ảnh gốc", "noisy": "Ảnh nhiễu", "mean": "Lọc trung bình", "median": "Lọc trung vị"}
NOISES = {"Gaussian": "gaussian", "Muối tiêu": "salt_pepper"}
MODES = {"Thử nghiệm tạo nhiễu": "simulate", "Khử nhiễu ảnh có sẵn": "existing"}


class DenoiseApp(tk.Tk):
    # Khởi tạo cửa sổ, ảnh mẫu, trạng thái và giao diện; đăng ký sự kiện và tác vụ đầu tiên.
    def __init__(self):
        super().__init__()
        self.title("Khử nhiễu ảnh • Bộ lọc không gian & Sobel")
        width, height = min(1380, self.winfo_screenwidth() - 60), min(940, self.winfo_screenheight() - 100)
        self.geometry(f"{width}x{height}")
        self.minsize(1000, 720)
        self.configure(bg="#f3f5f7")
        self.events = queue.Queue()
        self.busy = False
        self.result: RunResult | None = None
        self.original = demo_image()
        self.source = "Ảnh tổng hợp tự tạo"
        self.reference = None
        self.reference_source = None
        self.interactive = []
        self.disabled_states = []
        self.cards = {}
        self.scroll_panels = []
        self.view = SharedView(self)
        self.fullscreen = False
        self._style()
        self._layout()
        for variable in (self.sigma, self.amount, self.kernel, self.seed, self.threshold, self.noise):
            variable.trace_add("write", self._config_changed)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", self._exit_fullscreen)
        self.bind("<Control-o>", lambda _event: self._open_image() if not self.busy else None)
        self.after(100, self._poll)
        self.after(200, self._run)

    # Thiết lập phông chữ, màu sắc và kiểu hiển thị cho các thành phần Tkinter.
    def _style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background="#f3f5f7", foreground="#20344b")
        style.configure("TNotebook", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 12), font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")])
        style.configure("Card.TFrame", background="white")
        style.configure("Card.TLabel", background="white")
        style.configure("CardMuted.TLabel", background="white", foreground="#63758a", font=("Segoe UI", 9))
        style.configure("CardTitle.TLabel", background="white", font=("Segoe UI", 10, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Muted.TLabel", foreground="#63758a")
        style.configure("TButton", padding=(10, 8))
        style.configure("Accent.TButton", background="#147d64", foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#10634f"), ("disabled", "#acbdcc")])
        style.configure("Treeview", rowheight=30, background="white", fieldbackground="white")
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), padding=5)
        style.configure("TLabelframe", padding=12)

    # Đăng ký widget cần khóa trong lúc xử lý và trả lại widget để bố trí giao diện.
    def _track(self, widget):
        self.interactive.append(widget)
        return widget

    # Bố trí cột thiết lập có thể cuộn và ba tab kết quả.
    # Đặt nút xử lý và lưu ngoài vùng cuộn để luôn thao tác được.
    def _layout(self):
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        main = ttk.Frame(self, padding=6)
        main.grid(row=0, column=0, sticky="nsew")
        main.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        sidebar_host = self.sidebar_host = ttk.Frame(main, width=300)
        sidebar_host.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        sidebar_host.grid_propagate(False)
        sidebar_host.rowconfigure(0, weight=1)
        sidebar_host.columnconfigure(0, weight=1)
        sidebar_canvas = tk.Canvas(sidebar_host, bg="#f3f5f7", highlightthickness=0, width=286)
        sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        sidebar_scroll = ttk.Scrollbar(sidebar_host, orient="vertical", command=sidebar_canvas.yview)
        sidebar_scroll.grid(row=0, column=1, sticky="ns")
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        self.sidebar = ttk.Frame(sidebar_canvas)
        sidebar_id = sidebar_canvas.create_window(0, 0, window=self.sidebar, anchor="nw")
        self.sidebar.bind("<Configure>", lambda _event: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")))
        sidebar_canvas.bind("<Configure>", lambda event: sidebar_canvas.itemconfigure(sidebar_id, width=event.width))
        self.sidebar.columnconfigure(0, weight=1)
        self._controls()
        actions = ttk.Frame(sidebar_host, padding=(0, 10, 14, 0))
        actions.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.run_button = self._track(ttk.Button(actions, text="TẠO NHIỄU & SO SÁNH", style="Accent.TButton", command=self._run))
        self.run_button.pack(fill="x")
        self.save_button = self._track(ttk.Menubutton(actions, text="Lưu ảnh kết quả…", state="disabled", padding=(10, 8)))
        save_menu = tk.Menu(self.save_button, tearoff=False)
        save_menu.add_command(label="Lưu ảnh nhiễu…", command=lambda: self._save_image("noisy"))
        save_menu.add_command(label="Lưu ảnh trung vị…", command=lambda: self._save_image("median"))
        save_menu.add_command(label="Lưu ảnh trung bình…", command=lambda: self._save_image("mean"))
        self.save_button.configure(menu=save_menu)
        self.save_button.pack(fill="x", pady=(7, 0))
        self.scroll_panels.append((sidebar_host, sidebar_canvas))
        self.bind_all("<MouseWheel>", self._scroll_panel, add="+")
        self.tabs = ttk.Notebook(main)
        self.tabs.grid(row=0, column=1, sticky="nsew")
        compare, edges, metrics = (ttk.Frame(self.tabs, padding=5) for _ in range(3))
        self.tabs.add(compare, text="01  Kết quả lọc")
        self.tabs.add(edges, text="02  Bảo toàn biên · Sobel")
        self.tabs.add(metrics, text="03  Đánh giá · MSE / PSNR")
        self._comparison(compare)
        self._edges_tab(edges)
        self._metrics_tab(metrics)
        self.tabs.bind("<<NotebookTabChanged>>", lambda _event: self.view.redraw())

    # Tạo các trường chọn ảnh, chế độ, loại nhiễu, cửa sổ, seed và ảnh sạch đối chiếu.
    def _controls(self):
        parent = self.sidebar
        self.mode = tk.StringVar(value="Thử nghiệm tạo nhiễu")
        ttk.Label(parent, text="Chọn chế độ xử lý", style="Section.TLabel").grid(row=0, sticky="w", pady=(0, 6))
        mode_combo = self._track(ttk.Combobox(parent, textvariable=self.mode, values=list(MODES), state="readonly"))
        mode_combo.grid(row=1, sticky="ew", pady=(0, 16))
        mode_combo.bind("<<ComboboxSelected>>", self._mode_changed)
        ttk.Label(parent, text="1.  Ảnh đầu vào", style="Section.TLabel").grid(row=2, sticky="w", pady=(0, 7))
        buttons = ttk.Frame(parent)
        buttons.grid(row=3, sticky="ew")
        self.open_image_button = self._track(ttk.Button(buttons, text="Mở ảnh…", command=self._open_image))
        self.open_image_button.pack(side="left", fill="x", expand=True)
        self.demo_button = self._track(ttk.Button(buttons, text="Ảnh mẫu", command=self._demo))
        self.demo_button.pack(side="left", padx=(6, 0))
        self.info = tk.StringVar(value="Ảnh mẫu • 512 × 512 px • xám 8 bit")
        ttk.Label(parent, textvariable=self.info, wraplength=265, style="Muted.TLabel").grid(row=4, sticky="w", pady=(6, 15))
        self.filter_heading = ttk.Label(parent, text="2.  Nhiễu và bộ lọc", style="Section.TLabel")
        self.filter_heading.grid(row=5, sticky="w", pady=(0, 5))
        self.noise = tk.StringVar(value="Muối tiêu")
        self.sigma = tk.StringVar(value="20")
        self.amount = tk.StringVar(value="5")
        self.kernel = tk.StringVar(value="3")
        self.seed = tk.StringVar(value="42")
        self.threshold = tk.StringVar(value="20")
        self.noise_fields = []
        for row, title, variable, values in (
                (6, "Loại nhiễu", self.noise, list(NOISES)),
                (8, "Độ nhiễu Gaussian: σ (0–100)", self.sigma, None),
                (10, "Tỷ lệ điểm nhiễu muối tiêu (%)", self.amount, None),
                (12, "Cỡ cửa sổ (3 = 3×3, 5 = 5×5, 7 = 7×7)", self.kernel, (3, 5, 7))):
            label = ttk.Label(parent, text=title)
            label.grid(row=row, sticky="w", pady=(5, 3))
            widget = (ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
                      if values is not None else ttk.Entry(parent, textvariable=variable))
            self._track(widget).grid(row=row + 1, sticky="ew", ipady=2)
            if variable is not self.kernel:
                self.noise_fields.extend((label, widget))
            if variable is self.noise:
                self.noise_combo = widget
                widget.bind("<<ComboboxSelected>>", self._noise_state)
            elif variable is self.sigma:
                self.sigma_entry, self.sigma_label = widget, label
            elif variable is self.amount:
                self.amount_entry, self.amount_label = widget, label
        ttk.Label(parent, text="Hai bộ lọc dùng cùng cỡ cửa sổ. Cửa sổ lớn có thể làm mất chi tiết.",
                  style="Muted.TLabel", wraplength=265).grid(row=14, sticky="w", pady=(6, 10))
        self.advanced_box = ttk.Frame(parent)
        self.advanced_box.grid(row=16, sticky="ew")
        self.advanced_box.columnconfigure(0, weight=1)
        seed_label = ttk.Label(self.advanced_box, text="Mã tạo nhiễu (seed)")
        seed_label.grid(row=0, sticky="w", pady=(5, 3))
        self.seed_entry = self._track(ttk.Entry(self.advanced_box, textvariable=self.seed))
        self.seed_entry.grid(row=1, sticky="ew")
        self.noise_fields.extend((seed_label, self.seed_entry))
        ttk.Label(self.advanced_box, text="Ngưỡng biên Sobel để tính F1").grid(row=2, sticky="w", pady=(8, 3))
        self._track(ttk.Entry(self.advanced_box, textvariable=self.threshold)).grid(row=3, sticky="ew")
        self.reference_box = ttk.LabelFrame(parent, text="Ảnh sạch làm chuẩn (tùy chọn)")
        self.reference_box.grid(row=18, sticky="ew", pady=(12, 0))
        self._track(ttk.Button(self.reference_box, text="Chọn ảnh sạch đối chiếu…", command=self._open_reference)).pack(fill="x")
        self.clear_reference_button = self._track(ttk.Button(self.reference_box, text="Bỏ ảnh đối chiếu", command=self._clear_reference, state="disabled"))
        self.clear_reference_button.pack(fill="x", pady=(5, 0))
        self.reference_info = tk.StringVar(value="Không có ảnh chuẩn (tùy chọn).")
        ttk.Label(self.reference_box, textvariable=self.reference_info, style="Muted.TLabel", wraplength=265).pack(anchor="w", pady=(6, 0))
        self.reference_box.grid_remove()
        self.mode_hint = tk.StringVar(value="Cùng seed → cùng ảnh nhiễu. Mỗi lần chạy bắt đầu từ ảnh gốc.")
        ttk.Label(parent, textvariable=self.mode_hint, style="Muted.TLabel", wraplength=265).grid(row=17, sticky="w", pady=(10, 0))
        self._noise_state()

    # Ẩn, hiện và bật, tắt các trường theo chế độ xử lý và loại nhiễu đang chọn.
    def _noise_state(self, _event=None):
        simulate = MODES[self.mode.get()] == "simulate"
        for widget in self.noise_fields:
            widget.grid() if simulate else widget.grid_remove()
        for label, entry, visible in (
                (self.sigma_label, self.sigma_entry, simulate and self.noise.get() == "Gaussian"),
                (self.amount_label, self.amount_entry, simulate and self.noise.get() == "Muối tiêu")):
            for widget in (label, entry):
                widget.grid() if visible else widget.grid_remove()
        self.noise_combo.configure(state="readonly" if simulate else "disabled")
        self.seed_entry.configure(state="normal" if simulate else "disabled")
        self.sigma_entry.configure(state="normal" if simulate and self.noise.get() == "Gaussian" else "disabled")
        self.amount_entry.configure(state="normal" if simulate and self.noise.get() == "Muối tiêu" else "disabled")
        self.demo_button.configure(state="normal" if simulate else "disabled")
        self.clear_reference_button.configure(state="normal" if self.reference is not None else "disabled")

    # Chuyển sự kiện lăn chuột trong vùng thiết lập thành thao tác cuộn cột bên trái.
    def _scroll_panel(self, event):
        widget = event.widget
        while widget is not None:
            for host, canvas in self.scroll_panels:
                if widget == host:
                    canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
                    return "break"
            widget = getattr(widget, "master", None)

    # Đánh dấu tham số chưa áp dụng khi cấu hình nhập khác cấu hình đã chạy.
    def _config_changed(self, *_args):
        if self.result is None or self.busy:
            return
        try:
            changed = self._read_config() != self.result.config
        except ValueError:
            changed = True
        prefix = "THAM SỐ MỚI CHƯA ÁP DỤNG · " if changed else ""
        self.snapshot.set(prefix + self.result_caption)

    # Cập nhật nhãn và trường nhập khi đổi chế độ; xóa ảnh đối chiếu và kết quả cũ.
    def _mode_changed(self, _event=None):
        direct = MODES[self.mode.get()] == "existing"
        self.reference, self.reference_source = None, None
        self.reference_info.set("Không có ảnh chuẩn (tùy chọn).")
        if direct:
            self.reference_box.grid()
        else:
            self.reference_box.grid_remove()
        self.open_image_button.configure(text="Mở ảnh nhiễu…" if direct else "Mở ảnh…")
        self.run_button.configure(text="KHỬ NHIỄU & SO SÁNH" if direct else "TẠO NHIỄU & SO SÁNH")
        self.filter_heading.configure(text="2.  Thiết lập bộ lọc" if direct else "2.  Nhiễu và bộ lọc")
        self.mode_hint.set("Lọc trực tiếp ảnh đầu vào. Chọn ảnh sạch cùng nội dung, kích thước và vị trí để tính chỉ số."
                           if direct else "Mỗi lần chạy tạo lại nhiễu từ ảnh gốc.\nCùng seed → cùng ảnh nhiễu.")
        for (name, row), card in self.cards.items():
            title = ("Ảnh sạch đối chiếu" if name == "original" else "Ảnh nhiễu đầu vào") if direct and name in ("original", "noisy") else NAMES[name]
            card.set_title(title if row == 0 else f"Sobel · {title.lower()}")
        self._noise_state()
        self._clear_result()

    # Xóa chỉ số và ảnh kết quả cũ, khóa nút lưu và hiển thị lại ảnh đầu vào.
    # Cập nhật thông báo theo chế độ và tình trạng có ảnh sạch đối chiếu.
    def _clear_result(self):
        self.result = None
        self.save_button.configure(state="disabled")
        self.table.delete(*self.table.get_children())
        direct = MODES[self.mode.get()] == "existing"
        self.view.fit(redraw=False)
        for (name, row), card in self.cards.items():
            card.detail.configure(text=" ")
            image = None
            if name == ("noisy" if direct else "original"):
                image = self.original
            elif direct and name == "original":
                image = self.reference
            preview = image if row == 0 else gradient_display(sobel(image)) if image is not None else None
            card.set_array(preview,
                           "Không có ảnh chuẩn" if direct and name == "original" else "Chưa có kết quả")
        self.metric_note.set("Không có ảnh chuẩn: chưa tính MSE, PSNR, MSE Sobel và F1; Sobel dùng quan sát biên."
                             if direct and self.reference is None else "MSE/PSNR được tính với ảnh sạch chuẩn sau khi chạy xử lý.")
        self.insight.set("Chọn cửa sổ rồi bấm xử lý. Có thể chọn ảnh sạch đối chiếu ở cột bên trái."
                         if direct else "Chọn loại và mức nhiễu rồi bấm xử lý. Hai bộ lọc dùng cùng một ảnh nhiễu.")
        self.snapshot.set("Đã chọn ảnh/chế độ. Bấm KHỬ NHIỄU & SO SÁNH để lọc trực tiếp."
                          if direct else "Đã chọn ảnh/chế độ. Bấm TẠO NHIỄU & SO SÁNH để xử lý.")

    # Mở ảnh sạch đối chiếu và kiểm tra cùng kích thước, số kênh với ảnh đầu vào.
    def _open_reference(self):
        filename = filedialog.askopenfilename(parent=self, title="Chọn ảnh sạch cùng nội dung và vị trí với ảnh nhiễu",
                                              filetypes=[("Ảnh", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("Tất cả", "*.*")])
        if not filename:
            return
        try:
            reference = load_image(filename)
            if reference.shape != self.original.shape:
                raise ValueError("Ảnh sạch đối chiếu phải cùng kích thước với ảnh nhiễu. Không tự co giãn ảnh.")
        except (OSError, ValueError) as error:
            messagebox.showerror("Không mở được ảnh đối chiếu", str(error), parent=self)
            return
        self.reference, self.reference_source = reference, filename
        self.reference_info.set(f"Ảnh sạch: {Path(filename).name}")
        self._noise_state()
        self._clear_result()

    # Bỏ ảnh sạch đối chiếu và xóa kết quả cũ để tránh dùng sai nguồn tham chiếu.
    def _clear_reference(self):
        self.reference, self.reference_source = None, None
        self.reference_info.set("Không có ảnh chuẩn (tùy chọn).")
        self._noise_state()
        self._clear_result()

    # Tạo tab so sánh ảnh gốc, ảnh nhiễu, lọc trung bình và lọc trung vị.
    def _comparison(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        heading = ttk.Frame(parent)
        heading.grid(row=0, sticky="ew", pady=(0, 4))
        ttk.Label(heading, text="So sánh trước và sau lọc", style="Section.TLabel").pack(side="left")
        self._view_controls(heading)
        self.snapshot = tk.StringVar(value="Đang chuẩn bị thí nghiệm với ảnh mẫu…")
        snapshot = ttk.Label(parent, textvariable=self.snapshot, style="Muted.TLabel", wraplength=850)
        snapshot.grid(row=1, sticky="ew", pady=(0, 4))
        self.image_grid = ttk.Frame(parent)
        self.image_grid.grid(row=2, sticky="nsew")
        for name, title in NAMES.items():
            self.cards[(name, 0)] = ImageCard(self.image_grid, title, self.view)
        self.insight = tk.StringVar(value="Chọn ảnh sạch, chọn mức nhiễu rồi bấm TẠO NHIỄU & SO SÁNH.")
        parent.bind("<Configure>", lambda event: snapshot.configure(wraplength=max(100, event.width - 20)))
        self._arrange_images()

    # Sắp xếp bốn khung ảnh thành lưới 2×2 và vẽ lại với vùng xem đồng bộ.
    def _arrange_images(self, _event=None):
        for index, name in enumerate(NAMES):
            row, column = divmod(index, 2)
            self.image_grid.rowconfigure(row, weight=1, uniform="images")
            self.image_grid.columnconfigure(column, weight=1, uniform="images")
            self.cards[(name, 0)].grid(row=row, column=column, sticky="nsew", padx=4, pady=4)
        self.view.redraw()

    # Tạo tab gồm bốn khung Sobel dùng cùng mức zoom và vùng quan sát.
    def _edges_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        heading = ttk.Frame(parent)
        heading.grid(row=0, sticky="ew", pady=(0, 4))
        ttk.Label(heading, text="Biên Sobel · Thang hiển thị ×2", style="Section.TLabel").pack(side="left")
        self._view_controls(heading)
        grid = ttk.Frame(parent)
        grid.grid(row=1, sticky="nsew")
        for i, (name, title) in enumerate(NAMES.items()):
            r, c = divmod(i, 2)
            grid.rowconfigure(r, weight=1, uniform="edges")
            grid.columnconfigure(c, weight=1, uniform="edges")
            card = ImageCard(grid, f"Sobel · {title.lower()}", self.view)
            card.grid(row=r, column=c, sticky="nsew", padx=3, pady=3)
            self.cards[(name, 1)] = card

    # Tạo các nút thu nhỏ, phóng to, xem 100% và vừa khung.
    def _view_controls(self, parent):
        controls = ttk.Frame(parent)
        controls.pack(side="right")
        for text, command in (("−", lambda: self.view.zoom(0.8)),
                              ("+", lambda: self.view.zoom(1.25)),
                              ("100%", self.view.actual_size),
                              ("Vừa khung", self.view.fit)):
            ttk.Button(controls, text=text, command=command, width=9 if text == "Vừa khung" else 4,
                       padding=(3, 2)).pack(side="left", padx=2)

    # Tạo bảng MSE, PSNR, MSE Sobel, F1 và thời gian lọc cùng hướng dẫn đọc kết quả.
    def _metrics_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        ttk.Label(parent, text="CHẤT LƯỢNG SAU XỬ LÝ", style="Section.TLabel").grid(row=0, sticky="w", pady=(5, 14))
        ttk.Label(parent, text="MSE càng thấp, PSNR càng cao thì kết quả càng gần ảnh sạch chuẩn.",
                  style="Muted.TLabel", wraplength=650).grid(row=1, sticky="w", pady=(0, 16))
        columns = ("method", "mse", "psnr", "sobel", "f1", "time")
        self.table = ttk.Treeview(parent, columns=columns, show="headings", height=3, selectmode="none")
        for key, title, width in zip(columns, ("Phương pháp", "MSE ↓", "PSNR ↑ (dB)", "MSE Sobel ↓", "F1 biên ↑", "Lọc (ms)"),
                                     (155, 105, 110, 115, 90, 90)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, minwidth=70, anchor="w" if key == "method" else "center")
        self.table.grid(row=2, sticky="ew")
        scrollbar = ttk.Scrollbar(parent, orient="horizontal", command=self.table.xview)
        scrollbar.grid(row=3, sticky="ew")
        self.table.configure(xscrollcommand=scrollbar.set)
        self.table.tag_configure("best", background="#e0f2e9")
        self.metric_note = tk.StringVar(value="Chạy xử lý để xem chỉ số.")
        ttk.Label(parent, textvariable=self.metric_note, style="Muted.TLabel", wraplength=650).grid(row=4, sticky="w", pady=(15, 20))
        ttk.Label(parent, text="Cách đọc kết quả", style="Section.TLabel").grid(row=5, sticky="w")
        ttk.Label(parent, text="• MSE / PSNR: so sánh cường độ sáng với ảnh sạch.\n"
                  "• MSE Sobel / F1: so sánh gradient và biên với Sobel ảnh sạch; F1 phụ thuộc ngưỡng.\n"
                  "• Không có ảnh sạch chuẩn: không tính các chỉ số trên; vẫn xem và lưu được ảnh.\n"
                  "• PSNR = ∞ khi ảnh giống hệt ảnh chuẩn. Cần quan sát thêm chi tiết khi zoom 100%.",
                  style="Muted.TLabel", wraplength=650, justify="left").grid(row=6, sticky="w", pady=(8, 0))

        ttk.Label(parent, textvariable=self.insight, style="Section.TLabel",
                  wraplength=650).grid(row=7, sticky="w", pady=(16, 0))

    # Chuyển trạng thái toàn màn hình và ẩn hoặc khôi phục cột thiết lập.
    def _toggle_fullscreen(self, _event=None):
        self.fullscreen = not self.fullscreen
        self.attributes("-fullscreen", self.fullscreen)
        for widget in (self.sidebar_host,):
            widget.grid_remove() if self.fullscreen else widget.grid()

    # Thoát chế độ toàn màn hình khi đang bật.
    def _exit_fullscreen(self, _event=None):
        if self.fullscreen:
            self._toggle_fullscreen()

    # Đọc trường nhập thành RunConfig và kiểm tra hợp lệ.
    # Bỏ qua tham số nhiễu không áp dụng; đổi tỷ lệ muối tiêu từ phần trăm sang xác suất.
    def _read_config(self):
        try:
            direct = MODES[self.mode.get()] == "existing"
            config = RunConfig(input_mode=MODES[self.mode.get()], noise=NOISES[self.noise.get()],
                               sigma=float(self.sigma.get()) if not direct and self.noise.get() == "Gaussian" else 0,
                               amount=float(self.amount.get()) / 100 if not direct and self.noise.get() == "Muối tiêu" else 0,
                               kernel=int(self.kernel.get()),
                               seed=0 if direct else int(self.seed.get()), edge_threshold=float(self.threshold.get()))
            config.validate()
            return config
        except (ValueError, KeyError) as error:
            raise ValueError(f"Tham số không hợp lệ: {error}\nDùng dấu chấm cho số thập phân.") from error

    # Khóa các trường thao tác khi chạy tác vụ và khôi phục trạng thái khi hoàn tất.
    def _set_busy(self, active: bool):
        self.busy = active
        if active:
            self.disabled_states = [(widget, str(widget.cget("state"))) for widget in self.interactive]
            for widget, _ in self.disabled_states:
                widget.configure(state="disabled")
        else:
            for widget, state in self.disabled_states:
                widget.configure(state=state)
            self.save_button.configure(state="normal" if self.result else "disabled")
            self._noise_state()

    # Chạy tác vụ trên luồng nền; đưa kết quả hoặc lỗi vào hàng đợi để xử lý trên luồng chính.
    def _work(self, task, callback):
        self._set_busy(True)

        # Thực thi tác vụ nền, đưa callback và kết quả vào hàng đợi hoặc ghi nhận ngoại lệ.
        def worker():
            try:
                value = task()
                self.events.put(("done", (callback, value)))
            except Exception as error:
                traceback.print_exc()
                self.events.put(("error", str(error)))

        threading.Thread(target=worker, daemon=True).start()

    # Đọc hàng đợi trên luồng chính để cập nhật giao diện hoặc báo lỗi; hẹn lần kiểm tra tiếp.
    def _poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "done":
                    callback, result = value
                    self._set_busy(False)
                    callback(result)
                elif kind == "error":
                    self._set_busy(False)
                    messagebox.showerror("Không thể thực hiện", value, parent=self)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # Kiểm tra cấu hình, sao chép ảnh và khởi chạy pipeline trên luồng nền.
    def _run(self):
        if self.busy:
            return
        try:
            config = self._read_config()
        except ValueError as error:
            messagebox.showerror("Kiểm tra tham số", str(error), parent=self)
            return
        image = self.original.copy()
        reference = self.reference.copy() if self.reference is not None else None
        reference_source = self.reference_source if config.input_mode == "existing" else self.source
        self._work(lambda: run_pipeline(image, config, reference=reference, reference_source=reference_source), self._show_result)

    # Hiển thị ảnh và Sobel, điền bảng chỉ số và cập nhật mô tả cấu hình đã chạy.
    # Đánh dấu bộ lọc có PSNR tốt nhất khi cải thiện so với ảnh chưa lọc.
    def _show_result(self, result):
        self.result = result
        for (name, row), card in self.cards.items():
            image = result.images.get(name) if row == 0 else (gradient_display(result.gradients[name]) if name in result.gradients else None)
            card.set_array(image, "Không có ảnh chuẩn" if name == "original" else "Chưa có kết quả")
        self.table.delete(*self.table.get_children())
        best = max(row["psnr_db"] for row in result.metrics if row["method"] != "noisy") if result.has_reference else None
        baseline = result.metrics[0]["psnr_db"]
        # Định dạng chỉ số để hiển thị: dấu gạch khi thiếu, ký hiệu vô cực hoặc số làm tròn.
        def value(number, digits=3):
            return "—" if number is None else "∞" if np.isposinf(number) else f"{number:.{digits}f}"
        for row in result.metrics:
            name = row["method"]
            detail = (f"MSE {value(row['mse'], 2)}  ·  PSNR {value(row['psnr_db'], 2)} dB"
                      if result.has_reference else "Không có ảnh chuẩn để tính MSE/PSNR")
            self.cards[(name, 0)].detail.configure(text=detail)
            self.table.insert("", "end", values=(NAMES[name], value(row["mse"]),
                              value(row["psnr_db"]), value(row["sobel_mse"]), value(row["edge_f1"], 4),
                              "—" if name == "noisy" else f"{row['filter_ms']:.2f}"),
                              tags=("best",) if best is not None and best > baseline and name != "noisy" and row["psnr_db"] == best else ())
        self.cards[("original", 0)].detail.configure(text="Ảnh sạch dùng làm chuẩn" if result.has_reference else "Ảnh chuẩn là tùy chọn")
        config = result.config
        if config.input_mode == "existing":
            level = "Khử nhiễu ảnh có sẵn · " + ("Có ảnh sạch đối chiếu" if result.has_reference else "Không có ảnh chuẩn")
        else:
            level = (f"Gaussian σ={config.sigma:g}" if config.noise == "gaussian" else f"Muối tiêu {config.amount * 100:g}%") + f" · seed={config.seed}"
        self.result_caption = f"KẾT QUẢ: {level} · k={config.kernel} · ngưỡng Sobel={config.edge_threshold:g}"
        self.snapshot.set(self.result_caption)
        self.metric_note.set("Không có ảnh chuẩn: chưa tính MSE, PSNR, MSE Sobel và F1; Sobel dùng quan sát biên."
                             if not result.has_reference else "Hàng xanh: PSNR cao nhất giữa hai bộ lọc và cao hơn ảnh chưa lọc. F1 tham chiếu Sobel ảnh sạch; phụ thuộc ngưỡng.")
        if not result.has_reference:
            text = "Đã khử nhiễu bằng cả hai bộ lọc. Xem tab Đánh giá để so sánh chỉ số và thời gian lọc."
        elif best > baseline:
            winners = " / ".join(NAMES[r["method"]] for r in result.metrics if r["method"] != "noisy" and r["psnr_db"] == best)
            change = f"PSNR tăng {best - baseline:.2f} dB" if np.isfinite(best) else "Kết quả trùng ảnh chuẩn (PSNR ∞)"
            text = f"{winners}: {change} so với ảnh chưa lọc."
        else:
            text = "PSNR chưa cải thiện so với ảnh chưa lọc. Thử cửa sổ nhỏ hơn và quan sát chi tiết ở mức 100%."
        self.insight.set(text)
        self.save_button.configure(state="normal")

    # Cập nhật ảnh đầu vào và thông tin kích thước, màu; bỏ ảnh đối chiếu và kết quả cũ.
    def _load(self, image, source):
        self.original, self.source, self.result = image, source, None
        color = "ảnh màu RGB 8 bit/kênh" if image.ndim == 3 else "ảnh xám 8 bit"
        self.info.set(f"{Path(source).name}\n{image.shape[1]} × {image.shape[0]} px • {color}")
        self._clear_reference()

    # Mở hộp thoại chọn ảnh phù hợp chế độ và nạp ảnh; thông báo nếu không đọc được.
    def _open_image(self):
        if self.busy:
            return
        filename = filedialog.askopenfilename(parent=self, title="Chọn ảnh đã bị nhiễu" if MODES[self.mode.get()] == "existing" else "Chọn ảnh gốc sạch",
                                              filetypes=[("Ảnh", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("Tất cả", "*.*")])
        if not filename:
            return
        try:
            self._load(load_image(filename), filename)
        except (OSError, ValueError) as error:
            messagebox.showerror("Không mở được ảnh", str(error), parent=self)

    # Nạp ảnh tổng hợp mẫu và chạy xử lý với cấu hình hiện tại.
    def _demo(self):
        self._load(demo_image(), "Ảnh tổng hợp tự tạo")
        self._run()

    # Chọn nơi lưu ảnh kết quả PNG đúng độ phân giải và lưu trên luồng nền.
    # Kiểm tra đường dẫn để tránh ghi đè ảnh đầu vào hoặc ảnh sạch đối chiếu.
    def _save_image(self, method):
        if not self.result or self.busy:
            return
        filename = filedialog.asksaveasfilename(
            parent=self, title=f"Lưu ảnh {NAMES[method].lower()}", defaultextension=".png",
            initialfile=f"{method}_result.png", filetypes=[("Ảnh PNG", "*.png")])
        if not filename:
            return
        path = Path(filename)
        sources = [self.source, self.reference_source]
        if any(source and path.resolve() == Path(source).resolve() for source in sources):
            messagebox.showerror("Chọn tên tệp khác", "Hãy lưu kết quả vào tệp khác để giữ ảnh đầu vào.", parent=self)
            return
        image = self.result.images[method].copy()

        # Lưu bản sao ảnh kết quả thành PNG và trả về đường dẫn đã lưu.
        def save():
            Image.fromarray(image).save(path, format="PNG")
            return path

        self._work(save, lambda _path: None)

    # Đóng cửa sổ khi không có tác vụ đang chạy; thông báo chờ nếu còn bận.
    def _close(self):
        if self.busy:
            messagebox.showinfo("Đang xử lý", "Hãy đợi xử lý hoàn tất.", parent=self)
            return
        self.destroy()


# Khởi tạo ứng dụng và chạy vòng lặp sự kiện Tkinter.
def main():
    DenoiseApp().mainloop()


if __name__ == "__main__":
    main()
