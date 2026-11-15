# Giao diện Tkinter và điều khiển webcam.


import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk, ImageOps

from src.pipeline import process_frame
from src.metrics import PerformanceMonitor
from src.exporter import save_experiment


class EdgeDetectionApp:

    def __init__(self, root):
        self.root = root

        self.root.title(
            "De tai 20 - Xu ly bien webcam bang Sobel va Laplacian"
        )
        self.root.geometry("1280x780")
        self.root.minsize(1020, 610)

        # Trang thai webcam
        self.camera = None
        self.running = False
        self.paused = False
        self.after_id = None

        # Du lieu frame gan nhat
        self.last_frame = None
        self.last_result = None

        # Thong ke
        self.monitor = PerformanceMonitor(
            max_samples=120
        )
        self.previous_frame_time = None

        # Cac tham so
        self.gaussian_var = tk.StringVar(value="5")
        self.sobel_var = tk.IntVar(value=40)
        self.laplacian_var = tk.IntVar(value=40)
        self.kernel_var = tk.StringVar(value="4")
        self.binary_var = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(
            value="Webcam chua khoi dong"
        )

        self.fps_var = tk.StringVar(value="0.0")
        self.sobel_time_var = tk.StringVar(value="0.00 ms")
        self.laplacian_time_var = tk.StringVar(
            value="0.00 ms"
        )
        self.total_time_var = tk.StringVar(
            value="0.00 ms"
        )
        self.frames_var = tk.StringVar(value="0")

        # Luu doi tuong anh Tkinter de tranh bi mat anh
        self.photo_refs = {}
        self.displayed_images = {}
        self.canvas_image_ids = {}

        self.configure_theme()
        self.create_interface()
        self.update_button_states()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.on_close
        )

    # ==========================================
    # 1. GIAO DIEN
    # ==========================================

    
    def configure_theme(self):
        self.colors = {
            "bg": "#0B1220", "panel": "#151F30", "card": "#1D2A3D",
            "input": "#25364C", "text": "#F1F5FB", "muted": "#A7B5C9",
            "accent": "#38BDF8", "green": "#22C55E", "border": "#32445C"
        }
        c = self.colors
        self.root.configure(bg=c["bg"])
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background=c["bg"])
        style.configure("Panel.TFrame", background=c["panel"])
        style.configure("Card.TFrame", background=c["card"])
        style.configure("Title.TLabel", background=c["bg"], foreground=c["text"], font=("Segoe UI", 19, "bold"))
        style.configure("Subtitle.TLabel", background=c["bg"], foreground=c["muted"], font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=c["panel"], foreground=c["text"], font=("Segoe UI", 10, "bold"))
        style.configure("Field.TLabel", background=c["panel"], foreground=c["muted"], font=("Segoe UI", 9))
        style.configure("Value.TLabel", background=c["panel"], foreground=c["accent"], font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", background=c["panel"], foreground=c["muted"], font=("Segoe UI", 9), padding=(12, 8))
        style.configure("Preview.TLabelframe", background=c["panel"], bordercolor=c["border"], relief="solid", borderwidth=1)
        style.configure("Preview.TLabelframe.Label", background=c["panel"], foreground=c["text"], font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", background="#0284C7", foreground="#FFFFFF", borderwidth=0, padding=(8, 9), font=("Segoe UI", 9, "bold"))
        style.map("Accent.TButton", background=[("disabled", c["input"]), ("active", "#0369A1")], foreground=[("disabled", c["muted"])])
        style.configure("Neutral.TButton", background=c["input"], foreground=c["text"], borderwidth=0, padding=(8, 8))
        style.map("Neutral.TButton", background=[("active", "#36516D"), ("disabled", c["panel"])], foreground=[("disabled", c["muted"])])
        style.configure("TCombobox", fieldbackground=c["input"], background=c["input"], foreground=c["text"], arrowcolor=c["text"], padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", c["input"])], foreground=[("readonly", c["text"])])
        # Thanh truot cyan: thu gon va noi bat tren nen toi
        style.configure(
            "Accent.Horizontal.TScale",
            background=c["panel"],
            troughcolor=c["input"],
            bordercolor=c["panel"],
            lightcolor=c["accent"],
            darkcolor=c["accent"]
        )
        style.map(
            "Accent.Horizontal.TScale",
            background=[("active", c["accent"]), ("!disabled", c["accent"])],
            lightcolor=[("active", c["accent"])],
            darkcolor=[("active", c["accent"])]
        )
        style.configure("TCheckbutton", background=c["panel"], foreground=c["text"], font=("Segoe UI", 9))
        style.map("TCheckbutton", background=[("active", c["panel"])], foreground=[("active", c["text"])])

    def create_interface(self):
        c = self.colors
        main = ttk.Frame(self.root, style="App.TFrame", padding=14)
        main.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(main, style="App.TFrame")
        header.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(header, text="EDGE VISION  |  DE TAI 20", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Phat hien bien webcam bang Sobel va Laplacian  •  Xu ly thoi gian thuc", style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))

        status = ttk.Label(main, textvariable=self.status_var, style="Status.TLabel", anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        content = ttk.Frame(main, style="App.TFrame")
        content.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(content, style="Panel.TFrame", width=270)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        sidebar.pack_propagate(False)
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(0, weight=1)

        scroll_area = ttk.Frame(sidebar, style="Panel.TFrame")
        scroll_area.grid(row=0, column=0, sticky="nsew")
        scroll_area.grid_columnconfigure(0, weight=1)
        scroll_area.grid_rowconfigure(0, weight=1)

        canvas = tk.Canvas(scroll_area, bg=c["panel"], bd=0, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(scroll_area, orient=tk.VERTICAL, width=13, command=canvas.yview,
                                 bg=c["input"], troughcolor=c["panel"], activebackground=c["accent"])
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        controls = ttk.Frame(canvas, style="Panel.TFrame", padding=(10, 10, 9, 8))
        window_id = canvas.create_window((0, 0), window=controls, anchor="nw")

        def refresh_scroll(event=None):
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=bbox)

        def fit_width(event):
            canvas.itemconfigure(window_id, width=event.width)
            refresh_scroll()

        controls.bind("<Configure>", refresh_scroll)
        canvas.bind("<Configure>", fit_width)
        self.create_controls(controls)

        # Hai nut luon co dinh ben duoi sidebar
        footer = ttk.Frame(sidebar, style="Panel.TFrame", padding=(10, 8, 20, 12))
        footer.grid(row=1, column=0, sticky="ew")
        self.create_result_controls(footer)

        def on_wheel(event):
            delta = -1 if event.delta > 0 else 1
            canvas.yview_scroll(delta * 3, "units")
            return "break"

        def bind_wheel(widget):
            if not isinstance(widget, (tk.Scale, ttk.Combobox)):
                widget.bind("<MouseWheel>", on_wheel, add="+")
            for child in widget.winfo_children():
                bind_wheel(child)

        bind_wheel(controls)
        canvas.bind("<MouseWheel>", on_wheel, add="+")
        self.root.after_idle(refresh_scroll)

        preview = ttk.Frame(content, style="App.TFrame")
        preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.create_previews(preview)

    def create_controls(self, parent):
        c = self.colors
        ttk.Label(parent, text="WEBCAM", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        self.start_button = ttk.Button(parent, text="▶  Bat dau / Tiep tuc", command=self.start_camera, style="Accent.TButton")
        self.start_button.pack(fill=tk.X, pady=3)
        buttons = ttk.Frame(parent, style="Panel.TFrame")
        buttons.pack(fill=tk.X, pady=(3, 14))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        self.pause_button = ttk.Button(buttons, text="Ⅱ  Tam dung", command=self.pause_camera, style="Neutral.TButton")
        self.pause_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.stop_button = ttk.Button(buttons, text="■  Dung", command=self.stop_camera, style="Neutral.TButton")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ttk.Label(parent, text="THAM SO XU LY", style="Section.TLabel").pack(anchor="w", pady=(0, 10))
        ttk.Label(parent, text="Gaussian kernel", style="Field.TLabel").pack(anchor="w")
        gaussian = ttk.Combobox(parent, textvariable=self.gaussian_var, values=["3", "5", "7", "9"], state="readonly")
        gaussian.pack(fill=tk.X, pady=(4, 12))
        gaussian.bind("<<ComboboxSelected>>", self.on_settings_change)

        self.sobel_label = ttk.Label(parent, text="Nguong Sobel: 40", style="Field.TLabel")
        self.sobel_label.pack(anchor="w")
        self.sobel_scale = ttk.Scale(
            parent, from_=0, to=255, orient=tk.HORIZONTAL,
            variable=self.sobel_var, command=self.on_settings_change,
            style="Accent.Horizontal.TScale"
        )
        self.sobel_scale.pack(fill=tk.X, pady=(6, 13))

        self.laplacian_label = ttk.Label(parent, text="Nguong Laplacian: 40", style="Field.TLabel")
        self.laplacian_label.pack(anchor="w")
        self.laplacian_scale = ttk.Scale(
            parent, from_=0, to=255, orient=tk.HORIZONTAL,
            variable=self.laplacian_var, command=self.on_settings_change,
            style="Accent.Horizontal.TScale"
        )
        self.laplacian_scale.pack(fill=tk.X, pady=(6, 13))

        ttk.Label(parent, text="Laplacian kernel", style="Field.TLabel").pack(anchor="w")
        kernel = ttk.Combobox(parent, textvariable=self.kernel_var, values=["4", "8"], state="readonly")
        kernel.pack(fill=tk.X, pady=(4, 8))
        kernel.bind("<<ComboboxSelected>>", self.on_settings_change)
        ttk.Checkbutton(parent, text="Hien thi bien nhi phan", variable=self.binary_var,
                        command=self.on_display_mode_change).pack(anchor="w", pady=(2, 15))

        ttk.Label(parent, text="HIEU NANG", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        self.add_stat(parent, "FPS", self.fps_var)
        self.add_stat(parent, "Sobel", self.sobel_time_var)
        self.add_stat(parent, "Laplacian", self.laplacian_time_var)
        self.add_stat(parent, "Tong", self.total_time_var)
        self.add_stat(parent, "Frames", self.frames_var)

    def create_result_controls(self, parent):
        ttk.Label(parent, text="KET QUA", style="Section.TLabel").pack(anchor="w", pady=(0, 7))
        self.save_button = ttk.Button(parent, text="↓  Luu anh va CSV", command=self.save_results, style="Accent.TButton")
        self.save_button.pack(fill=tk.X, pady=(0, 6))
        self.reset_button = ttk.Button(parent, text="↺  Dat lai thong ke", command=self.reset_statistics, style="Neutral.TButton")
        self.reset_button.pack(fill=tk.X)

    def add_stat(self, parent, title, variable):
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=title, style="Field.TLabel").pack(side=tk.LEFT)
        ttk.Label(row, textvariable=variable, style="Value.TLabel").pack(side=tk.RIGHT)

    def create_previews(self, parent):
        """Tao 4 o video co gian theo kich thuoc cua so."""
        self.image_labels = {}
        names = [
            ("original", "01  |  WEBCAM GOC"),
            ("gaussian", "02  |  GAUSSIAN BLUR"),
            ("sobel", "03  |  SOBEL - CUSTOM"),
            ("laplacian", "04  |  LAPLACIAN - CUSTOM")
        ]
        for i, (key, title) in enumerate(names):
            panel = ttk.LabelFrame(
                parent, text=title, padding=8,
                style="Preview.TLabelframe"
            )
            panel.grid(
                row=i // 2, column=i % 2,
                sticky="nsew", padx=5, pady=5
            )
            # Canvas khong ep khung panel theo kich thuoc anh.
            canvas = tk.Canvas(
                panel, bg="#080E18", bd=0, highlightthickness=0,
                width=100, height=100
            )
            canvas.pack(fill=tk.BOTH, expand=True)
            self.image_labels[key] = canvas
            canvas.bind(
                "<Configure>",
                lambda event, name=key: self.redraw_image(name)
            )
        for i in range(2):
            parent.rowconfigure(i, weight=1, uniform="preview")
            parent.columnconfigure(i, weight=1, uniform="preview")

    def get_settings(self):

        return {
            "gaussian_size": int(
                self.gaussian_var.get()
            ),
            "sobel_threshold": self.sobel_var.get(),
            "laplacian_threshold":
                self.laplacian_var.get(),
            "laplacian_kernel": int(
                self.kernel_var.get()
            )
        }

    def on_settings_change(self, value=None):

        self.sobel_label.config(
            text=(
                f"Nguong Sobel: "
                f"{self.sobel_var.get()}"
            )
        )

        self.laplacian_label.config(
            text=(
                f"Nguong Laplacian: "
                f"{self.laplacian_var.get()}"
            )
        )

        # Khong tron thong ke giua cac cau hinh
        self.monitor.reset()
        self.previous_frame_time = None

        # Anh cu khong con khop voi tham so moi
        self.last_result = None

        self.update_statistics()
        self.update_button_states()

    def on_display_mode_change(self):
        self.status_var.set(
            "Da thay doi che do hien thi bien"
        )

    def update_statistics(self):

        stats = self.monitor.get_stats()

        self.fps_var.set(
            f"{stats['fps']:.1f}"
        )

        self.sobel_time_var.set(
            f"{stats['sobel_ms']:.2f} ms"
        )

        self.laplacian_time_var.set(
            f"{stats['laplacian_ms']:.2f} ms"
        )

        self.total_time_var.set(
            f"{stats['total_ms']:.2f} ms"
        )

        self.frames_var.set(
            str(stats["frames"])
        )

    def reset_statistics(self):

        self.monitor.reset()
        self.previous_frame_time = None
        self.update_statistics()

        self.status_var.set(
            "Da dat lai thong ke"
        )

    # ==========================================
    # 3. DIEU KHIEN WEBCAM
    # ==========================================

    def update_button_states(self):

        if not self.running:
            self.start_button.config(
                state=tk.NORMAL
            )
            self.pause_button.config(
                state=tk.DISABLED
            )
            self.stop_button.config(
                state=tk.DISABLED
            )

        elif self.paused:
            self.start_button.config(
                state=tk.NORMAL
            )
            self.pause_button.config(
                state=tk.DISABLED
            )
            self.stop_button.config(
                state=tk.NORMAL
            )

        else:
            self.start_button.config(
                state=tk.DISABLED
            )
            self.pause_button.config(
                state=tk.NORMAL
            )
            self.stop_button.config(
                state=tk.NORMAL
            )

        can_save = (
            self.last_frame is not None
            and self.last_result is not None
        )

        self.save_button.config(
            state=(
                tk.NORMAL if can_save
                else tk.DISABLED
            )
        )

    def start_camera(self):

        # Tiep tuc neu dang tam dung
        if self.running and self.paused:

            self.paused = False
            self.previous_frame_time = None

            self.status_var.set(
                "Webcam da tiep tuc"
            )

            self.update_button_states()
            self.schedule_next_frame()
            return

        if self.running:
            return

        # Mo webcam moi
        camera = cv2.VideoCapture(
            0,
            cv2.CAP_DSHOW
        )

        # Thu backend mac dinh neu DSHOW that bai
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            camera.release()

            messagebox.showerror(
                "Loi webcam",
                "Khong the mo webcam. "
                "Hay kiem tra camera va quyen truy cap."
            )
            return

        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )
        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

        self.camera = camera
        self.running = True
        self.paused = False

        self.last_frame = None
        self.last_result = None

        self.reset_statistics()

        self.status_var.set(
            "Webcam dang hoat dong"
        )

        self.update_button_states()
        self.schedule_next_frame()

    def pause_camera(self):

        if not self.running or self.paused:
            return

        self.paused = True

        if self.after_id is not None:
            self.root.after_cancel(
                self.after_id
            )
            self.after_id = None

        self.status_var.set(
            "Webcam dang tam dung"
        )

        self.update_button_states()

    def stop_camera(self):

        self.running = False
        self.paused = False

        if self.after_id is not None:
            self.root.after_cancel(
                self.after_id
            )
            self.after_id = None

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.previous_frame_time = None

        self.status_var.set(
            "Webcam da dung"
        )

        self.update_button_states()

    def schedule_next_frame(self):

        if self.running and not self.paused:
            self.after_id = self.root.after(
                10,
                self.update_frame
            )

    # ==========================================
    # 4. XU LY VA HIEN THI FRAME
    # ==========================================

    def update_frame(self):

        self.after_id = None

        if not self.running or self.paused:
            return

        try:
            # Moc thoi gian dau frame
            current_time = time.perf_counter()

            # Doc frame
            ret, frame = self.camera.read()

            if not ret:
                raise RuntimeError(
                    "Khong doc duoc frame webcam."
                )

            # Tinh FPS theo khoang cach frame
            if self.previous_frame_time is None:
                fps = None
            else:
                delta = (
                    current_time
                    - self.previous_frame_time
                )
                fps = (
                    1.0 / delta if delta > 0
                    else None
                )

            self.previous_frame_time = current_time

            # Lay tham so giao dien
            settings = self.get_settings()

            # Xu ly bang pipeline da xay dung
            result = process_frame(
                frame,
                **settings
            )

            # Cap nhat thong ke
            self.monitor.update(
                result["times"],
                fps
            )

            self.last_frame = frame.copy()
            self.last_result = result

            # Hien thi webcam va ket qua
            self.show_image(
                "original",
                frame
            )

            self.show_image(
                "gaussian",
                result["blurred"]
            )

            if self.binary_var.get():
                sobel = result["sobel_binary"]
                laplacian = result[
                    "laplacian_binary"
                ]
            else:
                sobel = result["sobel"]
                laplacian = result["laplacian"]

            self.show_image(
                "sobel",
                sobel
            )

            self.show_image(
                "laplacian",
                laplacian
            )

            self.update_statistics()
            self.update_button_states()

        except Exception as error:

            self.stop_camera()

            messagebox.showerror(
                "Loi xu ly",
                str(error)
            )
            return

        # Len lich cho frame ke tiep
        self.schedule_next_frame()

    def show_image(self, key, image):
        """Luu frame va hien thi trong canvas co gian."""
        self.displayed_images[key] = image
        self.redraw_image(key)

    def redraw_image(self, key):
        """Fit anh vao o hien tai, giu nguyen ty le, can giua."""
        image = self.displayed_images.get(key)
        if image is None:
            return
        canvas = self.image_labels[key]
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width < 20 or height < 20:
            return

        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image = ImageOps.contain(
            pil_image, (width, height),
            method=Image.Resampling.BILINEAR
        )
        photo = ImageTk.PhotoImage(image=pil_image, master=self.root)
        self.photo_refs[key] = photo
        center_x = width // 2
        center_y = height // 2
        if key not in self.canvas_image_ids:
            self.canvas_image_ids[key] = canvas.create_image(
                center_x, center_y, image=photo, anchor="center"
            )
        else:
            canvas.itemconfigure(self.canvas_image_ids[key], image=photo)
            canvas.coords(self.canvas_image_ids[key], center_x, center_y)

    # ==========================================
    # 5. LUU KET QUA
    # ==========================================

    def save_results(self):

        if (
            self.last_frame is None
            or self.last_result is None
        ):
            messagebox.showwarning(
                "Chua co du lieu",
                "Hay bat webcam va xu ly anh truoc."
            )
            return

        try:
            folder = save_experiment(
                frame=self.last_frame,
                result=self.last_result,
                stats=self.monitor.get_stats(),
                settings=self.get_settings()
            )

            self.status_var.set(
                f"Da luu: {folder}"
            )

            messagebox.showinfo(
                "Thanh cong",
                f"Da luu anh va CSV.\n\n{folder}"
            )

        except Exception as error:
            messagebox.showerror(
                "Loi luu du lieu",
                str(error)
            )

    # ==========================================
    # 6. DONG UNG DUNG
    # ==========================================

    def on_close(self):

        self.stop_camera()
        self.root.destroy()


def run_app():

    root = tk.Tk()

    app = EdgeDetectionApp(root)

    root.mainloop()


if __name__ == "__main__":
    run_app()
