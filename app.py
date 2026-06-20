from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.audio_features import mel_spectrogram
from src.model import load_checkpoint, predict_mel


AUDIO_TYPES = [("Audio", "*.wav *.flac *.mp3 *.ogg *.m4a *.aac"), ("All files", "*.*")]
MODEL_TYPES = [("PyTorch model", "*.pt"), ("All files", "*.*")]


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Детектор монтажа аудио")
        self.root.geometry("900x620")

        self.model_path = tk.StringVar(value=self.default_model())
        self.audio_path = tk.StringVar()
        self.result = tk.StringVar(value="Выберите модель и аудио")
        self.details = tk.StringVar(value="")
        self.canvas: FigureCanvasTkAgg | None = None

        self.build_ui()

    def default_model(self) -> str:
        models = sorted(Path("models").glob("*.pt"))
        return str(models[0]) if models else ""

    def build_ui(self) -> None:
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="Детектор монтажа аудио", font=("Arial", 18, "bold")).pack(anchor="w")

        model_row = tk.Frame(frame)
        model_row.pack(fill=tk.X, pady=(16, 4))
        tk.Label(model_row, text="Модель:", width=10, anchor="w").pack(side=tk.LEFT)
        tk.Entry(model_row, textvariable=self.model_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(model_row, text="Выбрать", command=self.choose_model).pack(side=tk.LEFT, padx=(8, 0))

        audio_row = tk.Frame(frame)
        audio_row.pack(fill=tk.X, pady=4)
        tk.Label(audio_row, text="Аудио:", width=10, anchor="w").pack(side=tk.LEFT)
        tk.Entry(audio_row, textvariable=self.audio_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(audio_row, text="Выбрать", command=self.choose_audio).pack(side=tk.LEFT, padx=(8, 0))

        tk.Button(frame, text="Проверить аудио", command=self.analyze, height=2).pack(fill=tk.X, pady=12)
        self.result_label = tk.Label(frame, textvariable=self.result, font=("Arial", 16, "bold"), anchor="w")
        self.result_label.pack(fill=tk.X)
        self.details_label = tk.Label(frame, textvariable=self.details, justify=tk.LEFT, anchor="w", height=2)
        self.details_label.pack(fill=tk.X, pady=(4, 10))

        self.plot_frame = tk.Frame(frame, borderwidth=1, relief=tk.SOLID)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        self.show_empty_plot()

    def choose_model(self) -> None:
        path = filedialog.askopenfilename(filetypes=MODEL_TYPES)
        if path:
            self.model_path.set(path)

    def choose_audio(self) -> None:
        path = filedialog.askopenfilename(filetypes=AUDIO_TYPES)
        if path:
            self.audio_path.set(path)

    def analyze(self) -> None:
        model_file = Path(self.model_path.get())
        audio_file = Path(self.audio_path.get())

        if not model_file.exists():
            messagebox.showerror("Ошибка", "Файл модели не найден")
            return
        if not audio_file.exists():
            messagebox.showerror("Ошибка", "Аудиофайл не найден")
            return

        try:
            self.result.set("Идет анализ...")
            self.details.set("")
            self.root.update()

            model, checkpoint = load_checkpoint(model_file, device="cpu")
            mel = mel_spectrogram(audio_file)
            probability = predict_mel(model, mel, device="cpu")
            threshold = float(checkpoint.get("threshold", 0.5))
            verdict = "Обнаружен монтаж" if probability >= threshold else "Похоже на оригинал"

            self.result.set(verdict)
            self.details.set(f"Вероятность монтажа: {probability:.3f}\nПорог: {threshold:.2f}")
            self.show_mel_plot(mel)
        except Exception as exc:
            self.result.set("Ошибка анализа")
            messagebox.showerror("Ошибка", str(exc))

    def show_empty_plot(self) -> None:
        fig = Figure(figsize=(7, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Здесь будет мел-спектрограмма", ha="center", va="center")
        ax.set_axis_off()
        self.set_plot(fig)

    def show_mel_plot(self, mel) -> None:
        fig = Figure(figsize=(7, 4), dpi=100)
        ax = fig.add_subplot(111)
        display_mel = self.crop_padded_mel(mel)
        ax.imshow(display_mel, origin="lower", aspect="auto", cmap="magma")
        ax.set_title("Mel spectrogram")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Mel bands")
        fig.tight_layout()
        self.set_plot(fig)

    def crop_padded_mel(self, mel):
        active_columns = mel.any(axis=0)
        if not active_columns.any():
            return mel
        last_active_column = active_columns.nonzero()[0][-1] + 1
        return mel[:, :last_active_column]

    def set_plot(self, fig: Figure) -> None:
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
