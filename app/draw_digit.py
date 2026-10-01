import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

CANVAS_SIZE = 280
BRUSH_SIZE = 20
MNIST_SIZE = 28


class KumoCanvas:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Kumo No Su")

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            cursor="cross"
        )

        self.canvas.pack(padx=20, pady=20)

        self.image = Image.new(
            "L",
            (CANVAS_SIZE, CANVAS_SIZE),
            0
        )

        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)
        
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        clear_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear
        )

        clear_button.pack(side=tk.LEFT, padx=5)

        preview_button = tk.Button(
            button_frame,
            text="Show Kumo Input",
            command=self.show_kumo_input
        )

        preview_button.pack(side=tk.LEFT, padx=5)

    def paint(self, event):
        radius = BRUSH_SIZE // 2
        x1 = event.x - radius
        y1 = event.y - radius
        x2 = event.x + radius
        y2 = event.y + radius

        self.canvas.create_oval(
            x1,
            y1,
            x2,
            y2,
            fill="white",
            outline="white"
        )

        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new(
            "L",
            (CANVAS_SIZE, CANVAS_SIZE),
            0
        )

        self.draw = ImageDraw.Draw(self.image)

    def get_kumo_input(self):
        image = self.image.resize((MNIST_SIZE, MNIST_SIZE), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=float)
        pixels = pixels / 255.0
        return pixels

    def show_kumo_input(self):
        pixels = self.get_kumo_input()
        print("\nKumo input shape:", pixels.shape)
        print("Minimum pixel:", pixels.min())
        print("Maximum pixel:", pixels.max())
        print("Flattened shape:", pixels.flatten().shape)

        plt.imshow(
            pixels,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        plt.title("What Kumo Will See")
        plt.axis("off")
        plt.show()

    def run(self):
        self.root.mainloop()

app = KumoCanvas()
app.run()