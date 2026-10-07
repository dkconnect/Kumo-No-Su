import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

from kumo.network import KumoNetwork
from kumo.losses import softmax


CANVAS_SIZE = 280
BRUSH_SIZE = 20
MNIST_SIZE = 28

MODEL_PATH = "models/mnist_full_kumo.npz"


class KumoCanvas:
    def __init__(self):

        self.root = tk.Tk()
        self.root.title("Kumo No Su")

        self.network = KumoNetwork.load(
            MODEL_PATH
        )

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            cursor="cross"
        )

        self.canvas.pack(
            padx=20,
            pady=20
        )

        self.image = Image.new(
            "L",
            (CANVAS_SIZE, CANVAS_SIZE),
            0
        )

        self.draw = ImageDraw.Draw(
            self.image
        )

        self.last_x = None
        self.last_y = None

        self.canvas.bind(
            "<Button-1>",
            self.start_stroke
        )

        self.canvas.bind(
            "<B1-Motion>",
            self.paint
        )

        self.canvas.bind(
            "<ButtonRelease-1>",
            self.end_stroke
        )

        self.result_label = tk.Label(
            self.root,
            text="Draw a digit",
            font=("Arial", 20)
        )

        self.result_label.pack(
            pady=5
        )

        button_frame = tk.Frame(
            self.root
        )

        button_frame.pack(
            pady=10
        )

        predict_button = tk.Button(
            button_frame,
            text="Predict",
            command=self.predict
        )

        predict_button.pack(
            side=tk.LEFT,
            padx=5
        )

        inspect_button = tk.Button(
            button_frame,
            text="Inspect Prediction",
            command=self.inspect_prediction
        )

        inspect_button.pack(
            side=tk.LEFT,
            padx=5
        )

        preview_button = tk.Button(
            button_frame,
            text="Show Kumo Input",
            command=self.show_kumo_input
        )

        preview_button.pack(
            side=tk.LEFT,
            padx=5
        )

        clear_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear
        )

        clear_button.pack(
            side=tk.LEFT,
            padx=5
        )

    def start_stroke(self, event):
        self.last_x = event.x
        self.last_y = event.y

        radius = BRUSH_SIZE // 2

        self.canvas.create_oval(
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius,
            fill="white",
            outline="white"
        )

        self.draw.ellipse(
            [
                event.x - radius,
                event.y - radius,
                event.x + radius,
                event.y + radius
            ],
            fill=255
        )

    def paint(self, event):
        if self.last_x is None:
            self.last_x = event.x
            self.last_y = event.y

        self.canvas.create_line(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            fill="white",
            width=BRUSH_SIZE,
            capstyle=tk.ROUND,
            smooth=True
        )

        self.draw.line(
            [
                self.last_x,
                self.last_y,
                event.x,
                event.y
            ],
            fill=255,
            width=BRUSH_SIZE
        )

        radius = BRUSH_SIZE // 2

        self.draw.ellipse(
            [
                event.x - radius,
                event.y - radius,
                event.x + radius,
                event.y + radius
            ],
            fill=255
        )

        self.last_x = event.x
        self.last_y = event.y

    def end_stroke(self, event):
        self.last_x = None
        self.last_y = None

    def clear(self):
        self.canvas.delete(
            "all"
        )

        self.image = Image.new(
            "L",
            (CANVAS_SIZE, CANVAS_SIZE),
            0
        )

        self.draw = ImageDraw.Draw(
            self.image
        )

        self.last_x = None
        self.last_y = None

        self.result_label.config(
            text="Draw a digit"
        )

    def get_kumo_input(self):
        image = self.image.copy()

        bbox = image.getbbox()

        if bbox is None:
            return np.zeros(
                (MNIST_SIZE, MNIST_SIZE),
                dtype=float
            )

        digit = image.crop(
            bbox
        )

        width, height = digit.size

        scale = min(
            20 / width,
            20 / height
        )

        new_width = max(
            1,
            int(width * scale)
        )

        new_height = max(
            1,
            int(height * scale)
        )

        digit = digit.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        centered = Image.new(
            "L",
            (MNIST_SIZE, MNIST_SIZE),
            0
        )

        x = (
            MNIST_SIZE - new_width
        ) // 2

        y = (
            MNIST_SIZE - new_height
        ) // 2

        centered.paste(
            digit,
            (x, y)
        )

        pixels = np.asarray(
            centered,
            dtype=float
        )

        total = np.sum(
            pixels
        )

        if total > 0:

            rows = np.arange(
                MNIST_SIZE
            )

            cols = np.arange(
                MNIST_SIZE
            )

            center_y = np.sum(
                rows[:, None] * pixels
            ) / total

            center_x = np.sum(
                cols[None, :] * pixels
            ) / total

            target = (
                MNIST_SIZE - 1
            ) / 2

            shift_x = int(
                round(
                    target - center_x
                )
            )

            shift_y = int(
                round(
                    target - center_y
                )
            )

            shifted = np.zeros_like(
                pixels
            )

            source_x1 = max(
                0,
                -shift_x
            )

            source_x2 = min(
                MNIST_SIZE,
                MNIST_SIZE - shift_x
            )

            source_y1 = max(
                0,
                -shift_y
            )

            source_y2 = min(
                MNIST_SIZE,
                MNIST_SIZE - shift_y
            )

            target_x1 = (
                source_x1 + shift_x
            )

            target_x2 = (
                source_x2 + shift_x
            )

            target_y1 = (
                source_y1 + shift_y
            )

            target_y2 = (
                source_y2 + shift_y
            )

            shifted[
                target_y1:target_y2,
                target_x1:target_x2
            ] = pixels[
                source_y1:source_y2,
                source_x1:source_x2
            ]

            pixels = shifted

        pixels = (
            pixels / 255.0
        )

        return pixels

    def predict(self):
        if self.image.getbbox() is None:

            self.result_label.config(
                text="Draw something first"
            )

            return

        pixels = self.get_kumo_input()

        x = pixels.flatten()

        logits = self.network.forward(
            x
        )

        probabilities = softmax(
            logits
        )

        prediction = int(
            np.argmax(
                probabilities
            )
        )

        confidence = probabilities[
            prediction
        ]

        self.result_label.config(
            text=(
                f"Kumo: {prediction}   "
                f"{confidence * 100:.1f}%"
            )
        )

        print(
            "\nKumo prediction:",
            prediction
        )

        print(
            "Confidence:",
            f"{confidence * 100:.2f}%"
        )

        print(
            "\nProbabilities:"
        )

        for digit in range(10):

            marker = ""

            if digit == prediction:
                marker = "  <---"

            print(
                f"{digit}: "
                f"{probabilities[digit] * 100:6.2f}%"
                f"{marker}"
            )

    def inspect_prediction(self):
        if self.image.getbbox() is None:

            self.result_label.config(
                text="Draw something first"
            )

            return

        pixels = self.get_kumo_input()

        x = pixels.flatten()

        activations = self.network.inspect(
            x
        )

        hidden = activations[1]
        logits = activations[2]

        probabilities = softmax(
            logits
        )

        prediction = int(
            np.argmax(
                probabilities
            )
        )

        confidence = probabilities[
            prediction
        ]

        output_layer = self.network.layers[
            1
        ]

        output_coefficients = (
            output_layer.C[
                :,
                prediction,
                :
            ]
        )

        output_degrees = np.arange(
            1,
            output_layer.degree + 1
        )

        hidden_powers = (
            hidden[:, None]
            ** output_degrees
        )

        hidden_contributions = np.sum(
            output_coefficients
            * hidden_powers,
            axis=1
        )

        important_hidden = int(
            np.argmax(
                np.abs(
                    hidden_contributions
                )
            )
        )

        hidden_layer = self.network.layers[
            0
        ]

        pixel_coefficients = (
            hidden_layer.C[
                :,
                important_hidden,
                :
            ]
        )

        pixel_degrees = np.arange(
            1,
            hidden_layer.degree + 1
        )

        pixel_powers = (
            x[:, None]
            ** pixel_degrees
        )

        pixel_contributions = np.sum(
            pixel_coefficients
            * pixel_powers,
            axis=1
        )

        hidden_reconstructed = (
            hidden_layer.b[
                important_hidden
            ]
            + np.sum(
                pixel_contributions
            )
        )

        output_reconstructed = (
            output_layer.b[
                prediction
            ]
            + np.sum(
                hidden_contributions
            )
        )

        hidden_difference = abs(
            hidden[
                important_hidden
            ]
            - hidden_reconstructed
        )

        output_difference = abs(
            logits[
                prediction
            ]
            - output_reconstructed
        )

        print(
            "\nKumo prediction trace"
        )

        print(
            "Prediction:",
            prediction
        )

        print(
            "Confidence:",
            f"{confidence * 100:.2f}%"
        )

        print(
            "\nStrongest hidden path:"
        )

        print(
            "Hidden node:",
            important_hidden
        )

        print(
            "Hidden activation:",
            hidden[
                important_hidden
            ]
        )

        print(
            "Contribution to prediction:",
            hidden_contributions[
                important_hidden
            ]
        )

        print(
            "\nHidden reconstruction difference:",
            hidden_difference
        )

        print(
            "Output reconstruction difference:",
            output_difference
        )

        contribution_image = (
            pixel_contributions.reshape(
                MNIST_SIZE,
                MNIST_SIZE
            )
        )

        limit = np.max(
            np.abs(
                contribution_image
            )
        )

        if limit == 0:
            limit = 1.0

        figure, axes = plt.subplots(
            1,
            3,
            figsize=(12, 4)
        )

        axes[0].imshow(
            pixels,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[0].set_title(
            f"Kumo input → {prediction}"
        )

        axes[0].axis(
            "off"
        )

        image = axes[1].imshow(
            contribution_image,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit
        )

        axes[1].set_title(
            (
                "Pixels → hidden "
                f"{important_hidden}"
            )
        )

        axes[1].axis(
            "off"
        )

        axes[2].bar(
            np.arange(
                len(hidden)
            ),
            hidden_contributions
        )

        axes[2].axhline(
            0,
            linewidth=1
        )

        axes[2].set_xlabel(
            "Hidden node"
        )

        axes[2].set_ylabel(
            "Contribution"
        )

        axes[2].set_title(
            f"Hidden → digit {prediction}"
        )

        axes[2].set_xticks(
            np.arange(
                len(hidden)
            )
        )

        figure.colorbar(
            image,
            ax=axes[1],
            label="Pixel contribution"
        )

        figure.suptitle(
            (
                f"Kumo prediction: {prediction} "
                f"({confidence * 100:.1f}%)"
            )
        )

        plt.tight_layout()

        plt.show()

    def show_kumo_input(self):
        pixels = self.get_kumo_input()

        print(
            "\nKumo input shape:",
            pixels.shape
        )

        print(
            "Minimum pixel:",
            pixels.min()
        )

        print(
            "Maximum pixel:",
            pixels.max()
        )

        print(
            "Flattened shape:",
            pixels.flatten().shape
        )

        plt.imshow(
            pixels,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        plt.title(
            "Kumo's vision"
        )

        plt.axis(
            "off"
        )

        plt.show()

    def run(self):
        self.root.mainloop()


app = KumoCanvas()
app.run()