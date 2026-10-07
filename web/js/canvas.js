class DrawingCanvas {
    constructor(canvas, onChange = null) {
        this.canvas = canvas;
        this.context = canvas.getContext("2d");
        this.onChange = onChange;
        this.drawing = false;
        this.lastX = 0;
        this.lastY = 0;
        this.setup();
        this.clear();
    }

    setup() {
        this.context.lineCap = "round";
        this.context.lineJoin = "round";
        this.context.lineWidth = 20;
        this.context.strokeStyle = "white";

        this.canvas.addEventListener(
            "pointerdown",
            event => this.start(event)
        );

        this.canvas.addEventListener(
            "pointermove",
            event => this.move(event)
        );

        window.addEventListener(
            "pointerup",
            () => this.stop()
        );

        this.canvas.addEventListener(
            "pointercancel",
            () => this.stop()
        );
    }

    position(event) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        return {
            x: (event.clientX - rect.left) * scaleX,
            y: (event.clientY - rect.top) * scaleY
        };
    }

    changed() {
        if (this.onChange !== null) {
            this.onChange();
        }
    }

    start(event) {
        event.preventDefault();

        const position = this.position(event);

        this.drawing = true;
        this.lastX = position.x;
        this.lastY = position.y;

        this.context.beginPath();

        this.context.arc(
            position.x,
            position.y,
            this.context.lineWidth / 2,
            0,
            Math.PI * 2
        );

        this.context.fillStyle = "white";
        this.context.fill();

        this.changed();
    }

    move(event) {
        if (!this.drawing) {
            return;
        }

        event.preventDefault();

        const position = this.position(event);

        this.context.beginPath();

        this.context.moveTo(
            this.lastX,
            this.lastY
        );

        this.context.lineTo(
            position.x,
            position.y
        );

        this.context.stroke();

        this.lastX = position.x;
        this.lastY = position.y;

        this.changed();
    }

    stop() {
        this.drawing = false;
    }

    clear() {
        this.context.fillStyle = "black";

        this.context.fillRect(
            0,
            0,
            this.canvas.width,
            this.canvas.height
        );
    }
}

export {
    DrawingCanvas
};