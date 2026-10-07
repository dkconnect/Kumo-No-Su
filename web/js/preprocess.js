function centerImage(pixels, width = 28, height = 28) {
    let total = 0;
    let weightedX = 0;
    let weightedY = 0;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const value = pixels[
                y * width + x
            ];

            total += value;
            weightedX += x * value;
            weightedY += y * value;
        }
    }

    if (total === 0) {
        return pixels.slice();
    }

    const centerX = weightedX / total;
    const centerY = weightedY / total;

    const targetX = (
        width - 1
    ) / 2;

    const targetY = (
        height - 1
    ) / 2;

    const shiftX = Math.round(
        targetX - centerX
    );

    const shiftY = Math.round(
        targetY - centerY
    );

    const centered = new Array(
        width * height
    ).fill(0);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const newX = x + shiftX;
            const newY = y + shiftY;

            if (
                newX >= 0
                && newX < width
                && newY >= 0
                && newY < height
            ) {
                centered[
                    newY * width + newX
                ] = pixels[
                    y * width + x
                ];
            }
        }
    }

    return centered;
}


function canvasToMNIST(canvas) {
    const size = 28;

    const smallCanvas = document.createElement(
        "canvas"
    );

    smallCanvas.width = size;
    smallCanvas.height = size;

    const context = smallCanvas.getContext(
        "2d"
    );

    context.fillStyle = "black";
    context.fillRect(
        0,
        0,
        size,
        size
    );

    context.drawImage(
        canvas,
        0,
        0,
        size,
        size
    );

    const imageData = context.getImageData(
        0,
        0,
        size,
        size
    );

    const pixels = new Array(
        size * size
    );

    for (
        let i = 0;
        i < pixels.length;
        i++
    ) {
        const offset = i * 4;

        const red = imageData.data[
            offset
        ];

        const green = imageData.data[
            offset + 1
        ];

        const blue = imageData.data[
            offset + 2
        ];

        pixels[i] = (
            red + green + blue
        ) / (3 * 255);
    }

    return centerImage(
        pixels,
        size,
        size
    );
}


export {
    centerImage,
    canvasToMNIST
};