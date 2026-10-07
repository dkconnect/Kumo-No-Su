function centerImage(pixels, width = 28, height = 28) {
    let total = 0;
    let weightedX = 0;
    let weightedY = 0;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const value = pixels[y * width + x];

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

    const targetX = (width - 1) / 2;
    const targetY = (height - 1) / 2;

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


function findBoundingBox(imageData) {
    const {
        data,
        width,
        height
    } = imageData;

    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const offset = (
                (y * width + x) * 4
            );

            const red = data[offset];
            const green = data[offset + 1];
            const blue = data[offset + 2];

            const value = (
                red + green + blue
            ) / 3;

            if (value > 0) {
                minX = Math.min(
                    minX,
                    x
                );

                minY = Math.min(
                    minY,
                    y
                );

                maxX = Math.max(
                    maxX,
                    x
                );

                maxY = Math.max(
                    maxY,
                    y
                );
            }
        }
    }

    if (maxX === -1) {
        return null;
    }

    return {
        x: minX,
        y: minY,
        width: maxX - minX + 1,
        height: maxY - minY + 1
    };
}


function canvasToMNIST(canvas) {
    const size = 28;
    const targetDigitSize = 20;

    const sourceContext = canvas.getContext(
        "2d"
    );

    const sourceImage = sourceContext.getImageData(
        0,
        0,
        canvas.width,
        canvas.height
    );

    const box = findBoundingBox(
        sourceImage
    );

    if (box === null) {
        return new Array(
            size * size
        ).fill(0);
    }

    const scale = Math.min(
        targetDigitSize / box.width,
        targetDigitSize / box.height
    );

    const newWidth = Math.max(
        1,
        Math.floor(
            box.width * scale
        )
    );

    const newHeight = Math.max(
        1,
        Math.floor(
            box.height * scale
        )
    );

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

    const x = Math.floor(
        (size - newWidth) / 2
    );

    const y = Math.floor(
        (size - newHeight) / 2
    );

    context.drawImage(
        canvas,
        box.x,
        box.y,
        box.width,
        box.height,
        x,
        y,
        newWidth,
        newHeight
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

    for (let i = 0; i < pixels.length; i++) {
        const offset = i * 4;

        const red = imageData.data[offset];
        const green = imageData.data[offset + 1];
        const blue = imageData.data[offset + 2];

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