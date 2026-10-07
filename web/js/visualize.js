function strongestContributions(
    contributions,
    count = 20
) {
    return contributions
        .map(
            (value, index) => ({
                index,
                value
            })
        )
        .sort(
            (a, b) =>
                Math.abs(b.value)
                - Math.abs(a.value)
        )
        .slice(
            0,
            count
        );
}


function drawHiddenWeb(
    canvas,
    input,
    hiddenIndex,
    hiddenValue,
    inputContributions,
    outputContributions,
    prediction
) {
    const context = canvas.getContext(
        "2d"
    );

    const width = canvas.width;
    const height = canvas.height;

    context.clearRect(
        0,
        0,
        width,
        height
    );

    context.fillStyle = "#111318";

    context.fillRect(
        0,
        0,
        width,
        height
    );

    const strongest = strongestContributions(
        inputContributions,
        20
    );

    drawTitle(
        context,
        hiddenIndex,
        hiddenValue
    );

    drawInputGrid(
        context,
        input,
        strongest
    );

    drawInputEdges(
        context,
        strongest
    );

    drawHiddenNode(
        context,
        hiddenIndex,
        hiddenValue
    );

    drawOutputEdges(
        context,
        outputContributions
    );

    drawOutputNodes(
        context,
        outputContributions,
        prediction
    );

    drawLegend(
        context
    );

    return strongest;
}


function drawTitle(
    context,
    hiddenIndex,
    hiddenValue
) {
    context.fillStyle = "#f4f4f5";
    context.font = "18px Arial";

    context.fillText(
        `Hidden ${hiddenIndex}`,
        30,
        35
    );

    context.fillStyle = "#a1a1aa";
    context.font = "14px monospace";

    context.fillText(
        `activation = ${hiddenValue.toFixed(4)}`,
        30,
        58
    );
}


function drawInputGrid(
    context,
    input,
    strongest
) {
    const startX = 30;
    const startY = 90;
    const cellSize = 10;

    const strongestMap = new Map();

    strongest.forEach(
        contribution => {
            strongestMap.set(
                contribution.index,
                contribution.value
            );
        }
    );

    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            const index = (
                y * 28 + x
            );

            const pixel = input[index];

            const brightness = Math.round(
                pixel * 200
            );

            context.fillStyle = (
                `rgb(${brightness}, ${brightness}, ${brightness})`
            );

            context.fillRect(
                startX + x * cellSize,
                startY + y * cellSize,
                cellSize - 1,
                cellSize - 1
            );

            if (strongestMap.has(index)) {
                const contribution = (
                    strongestMap.get(index)
                );

                context.strokeStyle = (
                    contribution >= 0
                        ? "#4ade80"
                        : "#f87171"
                );

                context.lineWidth = 2;

                context.strokeRect(
                    startX + x * cellSize,
                    startY + y * cellSize,
                    cellSize - 1,
                    cellSize - 1
                );
            }
        }
    }

    context.fillStyle = "#a1a1aa";
    context.font = "13px Arial";

    context.fillText(
        "28 × 28 Kumo input",
        startX,
        startY + 305
    );
}


function drawInputEdges(
    context,
    strongest
) {
    const gridX = 30;
    const gridY = 90;
    const cellSize = 10;

    const nodeX = 500;
    const nodeY = 230;

    let maximum = 0;

    strongest.forEach(
        contribution => {
            maximum = Math.max(
                maximum,
                Math.abs(
                    contribution.value
                )
            );
        }
    );

    if (maximum === 0) {
        maximum = 1;
    }

    strongest.forEach(
        contribution => {
            const pixelX = (
                contribution.index % 28
            );

            const pixelY = Math.floor(
                contribution.index / 28
            );

            const startX = (
                gridX
                + pixelX * cellSize
                + cellSize / 2
            );

            const startY = (
                gridY
                + pixelY * cellSize
                + cellSize / 2
            );

            const strength = (
                Math.abs(
                    contribution.value
                )
                / maximum
            );

            drawEdge(
                context,
                startX,
                startY,
                nodeX,
                nodeY,
                contribution.value,
                strength
            );
        }
    );
}


function drawHiddenNode(
    context,
    hiddenIndex,
    hiddenValue
) {
    const x = 500;
    const y = 230;
    const radius = 42;

    context.beginPath();

    context.arc(
        x,
        y,
        radius,
        0,
        Math.PI * 2
    );

    context.fillStyle = "#18181b";
    context.fill();

    context.strokeStyle = "#d4d4d8";
    context.lineWidth = 3;
    context.stroke();

    context.fillStyle = "#f4f4f5";
    context.textAlign = "center";
    context.font = "14px Arial";

    context.fillText(
        `Hidden ${hiddenIndex}`,
        x,
        y - 5
    );

    context.font = "12px monospace";

    context.fillText(
        hiddenValue.toFixed(3),
        x,
        y + 17
    );

    context.textAlign = "left";
}


function drawOutputEdges(
    context,
    contributions
) {
    const hiddenX = 500;
    const hiddenY = 230;

    let maximum = 0;

    contributions.forEach(
        value => {
            maximum = Math.max(
                maximum,
                Math.abs(value)
            );
        }
    );

    if (maximum === 0) {
        maximum = 1;
    }

    contributions.forEach(
        (value, digit) => {
            const position = outputPosition(
                digit
            );

            const strength = (
                Math.abs(value)
                / maximum
            );

            drawEdge(
                context,
                hiddenX,
                hiddenY,
                position.x,
                position.y,
                value,
                strength
            );
        }
    );
}


function drawOutputNodes(
    context,
    contributions,
    prediction
) {
    contributions.forEach(
        (value, digit) => {
            const position = outputPosition(
                digit
            );

            context.beginPath();

            context.arc(
                position.x,
                position.y,
                22,
                0,
                Math.PI * 2
            );

            context.fillStyle = "#18181b";
            context.fill();

            if (digit === prediction) {
                context.strokeStyle = "#facc15";
                context.lineWidth = 5;
            } else {
                context.strokeStyle = (
                    value >= 0
                        ? "#4ade80"
                        : "#f87171"
                );

                context.lineWidth = 2;
            }

            context.stroke();

            context.fillStyle = "#f4f4f5";
            context.textAlign = "center";
            context.font = "16px Arial";

            context.fillText(
                digit.toString(),
                position.x,
                position.y + 5
            );

            context.fillStyle = "#a1a1aa";
            context.font = "10px monospace";

            context.fillText(
                value.toFixed(2),
                position.x,
                position.y + 38
            );
        }
    );

    context.textAlign = "left";
}


function outputPosition(digit) {
    const column = (
        digit < 5
            ? 0
            : 1
    );

    const row = digit % 5;

    return {
        x: 690 + column * 130,
        y: 105 + row * 75
    };
}


function drawEdge(
    context,
    startX,
    startY,
    endX,
    endY,
    value,
    strength
) {
    context.beginPath();

    context.moveTo(
        startX,
        startY
    );

    context.lineTo(
        endX,
        endY
    );

    context.strokeStyle = (
        value >= 0
            ? `rgba(74, 222, 128, ${0.2 + strength * 0.8})`
            : `rgba(248, 113, 113, ${0.2 + strength * 0.8})`
    );

    context.lineWidth = (
        1 + strength * 4
    );

    context.stroke();
}


function drawLegend(
    context
) {
    context.textAlign = "left";
    context.font = "12px Arial";

    context.fillStyle = "#4ade80";

    context.fillText(
        "green = positive contribution",
        350,
        445
    );

    context.fillStyle = "#f87171";

    context.fillText(
        "red = negative contribution",
        350,
        465
    );

    context.fillStyle = "#a1a1aa";

    context.fillText(
        "thicker = larger |φ(x)|",
        350,
        485
    );
}


export {
    strongestContributions,
    drawHiddenWeb
};