import {
    KumoNetwork,
    softmax
} from "./kumo.js";

import {
    canvasToMNIST
} from "./preprocess.js";

import {
    DrawingCanvas
} from "./canvas.js";

import {
    strongestContributions,
    drawHiddenWeb
} from "./visualize.js";


const canvas = document.getElementById(
    "drawing-canvas"
);

const inputCanvas = document.getElementById(
    "input-canvas"
);

const inputContext = inputCanvas.getContext(
    "2d"
);

const webCanvas = document.getElementById(
    "web-canvas"
);

const webDetails = document.getElementById(
    "web-details"
);

const predictButton = document.getElementById(
    "predict-button"
);

const clearButton = document.getElementById(
    "clear-button"
);

const predictionElement = document.getElementById(
    "prediction"
);

const confidenceElement = document.getElementById(
    "confidence"
);

const probabilitiesElement = document.getElementById(
    "probabilities"
);

const hiddenNodesElement = document.getElementById(
    "hidden-nodes"
);


let network = null;
let detailedInspection = null;
let currentPrediction = null;
let selectedHiddenIndex = null;
let manualHiddenSelection = false;
let liveFrame = null;


function scheduleLivePrediction() {
    if (
        network === null
        || liveFrame !== null
    ) {
        return;
    }

    liveFrame = requestAnimationFrame(
        () => {
            liveFrame = null;
            predict();
        }
    );
}


const drawingCanvas = new DrawingCanvas(
    canvas,
    scheduleLivePrediction
);


async function loadModel() {
    const response = await fetch(
        "model/kumo_mnist.json"
    );

    if (!response.ok) {
        throw new Error(
            "Could not load Kumo model"
        );
    }

    const model = await response.json();

    network = new KumoNetwork(
        model
    );

    console.log(
        "Kumo model loaded"
    );
}


function showInput(input) {
    const imageData = inputContext.createImageData(
        28,
        28
    );

    for (
        let i = 0;
        i < input.length;
        i++
    ) {
        const value = Math.round(
            input[i] * 255
        );

        const offset = i * 4;

        imageData.data[offset] = value;
        imageData.data[offset + 1] = value;
        imageData.data[offset + 2] = value;
        imageData.data[offset + 3] = 255;
    }

    inputContext.putImageData(
        imageData,
        0,
        0
    );
}


function clearInput() {
    inputContext.fillStyle = "black";

    inputContext.fillRect(
        0,
        0,
        inputCanvas.width,
        inputCanvas.height
    );
}


function clearWeb() {
    const context = webCanvas.getContext(
        "2d"
    );

    context.clearRect(
        0,
        0,
        webCanvas.width,
        webCanvas.height
    );

    context.fillStyle = "#111318";

    context.fillRect(
        0,
        0,
        webCanvas.width,
        webCanvas.height
    );

    webDetails.textContent = (
        "Start drawing to inspect Kumo."
    );
}


function findBestHiddenNode() {
    if (
        detailedInspection === null
        || currentPrediction === null
    ) {
        return null;
    }

    const outputLayer = (
        detailedInspection.layers[1]
    );

    let bestIndex = 0;

    let bestStrength = Math.abs(
        outputLayer.edgeContributions[0][
            currentPrediction
        ]
    );

    for (
        let i = 1;
        i < outputLayer.edgeContributions.length;
        i++
    ) {
        const strength = Math.abs(
            outputLayer.edgeContributions[i][
                currentPrediction
            ]
        );

        if (strength > bestStrength) {
            bestStrength = strength;
            bestIndex = i;
        }
    }

    return bestIndex;
}


function showHiddenNodes(hidden) {
    hiddenNodesElement.innerHTML = "";

    hidden.forEach(
        (value, index) => {
            const node = document.createElement(
                "button"
            );

            node.type = "button";
            node.className = "hidden-node";

            if (index === selectedHiddenIndex) {
                node.classList.add(
                    "selected"
                );
            }

            const number = document.createElement(
                "div"
            );

            number.className = "hidden-node-number";

            number.textContent = (
                `Hidden ${index}`
            );

            const activation = document.createElement(
                "div"
            );

            activation.className = "hidden-node-value";

            activation.textContent = (
                value.toFixed(4)
            );

            node.appendChild(
                number
            );

            node.appendChild(
                activation
            );

            node.addEventListener(
                "click",
                () => {
                    selectHiddenNode(
                        index
                    );
                }
            );

            hiddenNodesElement.appendChild(
                node
            );
        }
    );
}


function selectHiddenNode(index) {
    selectedHiddenIndex = index;
    manualHiddenSelection = true;

    const nodes = hiddenNodesElement.querySelectorAll(
        ".hidden-node"
    );

    nodes.forEach(
        (node, nodeIndex) => {
            node.classList.toggle(
                "selected",
                nodeIndex === index
            );
        }
    );

    drawSelectedHiddenWeb();
}


function drawSelectedHiddenWeb() {
    if (
        detailedInspection === null
        || selectedHiddenIndex === null
    ) {
        return;
    }

    const index = selectedHiddenIndex;

    const hiddenLayer = (
        detailedInspection.layers[0]
    );

    const inputContributions = (
        hiddenLayer.edgeContributions.map(
            edges => edges[index]
        )
    );

    const strongest = strongestContributions(
        inputContributions,
        20
    );

    const outputLayer = (
        detailedInspection.layers[1]
    );

    const outputContributions = (
        outputLayer.edgeContributions[index]
    );

    drawHiddenWeb(
        webCanvas,
        detailedInspection.input,
        index,
        hiddenLayer.output[index],
        inputContributions,
        outputContributions,
        currentPrediction
    );

    const mode = (
        manualHiddenSelection
            ? "MANUAL"
            : "AUTO"
    );

    webDetails.textContent = (
        `${mode} — Hidden ${index}. `
        + `Showing its 20 strongest input contributions `
        + `and its contributions to all 10 output digits. `
        + `Yellow marks Kumo's current prediction.`
    );

    console.log(
        `Strongest edges into hidden ${index}:`,
        strongest
    );
}


function showProbabilities(probabilities) {
    probabilitiesElement.innerHTML = "";

    probabilities.forEach(
        (probability, digit) => {
            const row = document.createElement(
                "div"
            );

            row.textContent = (
                `${digit}: `
                + `${(probability * 100).toFixed(2)}%`
            );

            probabilitiesElement.appendChild(
                row
            );
        }
    );
}


function predict() {
    if (network === null) {
        return;
    }

    const input = canvasToMNIST(
        canvas
    );

    showInput(
        input
    );

    detailedInspection = network.inspectDetailed(
        input
    );

    const hidden = (
        detailedInspection.layers[0].output
    );

    const logits = (
        detailedInspection.output
    );

    const probabilities = softmax(
        logits
    );

    let prediction = 0;

    for (
        let i = 1;
        i < probabilities.length;
        i++
    ) {
        if (
            probabilities[i]
            > probabilities[prediction]
        ) {
            prediction = i;
        }
    }

    currentPrediction = prediction;

    if (!manualHiddenSelection) {
        selectedHiddenIndex = (
            findBestHiddenNode()
        );
    }

    const confidence = (
        probabilities[prediction]
        * 100
    );

    predictionElement.textContent = (
        prediction
    );

    confidenceElement.textContent = (
        `${confidence.toFixed(2)}% confidence`
    );

    showProbabilities(
        probabilities
    );

    showHiddenNodes(
        hidden
    );

    if (selectedHiddenIndex !== null) {
        drawSelectedHiddenWeb();
    } else {
        clearWeb();
    }
}


function clear() {
    if (liveFrame !== null) {
        cancelAnimationFrame(
            liveFrame
        );

        liveFrame = null;
    }

    drawingCanvas.clear();
    clearInput();
    clearWeb();

    detailedInspection = null;
    currentPrediction = null;
    selectedHiddenIndex = null;
    manualHiddenSelection = false;

    predictionElement.textContent = "-";

    confidenceElement.textContent = (
        "Draw a digit to begin."
    );

    probabilitiesElement.innerHTML = "";
    hiddenNodesElement.innerHTML = "";
}


predictButton.addEventListener(
    "click",
    predict
);

clearButton.addEventListener(
    "click",
    clear
);


clearInput();
clearWeb();

loadModel().catch(
    error => {
        console.error(
            error
        );

        confidenceElement.textContent = (
            "Failed to load Kumo model."
        );
    }
);