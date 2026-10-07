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


const drawingCanvas = new DrawingCanvas(
    canvas
);

let network = null;
let detailedInspection = null;
let currentPrediction = null;


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
        "Predict a digit, then select a hidden node."
    );
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
                    inspectHiddenNode(
                        index,
                        node
                    );
                }
            );

            hiddenNodesElement.appendChild(
                node
            );
        }
    );
}


function inspectHiddenNode(
    index,
    selectedNode
) {
    if (detailedInspection === null) {
        return;
    }

    const nodes = hiddenNodesElement.querySelectorAll(
        ".hidden-node"
    );

    nodes.forEach(
        node => {
            node.classList.remove(
                "selected"
            );
        }
    );

    selectedNode.classList.add(
        "selected"
    );

    const layer = (
        detailedInspection.layers[0]
    );

    const contributions = (
        layer.edgeContributions.map(
            edges => edges[index]
        )
    );

    const strongest = strongestContributions(
        contributions,
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
        layer.output[index],
        contributions,
        outputContributions,
        currentPrediction
    );

    webDetails.textContent = (
        `Showing the 20 strongest input edges `
        + `affecting Hidden ${index}. `
        + `Yellow marks Kumo's predicted digit.`
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

    const activations = network.inspect(
        input
    );

    const hidden = activations[1];
    const logits = activations[2];

    showHiddenNodes(
        hidden
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

    clearWeb();

    console.log(
        "Kumo input:",
        input
    );

    console.log(
        "Kumo hidden activations:",
        hidden
    );

    console.log(
        "Kumo logits:",
        logits
    );
}


function clear() {
    drawingCanvas.clear();
    clearInput();
    clearWeb();

    detailedInspection = null;
    currentPrediction = null;

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