import {
    KumoNetwork,
    softmax
} from "./kumo.js";


function maximumDifference(a, b) {
    let maximum = 0;

    for (let i = 0; i < a.length; i++) {
        const difference = Math.abs(
            a[i] - b[i]
        );

        if (difference > maximum) {
            maximum = difference;
        }
    }

    return maximum;
}


async function runParityTest() {
    const modelResponse = await fetch(
        "model/kumo_mnist.json"
    );

    const fixtureResponse = await fetch(
        "model/parity_test.json"
    );
    
    if (
        !modelResponse.ok
        || !fixtureResponse.ok
    ) {
        throw new Error(
            "Could not load parity test files"
        );
    }

    const model = await modelResponse.json();
    const fixture = await fixtureResponse.json();

    const network = new KumoNetwork(
        model
    );

    const activations = network.inspect(
        fixture.input
    );

    const hidden = activations[1];
    const logits = activations[2];
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

    const hiddenDifference = maximumDifference(
        hidden,
        fixture.hidden
    );

    const logitsDifference = maximumDifference(
        logits,
        fixture.logits
    );

    const probabilityDifference = maximumDifference(
        probabilities,
        fixture.probabilities
    );

    console.log(
        "Kumo Python ↔ JavaScript parity test"
    );

    console.log(
        "Python prediction:",
        fixture.prediction
    );

    console.log(
        "JavaScript prediction:",
        prediction
    );

    console.log(
        "Max hidden difference:",
        hiddenDifference
    );

    console.log(
        "Max logits difference:",
        logitsDifference
    );

    console.log(
        "Max probability difference:",
        probabilityDifference
    );

    const passed = (
        prediction === fixture.prediction
        && hiddenDifference < 1e-10
        && logitsDifference < 1e-10
        && probabilityDifference < 1e-10
    );

    console.log(
        passed
            ? "PARITY TEST PASSED"
            : "PARITY TEST FAILED"
    );
}


runParityTest().catch(
    error => {
        console.error(
            error
        );
    }
);