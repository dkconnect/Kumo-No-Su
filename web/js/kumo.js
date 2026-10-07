class KumoLayer {
    constructor(config) {
        this.nInputs = config.n_inputs;
        this.nOutputs = config.n_outputs;
        this.degree = config.degree;
        this.C = config.coefficients;
        this.b = config.biases;
    }

    forward(input) {
        if (input.length !== this.nInputs) {
            throw new Error(
                `Expected ${this.nInputs} inputs, received ${input.length}`
            );
        }

        const output = new Array(
            this.nOutputs
        ).fill(0);

        for (let j = 0; j < this.nOutputs; j++) {
            output[j] = this.b[j];

            for (let i = 0; i < this.nInputs; i++) {
                const x = input[i];

                for (let d = 1; d <= this.degree; d++) {
                    output[j] += (
                        this.C[i][j][d - 1]
                        * Math.pow(x, d)
                    );
                }
            }
        }

        return output;
    }

    inspect(input) {
        if (input.length !== this.nInputs) {
            throw new Error(
                `Expected ${this.nInputs} inputs, received ${input.length}`
            );
        }

        const edgeContributions = Array.from(
            {
                length: this.nInputs
            },
            () => new Array(
                this.nOutputs
            ).fill(0)
        );

        const output = this.b.slice();

        for (let i = 0; i < this.nInputs; i++) {
            const x = input[i];

            for (let j = 0; j < this.nOutputs; j++) {
                let contribution = 0;

                for (let d = 1; d <= this.degree; d++) {
                    contribution += (
                        this.C[i][j][d - 1]
                        * Math.pow(x, d)
                    );
                }

                edgeContributions[i][j] = contribution;

                output[j] += contribution;
            }
        }

        return {
            input: input.slice(),
            output,
            bias: this.b.slice(),
            edgeContributions
        };
    }
}


class KumoNetwork {
    constructor(model) {
        this.layers = model.layers.map(
            config => new KumoLayer(config)
        );
    }

    forward(input) {
        let output = input;

        for (const layer of this.layers) {
            output = layer.forward(
                output
            );
        }

        return output;
    }

    inspect(input) {
        const activations = [
            input.slice()
        ];

        let output = input;

        for (const layer of this.layers) {
            output = layer.forward(
                output
            );

            activations.push(
                output.slice()
            );
        }

        return activations;
    }

    inspectDetailed(input) {
        const layers = [];

        let output = input.slice();

        for (const layer of this.layers) {
            const inspection = layer.inspect(
                output
            );

            layers.push(
                inspection
            );

            output = inspection.output;
        }

        return {
            input: input.slice(),
            layers,
            output: output.slice()
        };
    }
}


function softmax(logits) {
    const maximum = Math.max(
        ...logits
    );

    const exponentials = logits.map(
        value => Math.exp(
            value - maximum
        )
    );

    const total = exponentials.reduce(
        (sum, value) => sum + value,
        0
    );

    return exponentials.map(
        value => value / total
    );
}


export {
    KumoLayer,
    KumoNetwork,
    softmax
};