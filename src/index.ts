type ActivateFunction = (val: number) => number;

class NeuralNetwork {
  activateFunction: Function;
  constructor(neurons: Neuron[], ridges: Ridge[], activate: ActivateFunction) {
    this.neurons = neurons;
    this.ridges = ridges;
    this.activateFunction = activate;
  }
  // input: number[] = [];
  ridges: Ridge[];
  neurons: Neuron[];

  getNeuronById = (id: number) => this.neurons.find((n) => n.id === id);

  findInRidgesForNeuron = (id: number) =>
    this.ridges.filter((r) => r.outNeuron.id === id);

  fit(input_values: number[]) {
    this.ridges.forEach((ridge, index) => {
      if (ridge.isInput) {
        let n = new Neuron(0 - index, true);
        n._value = input_values[ridge.inId];
        this.ridges[index].inNeuron = n;
      }
    });

    for (let i = 0; i < this.neurons.length; i++) {
      let n = this.neurons[i];
      let ridges = this.findInRidgesForNeuron(n.id);

      let neurVal = 0;

      ridges.forEach((r) => {
        if (r.bias) {
          neurVal += r.weight * 1;
        } else {
          neurVal += r.weight * r.inNeuron._value;
        }
      });

      n._value = this.activateFunction(neurVal);
    }

    console.log(this.neurons);

    return this.neurons[this.neurons.length - 1]._value;
  }
}

class Neuron {
  id: number;
  inputLayer: boolean;
  _value: number;

  constructor(id: number, inputLayer: boolean = false) {
    this.id = id;
    this.inputLayer = inputLayer;
  }
}

class Ridge {
  id: number;
  weight: number;
  bias: boolean;
  isInput: boolean;
  inId: number | null; // input Id

  inNeuron: Neuron | null;
  outNeuron: Neuron | null;

  constructor(
    id: number,
    weight: number,
    inputNeuron: Neuron = null,
    outputNeuron: Neuron = null,
    bias: boolean = false,
    isInput: boolean = false,
    inId: number | null = null
  ) {
    this.id = id;
    this.weight = weight;
    this.inNeuron = inputNeuron;
    this.outNeuron = outputNeuron;
    this.bias = bias;
    this.isInput = isInput;
    this.inId = inId;
  }
}

function init() {
  let neurons: Neuron[] = [
    new Neuron(0),
    new Neuron(1),
    new Neuron(2),
    new Neuron(3),
    new Neuron(4),
  ];

  let ridges: Ridge[] = [
    new Ridge(100, -0.5, null, neurons[0], true, false),
    new Ridge(101, 0.6, null, neurons[0], false, true, 0),
    new Ridge(102, 0.3, null, neurons[0], false, true, 1),
    new Ridge(110, 0.8, null, neurons[1], false, true, 0),
    new Ridge(111, -0.8, null, neurons[1], false, true, 1),
    new Ridge(112, 0.1, null, neurons[1], true, false),
    new Ridge(200, 0.7, null, neurons[2], true),
    new Ridge(201, 0.4, neurons[0], neurons[2]),
    new Ridge(202, -0.3, neurons[1], neurons[2]),
    new Ridge(210, 1, neurons[0], neurons[3]),
    new Ridge(211, 0.7, neurons[1], neurons[3]),
    new Ridge(212, -0.2, null, neurons[3], true),
    new Ridge(300, -0.3, null, neurons[4], true),
    new Ridge(301, 0.9, neurons[2], neurons[4]),
    new Ridge(302, 0.1, neurons[3], neurons[4]),
  ];

  const activateF: ActivateFunction = (val) => {
    return val > 0 ? val : 0;
  };

  let n = new NeuralNetwork(neurons, ridges, activateF);
  console.log(n.fit([0.5, 0.9]));
}

init();
