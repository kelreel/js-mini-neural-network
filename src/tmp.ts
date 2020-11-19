let neurons: Neuron[] = [
  new Neuron(1),
  new Neuron(2),
  new Neuron(3),
  new Neuron(4),
  new Neuron(5),
];

let ridges: Ridge[] = [
  new Ridge(1, 0.9, null, neurons[0], false, true),
  new Ridge(2, 0.3, null, neurons[0], true),
  new Ridge(3, -0.5, neurons[0], neurons[1], false),
  new Ridge(4, 0.2, null, neurons[1], true),
];
