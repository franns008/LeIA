import random
from .engine import Value
from abc import ABC, abstractmethod


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)

    @abstractmethod
    def __call__(self, x):
        pass

    def parameters(self):
        return self.w + [self.b]

    @abstractmethod
    def __repr__(self):
        pass


class TanhNeuron(Neuron):

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def __repr__(self):
        return f"{'Tanh '}Neuron({len(self.w)})"


class SigmoidNeuron(Neuron):

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.sigmoid()

    def __repr__(self):
        return f"{'Sigmoid '}Neuron({len(self.w)})"


class ReLUNeuron(Neuron):

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()

    def __repr__(self):
        return f"{'ReLU '}Neuron({len(self.w)})"


class LinearNeuron(Neuron):

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act

    def __repr__(self):
        return f"{'Linear '}Neuron({len(self.w)})"


class Layer(Module):

    def __init__(self, nin, nout, function="Linear", **kwargs):
        match function:
            case "Linear":
                self.neurons = [LinearNeuron(nin, **kwargs) for _ in range(nout)]
            case "ReLU":
                self.neurons = [ReLUNeuron(nin, **kwargs) for _ in range(nout)]
            case "Sigmoid":
                self.neurons = [SigmoidNeuron(nin, **kwargs) for _ in range(nout)]
            case "Tanh":
                self.neurons = [TanhNeuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [[nin]] + nouts
        # [1, (2, "ReLU"), (2, "Sigmoid"), (1)]
        print(sz[1][0])
        self.layers = [
            Layer(
                sz[i][0],
                sz[i + 1][0],
                sz[i + 1][1] if len(sz[i + 1]) == 2 else "Linear",
            )
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
