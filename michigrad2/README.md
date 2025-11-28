# Michigrad
Pequeño Autograd con fines educativos.

![gatite](images/gatite.png)

Clon de Micrograd de [Andrej Karpathy](https://github.com/karpathy/micrograd) y básicamente comparte la misma base de código. Se mejoraron algunos aspectos de la visualización, orientados al curso de Nociones de Deep Learning para Inteligencia Artificial Generativa de Texto de [Purrfect AI](https://purrfectai.online)

## Características
Michigrad es un motor de cálculo de gradientes para valores escalares. Permite representar valores numéricos envolviendolos en objetos `Value`. Estos objetos soportan algunas de las operaciones análogas a las de los números, como la suma, la multiplicación, la división, la exponenciación, entre otras. Michigrad permite conocer el resultado de aplicar esas operaciones sobre los Values, lo que se conoce como forward pass, pero además permite generar el grafo de operaciones y dependencias necesarios para llegar al Value resultado. Este grafo puede usarse para calcular los gradientes de cualquier Value del grafo con respecto al resultado mediante el algoritmo de backpropagation que Michigrad también implenta. Esta información puede usarse para modificar los pesos W de una red neuronal respecto a una función de perdida L, con el objetivo de minimizar la función de perdida y entrenar la red neuronal.

## Uso de Michigrad

```python
import numpy as np
from michigrad.engine import Value
from michigrad.visualize import show_graph

# Definición de los pesos
np.random.seed(42)
W0 = Value(np.random.random(), name='W₀')
W1 = Value(np.random.random(), name='W₁')
b = Value(np.random.random(), name='b')
print(W0)  # imprime Value(data=0.3745401188473625, grad=0, name=W₀)

# definición del dataset de entrenamiento
x0 = Value(.5, name="x₀")
x1 = Value(1., name="x₁")
y = Value(2., name="y")

# forward pass
yhat = x0*W0 + x1*W1 + b
yhat.name = "ŷ"
print(yhat)  # imprime Value(data=1.8699783076450025, grad=0, name=ŷ)

L = (y - yhat) ** 2
L.name = "L"
print(L)  # imprime Value(data=0.016905640482857615, grad=0, name=L)

# backward pass
L.backward()

print(L)  # imprime Value(data=0.016905640482857615, grad=1, name=L)
print(W0)  # imprime Value(data=0.3745401188473625, grad=-0.1300216923549975, name=W₀)

# update de los pesos en la dirección contraria al gradiente de los W

show_graph(L, rankdir="TB",format="png")
```
![graph](images/graph.png)
