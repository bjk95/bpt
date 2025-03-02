import math
import random
from typing import Self


class Value:
    def __init__(
        self, 
        data: float, 
        _children=set(), 
        _operation: str = '', 
        label: str = '',
        grad: float = 0,
        ) -> None:
        
        self.data: float = data
        self._prev = set(_children)
        self._operation = _operation
        self.label = label
        self.grad = grad
        self._backward = lambda: None
    
    def __repr__(self) -> str:
        return f'Value({self.label + ", " if self.label != "" else ""}{self.data})'
    
    # Basic expressions
    def __add__(self, other: Self | float):
        other: Self = other if isinstance(other, Value) else Value(other)  # type: ignore
        out = Value(self.data + other.data, (self, other), '+') # type: ignore
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
            
        out._backward = _backward
         
        return out
    
    def __mul__(self, other: Self | float):
        other: Self = other if isinstance(other, Value) else Value(other) # type: ignore
        out = Value(self.data * other.data, (self,other), "*") # type: ignore
        
        def _backward():
            self.grad += out.grad * other.data 
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def exp(self):
        out = Value(math.exp(self.data))
        
        def _backward():
            self.grad += out.data * out.grad
            
        out._backward = _backward
        
        return out
    
    def __pow__(self, other: int | float):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
    
    # Activation functions
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    # Convenience functions
    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __radd__(self, other): # other + self
        return self + other
    
    def __rmul__(self, other: Self | float):
        return self * other
    
    def __truediv__(self, other: Self | float):
        return self * other**-1
    
    # Run back propagation
    def backward(self):
        topologically_sorted = []
        visited: set[Self] = set()
        def _build_topo(v: Self):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    _build_topo(child)
                topologically_sorted.append(v)
        _build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topologically_sorted):
            node._backward()
            
class Neuron:
    def __init__(self, number_of_inputs: int) -> None:
        # Randomly initialise learnable parameters -> weights and bias
        self.w = [Value(random.uniform(-1, 1)) for _ in range(number_of_inputs)]
        self.b = Value(random.uniform(-1,1))
        
    def __call__(self, x: list[Value] | list[float]) -> Value:
        # activation = sum(wi*xi for xi, wi in zip(x, self.w)) + self.b # type: ignore
        activation = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b # type: ignore
        return activation.tanh()
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"Neuron({len(self.w)})"
    
class Layer:
    # Number of inputs is the number of features for the first layer and number size of the previous layer if not
    # Number of outputs is the number of neurons in this layer, that then get integrated with the following layer
    def __init__(self, number_of_inputs: int, number_of_outputs: int) -> None:
        self.neurons = [Neuron(number_of_inputs=number_of_inputs) for _ in range(number_of_outputs)]
        
    def __call__(self, x: list[Value] | list[float]) -> Value | list[Value]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MultiLayerPerceptron:
    def __init__(self, number_of_features: int, number_of_neurons_per_hidden_layer: list[int]) -> None:
        sizes = [number_of_features] + number_of_neurons_per_hidden_layer
        self.layers = [Layer(sizes[n], sizes[n+1]) for n in range(len(number_of_neurons_per_hidden_layer))]
    
    def __call__(self, x: list[float]) -> Value | list[Value]:
        for layer in self.layers:
            x= layer(x) # type: ignore
        return x # type: ignore
    
    def parameters(self) -> list[Value]:
        return [p for l in self.layers for p in l.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
def train(mlp: MultiLayerPerceptron, X: list[list[float]], y: list[float], epochs: int, learning_rate: float, log_every: int | None = None) -> list[Value | list[Value]]:
    if log_every is not None:
        print(mlp)
    for k in range(epochs):
  
        # forward pass
        y_pred = [mlp(x) for x in X]
        loss: Value = sum((yout - ygt)**2 for ygt, yout in zip(y, y_pred)) # type: ignore
        
        # backward pass
        for p in mlp.parameters():
            p.grad = 0.0
        loss.backward()
        
        # update
        for p in mlp.parameters():
            p.data += -learning_rate * p.grad
            
        if log_every is not None and k % log_every == 0:
            print(k, loss.data)
            
    return y_pred
        
        
        