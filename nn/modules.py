from core.tensor import Tensor 
import random
import numpy as np

class Neuron:
    
    def __init__(self, number_of_input):
        self.weights = [Tensor(random.uniform(-1,1)) for _ in range(number_of_input)] # we know, # of input = weight
        self.bias = Tensor(random.uniform(-1, 1))
        
    def __call__(self, input):
        # weight*input + bias
        mul_and_sum = sum((weight_i*input_i for weight_i, input_i in zip(self.weights, input)), self.bias)
        add_activation = mul_and_sum.tanh()
        
        return add_activation
    
    def parameters(self):
        return self.weights + [self.bias]
    

class Layer:
    
    def __init__(self, number_of_input, number_of_neurons):
        self.neurons = [Neuron(number_of_input) for _ in range(number_of_neurons)]
        
    def __call__(self, input):
        outputs = [neuron(input) for neuron in self.neurons]
        return outputs
    
    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]
    
class MLP:
    
    def __init__(self, number_of_input, number_of_layer):
        all_layers = [number_of_input] + number_of_layer
        self.layers = [Layer(all_layers[i], all_layers[i+1]) for i in range(len(number_of_layer))]
        
    def __call__(self, input):
        for layer in self.layers:
            input = layer(input)
        return input
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
        
    