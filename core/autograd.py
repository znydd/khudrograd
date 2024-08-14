import numpy as np
from .tensor import Tensor


class Add:
    
    @staticmethod
    def apply(*args):
        result = Tensor(Add.forward(*args))
        result.childrens.add(args[0])
        result.childrens.add(args[1])
        result.grad_fn = lambda grad: Add.backward(*args, grad)
        return result
     
    @staticmethod
    def forward(a, b):
        return a.data + b.data
    
    @staticmethod
    def backward(a, b, grad_output):
        a.grad += grad_output
        b.grad += grad_output
            
class Mul:
    
    @staticmethod
    def apply(*args):
        result = Tensor(Mul.forward(*args))
        result.childrens.add(args[0])
        result.childrens.add(args[1])
        result.grad_fn = lambda grad: Mul.backward(*args, grad)
        return result
    
    @staticmethod
    def forward(a, b):
        return a.data * b.data
    
    @staticmethod
    def backward(a, b, grad_output):
        a.grad += grad_output * b.data
        b.grad += grad_output * a.data


class Tanh:
    
    @staticmethod
    def apply(*args):
        result = Tensor(Tanh.forward(*args))
        result.childrens.add(args[0])
        result.grad_fn = lambda grad: Tanh.backward(*args, grad)
        return result
    
    @staticmethod
    def forward(a):
        return ((np.exp(2*a.data) - 1)/(np.exp(2*a.data) + 1))
    
    @staticmethod
    def backward(a, grad_output):
        tanh = ((np.exp(2*a.data) - 1)/(np.exp(2*a.data) + 1))
        a.grad += (1 - tanh**2) * grad_output

