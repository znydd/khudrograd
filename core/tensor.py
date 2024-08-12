import numpy as np

    
class Tensor:

    def __init__(self, data):
        self.data = np.array(data)
        self.grad = 0.0
        self.grad_fn = None
        self.childrens = set()

    # Addition
    def __add__(self, other):
        return Add.apply(self, other)
        
    # Multiplication
    def __mul__(self,other):             
        return Mul.apply(self, other)

    # Tanh
    def tanh(self):
        # self.data = np.array(self.data)
        return Tanh.apply(self)
    
    # Backward
    def backward(self):        
        self.grad = 1.0
        
        topo = []
        visited = set()
        def build_topo(v):
          if v not in visited:
            visited.add(v)
            for child in v.childrens:
              build_topo(child)
            topo.append(v)
        build_topo(self)

        for node in reversed(topo):
            if node.grad_fn:
                node.grad_fn(node.grad)            

    
    def __repr__(self):
        return f"Tensor({self.data})"
    


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


    

# input     
x1 = Tensor([2.0])
x2 = Tensor([0.0])

# weights
w1 = Tensor([-3.0])
w2 = Tensor([1.0])

# bias
b = Tensor([6.8813735870195432])

# x1w1 + x2w2 + b
x1w1 = x1*w1
x2w2 = x2*w2

x1w1x2w2 = x1w1 + x2w2

n = x1w1x2w2 + b

o = n.tanh()

o.backward()


print(o.data)

print("n.grad", n.grad)