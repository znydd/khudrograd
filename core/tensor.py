import numpy as np

class Tensor:

    def __init__(self, data):
        self.data = np.array(data)
        self.grad = 0.0
        self.grad_fn = None
        self.childrens = set()

    # Addition
    def __add__(self, other):
        from .autograd import Add
        return Add.apply(self, other)
        
    # Multiplication
    def __mul__(self,other):
        from .autograd import Mul             
        return Mul.apply(self, other)

    # Tanh
    def tanh(self):
        # self.data = np.array(self.data)
        from .autograd import Tanh        
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