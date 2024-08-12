import numpy as np

class Tensor:

    def __init__(self, data, label=''):
        self.data = data
        self.grad = 0.0
        self.childrens = []
        self.operation = ''
        self._backward = lambda:None
        self.label = label

    # Addition
    def __add__(self,other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        res_obj = Tensor(self.data + other.data)
        res_obj.childrens+=[self, other]
        res_obj.operation = '+'
        
        def _backward():
            self.grad = res_obj.grad * 1.0
            other.grad = res_obj.grad * 1.0
        res_obj._backward = _backward
        
        return res_obj
    
    def __radd__(self, other):
        other = Tensor(other)
        res_obj = Tensor(self.data + other.data)
        res_obj.childrens+=[other, self]
        res_obj.operation = '+'
        
        return res_obj
    
    # Multiplication
    def __mul__(self,other):             
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        res_obj = Tensor(self.data * other.data) 
        res_obj.childrens+=[self, other]
        res_obj.operation = '*'
        
        def _backward():
            self.grad = res_obj.grad * other.data
            other.grad = res_obj.grad * self.data
        res_obj._backward = _backward
        
        return res_obj
    
    def __rmul__(self, other):
        other = Tensor(other)
        res_obj = Tensor(self.data * other.data)
        res_obj.childrens+=[other, self]
        res_obj.operation = '*'
        return res_obj
    
    
    def __repr__(self):
        return f"Tensor({self.data})"



    









if __name__ == "__main__":
    x = Tensor(3)
    y = Tensor(9)
    z = x*y 
    print(z)





