from core.tensor import Tensor
from nn.modules import MLP

if __name__ == "__main__":
    
    x = [Tensor(2.0), Tensor(3.0), Tensor(-1.0)]
    n = MLP(3, [4, 4, 1])
    print(n(x))
    
    
    
    
    # input     
    # x1 = Tensor([2.0, 4.0])
    # x2 = Tensor([0.0, 1.0])

    # # weights
    # w1 = Tensor([-3.0, 9.0])
    # w2 = Tensor([1.0, 3.0])

    # # bias
    # b = Tensor([6.8813735870195432, 5])

    # # x1w1 + x2w2 + b
    # x1w1 = x1*w1
    # x2w2 = x2*w2

    # x1w1x2w2 = x1w1 + x2w2

    # n = x1w1x2w2 + b

    # o = n.tanh()

    # o.backward()


    # print(o.data)

    # print("n.grad", n.grad)