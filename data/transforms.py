from torchvision import transform as T
import albumentation as A

class Transform:
    def __init__(self, config):
        self.ops = []
    
    def apply(self, input):
        for op in self.ops:
            input = op(input)
        return input
    
    def __call__(self, input):
        return self.apply(input)