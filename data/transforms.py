from torchvision import transforms as T
import albumentations as A


class Transform:
    def __init__(self, config):
        self.ops = []

    def apply(self, inputs):
        for op in self.ops:
            inputs = op(inputs)
        return inputs

    def __call__(self, inputs):
        return self.apply(inputs)
