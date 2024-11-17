import random
from PIL import ImageFilter

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, random_transform):
        self.random_transform = random_transform

    def __call__(self, x):
        q = self.random_transform(x)
        k = self.random_transform(x)
        return [q, k]


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x