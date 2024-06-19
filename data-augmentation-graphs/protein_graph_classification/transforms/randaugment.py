from transforms.basics import DropNodes, PermuteEdges, Subgraph, MaskNodes
import random

from torch_geometric.transforms import Compose

class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = [DropNodes, PermuteEdges, Subgraph, MaskNodes]

    def __call__(self, data):
        ops = random.choices(self.augment_list, k=self.n)
        transform = Compose([op(self.m) for op in ops])
        return transform(data) 