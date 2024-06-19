from transforms.basics import DropNodes, PermuteEdges, Subgraph, MaskNodes, Identity, \
    AddNodes, AddEdges, DropEdges, MaskChannels, GaussianNoise
from transforms.simclr import SimCLRTransfrom, SimCLRTransfrom_v2
from transforms.randaugment import RandAugment
from transforms.composition import SequentialAugmentation