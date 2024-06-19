import transforms
from torch_geometric.transforms import Compose
import random
import numpy as np

class SequentialAugmentation(object):

    def __init__(self, transform_names, ratios, probs=None, tree_idxes=None, **kwargs) -> None:
        '''
        A generic class for simclr data augmentation that takes in a list of transform names and ratios
        '''
        self.transforms = [getattr(transforms, name)(ratios[i]) for i, name in enumerate(transform_names)]
        self.probs = probs
        
        self.max_idx = np.max(tree_idxes)+1 if tree_idxes is not None else None
        self.tree_idxes = tree_idxes
        self.idx_to_transform = self.generate_idx_to_transforms(tree_idxes)

    def generate_idx_to_transforms(self, tree_idxes):
        if tree_idxes is None:
            return None
        transformed_idxes = np.ones(self.max_idx)*-1
        for i, idx in enumerate(tree_idxes):
            transformed_idxes[idx] = i
        return transformed_idxes.astype(int)

    def apply_transform(self, data):
        if self.probs is None:
            return Compose(self.transforms)(data)
        elif self.tree_idxes is None:
            for prob, transform in zip(self.probs, self.transforms):
                if random.random() < prob:
                    data = transform(data)
            return data
        else: 
            cur_idx = 0
            cur_prob = self.probs[self.idx_to_transform[cur_idx]]
            if random.random() >= cur_prob: # probability of index 0
                return data

            # appy transforms based on the tree
            while cur_idx < self.max_idx:
                if self.idx_to_transform[cur_idx] == -1:
                    break
                cur_transform = self.transforms[self.idx_to_transform[cur_idx]]
                cur_prob = self.probs[self.idx_to_transform[cur_idx]]
                data = cur_transform(data)

                if (cur_idx*2 + 1) < len(self.idx_to_transform) and self.idx_to_transform[cur_idx*2+1] != -1:
                    next_idx = cur_idx*2+1
                elif (cur_idx*2 + 2) < len(self.idx_to_transform) and self.idx_to_transform[cur_idx*2+2] != -1:
                    next_idx = cur_idx*2+2
                else:
                    break

                next_prob = self.probs[self.idx_to_transform[next_idx]]
                if random.random() < next_prob:
                    cur_idx = cur_idx*2 + 1
                else:
                    cur_idx = cur_idx*2 + 2
            return data

    def __call__(self, data):
        return self.apply_transform(data)