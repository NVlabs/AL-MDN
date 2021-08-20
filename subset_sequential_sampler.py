# Updated for returning list of indices
# Adapted from https://github.com/pytorch/pytorch


from torch.utils.data.sampler import Sampler


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
