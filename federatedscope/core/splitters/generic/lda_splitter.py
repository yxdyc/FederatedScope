import numpy as np
from federatedscope.core.splitters.utils import \
    dirichlet_distribution_noniid_slice


class LDASplitter(object):
    def __init__(self, client_num, alpha=0.5, min_size=10, smooth=True):
        self.client_num = client_num
        self.alpha = alpha
        self.min_size = min_size
        self.smooth = smooth

    def __call__(self, dataset, prior=None):
        dataset = [ds for ds in dataset]
        label = np.array([y for x, y in dataset])
        idx_slice = dirichlet_distribution_noniid_slice(label,
                                                        self.client_num,
                                                        self.alpha,
                                                        prior=prior,
                                                        min_size=self.min_size,
                                                        smooth=self.smooth)
        data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list

    def __repr__(self):
        return f'{self.__class__.__name__}(client_num={self.client_num}, ' \
               f'alpha={self.alpha})'
