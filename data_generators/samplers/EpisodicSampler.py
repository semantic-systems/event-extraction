# coding=utf-8
import copy
import random
from collections import Sized

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset


# from https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/prototypical_batch_sampler.py


class EpisodicBatchSampler(Sampler[int]):

    def __init__(self, data_source: Sized, n_way, k_shot, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(EpisodicBatchSampler, self).__init__(data_source)
        self.labels = data_source['label']
        self.classes_per_it = n_way
        self.sample_per_class = k_shot
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.from_numpy(np.asarray(self.classes))

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        """
        yield a batch of indexes
        """
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exist
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations


class CategoricalSampler(Sampler):
    # stolen from https://github.com/fiveai/on-episodes-fsl/blob/master/src/datasets/sampler.py
    def __init__(self, label, replacement, n_iter, n_way, n_shot, n_query):
        super(CategoricalSampler, self).__init__()

        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.replacement = replacement

        label = np.array(label)
        unique = np.unique(label)
        unique = np.sort(unique)

        self.m_ind = []
        self.labels = unique
        # dictionary to keep track of which images belong to which class
        self.class2imgs = {}

        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
            self.class2imgs[i] = list(ind.numpy())

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        if self.replacement:
            for i in range(self.n_iter):
                batch_gallery = []
                batch_query = []
                classes = torch.randperm(len(self.m_ind))[:self.n_way]
                for c in classes:
                    l = self.m_ind[c.item()]
                    pos = torch.randperm(l.size()[0])
                    batch_gallery.append(l[pos[:self.n_shot]])
                    batch_query.append(l[pos[self.n_shot:self.n_shot + self.n_query]])
                batch = torch.cat(batch_gallery + batch_query)
                yield batch

        else:
            n_to_sample = (self.n_query + self.n_shot)
            batch_size = self.n_way*(self.n_query + self.n_shot)

            remaining_classes = list(self.labels)

            copy_class2imgs = copy.deepcopy(self.class2imgs)

            while len(remaining_classes) > self.n_way - 1:
                # randomly select classes
                classes = random.sample(remaining_classes, self.n_way)

                batch_gallery = []
                batch_query = []

                # construct the batch
                for c in classes:
                    # sample correct numbers
                    l = random.sample(copy_class2imgs[c], n_to_sample)
                    batch_gallery.append(torch.tensor(l[:self.n_shot], dtype=torch.int32))
                    batch_query.append(torch.tensor(l[self.n_shot:self.n_shot + self.n_query],
                                                    dtype=torch.int32))

                    # remove values if used (sampling without replacement)
                    for value in l:
                        copy_class2imgs[c].remove(value)

                    # if not enough elements remain,
                    # remove key from dictionary and remaining classes
                    if len(copy_class2imgs[c]) < n_to_sample:
                        del copy_class2imgs[c]
                        remaining_classes.remove(c)

                batch = torch.cat(batch_gallery + batch_query)
                yield batch