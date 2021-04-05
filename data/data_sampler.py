from torch.utils.data.sampler import *
import math
import numpy as np


class BatchMergeDatasetSampler(Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """

    def __init__(self, dataset, dataset_batch_size):
        super(BatchMergeDatasetSampler, self).__init__(data_source=dataset)
        self.dataset = dataset

        self.dataset_batch_size = dataset_batch_size
        self.batch_size = sum(self.dataset_batch_size)
        self.number_of_datasets = len(dataset.datasets)

        self.dataset_iter_num = [math.ceil(len(self.dataset.datasets[i]) / self.dataset_batch_size[i]) for i in
                                 range(self.number_of_datasets)]

    def __len__(self):
        return max(self.dataset_iter_num) * self.batch_size

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size

        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        max_idx = np.argmax(self.dataset_iter_num)
        epoch_samples = self.dataset_iter_num[max_idx] * self.batch_size

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []

                samples_to_grab = self.dataset_batch_size[i]
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)
