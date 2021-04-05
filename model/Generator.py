import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, feature_len=256, device='gpu'):
        super(Generator, self).__init__()

        self.en1 = nn.Linear(in_features=feature_len, out_features=256, bias=True)
        self.en_mean = nn.Linear(in_features=256, out_features=128, bias=True)
        self.en_var = nn.Linear(in_features=256, out_features=128, bias=True)

        self.de1 = nn.Linear(in_features=128, out_features=256, bias=True)
        self.de2 = nn.Linear(in_features=256, out_features=feature_len, bias=True)

        self.device = device

        self.RELU = nn.ReLU()

    def forward(self, x, gen_weight=False):
        x = self.RELU(self.en1(x))

        x_mean = self.en_mean(x)
        x_var = self.en_var(x)
        z = self.re_sample(x_mean, x_var)

        if gen_weight:
            x = self.RELU(self.de1(x_mean))
        else:
            x = self.RELU(self.de1(z))
        x = self.de2(x)

        return x, x_mean, x_var

    def re_sample(self, x_mean, x_var):
        data_shape = x_mean.shape
        if self.device == 'gpu':
            std_gaussian = torch.normal(mean=torch.zeros(data_shape), std=torch.ones(data_shape)).cuda()
        else:
            std_gaussian = torch.normal(mean=torch.zeros(data_shape), std=torch.ones(data_shape)).cpu()
        return x_mean + torch.exp(x_var / 2) * std_gaussian

    def set_device(self, device):
        self.device = device

    def gen_gen_feat(self, x, sample_count):
        x = self.RELU(self.en1(x))
        x_mean = self.en_mean(x)
        x_var = self.en_var(x)

        gen_list = []
        for i in range(sample_count):
            z = self.re_parameter(x_mean, x_var)
            x = self.RELU(self.de1(z))
            x = self.de2(x)
            gen_list.append(F.normalize(x))

        gen_array = torch.cat(gen_list, dim=0)
        return gen_array
