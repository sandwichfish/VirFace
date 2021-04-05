from torch import nn
import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter


class ArcMarginProduct_virface(nn.Module):
    def __init__(self, in_features=512, out_features=84281, s=32, m=0.5, easy_margin=False, device='cuda'):
        super(ArcMarginProduct_virface, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.dropout = nn.Dropout(0.5)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.device = device

    def forward(self, x, label, unlabel_x=None, unlabel_aug=None, overlap=False):
        if unlabel_x is not None:
            # filter overlap
            if overlap:
                unlabel_dot_w = F.linear(F.normalize(unlabel_x.detach()),
                                         F.normalize(self.weight.detach())) * self.s
                prob = F.softmax(unlabel_dot_w, dim=1)
                max_prob = torch.max(prob, dim=1)[0]
                idx_lt = max_prob.lt(0.8)

                idx = idx_lt
            else:
                idx = torch.ones(unlabel_x.shape[0]).bool().cuda()

            unlabel_data = unlabel_x[idx]

            weight_all = torch.cat([F.normalize(self.weight), F.normalize(unlabel_data)], dim=0)
            weight_all_fix = torch.cat([F.normalize(self.weight), F.normalize(unlabel_data).detach()], dim=0)
        else:  # no unlabel data, return arcface
            weight_all = F.normalize(self.weight)

        # virclass
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if unlabel_x is not None:
            x = self.dropout(x)
        cosine_label = F.linear(F.normalize(x), weight_all)
        sine_label = 1.0 - torch.pow(cosine_label, 2)
        sine_label = torch.where(sine_label > 0, sine_label, torch.zeros(sine_label.size(), device=self.device))
        sine_label = torch.sqrt(sine_label)

        phi_label = cosine_label * self.cos_m - sine_label * self.sin_m
        if self.easy_margin:
            phi_label = torch.where(cosine_label > 0, phi_label, cosine_label)
        else:
            phi_label = torch.where((cosine_label - self.th) > 0, phi_label, cosine_label - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine_label.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output_label = (one_hot * phi_label) + ((1.0 - one_hot) * cosine_label)
        output_label *= self.s

        if unlabel_aug is None or unlabel_x is None:  # no aug, return virclass or arcface
            return output_label, None, None

        # virinstance
        aug_data = torch.cat([unlabel_aug[i * unlabel_x.shape[0]: (i + 1) * unlabel_x.shape[0]][idx] for i in range(int(unlabel_aug.shape[0]/unlabel_x.shape[0]))], dim=0)
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine_unlabel = F.linear(F.normalize(aug_data), weight_all_fix)
        sine_unlabel = 1.0 - torch.pow(cosine_unlabel, 2)
        sine_unlabel = torch.where(sine_unlabel > 0, sine_unlabel, torch.zeros(sine_unlabel.size(), device=self.device))
        sine_unlabel = torch.sqrt(sine_unlabel)

        phi_unlabel = cosine_unlabel * self.cos_m - sine_unlabel * self.sin_m
        if self.easy_margin:
            phi_unlabel = torch.where(cosine_unlabel > 0, phi_unlabel, cosine_unlabel)
        else:
            phi_unlabel = torch.where((cosine_unlabel - self.th) > 0, phi_unlabel, cosine_unlabel - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        unlabel_label = torch.arange(self.out_features, self.out_features + unlabel_data.size(0), device=self.device).repeat(int(unlabel_aug.shape[0]/unlabel_x.shape[0]))

        one_hot_unlabel = torch.zeros(cosine_unlabel.size(), device=self.device)
        one_hot_unlabel.scatter_(1, unlabel_label.view(-1, 1).long(), 1)

        output_unlabel = (one_hot_unlabel * phi_unlabel) + ((1.0 - one_hot_unlabel) * cosine_unlabel)
        output_unlabel *= self.s

        return output_label, output_unlabel, unlabel_label
