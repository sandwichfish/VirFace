import torch
import numpy as np
import torch.nn.functional as F


def extract_feature(test_loader, model, norm=True):
    result = None
    model.eval()
    pre = 0
    for i, data in enumerate(test_loader):
        if i % 100 == 0:
            print(i, 'vs', test_loader.__len__())
        data = data.cuda()
        output = model(data)
        if norm:
            output = F.normalize(output, dim=0)

        if result is None:
            size = np.array(output.data.cpu().size())
            n = size[0]
            size[0] = test_loader.dataset.__len__()
            result = torch.FloatTensor(*size).zero_()

        result[pre:pre + n, :] = output.data.cpu().clone()
        pre = pre + n

    return result
