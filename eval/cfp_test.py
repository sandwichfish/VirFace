import os
import torch
import util.data_transformer as transforms
from data.test_dataset import ImageFolder
import util.extract_feature as ext_feat
from eval.eval_1v1 import *


def cfp_test_gpu(cfp_path, model, method='FP'):
    batch_size = 200
    workers = 8
    method_set = 'CFP'
    cfp_data_path = os.path.join(cfp_path, 'Align_180_220/')
    meta_path = os.path.join(cfp_path, 'myMeta/%s_meta.txt' % method)
    model.eval()

    with torch.no_grad():
        # test dataloader
        transform = transforms.Compose([
            transforms.CenterCropWithOffset(150, 150, 0, 20, 0, 0, ignore_fault=True),
            transforms.Scale((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        test_dataset = ImageFolder(
            root=cfp_data_path, proto=os.path.join(cfp_path, 'myMeta/Pair_list_A.txt'),
            transform=transform, method=method_set
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
        feature = ext_feat.extract_feature(test_dataloader, model)  # feature extraction

        feature_np = feature.numpy().astype(np.float32)
        best_acc, best_thresh = base_1v1(feature_np, meta_path, method_set, similarity=cosine_similarity)

        print('CFP-%s acc is %f at threshold %f' % (method, best_acc, best_thresh))
        return best_acc, best_thresh
