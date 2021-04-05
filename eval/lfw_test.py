import os
import torch
import util.data_transformer as transforms
from data.test_dataset import ImageFolder
import util.extract_feature as ext_feat
from eval.eval_1v1 import *


def lfw_test_gpu(lfw_path, model):
    batch_size = 400
    workers = 8
    method = 'lfw'
    lfw_data_path = os.path.join(lfw_path, 'lfw_manual_remove_error_detection_align/lfw_manual_remove_error_detection/')
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
            root=lfw_data_path, proto=os.path.join(lfw_path, 'lfw_pairs.txt'),
            transform=transform, method=method
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
        feature = ext_feat.extract_feature(test_dataloader, model)  # feature extraction

        feature_np = feature.numpy().astype(np.float32)
        best_acc, best_thresh = base_1v1(feature_np, os.path.join(lfw_path, 'lfw_pairs.txt'), method, similarity=cosine_similarity)

        print('LFW acc is %f at threshold %f' % (best_acc, best_thresh))
        return best_acc, best_thresh
