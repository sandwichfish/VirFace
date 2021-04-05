import torch
import shutil


def save_checkpoint(state, is_best=False, filename='checkpoint.pytorch'):
    filename += '.pytorch'
    print('saving snapshot in', filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pytorch')
    print('saving complete')
