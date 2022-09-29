import sys
import os
import torch.backends.cudnn as cudnn
import argparse

import model.resnet as resnet
import model.ArcFace as arc
import model.Generator as gen
from config import *
from learner import *
from data.datasets import ImageLabelFolder
from data.data_sampler import BatchMergeDatasetSampler
from eval.lfw_test import lfw_test_gpu
from eval.cfp_test import cfp_test_gpu
from util import target_transformer
import util.data_transformer as transforms
from util.logger import Logger
from util.lr_policy import *
from util.save import save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default='0, 1, 2, 3', help='index of used gpu.')
parser.add_argument('--method', type=str, default='pretrain',
                    help='choose training stage. [pretrain, generator, virclass, virface]')
# dataset
parser.add_argument('--label_batch_size', type=int, default=128, help='label data batch size.')  # 256 for pretrain
parser.add_argument('--unlabel_batch_size', type=int, default=128, help='unlabel data batch size.')
# network architecture
parser.add_argument('--arch', type=str, default='resnet50',
                    help='choose architecture of backbone. [resnet18, resnet34, resnet50, resnet101, resnet152, usr]')
parser.add_argument('--feat_len', type=int, default=512, help='length of embedding feature.')
parser.add_argument('--num_ids', type=int, default=84282, help='number of identities in labeled dataset.')
parser.add_argument('--gen_num', type=int, default=5, help='the number of generated features via generator.')
# lr_obj, optimizer
parser.add_argument('--lr', type=float, default=0.1, help='learning rate.')  # 0.01 for pretrain
parser.add_argument('--g_lr', type=float, default=0.1, help='learning rate for generator pretrain.')
parser.add_argument('--lr_policy', type=str, default='multistep', help='lr policy. [multistep]')
parser.add_argument('--gamma', type=float, default=0.1, help='optimizer gamma.')
parser.add_argument('--decay_steps', type=list, default=[8, 12, 15, 18], help='lr decay steps.')  # [4,6,8] for virclass
parser.add_argument('--g_decay_steps', type=list, default=[3, 5, 7, 9], help='lr decay steps for generator pretrain.')
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum.')
parser.add_argument('--weight_decay', type=float, default=0.001, help='optimizer weight decay.')
parser.add_argument('--max_epoch', type=int, default=10, help='max training epochs.') # 20 for pretrain
parser.add_argument('--KL', type=float, default=2.0, help='weight of KL loss.')
parser.add_argument('--L2', type=float, default=5e3, help='weight of MSE loss.')
# resume & pretrain
parser.add_argument('--resume', type=bool, default=False, help='resume flag.')
parser.add_argument('--resume_file', type=str, default='', help='resume checkpoint path.')
parser.add_argument('--pretrained_file', type=str, default='',
                    help='pretrained checkpoint path. is necessary if method is not "pretrain".')
# tensorboard & checkpoint
parser.add_argument('--tensorboard', type=bool, default=True, help='use tensorboard or not.')
parser.add_argument('--snapshot_prefix', type=str, default='./snapshot/', help='path to save checkpoint.')
# eval
parser.add_argument('--eval', type=bool, default=True, help='whether evaluate for each epoch on LFW, CFP-FF, CFP-FP')

if __name__ == '__main__':
    ################### Config ###################
    args = parser.parse_args()

    # checkpoint path
    if not os.path.exists(args.snapshot_prefix):
        os.mkdir(args.snapshot_prefix)

    snapshot_prefix = os.path.join(args.snapshot_prefix, args.arch + '_' + args.method)

    if not os.path.exists(snapshot_prefix):
        os.mkdir(snapshot_prefix)

    # GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # tensorboardX
    if args.tensorboard:
        from tensorboardX import SummaryWriter

        tf_record = './runs/'
        if not os.path.exists(tf_record):
            os.mkdir(tf_record)

        tf_record = os.path.join(tf_record, args.arch + '_' + args.method)
        writer = SummaryWriter(tf_record)
    else:
        writer = None

    # batch size
    train_batchSize = [args.label_batch_size, args.unlabel_batch_size]

    # backbone architecture
    if args.arch == 'resnet18':
        backbone = resnet.resnet18(feature_len=args.feat_len)
    elif args.arch == 'resnet34':
        backbone = resnet.resnet34(feature_len=args.feat_len)
    elif args.arch == 'resnet50':
        backbone = resnet.resnet50(feature_len=args.feat_len)
    elif args.arch == 'resnet101':
        backbone = resnet.resnet101(feature_len=args.feat_len)
    elif args.arch == 'resnet152':
        backbone = resnet.resnet152(feature_len=args.feat_len)
    elif args.arch == 'usr':
        backbone = model_usr
    else:
        raise NameError(
            'Arch %s is not support. Please enter from [resnet18, resnet34, resnet50, resnet101, resnet152, usr]' % args.arch)

    # head
    model_head = arc.ArcMarginProduct_virface(in_features=args.feat_len, out_features=args.num_ids, s=32, m=0.5,
                                              device='cuda')
    # generator, model
    if args.method == 'pretrain':
        model = {'backbone': backbone, 'head': model_head}
        pretrain_flag = False
        criterion = [torch.nn.CrossEntropyLoss()]
    elif args.method == 'virclass':
        model = {'backbone': backbone, 'head': model_head}
        pretrain_flag = True
        criterion = [torch.nn.CrossEntropyLoss()]
    elif args.method == 'virface':
        model_gen = gen.Generator(feature_len=args.feat_len)
        model = {'backbone': backbone, 'head': model_head, 'generator': model_gen}
        pretrain_flag = True
        criterion = [torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss(), torch.nn.MSELoss()]
    elif args.method == 'generator':
        model_gen = gen.Generator(feature_len=args.feat_len)
        model = {'backbone': backbone, 'head': model_head, 'generator': model_gen}
        pretrain_flag = True
        criterion = [torch.nn.MSELoss()]
    else:
        raise NameError(
            'method %s is not support. Please enter from [pretrain, generator, virclass, virface]' % args.method)

    sys.stdout = Logger(os.path.join(snapshot_prefix, str(time.time()) + '.log'))

    ################### Data Loading ###################
    print('Data loading...')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    train_dataset_label = ImageLabelFolder(
        root=trainRoot_label, proto=trainProto_label,
        transform=transforms.Compose([
            transforms.CenterCropWithOffset(150, 150, 0, 20, 1, 1),
            transforms.RandomHorizontalFlip(),
            transforms.Scale((112, 112)),
            transforms.ToTensor(),
            normalize,
        ]),
        target_transform=target_transformer.ToInt()
    )
    if args.method != 'pretrain' and args.method != 'generator':
        train_dataset_unlabel = ImageLabelFolder(
            root=trainRoot_unlabel, proto=trainProto_unlabel,
            transform=transforms.Compose([
                transforms.CenterCropWithOffset(150, 150, 0, 20, 1, 1),
                transforms.RandomHorizontalFlip(),
                transforms.Scale((112, 112)),
                transforms.ToTensor(),
                normalize,
            ]),
            target_transform=target_transformer.ToInt()
        )

        train_sampler = BatchMergeDatasetSampler(
            dataset=torch.utils.data.ConcatDataset([train_dataset_label, train_dataset_unlabel]),
            dataset_batch_size=train_batchSize)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset([train_dataset_label, train_dataset_unlabel]),
            batch_size=sum(train_batchSize),
            num_workers=workers, pin_memory=True, sampler=train_sampler, drop_last=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset_label, batch_size=args.label_batch_size, shuffle=True,
            num_workers=workers, pin_memory=True, sampler=None, drop_last=True
        )

    print('Data loading complete')
    ###################  MODEL INIT ###################
    print('Model Initing...')
    cudnn.benchmark = cudnn_use
    pretrain_flag_generator = False

    # model cuda
    if len(args.gpus.split(',')) > 1:
        for k in model.keys():
            if k == 'head':
                model[k] = model[k].cuda()
            else:
                model[k] = torch.nn.DataParallel(model[k]).cuda()
    else:
        for k in model.keys():
            model[k] = model[k].cuda()

    # loss cuda
    for i in range(len(criterion)):
        criterion[i] = criterion[i].cuda()

    # optimizer
    lr_obj = lr_class(base_lr=args.lr, gamma=args.gamma, lr_policy=args.lr_policy, steps=args.decay_steps)
    if args.method == 'pretrain' or args.method == 'virclass':
        optimizer = torch.optim.SGD([{'params': model['backbone'].parameters(), "lr": lr_obj.base_lr},
                                     {'params': model['head'].parameters(), "lr": lr_obj.base_lr}],
                                    lr=lr_obj.base_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.method == 'virface':
        optimizer = torch.optim.SGD([{'params': model['backbone'].parameters(), "lr": lr_obj.base_lr},
                                     {'params': model['head'].parameters(), "lr": lr_obj.base_lr},
                                     {'params': model['generator'].parameters(), "lr": lr_obj.base_lr}],
                                    lr=lr_obj.base_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.method == 'generator':
        optimizer = torch.optim.SGD([{'params': model['generator'].parameters(), "lr": lr_obj.base_lr}],
                                    lr=lr_obj.base_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # resume & pretrain
    if args.resume:
        if os.path.isfile(args.resume_file):
            print("resume => loading checkpoint '{}'".format(args.resume_file))
            checkpoint = torch.load(args.resume_file)
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            for k in model.keys():
                if k not in checkpoint:
                    continue
                resume_dict = checkpoint[k]
                model_dict = model[k].state_dict()
                update_dict = {k: v for k, v in resume_dict.items() if k in model_dict}
                unupdate_dict = {k for k in model_dict if k not in resume_dict.keys()}  # for testing
                model_dict.update(update_dict)
                model[k].load_state_dict(model_dict)
                # testing
                for it in unupdate_dict:
                    print('===> model %s no updating param:' % k, it)
        else:
            print("resume => no checkpoint found at '{}'".format(args.resume_file))
            exit()
    elif pretrain_flag:
        if os.path.isfile(args.pretrained_file):
            print("pretrain => loading checkpoint '{}'".format(args.pretrained_file))
            checkpoint = torch.load(args.pretrained_file)
            for k in model.keys():
                if k not in checkpoint:
                    continue
                pretrain_dict = checkpoint[k]
                model_dict = model[k].state_dict()
                update_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
                unupdate_dict = {k for k in model_dict if k not in pretrain_dict.keys()}  # for testing
                model_dict.update(update_dict)
                model[k].load_state_dict(model_dict)
                # generator pretrain flag
                if k == 'generator':
                    pretrain_flag_generator = True
                # testing
                for it in unupdate_dict:
                    print('===> model %s no updating param:' % k, it)
        else:
            print("pretrain => no checkpoint found at '{}'".format(args.pretrained_file))
            exit()

    print('Model Initing complete')

    ###################  TRAINING  ###################
    cfp_fp_best = 0
    if args.tensorboard:
        acc_text = '## Evaluating Results\n'
        acc_text += '|Epoch|LFW|CFP-FF|CFP-FP|\n|----|----|----|----|\n'

    if args.eval:
        print('Init evaluating...')
        with torch.no_grad():
            lfw_prec = lfw_test_gpu(lfw_path, model['backbone'])
            cfp_prec_ff = cfp_test_gpu(cfp_path, model['backbone'], method='FF')  # cfp
            cfp_prec_fp = cfp_test_gpu(cfp_path, model['backbone'], method='FP')  # cfp

        if args.tensorboard:
            acc_text += '|init|%.4f @t = %.4f|' % lfw_prec + \
                        '%.4f @t = %.4f|' % cfp_prec_ff + \
                        '%.4f @t = %.4f|\n' % cfp_prec_fp
            writer.add_text('Test Results', acc_text, 0)

    for epoch in range(start_epoch, args.max_epoch + 1):
        is_best = False
        print('Training... epoch:', epoch)
        adjust_learning_rate(lr_obj, optimizer, epoch)

        train(args, train_loader, model, criterion, optimizer, epoch, pretrain_flag_generator,
              train_batchSize, train_iter_per_epoch, display, IterationNum, writer)

        if args.eval and args.method != 'generator':
            print('Evaluating... epoch:', epoch)
            with torch.no_grad():
                lfw_prec = lfw_test_gpu(lfw_path, model['backbone'])
                cfp_prec_ff = cfp_test_gpu(cfp_path, model['backbone'], method='FF')  # cfp
                cfp_prec_fp = cfp_test_gpu(cfp_path, model['backbone'], method='FP')  # cfp

            if cfp_prec_fp[0] > cfp_fp_best:
                is_best = True

            if args.tensorboard:
                acc_text += '|%d|' % int(epoch) + '%.4f @t = %.4f|' % lfw_prec + \
                            '%.4f @t = %.4f|' % cfp_prec_ff + \
                            '%.4f @t = %.4f|\n' % cfp_prec_fp
                writer.add_text('Test Results', acc_text, epoch)

        # save per epoch
        IterationNum[0] = 0
        save_dict = {'epoch': epoch + 1}
        for k in model.keys():
            save_dict.update({k: model[k].state_dict()})
        save_checkpoint(save_dict, is_best, os.path.join(snapshot_prefix, '_epoch_' + str(epoch)))

    print('Training complete')
