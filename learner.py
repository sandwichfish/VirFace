import time
import torch
import torch.nn.functional as F
from util.generator_eval_func import compute_correct


def train(args, train_loader, model, criterion, optimizer, epoch, pretrain_flag_generator,
          batch_size, train_iter_per_epoch, display, IterationNum, writer=None, gpu=None):
    if args.method == 'generator':
        for k in model.keys():
            model[k].eval()
        model['generator'].train()
    else:
        for k in model.keys():
            model[k].train()

    print('===Training===\n')
    print('Training length is {}'.format(len(train_loader)))
    end = time.time()
    if args.method == 'virface' and pretrain_flag_generator == False:
        print('Please pre-train generator first.')
        exit()

    for data, target in train_loader:
        if gpu is None:
            target = target.cuda()
            data = data.cuda()
        else:
            target = target.cuda(gpu, non_blocking=True)
            data = data.cuda(gpu, non_blocking=True)

        feature = model['backbone'](data)

        if args.method != 'pretrain' and args.method != 'generator':
            unlabel_feature = feature.narrow(0, batch_size[0], batch_size[1])
            label_feature = feature.narrow(0, 0, batch_size[0])
            label = target.narrow(0, 0, batch_size[0])
        else:
            label_feature = feature
            label = target
            unlabel_feature = None

        if args.method == 'virface':
            gen_feature = model['generator'].gen_aug_feat(F.normalize(unlabel_feature.detach()), args.gen_num)
            gen_feature_label, mean_label, var_label = model['generator'](F.normalize(label_feature.detach()))
            gt_weight = model['head'].weight[label, :]
        elif args.method == 'generator':
            gen_feature, mean_label, var_label = model['generator'](F.normalize(label_feature))
            gt_weight = model['head'].weight[label, :]
        else:
            gen_feature = None

        if args.method != 'generator':
            output_label, output_unlabel, unlabel_label = model['head'](label_feature, label, unlabel_feature, gen_feature)

        if args.method == 'pretrain' or args.method == 'virclass':
            loss = criterion[0](output_label, label)
        elif args.method == 'virface':
            loss_label = criterion[0](output_label, label)
            loss_unlabel = criterion[1](output_unlabel, unlabel_label)
            loss_MSE = criterion[2](F.normalize(gen_feature_label), F.normalize(gt_weight))
            KL_loss = -0.5 * torch.mean(1 + var_label - torch.square(mean_label) - torch.exp(var_label))
            loss = loss_label + loss_unlabel + loss_MSE * args.MSE + KL_loss * args.KL
        else:
            loss_MSE = criterion[0](F.normalize(gen_feature), F.normalize(gt_weight))
            KL_loss = -0.5 * torch.mean(1 + var_label - torch.square(mean_label) - torch.exp(var_label))
            loss = loss_MSE * args.MSE + KL_loss * args.KL
            with torch.no_grad():
                weight = model['head'].weight
                cosine = F.linear(F.normalize(gen_feature), F.normalize(weight))
                correct = compute_correct(cosine, target)
                acc = float(correct[0].item()) / args.label_batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_show = loss.item()
        if args.method == 'virface':
            loss_label_show = loss_label.item()
            loss_unlabel_show = loss_unlabel.item()
            loss_MSE_show = loss_MSE.item()
            KL_loss_show = KL_loss.item()
        elif args.method == 'generator':
            loss_MSE_show = loss_MSE.item()
            KL_loss_show = KL_loss.item()

        run_time = time.time() - end
        end = time.time()
        IterationNum[0] += 1

        if IterationNum[0] % display == 0:
            print('Device_ID:' + str(torch.cuda.current_device()) + ', Epoch:' + str(epoch) + ', ' +
                  'Iteration:' + str(IterationNum[0]) + ', ' +
                  'TrainSpeed = ' + str(run_time) + ', ' +
                  'lr = ' + str(optimizer.param_groups[0]['lr']) + ', ' +
                  'Trainloss = ' + str(loss_show))

            writer.add_scalar('train_loss', loss_show,
                              epoch * min(train_iter_per_epoch, len(train_loader)) + IterationNum[0])

            if args.method == 'virface':
                writer.add_scalar('label_loss', loss_label_show,
                                  epoch * min(train_iter_per_epoch, len(train_loader)) + IterationNum[0])
                writer.add_scalar('unlabel_loss', loss_unlabel_show,
                                  epoch * min(train_iter_per_epoch, len(train_loader)) + IterationNum[0])
                writer.add_scalar('MSE_loss', loss_MSE_show,
                                  epoch * min(train_iter_per_epoch, len(train_loader)) + IterationNum[0])
                writer.add_scalar('KL_loss', KL_loss_show,
                                  epoch * min(train_iter_per_epoch, len(train_loader)) + IterationNum[0])
            elif args.method == 'generator':
                writer.add_scalar('MSE_loss', loss_MSE_show,
                                  epoch * min(train_iter_per_epoch, len(train_loader)) + IterationNum[0])
                writer.add_scalar('KL_loss', KL_loss_show,
                                  epoch * min(train_iter_per_epoch, len(train_loader)) + IterationNum[0])
                writer.add_scalar('acc', acc,
                                  epoch * min(train_iter_per_epoch, len(train_loader)) + IterationNum[0])

        if train_iter_per_epoch != 0 and IterationNum[0] % train_iter_per_epoch == 0:
            break
