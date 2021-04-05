import numpy as np


def cosine_similarity(feature1, feature2):
    dot = sum(a * b for a, b in zip(feature1, feature2))
    norm_1 = sum(a * a for a in feature1) ** 0.5
    norm_2 = sum(b * b for b in feature2) ** 0.5
    cos_sim = dot / (norm_1 * norm_2)

    return cos_sim


def l2_similarity(feature1, feature2):
    dist = np.linalg.norm(feature1 - feature2)

    return dist


def base_1v1(feature, probe, method, similarity=cosine_similarity):
    # load probe
    fid = open(probe, 'r')
    if method == 'YTF' or method == 'CFP':
        _ = fid.readline()
    lines = fid.readlines()
    fid.close()

    # pair num
    half_list = ['lfw', 'CFP', 'YTF']
    if method in half_list:
        pair_num = len(lines)
        half_flag = False
    else:
        pair_num = int(len(lines) / 2)
        half_flag = True

    score = []
    label = []

    for index in range(pair_num):
        if half_flag:
            line = lines[index * 2].strip()
        else:
            line = lines[index].strip()

        if method == 'CFP':
            line = line.split(',')
        elif method == 'YTF':
            line = line.split(', ')
        else:
            # lfw, calfw, cplfw, sllfw
            line = line.split()

        # calculate similarity
        if method == 'CFP':
            dist = similarity(feature[int(line[2]) - 1], feature[int(line[3]) - 1])
        else:
            # lfw, calfw, cplfw, sllfw, ytf,
            dist = similarity(feature[2 * index], feature[2 * index + 1])

        score.append(dist)
        # same/not same
        if method == 'lfw':
            if len(line) == 3:
                label.append(1)
            elif len(line) == 4:
                label.append(0)
            else:
                print('lfw error at %d' % index)
                label.append(-1)
                continue
        else:
            if half_flag:
                if method == 'SLLFW':
                    id1 = line[0].split('/')[0]
                    id2 = lines[index * 2 + 1].strip().split()[0].split('/')[0]
                    label.append(id1 == id2)
                else:
                    # calfw, cplfw
                    label.append(int(line[-1]) > 0)
            else:
                # cfp, ytf
                label.append(int(line[-1]))

    # cal threshold acc
    scores = np.array(score)
    labels = np.array(label)

    best_thresh = 0.0
    best_acc = 0.0
    for i in range(pair_num):
        th = scores[i]
        scoreTest = (scores >= th)
        acc = np.mean((scoreTest == labels))
        if acc > best_acc:
            best_acc = acc
            best_thresh = th

    # print('=> %s acc: %f @ %f' % (method, best_acc, best_thresh))
    return best_acc, best_thresh
