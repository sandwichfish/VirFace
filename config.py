"""
user backbone network setting
"""
# import ...
model_usr = None

# common args
cudnn_use = True
workers = 16
IterationNum = [0]
display = 1
train_iter_per_epoch = 100000
start_epoch = 0

# training data root & meta
trainRoot_label = r''  # label data root
trainProto_label = r''  # label data meta
trainRoot_unlabel = r''  # unlabel data root
trainProto_unlabel = r''  # unlabel data meta

# LFW, CFP root
lfw_path = r''
cfp_path = r''
