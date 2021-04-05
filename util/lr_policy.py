class lr_class:
    def __init__(self, base_lr=0.1, gamma=0.1, lr_policy="multistep", steps=None):
        if steps is None:
            steps = [8, 12, 15, 18]
        self.base_lr = base_lr
        self.gamma = gamma
        self.lr_policy = lr_policy
        self.steps = steps


def adjust_learning_rate(lr_obj, optimizer, Iteration):
    lr_policy = lr_obj.lr_policy
    base_lr = lr_obj.base_lr
    if lr_policy == 'multistep':
        steps = lr_obj.steps
        gamma = lr_obj.gamma
        multistep(base_lr, gamma, optimizer, Iteration, steps)


def multistep(base_lr, gamma, optimizer, Iteration, steps=None):
    if steps is None:
        steps = []
    current_step = 0
    for step in steps:
        if Iteration >= step:
            current_step += 1
    lr = pow(gamma, current_step) * base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
