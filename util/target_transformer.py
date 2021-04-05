class Reshape:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        return data.view(self.shape)


class ToInt:
    def __call__(self, data):
        return int(data[0])


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, label):
        for t in self.transforms:
            label = t(label)
        return label
