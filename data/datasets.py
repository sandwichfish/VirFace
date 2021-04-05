import os
import torch
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def default_loader(path):
    return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def opencv_loader(path):
    import cv2
    return cv2.imread(path)


class ImageLabelFolder(data.Dataset):
    class imageLabel:
        def __init__(self, image_path, label_len, meta=None):
            self.image = image_path
            self.meta = meta
            self.labels = None
            self.label_len = label_len
            if os.path.exists(image_path):
                self.success = True
            else:
                self.success = False

    def __init__(self, root, proto, transform=None, target_transform=None, label_len=1, sign_imglist=False,
                 loader=default_loader, key_index=0, ignore_fault=False):
        protoFile = open(proto, encoding="utf8", errors='ignore')
        content = protoFile.readlines()
        self.imageLabel_list = []
        self.label_len = label_len
        self.loader = loader
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.sign_imglist = sign_imglist
        self.ignore_fault = ignore_fault
        errorwriter = open('./errorImage.txt', 'a', encoding='utf8')

        for line in tqdm(content):
            line = line.strip()
            line_list = line.split(' ')
            imagePath = ''
            for i in range(len(line_list)):
                if i < len(line_list) - self.label_len:
                    imagePath += line_list[i] + ' '
            imagePath_ = imagePath[:-1]
            imagePath = self.root + imagePath_
            imagePath = os.path.normpath(imagePath)

            cur_imageLabel = self.imageLabel(imagePath, line_list.__len__() - self.label_len)
            if self.sign_imglist:
                cur_imageLabel.meta = imagePath_

            label_list = []
            for i in range(line_list.__len__() - self.label_len, line_list.__len__()):
                label_list.append(float(line_list[i]))

            cur_imageLabel.labels = torch.FloatTensor(label_list)
            if cur_imageLabel.success:
                self.imageLabel_list.append(cur_imageLabel)
            else:
                errorwriter.write(line + '\n')

        self.labels = torch.FloatTensor((self.imageLabel_list.__len__()))
        self.labelgenerator(key_index)
        errorwriter.close()
        print('data size is ', self.imageLabel_list.__len__(), content.__len__())

    def labelgenerator(self, key_index=0):
        for i, imglabel in enumerate(self.imageLabel_list):
            self.labels[i] = imglabel.labels[key_index]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        cur_imageLabel = self.imageLabel_list[index]
        if self.ignore_fault:
            try:
                img = self.loader(cur_imageLabel.image)
            except:
                img = Image.new("RGB", (300, 300), (0, 0, 0))
        else:
            img = self.loader(cur_imageLabel.image)
        labels = cur_imageLabel.labels
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            labels = self.target_transform(labels)

        if self.sign_imglist:
            return img, labels, cur_imageLabel.meta
        else:
            return img, labels

    def __len__(self):
        return len(self.imageLabel_list)
