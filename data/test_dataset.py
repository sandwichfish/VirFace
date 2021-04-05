import os
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


def opencv_loader(path):
    import cv2
    return cv2.imread(path)


class ImageFolder(data.Dataset):
    class image:
        def __init__(self, image_path):
            self.image = image_path
            if os.path.exists(image_path):
                self.success = True
            else:
                self.success = False

    def __init__(self, root, proto, transform=None, method=None,
                 loader=default_loader, ignore_fault=False):
        if method == 'YTF':
            content = os.listdir(os.path.join(root, proto))
        else:
            protoFile = open(proto, 'r')
            content = protoFile.readlines()

        self.image_list = []
        self.loader = loader
        self.root = root
        self.transform = transform
        self.ignore_fault = ignore_fault
        errorwriter = open('./errorTestImage.txt', 'a', encoding='utf8')
        errorwriter.write('==============' + method + '==============\n')

        for line in tqdm(content):
            line = line.strip()
            imagePath = []
            if method == 'CFP':
                imageName = line.split(',')[-1].split('Images/')[-1]
                imagePath.append(os.path.join(self.root, imageName))
            elif method == 'lfw':
                a = line.split()
                if len(a) == 3:
                    # genius
                    name1 = "%s/%s_%.4d.jpg" % (a[0], a[0], int(a[1]))
                    name2 = "%s/%s_%.4d.jpg" % (a[0], a[0], int(a[2]))
                elif len(a) == 4:
                    # imposter
                    name1 = "%s/%s_%.4d.jpg" % (a[0], a[0], int(a[1]))
                    name2 = "%s/%s_%.4d.jpg" % (a[2], a[2], int(a[3]))
                else:
                    print('lfw read error: ' + line)
                    errorwriter.write(line + '\n')
                    continue

                imagePath.append(os.path.join(self.root, name1))
                imagePath.append(os.path.join(self.root, name2))
            elif method == 'YTF':
                imagePath.append(os.path.join(self.root, proto, line))
            else:
                # ijb, calfw, cplfw,sllfw
                line_list = line.split()
                imagePath.append(os.path.join(self.root, line_list[0]))

            for i in imagePath:
                cur_image = self.image(i)
                if cur_image.success:
                    self.image_list.append(cur_image)
                else:
                    print('read error: ' + line)
                    errorwriter.write(line + '\n')

        errorwriter.close()
        print('data size is ', self.image_list.__len__(), content.__len__())

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        cur_image = self.image_list[index]
        if self.ignore_fault:
            try:
                img = self.loader(cur_image.image)
            except:
                img = Image.new("RGB", (300, 300), (0, 0, 0))
        else:
            img = self.loader(cur_image.image)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_list)
