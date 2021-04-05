from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageFilter

try:
    import accimage
except ImportError:
    accimage = None
import scipy.ndimage as ndimage
import numpy as np
import numbers
import types
import collections


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
         transforms.Compose([
             transforms.CenterCrop(10),
             transforms.ToTensor(),
         ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(255)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


class ToPILImage(object):
    """Convert a tensor to PIL Image.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL.Image while preserving the value range.
    """

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL.Image.

        Returns:
            PIL.Image: Image converted to PIL.Image.

        """
        npimg = pic
        mode = None
        if isinstance(pic, torch.FloatTensor):
            pic = pic.mul(255).byte()
        if torch.is_tensor(pic):
            npimg = np.transpose(pic.numpy(), (1, 2, 0))
        assert isinstance(npimg, np.ndarray), 'pic should be Tensor or ndarray'
        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]

            if npimg.dtype == np.uint8:
                mode = 'L'
            if npimg.dtype == np.int16:
                mode = 'I;16'
            if npimg.dtype == np.int32:
                mode = 'I'
            elif npimg.dtype == np.float32:
                mode = 'F'
        else:
            if npimg.dtype == np.uint8:
                mode = 'RGB'
        assert mode is not None, '{} is not supported'.format(npimg.dtype)
        return Image.fromarray(npimg, mode=mode)


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


class Scale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


class CenterCrop(object):
    """Crops the given PIL.Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))


class Pad(object):
    """Pad the given PIL.Image on all sides with the given "pad" value.

    Args:
        padding (int or sequence): Padding on each border. If a sequence of
            length 4, it is used to pad left, top, right and bottom borders respectively.
        fill: Pixel fill value. Default is 0.
    """

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be padded.

        Returns:
            PIL.Image: Padded image.
        """
        return ImageOps.expand(img, border=self.padding, fill=self.fill)


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class RandomCrop(object):
    """Crop the given PIL.Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.random() < self.flip_prob:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomSizedCrop(object):
    """Crop the given PIL.Image to random size and aspect ratio.

    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))


class CenterCropWithOffset(object):
    """Crops the given PIL.Image at the center.

        Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, x_size, y_size, x_offset, y_offset, x_joggle, y_joggle, ignore_fault=False):
        self.x_size = x_size
        self.y_size = y_size
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.x_joggle = x_joggle
        self.y_joggle = y_joggle
        self.ignore_fault = ignore_fault

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th = self.y_size
        tw = self.x_size

        x1 = int(round((w - tw) / 2.)) + self.x_offset + random.uniform(0, self.x_joggle) * (
                    random.uniform(0, 1) - 0.5) * 2
        y1 = int(round((h - th) / 2.)) + self.y_offset + random.uniform(0, self.y_joggle) * (
                    random.uniform(0, 1) - 0.5) * 2

        x1 = max(0, x1)
        x2 = min(x1 + tw, w)
        y1 = max(0, y1)
        y2 = min(y1 + th, h)
        if y2 - y1 != self.y_size:
            if self.ignore_fault:
                y1 = 0
                th = y2
            else:
                raise (RuntimeError('(data_transformer)Size Error:y2-y1 != self.y_size:' +
                                    str(y2 - y1) + 'vs' + str(self.y_size)))
        if x2 - x1 != self.x_size:
            if self.ignore_fault:
                x1 = 0
                tw = x2
            else:
                raise (RuntimeError('(data_transformer)Size Error:x2-x1 != self.x_size:' +
                                    str(x2 - x1) + 'vs' + str(self.x_size)))

        return img.crop((x1, y1, x1 + tw, y1 + th))


class GammaCorrection(object):
    """GammaCorrection
    Args:
        ratio: ratio to use the projection
        low_0, high_0. different type of parameter
    """

    def __init__(self, ratio=0, low_0=0.5, high_0=1.8, low_1=0.5, high_1=3, low_2=0.5, high_2=1, projection_type=None):
        self.ratio = ratio
        self.low_0 = low_0
        self.high_0 = high_0
        self.low_1 = low_1
        self.high_1 = high_1
        self.low_2 = low_2
        self.high_2 = high_2

        self.projection_type = projection_type

    def GammaCorrection_uniform(self, img, fGamma):
        data = np.array(img)
        lut = [max(min(math.pow(i / 255.0, fGamma) * 255.0, 255), 0) for i in range(256)]
        lut = np.array(lut)
        data[:, :, :] = lut[data[:, :, :]]

        return Image.fromarray(data)

    def GammaCorrection_Horizontal(self, img, low, high):
        data = np.array(img)

        step = (high - low) / data.shape[1]
        orientationFlag = int(random.uniform(0, 1) + 0.5)

        if orientationFlag == 1:
            gamma = low
            for j in range(data.shape[1]):
                lut = [max(min(math.pow(i / 255.0, gamma) * 255.0, 255), 0) for i in range(256)]
                lut = np.array(lut)
                for i in range(data.shape[0]):
                    data[i, j, :] = lut[data[i, j, :]]

                gamma += step

        else:
            gamma = low
            for j in range(data.shape[1]):
                lut = [max(min(math.pow(i / 255.0, gamma) * 255.0, 255), 0) for i in range(256)]
                lut = np.array(lut)
                for i in range(data.shape[0]):
                    data[i, data.shape[1] - 1 - j, :] = lut[data[i, data.shape[1] - 1 - j, :]]

                gamma += step

        return Image.fromarray(data)

    def GammaCorrection_channel(self, img, fGamma):
        data = np.array(img)
        lut = [max(min(math.pow(i / 255.0, fGamma) * 255.0, 255), 0) for i in range(256)]
        lut = np.array(lut)

        ind_channel = int(random.uniform(0, 2) + 0.5)
        data[:, :, ind_channel] = lut[data[:, :, ind_channel]]

        return Image.fromarray(data)

    def __call__(self, img):
        if random.uniform(0, 1) >= self.ratio:
            return img

        if self.projection_type is None:
            projection_type = int(random.uniform(0, 2) + 0.5)
        if isinstance(self.projection_type, list):
            random.shuffle(self.projection_type)
            projection_type = self.projection_type[0]

        if projection_type == 0:
            gamma = random.uniform(self.low_0, self.high_0)
            img = self.GammaCorrection_uniform(img, gamma)
        elif projection_type == 1:
            low = self.low_1
            high = random.uniform(low, self.high_1)
            img = self.GammaCorrection_Horizontal(img, low, high)
        elif projection_type == 2:
            gamma = random.uniform(self.low_2, self.high_2)
            img = self.GammaCorrection_channel(img, gamma)
        else:
            raise (RuntimeError('projection_type should be in [0,1,2]' + str(self.projection_type)))
        return img


class BlurProjection(object):
    """BlurProjection
    Args:
        ratio: ratio to use the projection
    """

    def __init__(self, ratio=0, guassian_low=0, guassian_high=3, downsample_low=0.25, downsample_high=5, psf_len_low=2,
                 psf_len_high=5, psf_ang_low=1, psf_ang_high=180, projection_type=None):
        self.ratio = ratio
        self.guassian_low = guassian_low
        self.guassian_high = guassian_high
        self.downsample_low = downsample_low
        self.downsample_high = downsample_high
        self.psf_len_low = psf_len_low
        self.psf_len_high = psf_len_high
        self.psf_ang_low = psf_ang_low
        self.psf_ang_high = psf_ang_high

        self.projection_type = projection_type

    def Gaussian(self, img, radius):
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

    def DownSample(self, img, downRate):
        rows, cols = img.size
        rows_new = int(rows * downRate)
        cols_new = int(cols * downRate)

        img = img.resize((rows_new, cols_new), resample=Image.BILINEAR)
        img = img.resize((rows, cols), resample=Image.BILINEAR)

        return img

    def Psf(self, img, len_, angle):
        EPS = 0.00000001
        if int(angle) % 90 == 0:
            angle -= 1

        half = len_ / 2
        alpha = (angle - (angle / 180) * 180) / 180 * math.pi
        cosalpha = math.cos(alpha)
        sinalpha = math.sin(alpha)

        if cosalpha < 0:
            xsign = -1
        else:
            if angle == 90:
                xsign = 0
            else:
                xsign = 1

        psfwdt = 1
        sx = int(abs(half * cosalpha + psfwdt * xsign - len_ * EPS) + 0.5)
        sy = int(abs(half * sinalpha + psfwdt - len_ * EPS) + 0.5)
        psf1 = np.zeros((sy, sx))
        psf2 = np.zeros((sy * 2, sx * 2))

        # row = 2 * sy
        # col = 2 * sx
        for i in range(sy):
            for j in range(sx):
                psf1[i, j] = i * abs(cosalpha) - j * sinalpha
                rad = math.sqrt(i * i + j * j)
                if rad >= half and abs(psf1[i, j]) <= psfwdt:
                    tmp = half - abs((j + psf1[i, j] * sinalpha) / (cosalpha + EPS))
                    psf1[i, j] = math.sqrt(psf1[i, j] * psf1[i, j] + tmp * tmp)

        psf2[:sy, :sx] = psf1[:sy, :sx]
        for i in range(sy):
            for j in range(sx):
                psf2[2 * sy - 1 - i, 2 * sx - 1 - j] = psf1[i, j]
                psf2[sy + i][j] = 0
                psf2[i][sx + j] = 0

        sum_ = psf2.sum()
        psf2 = psf2 / (sum_ + EPS)
        if cosalpha > 0:
            for i in range(sy):
                tmp = np.copy(psf2[i])
                psf2[i] = np.copy(psf2[2 * sy - i - 1])
                psf2[2 * sy - i - 1] = np.copy(tmp)

        data = np.array(img)
        data[:, :, 0] = ndimage.convolve(data[:, :, 0], psf2)
        data[:, :, 1] = ndimage.convolve(data[:, :, 1], psf2)
        data[:, :, 2] = ndimage.convolve(data[:, :, 2], psf2)

        return Image.fromarray(data)

    def __call__(self, img):
        if random.uniform(0, 1) >= self.ratio:
            return img

        if self.projection_type is None:
            projection_type = int(random.uniform(0, 2) + 0.5)
        elif isinstance(self.projection_type, list):
            random.shuffle(self.projection_type)
            projection_type = self.projection_type[0]
        else:
            projection_type = self.projection_type

        if projection_type == 0:
            radius = random.uniform(self.guassian_low, self.guassian_high)
            img = self.Gaussian(img, radius)
        elif projection_type == 1:
            downsample = random.uniform(self.downsample_low, self.downsample_high)
            img = self.DownSample(img, downsample)
        elif projection_type == 2:
            len_ = random.uniform(self.psf_len_low, self.psf_len_high)
            ang = random.uniform(self.psf_ang_low, self.psf_ang_high)
            img = self.Psf(img, len_, ang)
        else:
            raise (RuntimeError('projection_type should be in [0,1,2]' +
                                str(self.projection_type)))
        return img


class ConvertGray(object):
    def __init__(self, ratio=0):
        self.ratio = ratio

    def __call__(self, img):
        if random.uniform(0, 1) >= self.ratio:
            return img

        return img.convert('L').convert('RGB')


class MultiCropTransform(object):
    """ Create multi crops of the same image """
    def __init__(self, transform, ori_transform=None, n=2):
        self.transform = transform
        self.ori_transform = ori_transform
        self.n = n

    def __call__(self, x):
        out = []
        if self.ori_transform is None:
            for i in range(self.n):
                out.append(self.transform(x))
        else:
            out.append(self.ori_transform(x))
            if self.n > 2:
                for i in range(self.n - 1):
                    out.append(self.transform(x))
            else:
                out.append(self.transform(x))

        return out
