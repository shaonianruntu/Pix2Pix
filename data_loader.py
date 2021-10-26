# 模型参数
import os
import random
import numpy as np
import scipy.io as sio
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

IMG_EXTEND = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def is_img_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTEND)


def mat_process(img_fl):
    """
    process mat, 11 channel to 8 channel
    :param img_fl:
    :return:
    """
    img_fl = img_fl.astype(np.float32)
    temp = img_fl
    lists = []
    refen = [(0, 0), (1, 1), (2, 3), (4, 5), (6, 6), (7, 9), (8, 8), (10, 10)]
    for item in refen:
        aa, bb = item
        if aa == bb:
            ll = temp[aa, :, :]
        else:
            ll = temp[aa, :, :] + temp[bb, :, :]
            ll = np.where(ll > 1, 1, ll)
        lists.append(ll.reshape(1, ll.shape[0], ll.shape[1]))
    parsing = np.concatenate(lists, 0)

    return parsing


def make_dataset(dir, file):
    imgA = []
    imgB = []
    file = os.path.join(dir, file)

    fimg = open(file, 'r')
    for line in fimg:
        line = line.strip('\n')
        line = line.rstrip()
        word = line.split("||")
        imgA.append(os.path.join(dir, word[0].lstrip('/')))
        imgB.append(os.path.join(dir, word[1].lstrip('/')))

    return imgA, imgB


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(data.Dataset):
    def __init__(self,
                 args,
                 isTrain=0,
                 transform=None,
                 return_paths=None,
                 loader=default_loader):
        super(MyDataset, self).__init__()
        self.args = args
        imgs = make_dataset(self.args.dataroot, self.args.datalist)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + self.args.dataroot +
                                dir + "\n"
                                "Supported image extensions are: " +
                                ",".join(IMG_EXTEND)))

        self.isTrain = isTrain
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path_A = self.imgs[0][index]
        path_B = self.imgs[1][index]

        imgA = self.load_img(path_A, self.args.input_nc)
        imgB = self.load_img(path_B, self.args.output_nc)

        if self.isTrain == 0:

            imgA = self.load_padding(imgA)
            imgB = self.load_padding(imgB)

            imgA, imgB = self.random_crop(imgA, imgB)

            imgA = self.normalize(imgA, self.args.input_nc)
            imgB = self.normalize(imgB, self.args.output_nc)

        else:

            imgA = self.fine_padding(imgA)
            imgB = self.fine_padding(imgB)

            imgA = self.normalize(imgA, self.args.input_nc)
            imgB = self.normalize(imgB, self.args.output_nc)

        return imgA, imgB

    def __len__(self):
        return len(self.imgs[1])

    def load_img(self, path, nc):
        if nc == 3:
            img = Image.open(path).convert('RGB')
        else:
            img = Image.open(path).convert('L')
        return img

    def load_padding(self, img):
        transform = transforms.Compose([
            transforms.Resize((self.args.img_height, self.args.img_width)),
            transforms.Pad((self.args.load_pad_w, self.args.load_pad_h),
                           fill=0,
                           padding_mode='constant'),
        ])
        img = transform(img)
        return img

    def fine_padding(self, img):
        transform = transforms.Compose([
            transforms.Resize((self.args.img_height, self.args.img_width)),
            transforms.Pad((self.args.fine_pad_w, self.args.fine_pad_h),
                           fill=0,
                           padding_mode='constant'),
        ])
        img = transform(img)
        return img

    def random_crop(self, imgA, imgB):
        i = random.randint(0, self.args.fill_h)
        j = random.randint(0, self.args.fill_w)
        imgA = imgA.crop(
            (j, i, j + self.args.fine_size, i + self.args.fine_size))
        imgB = imgB.crop(
            (j, i, j + self.args.fine_size, i + self.args.fine_size))
        return imgA, imgB

    def normalize(self, img, nc=3):
        if nc == 3:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5)
            ])
        img = transform(img)
        return img

    def process_img(self, img, i, j, padding, mode='RGB'):
        img = padding(img)
        img = img.crop((j, i, j + self.args.fineSize, i + self.args.fineSize))
        img = transforms.ToTensor()(img)
        # if self.isTrain == 0:
        if mode == 'RGB':
            img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        else:
            img = transforms.Normalize(0.5, 0.5)(img)
        return img

    def process_parsing(self, mat_path, i, j, w, h):
        facelabel = sio.loadmat(mat_path)
        parsing = facelabel['res_label']
        parsing = np.transpose(parsing, (2, 1, 0))
        parsing = np.minimum(parsing, 1)
        parsing = np.maximum(parsing, 0)
        parsing = np.pad(parsing, ((0, 0), (w, w), (h, h)), 'edge')
        parsing = parsing[:, i:i + self.args.fineSize,
                          j:j + self.args.fineSize]
        # parsing = np.where(parsing > 0.5, 1, 0)  # 二值化parsing

        parsing = parsing.astype('float32')
        torch_parsing = torch.from_numpy(parsing)

        return torch_parsing


if __name__ == '__main__':
    pass
