# download dataset from links shown below

import os
import collections
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

class NYUv2Loader(data.Dataset):
    """
    NYUv2 loader
    Download From (only 13 classes):
    test source: http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz
    train source: http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz
    test_labels source:
      https://github.com/ankurhanda/nyuv2-meta-data/raw/master/test_labels_13/nyuv2_test_class13.tgz
    train_labels source:
      https://github.com/ankurhanda/nyuv2-meta-data/raw/master/train_labels_13/nyuv2_train_class13.tgz
    """

    def __init__(
        self,
        root,
        split="training",
        is_transform=False,
        img_size=(480, 640),
        augmentations=None,
        img_norm=True,
        test_mode=False,
    ):
        self.root = root
        self.is_transform = is_transform
        self.n_classes = 14
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.cmap = self.color_map(normalized=False)

        split_map = {"training": "train", "val": "test"}
        self.split = split_map[split]

        for split in ["train", "test"]:
            file_list = recursive_glob(rootdir=self.root + split + "/", suffix="png")
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        img_number = img_path.split("_")[-1][:4]
        lbl_path = os.path.join(
            self.root, self.split + "_annot", "new_nyu_class13_" + img_number + ".png"
        )

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.uint8)

        if not (len(img.shape) == 3 and len(lbl.shape) == 2):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)
        assert np.all(classes == np.unique(lbl))

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def color_map(self, N=256, normalized=False):
        """
        Return Color Map in PASCAL VOC format
        """

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        dtype = "float32" if normalized else "uint8"
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255.0 if normalized else cmap
        return cmap

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.cmap[l, 0]
            g[temp == l] = self.cmap[l, 1]
            b[temp == l] = self.cmap[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb
