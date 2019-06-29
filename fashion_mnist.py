from torch.utils.data import Dataset
import os
import numpy as np
import gzip
from urllib import request



class FashionMnistDataset(Dataset):
    def __init__(self, root, train = True, download=True):

        super(FashionMnistDataset, self).__init__()

        self.root = root
        self.train = train
        self.download = download

        self.train_images_link = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz'
        self.train_labels_link = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz'

        self.test_images_link = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz'
        self.test_labels_link = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'

        if self.train:
            self.images_link = self.train_images_link
            self.labels_link = self.train_labels_link
        else:
            self.images_link = self.test_images_link
            self.labels_link = self.test_labels_link

        images_filename = self.images_link.split('/')[-1]
        self.images_path = os.path.join(self.root, images_filename)

        labels_filename = self.labels_link.split('/')[-1]
        self.labels_path = os.path.join(self.root, labels_filename)

        if download:
            self.do_download()

        self.load_data()

    def load_data(self):

        if not os.path.exists(self.images_path):
            print(f"Images file in {self.images_path} does not exist")
            exit()

        if not os.path.exists(self.labels_path):
            print(f"Images file in {self.labels_path} does not exist")
            exit()

        with gzip.open(self.labels_path, 'rb') as lbpath:
            self.labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(self.images_path, 'rb') as imgpath:
            self.images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(self.labels), 784)

    def do_download(self):

        if not os.path.exists(self.root):
            os.mkdir(self.root)

        if not os.path.exists(self.images_path):
            response_images = request.urlopen(self.images_link)
            with open(self.images_path, "wb") as f:
                f.write(response_images.read())

        if not os.path.exists(self.labels_path):
            response_labels = request.urlopen(self.labels_link)
            with open(self.labels_path, "wb") as f:
                f.write(response_labels.read())

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.labels[idx], (self.images[idx].reshape(28,28) / 255)
