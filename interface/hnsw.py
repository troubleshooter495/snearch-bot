import numpy as np
import torch
from skimage import io
import torchvision.transforms as transforms
import hnswlib
import tqdm
import cv2


class Hnsw:
    def __init__(self, paths, model):
        self.model = model
        self.paths = paths
        self.encoded = self.encodepaths(paths)
        print("hnsw is loaded")

    def encodepaths(self, paths):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        encoded = []

        for p in tqdm.tqdm(paths):
            img = io.imread(p)
            if img.shape[-1] != 3:
                continue
            img = transform(cv2.resize(img, (102, 136)))
            img = img.view(1, 3, 102, 136)
            with torch.no_grad():
                encoded.append(torch.flatten(self.model.encode(img)))

        for i in range(len(encoded)):
            encoded[i] = encoded[i].numpy()

        return np.array(encoded)

    def knn(self, likes, n_top):
        encodlikes = self.encodepaths(likes)
        samples = np.concatenate((self.encoded, encodlikes))
        ppaths = np.concatenate((self.paths, likes))
        num_elements, dim = samples.shape

        data1 = samples[:num_elements // 2]
        data2 = samples[num_elements // 2:]

        p = hnswlib.Index(space='l2', dim=dim)
        p.init_index(max_elements=num_elements // 2, ef_construction=100, M=16)

        p.set_ef(10)
        p.set_num_threads(4)
        p.add_items(data1)

        index_path = 'first_half.bin'
        p.save_index(index_path)
        del p

        p = hnswlib.Index(space='l2', dim=dim)
        p.load_index("first_half.bin", max_elements=num_elements)

        p.add_items(data2)

        labels, distances = p.knn_query(encodlikes, k=n_top)

        best_paths = ['' for i in range(labels.shape[0] * n_top)]
        probs = np.empty(labels.shape[0] * n_top )
        j = 0
        for i in range(labels.shape[0]):
            for rec in range(n_top):
                best_paths[j] = ppaths[labels[i][rec]]
                probs[j] = distances[i][rec]
                j += 1
        for i in range(len(probs)):
            probs[i] = 1 / probs[i] if probs[i] else 0

        probs /= probs.sum()
        return best_paths, probs

# TESTING
#
# ddb = db.ServerDB("../data/base", lambda: random.randint(1, 100) > 13)
# m = interface.autoencoder.load_model("../data/model.pt")
# t = hnsw(ddb.getimgpaths(), m)
# likes = ['../data/base/Shoes/Sneakers and Athletic Shoes/adidas Originals/7183714.7968.jpg',
#          '../data/base/Shoes/Sneakers and Athletic Shoes/adidas Originals/7187046.742.jpg',
#          '../data/base/Shoes/Sneakers and Athletic Shoes/adidas Originals//7289976.56026.jpg']
#
# a,b = t.knn(likes, 3)
# print(a,sep='\n')
#
# print(np.random.choice(a,p=b))
