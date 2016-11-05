import numpy as np
import pickle
import cv2

# CIFAR-10
# size: 32x32
# number of class: 10(
# number of image: 6000/class

def unpickle(f):
    with open(f, 'rb') as fo:
        d = pickle.load(fo, encoding='latin-1')
    return d

base = './data/cifar-10-batches-py'
label_names = unpickle(base+"/batches.meta")["label_names"]
d = unpickle(base+"/data_batch_1")
data = d["data"]
labels = np.array(d["labels"])
nsamples = len(data)

print(label_names)

nclasses = 10
pos = 1
for i in range(nclasses):
    # クラスiの画像のインデックスリストを取得
    targets = np.where(labels == i)[0]
    np.random.shuffle(targets)
    img = data[targets[0]].reshape((3,32,32)).transpose(1,2,0)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
