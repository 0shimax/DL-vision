import cv2
import numpy as np
from sklearn.decomposition import PCA
from scipy import linalg
from sklearn.utils.extmath import svd_flip
from sklearn.utils.extmath import fast_dot
from sklearn.utils import check_array
from math import sqrt


class ImageNormalizer(object):
    def __init__(self):
        pass

    def zca_whitening(self, image, eps):
        """
        N = 1
        X = image[:,:].reshape((N, -1)).astype(np.float64)

        X = check_array(X, dtype=[np.float64], ensure_2d=True, copy=True)

        # Center data
        self.mean_ = np.mean(X, axis=0)
        print(X.shape)
        X -= self.mean_

        U, S, V = linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, V = svd_flip(U, V)

        zca_matrix = U.dot(np.diag(1.0/np.sqrt(np.diag(S) + 1))).dot(U.T) #ZCA Whitening matrix

        return fast_dot(zca_matrix, X).reshape(image.shape)   #Data whitening
        """
        image = self.local_contrast_normalization(image)
        N = 1
        X = image.reshape((N, -1))

        pca = PCA(whiten=True, svd_solver='full', n_components=X.shape[-1])
        transformed = pca.fit_transform(X)  # return U
        pca.whiten = False
        # zca = fast_dot(transformed, pca.components_+eps) + pca.mean_
        zca = pca.inverse_transform(transformed)
        return zca.reshape(image.shape)


    def local_contrast_normalization(self, image, args=None):
        def __histogram_equalize(img):
            h, w = image.shape[:2]
            img = img.astype(np.uint8)
            cn_channels = tuple(cv2.equalizeHist(d_ch) for d_ch in cv2.split(img))

            if len(cn_channels)==3:
                return cv2.merge(cn_channels)
            elif len(cn_channels)==1:
                return cn_channels[0].reshape((h, w, 1))
        return __histogram_equalize(image)

    def one_pixel_normalize(self, orig_img, pixel_means, max_size=1000, scale=600):
        img = orig_img.astype(np.float32, copy=True)
        img -= pixel_means
        img /= 255
        im_size_min = np.min(img.shape[0:2])
        im_size_max = np.max(img.shape[0:2])
        im_scale = float(scale) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

        return img.transpose([2, 0, 1]).astype(np.float32), im_scale

    def global_contrast_normalization(self, image, args=None):
        mean = np.mean(image)
        var = np.var(image)
        a=(image-mean)/float(sqrt(var))
        print(a.shape)
        return (image-mean)/float(sqrt(var))
