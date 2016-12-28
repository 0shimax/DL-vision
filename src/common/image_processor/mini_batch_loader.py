import os, sys
sys.path.append('./src/common/image_processor')
import chainer
import cv2
import numpy as np
from contextlib import ExitStack
import numbers
from image_normalizer import ImageNormalizer


class DatasetPreProcessor(chainer.dataset.DatasetMixin):
    def __init__(self, args):
        """args type is EasyDict class 
        """
        self.args = args
        self.gray = args.converse_gray
        self.image_normalizer = ImageNormalizer()
        self.pairs = self.load_pairs()
        self.counter = 0
        self.image_size_in_batch = [None, None]  # height, width

    def __len__(self):
        return len(self.pairs)

    def __init_batch_counter(self):
        if self.args.train and self.counter==self.args.training_params.batch_size:
            self.counter = 0
            self.image_size_in_batch = [None, None]

    def __set_image_size_in_batch(self, image):
        if self.counter==1:
            resized_h, resized_w = image.shape[:2]
            self.image_size_in_batch = [resized_h, resized_w]

    def get_example(self, index):
        self.counter += 1
        if self.args.debug_mode:
            if self.counter>15:
                assert False, 'stop test'

        image, label = self.pairs[index]

        # gray transform if converse_gray is True
        image = self.color_trancefer(image)
        h, w, ch = image.shape

        if image is None:
            raise RuntimeError("invalid image: {}".format(path))

        # resizing image
        if self.args.do_resize:
            if self.counter>1:
                # augmentas is ordered w,h in resize method of openCV
                scale = self.image_size_in_batch[1]/w, self.image_size_in_batch[0]/h
                image= self.resize_image(image, scale)
            else:
                image= self.resize_image(image)
        elif self.args.crop_params.flag:
            image = self.crop_image(image)
        # print('resizing done-------------')

        # augmentat image
        if self.args.aug_params.do_augment:
            image = self.augment_image(image)

        # store image size
        # because dimension must be equeal per batch
        self.__set_image_size_in_batch(image)
        # print('augmentation done-------------')

        # image normalize
        image = getattr(self.image_normalizer, \
            self.args.im_norm_type.method)(image, self.args.im_norm_type.opts)
        # print('normalization done-------------')

        if self.args.debug_mode:
            cv2.imshow('image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(image.shape)
            print('label:', label)

        # transpose for chainer
        image = image.transpose(2, 0, 1)
        # initialize batch counter
        self.__init_batch_counter()

        return image.astype(np.float32), np.array(label, dtype=np.int32)

    def load_pairs(self):
        image_label_pairs = []
        for image, label in self.image_label_pair_generator():
            image_label_pairs.append((image, label))
        return image_label_pairs

    def image_label_pair_generator(self):
        import pickle
        def __unpickle(file_name):
            with open(file_name, 'rb') as f_cifar10:
                pairs = pickle.load(f_cifar10, encoding='latin-1')
            return pairs

        pair = __unpickle(self.args.image_pointer_path)
        for image, label in zip(pair["data"], pair['labels']):
            yield image.reshape((3,32,32)).transpose(1,2,0), label

    def color_trancefer(self, image):
        h, w, _ = image.shape
        if self.args.converse_gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape((h,w,1))
        # elif self.yuv:
        #     image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        else:
            image = image.astype(np.float32)
        return image

    def augment_image(self, image):
        if self.args.aug_params.params.do_scale and self.counter==1:
            image = self.scaling(image)

        if self.args.aug_params.params.do_flip:
            image = self.flip(image)

        if self.args.aug_params.params.change_britghtness:
            image = self.random_brightness(image)

        if self.args.aug_params.params.change_contrast:
            image = self.random_contrast(image)

        if self.args.aug_params.params.do_rotate:
            image = self.rotate_image(image)

        if self.args.aug_params.params.do_shift:
            image = self.shift_image(image)

        return image

    def resize_image(self, image, scale=None):
        xh, xw = image.shape[:2]

        if scale is None:
            # if scale is not difinded, calculate scale as closest multiple number.
            scale = float(xh)/(xh//self.args.multiple)/self.args.multiple
                    #float(xw)/(xw//self.args.multiple)/self.args.multiple,
            scale = scale, scale
        elif isinstance(scale, numbers.Number):
            scale = scale, scale
        elif isinstance(scale, tuple) and len(scale)>2:
            raise InvalidArgumentError

        new_sz = (int(xw*scale[0])+1, int(xh*scale[1])+1)  # specification of opencv, argments is recepted (w, h)
        # image = image[::inv_resize_scale,::inv_resize_scale]
        image = cv2.resize(image, new_sz)

        xh, xw = image.shape[:2]
        m0, m1 = xh % self.args.multiple, xw % self.args.multiple
        d0, d1 = np.random.randint(m0+1), np.random.randint(m1+1)
        image = image[d0:(image.shape[0] - m0 + d0), d1:(image.shape[1] - m1 + d1)]

        if len(image.shape)==2:
            return image.reshape((image.shape[0], image.shape[1], 1))
        else:
            return image

    def flip(self, image):
        do_flip_xy = np.random.randint(0, 2)
        do_flip_x = np.random.randint(0, 2)
        do_flip_y = np.random.randint(0, 2)

        if do_flip_xy: # Transpose X and Y axis
            image = image[::-1, ::-1, :]
        elif do_flip_x: # Flip along Y-axis
            image = image[::-1, :, :]
        elif do_flip_y: # Flip along X-axis
            image = image[:, ::-1, :]
        return image

    def scaling(self, image):
        do_scale = np.random.randint(0, 2)
        if do_scale:
            scale = self.args.aug_params.params.scale[ \
                np.random.randint(0,len(self.args.aug_params.params.scale))]
            return self.resize_image(image, scale)
        else:
            return image

    def random_brightness(self, image, max_delta=63, seed=None):
        brightness_flag = np.random.randint(0, 2)
        if brightness_flag:
            delta = np.random.uniform(-max_delta, max_delta)
            return image + delta
        else:
            return image

    def random_contrast(self, image, lower=0.2, upper=1.8, seed=None):
        contrast_flag = np.random.randint(0, 2)
        if contrast_flag:
            factor = np.random.uniform(-lower, upper)
            im_mean = image.mean(axis=2)
            return ((image.transpose(2, 0, 1) - im_mean)*factor + im_mean).transpose(1,2,0).astype(np.uint8)
        else:
            return image

    def shift_image(self, image):
        do_shift_xy = np.random.randint(0, 2)
        do_shift_x = np.random.randint(0, 2)
        do_shift_y = np.random.randint(0, 2)

        if do_shift_xy:
            lr_shift = self.args.aug_params.params.lr_shift[ \
                np.random.randint(0,len(self.args.aug_params.params.lr_shift))]
            ud_shift = self.args.aug_params.params.ud_shift[ \
                np.random.randint(0,len(self.args.aug_params.params.ud_shift))]
        elif do_shift_y:
            lr_shift = 0
            ud_shift = self.args.aug_params.params.ud_shift[ \
                np.random.randint(0,len(self.args.aug_params.params.ud_shift))]
        elif do_shift_x:
            lr_shift = self.args.aug_params.params.lr_shift[ \
                np.random.randint(0,len(self.args.aug_params.params.lr_shift))]
            ud_shift = 0

        if do_shift_xy or do_shift_y or do_shift_y:
            h, w, ch = image.shape
            affine_matrix = np.float32([[1,0,lr_shift],[0,1,ud_shift]])  # 横、縦
            image = cv2.warpAffine(image, affine_matrix, (w,h))
            return self.__reshpe_channel(image, (h,w,ch))
        else:
            return image

    def rotate_image(self, image):
        do_rotate = np.random.randint(0, 2)
        if do_rotate:
            h, w, ch = image.shape
            rotation_angle = self.args.aug_params.params.rotation_angle[ \
                np.random.randint(0,len(self.args.aug_params.params.rotation_angle))]
            affine_matrix = cv2.getRotationMatrix2D((h/2, w/2), rotation_angle, 1)

            image = cv2.warpAffine(image, affine_matrix, (w,h))
            return self.__reshpe_channel(image, (h,w,ch))
        else:
            return image

    def crop_image(self, image):
        h, w, ch = image.shape
        top = int((w-self.args.crop_params.size)/2)
        left = int((h-self.args.crop_params.size)/2)
        return image[left:left+self.args.crop_params.size,top:top+self.args.crop_params.size,:]

    def __reshpe_channel(self, image, im_shape ):
        if len(image.shape)==2:
            return image.reshape(im_shape)
        else:
            return image
