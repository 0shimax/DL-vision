import sys, os
sys.path.append('./src/common')
sys.path.append('./src/common/image_processor')
sys.path.append('./src/common/model_preparator')
sys.path.append('./src/net')
sys.path.append('./experiment_settings')
from mini_batch_loader import DatasetPreProcessor
from visualization_settings import get_args
from model_loader import prepare_model

import chainer
from chainer import Variable, cuda
import numpy as np
import cv2
from itertools import product
from math import ceil, floor


class AttentionVisualizer(object):
    """
    see OBJECT DETECTORS EMERGE IN DEEP SCENE CNNs, Zhou+, '14
    https://arxiv.org/abs/1412.6856
    """
    def __init__(self, args):
        self.args = args
        self.xp = cuda.cupy if args.gpu>=0 else np

    def calculate_slice_idxs(self, size, x, y, h, w):
        patch_harf_slide = ceil(size/2)
        sl_strt_x = int(max(0, x - patch_harf_slide))
        sl_end_x  = int(min(h, x - patch_harf_slide + size))
        sl_strt_y = int(max(0, y - patch_harf_slide))
        sl_end_y = int(min(w, y - patch_harf_slide + size))
        return sl_strt_x, sl_end_x, sl_strt_y, sl_end_y

    def calculate_target_class(self, y, gt):
        if self.args.view_type=='gt':
            prob = y[0, gt]
            target_class = gt
        elif self.args.view_type=='infer':
            max_class_idx = int(self.xp.argmax(y[0, :]))
            prob = y[0, max_class_idx]
            target_class = max_class_idx
        return prob, target_class,

    def crop_patches(self, image):
        n_img, ch, h, w = image.shape

        num_occluded_img = \
            ((h - 1)//self.args.stride+1) * ((w - 1)//self.args.stride+1)
        patches = self.xp.zeros( \
            (num_occluded_img, ch, self.args.size, self.args.size), \
            dtype=np.float32)

        window_pos = []
        idx = 0

        for x, y in product(range(self.args.size//2, h, self.args.stride), \
                            range(self.args.size//2, w, self.args.stride)):
            _img = image.copy()
            sl_strt_x, sl_end_x, sl_strt_y, sl_end_y = \
                            self.calculate_slice_idxs(self.args.size, x, y, h, w)

            patch = image[:, :, sl_strt_x:sl_end_x, sl_strt_y:sl_end_y]
            if min(patch.shape[2:])<self.args.size:
                continue
            patches[idx] = patch
            window_pos.append([x, y])
            idx += 1
        window_pos = self.xp.array(window_pos, dtype=np.int32)
        return patches[:idx], window_pos

    def compute_one_batch_mask( \
            self, mask, patches, prob, target_class, w_pos, idx):

        n_img, ch, h, w = mask.shape
        x_batch = Variable(patches[idx:idx+self.args.patch_batchsize])
        self.args.net(x_batch, self.xp.array( \
                        [target_class]*len(x_batch.data), np.int32))
        y_batch = self.args.net.prob.data
        patches_prob = y_batch[:, target_class]

        # attention score
        diff = patches_prob - prob
        if self.args.gpu>=0:
            threshold = np.percentile(self.xp.asnumpy(diff), self.args.percentile)
        else:
            threshold = self.xp.percentile(diff, self.args.percentile)

        batch_w_pos = w_pos[idx:idx+self.args.patch_batchsize]
        crux_coordinate = self.xp.array([batch_w_pos[idx] for idx, flag in \
                        enumerate(diff > threshold) if flag], dtype=np.float32)

        for x, y in crux_coordinate:
            sl_strt_x, sl_end_x, sl_strt_y, sl_end_y = \
                    self.calculate_slice_idxs(self.args.size, x, y, h, w)
            mask[:, :, sl_strt_x:sl_end_x, sl_strt_y:sl_end_y] = 1.
        return mask

    def compute_attention_mask(self, image, gt):
        ch, h, w = image.shape
        image = image.reshape(1, ch, h, w).astype(np.float32)

        if self.args.gpu >= 0:
            image = cuda.to_gpu(image, device=self.args.gpu)

        x = Variable(image)
        self.args.net(x, self.xp.array([gt], np.int32))
        y = self.args.net.prob.data
        prob, target_class = self.calculate_target_class(y, gt)

        patches, w_pos = self.crop_patches(image)
        if self.args.gpu>=0:
            patches = cuda.to_gpu(patches, device=self.args.gpu)

        mask = self.xp.zeros_like(image)
        for idx in range(0, len(patches), self.args.patch_batchsize):
            mask = self.compute_one_batch_mask( \
                        mask, patches, prob, target_class, w_pos, idx)
        return mask, target_class

    def visualize_attention(self, raw_image, preprocessed_image, gt):
        h, w, _ = raw_image.shape

        mask, target_class = \
            self.compute_attention_mask(preprocessed_image, gt)
        mask = mask[0].transpose(1,2,0).astype(np.uint8)

        xp = cuda.get_array_module(mask)
        if xp!=np:
            mask = self.xp.asnumpy(mask)
        mask = cv2.resize(mask, (w, h))
        return raw_image * mask, target_class


if __name__=='__main__':
    output_path_base = 'visualized_attention_images'
    args = get_args('test')
    mini_batch_loader = DatasetPreProcessor(args)

    # patch config
    args.size = 16  # size>=40, for pyramid spacial pooling
    args.stride = 3
    args.converse_gray = False
    args.in_ch = 3
    args.percentile = 95  # percentile value
    args.patch_batchsize = 4096
    args.view_type = 'infer'  # select 'infer' or 'gt'.

    visualizer = AttentionVisualizer(args)

    _, model_eval = prepare_model(get_args('train'))
    args.net = model_eval

    for idx, (raw_image, label) in enumerate(mini_batch_loader.pairs):
        preprocessed_image, _ = mini_batch_loader.get_example(idx)
        attention_view, target_class = \
            visualizer.visualize_attention(raw_image, preprocessed_image, label)

        image_fname = 'image'+str(idx)+'_'+str(label)+'_'+str(target_class)+'.jpg'
        output_path = os.path.join(args.output_path, output_path_base)
        if not os.path.exists(output_path):
            print("create directory:", output_path)
            os.mkdir(output_path)
        output_path = os.path.join(output_path, image_fname)
        cv2.imwrite(output_path, attention_view)
