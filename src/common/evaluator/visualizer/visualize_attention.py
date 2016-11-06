import sys, os
sys.path.append('./src/common')
sys.path.append('./src/common/image_processor')
sys.path.append('./src/common/model_preparator')
sys.path.append('./src/net')
sys.path.append('./experiment_settings')
from mini_batch_loader import DatasetPreprocessor
from visualization import get_args
from model_loader import prepare_model

import chainer
from chainer import Variable, cuda
import numpy as np
import cv2
from itertools import product
from math import ceil, floor


"""
see OBJECT DETECTORS EMERGE IN DEEP SCENE CNNs, Zhou+, '15
https://arxiv.org/abs/1412.6856
"""


def calculate_slice_idxs(size, x, y, h, w):
    patch_harf_slide = ceil(size/2)
    sl_strt_x = int(max(0, x - patch_harf_slide))
    sl_end_x  = int(min(h, x - patch_harf_slide + size))
    sl_strt_y = int(max(0, y - patch_harf_slide))
    sl_end_y = int(min(w, y - patch_harf_slide + size))
    return sl_strt_x, sl_end_x, sl_strt_y, sl_end_y


def calculate_target_class(args, y, gt, xp):
    if args.view_type=='gt':
        prob = y[0, gt]
        target_class = gt
    elif args.view_type=='infer':
        max_class_idx = int(xp.argmax(y[0, :]))
        prob = y[0, max_class_idx]
        target_class = max_class_idx
    return prob, target_class,


def crop_patches(image, size, stride):
    '''
    occluded images is so time consuming.
    create image patches instead of occluded images.
    thus, score need to be reversed.
    do not forget.
    '''
    xp = cuda.get_array_module(image)
    n_img, ch, h, w = image.shape

    num_occluded_img = ((h - 1)//stride+1) * ((w - 1)//stride+1)
    patches = xp.zeros((num_occluded_img, ch, size, size), dtype=np.float32)

    window_pos = []
    idx = 0

    for x, y in product(range(size//2, h, stride), range(size//2, w, stride)):
        _img = image.copy()
        sl_strt_x, sl_end_x, sl_strt_y, sl_end_y = \
                        calculate_slice_idxs(size, x, y, h, w)

        patch = image[:, :, sl_strt_x:sl_end_x, sl_strt_y:sl_end_y]
        if min(patch.shape[2:])<size:
            continue
        patches[idx] = patch
        window_pos.append([x, y])
        idx += 1
    window_pos = xp.array(window_pos, dtype=np.int32)

    return patches[:idx], window_pos


def compute_attention_mask(args, image, gt, net, percentile, batchsize=256):
    '''
    create mask without attention.

    image     : (3, height, width)
    gt        : integer
    percentile: degree of interest
    '''

    ch, h, w = image.shape
    image = image.reshape(1, ch, h, w).astype(np.float32)

    if args.gpu >= 0:
        image = cuda.to_gpu(image, device=args.gpu)

    xp = cuda.get_array_module(image)

    x = Variable(image)
    net(x, xp.array([gt], np.int32))
    y = net.prob.data
    prob, target_class = calculate_target_class(args, y, gt, xp)

    patches, w_pos = crop_patches(image, size=args.size, stride=args.stride)
    if args.gpu>=0:
        patches = cuda.to_gpu(patches, device=args.gpu)

    mask = xp.zeros_like(image)

    for i in range(0, len(patches), batchsize):
        x_batch = Variable(patches[i:i+batchsize])
        net(x_batch, xp.array([target_class]*len(x_batch.data), np.int32))
        y_batch = net.prob.data
        patches_prob = y_batch[:, target_class]

        # attention score
        diff = patches_prob - prob
        if xp==np:
            threshold = xp.percentile(diff, percentile)
        else:
            threshold = np.percentile(xp.asnumpy(diff), percentile)

        batch_w_pos = w_pos[i:i+batchsize]
        crux_coordinate = xp.array([batch_w_pos[idx] for idx, flag in \
                        enumerate(diff > threshold) if flag], dtype=np.float32)

        for x, y in crux_coordinate:
            sl_strt_x, sl_end_x, sl_strt_y, sl_end_y = \
                    calculate_slice_idxs(args.size, x, y, h, w)
            mask[:, :, sl_strt_x:sl_end_x, sl_strt_y:sl_end_y] = 1.

    return mask, target_class


def visualize_attention(args, path, gt, image_loader, index, net, percentile):
    '''
    gt        : integer
    percentile: degree of interest
    '''
    raw_image = cv2.imread(path)
    image, _ = image_loader.get_example(index)
    h, w, _ = raw_image.shape

    mask, target_class = \
        compute_attention_mask( \
            args, image, gt, net, percentile, args.patch_batchsize)
    mask = mask[0].transpose(1,2,0).astype(np.uint8)

    xp = cuda.get_array_module(mask)
    if xp!=np:
        mask = xp.asnumpy(mask)
    mask = cv2.resize(mask, (w, h))

    return raw_image * mask, target_class


def run_visualize(args, image_path, image_loader, index, label, \
                    output_base_path, model_eval):
    rf, target_class = visualize_attention(args, image_path, label, \
                        image_loader, index, model_eval, percentile=args.percentile)
    image_fname = os.path.basename(image_path)
    name, extension = image_fname.split('.')
    image_fname = name+'_'+str(label)+'_'+str(target_class)+'.'+extension
    output_path = os.path.join(output_base_path, image_fname)
    cv2.imwrite(output_path, rf)


if __name__=='__main__':
    args = get_args('test')
    postfix_output_path = 'visualized_attention_images'
    output_base_path = os.path.join(args.output_path, postfix_output_path)
    if not os.path.exists(output_base_path):
        os.mkdir(output_base_path)

    mini_batch_loader = DatasetPreprocessor(args)

    # patch config
    args.size = 8*4  # size>=40, for pyramid spacial pooling
    args.stride = 3
    args.converse_gray = False
    args.in_ch = 3
    args.percentile = 95  # percentile value
    args.patch_batchsize = 4096
    args.view_type = 'infer'  # select 'infer' or 'gt'.

    labels = list('ABCDEFGHIJKLMNOPQRS')
    label2clsval = {l:i for i,l in enumerate(labels)}

    _, model_eval = prepare_model(get_args('train'))
    for idx, (image_path, label) in enumerate(mini_batch_loader.pairs):
        print('computing:',os.path.basename(image_path))

        if isinstance(label, str):
            label = label2clsval[label]
        run_visualize(args, image_path, mini_batch_loader, idx, label, \
                        output_base_path, model_eval)
