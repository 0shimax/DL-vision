import os
from easydict import EasyDict as edict
from math import sqrt
import platform
import inflection


#
# immortal params
local_os_name = 'Darwin'
data_root_path = './data' if platform.system()==local_os_name else '/data'

# mult_dir's key is module name
mult_dir = {'n_i_n': 32,
            'squeeze_net': 16,
            'cifar10': 4}

augmentation_params = {
                       'scale':[0.5, 0.75, 1.25],
                       'lr_shift':[-64, -32, -16, 16, 32, 64],
                       'ud_shift':[-64, -32, -16, 16, 32, 64],
                       'rotation_angle': list(range(5,360,5))
                      }

net_dir = {net_name:{'module_name':net_name, \
                    'class_name':inflection.camelize(net_name)} \
                        for net_name, _ in mult_dir.items()}

image_normalize_types_dir = {'ZCA': {'method':'zca_whitening', 'opts':{'eps':1e-5}},
                             'LCN': {'method':'local_contrast_normalization', 'opts':None},
                             'GCN': {'method':'global_contrast_normalization', 'opts':None}
                            }

# hand making params
debug_mode = False
converse_gray = False
gpu = -1 if platform.system()==local_os_name else 1
use_net = 'cifar10'
n_class = 10  # number of class is 2, if you use ne_class classifier.
crop_size = 224
normalize_type = 'LCN'
output_path = os.path.join(data_root_path+'/results', use_net)
im_norm_type = image_normalize_types_dir[normalize_type]
model_module = net_dir[use_net]
experiment_criteria = ''
initial_model = os.path.join(data_root_path+'/results'+'/'+use_net+experiment_criteria, 'model_iter_3868')
resume = os.path.join(data_root_path+'/results'+'/'+use_net+experiment_criteria, 'snapshot_iter_xxx')
aug_flags = {'do_scale':False, 'do_flip':False,
             'change_britghtness':False, 'change_contrast':False,
             'do_shift':False, 'do_rotate':False}


trainig_params = { \
        'lr': 1e-5,
        'batch_size': 256,
        'weight_decay': 0.0005,
        'epoch': 100,
        'decay_factor': 0.90,  # as lr time decay
        'decay_epoch': 50,
        'snapshot_epoch': 3,
        'report_epoch': 1,
    }


# a body of args
train_args = \
    {
        'train': True,
        'debug_mode': debug_mode,
        'gpu': gpu,
        'n_class': n_class,
        'in_ch': 1 if converse_gray else 3,
        'image_pointer_path': data_root_path+'/cifar-10-batches-py/data_batch_1',
        'output_path': output_path,
        'initial_model': initial_model,
        'resume': resume,
        'im_norm_type': im_norm_type,
        'archtecture': model_module,
        'converse_gray': converse_gray,
        'do_resize': True,
        'crop_params': {'flag':False, 'size': crop_size},
        'multiple': mult_dir[use_net],  # total stride multiple
        'aug_params': {'do_augment':True,
                       'params': dict(augmentation_params, **aug_flags),
                      },
        'shuffle': True,  # data shuffle in SerialIterator
        'training_params': trainig_params
    }

test_args = \
    {
        'train': False,
        'debug_mode': debug_mode,
        'gpu': gpu,
        'n_class': n_class,
        'in_ch': 1 if converse_gray else 3,
        'image_pointer_path': data_root_path+'/cifar-10-batches-py/data_batch_1',  #test_batch
        'output_path': output_path,
        'initial_model': initial_model,
        'im_norm_type': im_norm_type,
        'archtecture': model_module,
        'converse_gray': converse_gray,
        'do_resize': True,
        'crop_params': {'flag':False, 'size': crop_size},
        'multiple': mult_dir[use_net],  # total stride multiple
        'aug_params': {'do_augment':False},
        'multiple': mult_dir[use_net],  # total stride multiple
    }


def get_args(args_type='train'):
    if args_type=='train':
        return edict(train_args)
    else:
        return edict(test_args)
