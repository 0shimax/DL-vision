import sys, os
sys.path.append('./src/common')
sys.path.append('./src/common/image_processor')
sys.path.append('./src/common/model_preparator')
sys.path.append('./src/net')
sys.path.append('./experiment_settings')
from mini_batch_loader import DatasetPreProcessor
from visualization import get_args
from model_loader import prepare_model

import chainer
import chainer.functions as F
from chainer import serializers
from chainer import cuda, optimizers, Variable
from chainer import training
from chainer.training import extensions
from chainer import Reporter, report, report_scope

import cv2
import importlib
import numpy as np


def prepare_optimizer(model, args):
    optimizer = chainer.optimizers.RMSpropGraves(args.training_params.lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.training_params.weight_decay))
    return optimizer


def prepare_dataset():
    train_args = get_args('train')
    # load dataset
    train_mini_batch_loader = DatasetPreProcessor(train_args)
    test_mini_batch_loader = DatasetPreProcessor(get_args('test'))
    print("---set mini_batch----------")
    train_it = chainer.iterators.SerialIterator( \
                            train_mini_batch_loader, \
                            train_args.training_params.batch_size, \
                            shuffle=train_args.shuffle)
    val_it = chainer.iterators.SerialIterator( \
                            test_mini_batch_loader, \
                            1, repeat=False, shuffle=False)
    return train_it, val_it, train_mini_batch_loader.__len__()


def main(args):
    # load model
    model, model_for_eval = prepare_model(args)
    print("---set model----------")

    # Setup optimizer
    optimizer = prepare_optimizer(model, args)
    print("---set optimzer----------")

    # load data
    train_it, val_it, train_data_length = prepare_dataset()
    print("---set data----------")

    updater = training.StandardUpdater(train_it, optimizer, device=args.gpu)
    print("---set updater----------")


    val_interval = args.training_params.report_epoch, 'epoch'
    val_snapshot_interval = args.training_params.snapshot_epoch, 'epoch'
    log_interval = args.training_params.report_epoch, 'epoch'

    trainer = training.Trainer( \
        updater, (args.training_params.epoch, 'epoch'), out=args.output_path)
    trainer.extend( \
        extensions.Evaluator(val_it, model_for_eval, device=args.gpu), \
        trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_snapshot_interval)
    trainer.extend(extensions.snapshot_object( \
        model, 'model_iter_{.updater.iteration}'), \
        trigger=val_snapshot_interval)
    trainer.extend(extensions.ExponentialShift( \
        'lr', args.training_params.decay_factor), \
        trigger=(args.training_params.decay_epoch, 'epoch'))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([ \
        'epoch', 'iteration', 'main/loss', 'validation/main/loss', \
        'main/accuracy', 'validation/main/accuracy', \
        ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=1))
    print("---set trainer----------")

    if os.path.exists(args.resume):
        print('resume trainer:{}'.format(args.resume))
        # Resume from a snapshot
        serializers.load_npz(args.resume, trainer)
    trainer.run()


if __name__ == '__main__':
    print("-------traing")
    args = get_args('train')
    main(args)
