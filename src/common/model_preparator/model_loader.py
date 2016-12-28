import chainer
import importlib
import os, sys
from chainer import serializers


def prepare_model(args):
    model = getattr(
        importlib.import_module(args.archtecture.module_name),
                                args.archtecture.class_name)(args.n_class, args.in_ch)
    if os.path.exists(args.initial_model):
        print('Load model from', args.initial_model, file=sys.stderr)
        serializers.load_npz(args.initial_model, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
    model.train = True
    model.n_class = args.n_class
    m_eval = model.copy()
    m_eval.train = False

    return model, m_eval
