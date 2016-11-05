import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


class Fire(chainer.Chain):
    def __init__(self, in_size, s1, e1, e3):
        super().__init__(
            conv1=L.Convolution2D(in_size, s1, 1),
            conv2=L.Convolution2D(s1, e1, 1),
            conv3=L.Convolution2D(s1, e3, 3, pad=1),
        )

    def __call__(self, x):
        h = F.elu(self.conv1(x))
        h_1 = self.conv2(h)
        h_3 = self.conv3(h)
        h_out = F.concat([h_1, h_3], axis=1)
        return F.elu(h_out)


class SqueezeNet(chainer.Chain):
    def __init__(self, n_class, in_ch):
        super().__init__(
            conv1=L.Convolution2D(in_ch, 96, 7, stride=2, pad=3),
            fire2=Fire(96, 16, 64, 64),
            fire3=Fire(128, 16, 64, 64),
            fire4=Fire(128, 16, 128, 128),
            fire5=Fire(256, 32, 128, 128),
            fire6=Fire(256, 48, 192, 192),
            fire7=Fire(384, 48, 192, 192),
            fire8=Fire(384, 64, 256, 256),
            fire9=Fire(512, 64, 256, 256),
            conv10=L.Convolution2D(512, n_class, 2)
        )

        self.train = True
        self.n_class = n_class

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        x.volatile = not self.train

        h = F.elu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = self.fire2(h)
        # h = self.fire3(h)
        h = self.fire4(h)

        h = F.max_pooling_2d(h, 2, stride=2)

        # h = self.fire5(h)
        h = self.fire6(h)
        # h = self.fire7(h)
        h = self.fire8(h)

        h = F.max_pooling_2d(h, 2, stride=2)
        h = self.fire9(h)

        h = F.reshape(self.conv10(h), (len(x.data), self.n_class))

        self.prob = F.softmax(h)
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        chainer.report({'loss': self.loss, 'accuracy': self.accuracy}, self)
        return self.loss
