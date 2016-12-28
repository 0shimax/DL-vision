import chainer
import chainer.functions as F
import chainer.links as L


class Cifar10(chainer.Chain):

    def __init__(self, n_class, in_ch):
        super().__init__(
            conv1=L.Convolution2D(in_ch, 32, 5, pad=2),
            conv2=L.Convolution2D(32, 32, 5, pad=2),
            conv3=L.Convolution2D(32, 64, 5, pad=2),
            fc4=F.Linear(1344, 4096),
            fc5=F.Linear(4096, n_class),
        )
        self.train = True
        self.n_class = n_class

    def __call__(self, x, t):
        x.volatile = True

        h = F.max_pooling_2d(F.elu(self.conv1(x)), 3, stride=2)
        h = F.max_pooling_2d(F.elu(self.conv2(h)), 3, stride=2)
        h = F.elu(self.conv3(h))

        h.volatile = False
        h = F.spatial_pyramid_pooling_2d(h, 3, F.MaxPooling2D)
        h = F.dropout(F.elu(self.fc4(h)), ratio=0.5, train=self.train)
        h = self.fc5(h)

        self.prob = F.softmax(h)
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        chainer.report({'loss': self.loss, 'accuracy': self.accuracy}, self)

        return self.loss
