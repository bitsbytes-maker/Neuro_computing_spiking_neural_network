import argparse
import os
from snnbuilder.models.mini import Mini_Network
from snnbuilder.models.autoencoders import Autoencoder_Mnist
from snnbuilder.models.catsvsdogs import CNN_CatsVsDogs
from snnbuilder.models.alexnet import Alexnet_Mnist224, AlexNet_CatsVsDogs
from snnbuilder.models.mnist import CNN_Mnist, LeNet_Mnist, LeNet_padded_Mnist, Zambrano_Mnist, MultiLayerPerceptron_Mnist
from snnbuilder.models.cifar10 import LeNet_Cifar10, Rueckauer_Cifar10
from snnbuilder.models.cifar100 import LeNet_Cifar100
from snnbuilder.models.squeezenet import SqueezeNet_CatsVsDogs, SqueezeNet_Fruits360
from snnbuilder.models.fruits import CNN_Fruits360


if __name__ == "__main__":
    # output_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dev_outputs'))
    output_path = None

    # get command line args
    models = ['cnn_mnist', 'cifar_lenet', 'mnist_zambrano', 'mini_network', 'mnist_lenet', 'mnist_lenet_padded',
              'rueck_cifar', 'alex_mnist', 'cnn_catdog', 'mnist_auto', 'mlp', 'squeeze_mnist', 'squeeze_catdog']

    model_options = 'model options: ' + ' '.join(models)
    parser = argparse.ArgumentParser(description='builds, trains, converts, and simulates neural networks')
    parser.add_argument('model', type=str, nargs='?', help=model_options, default=None)
    parser.add_argument('--path', type=str, nargs='?', help='relative output path', default=None)
    parser.add_argument('-t', '--train', help='enable train', action='store_true')
    parser.add_argument('-p', '--parse', help='enable parse', action='store_true')
    parser.add_argument('-s', '--sim', help='enable sim', action='store_true')
    parser.add_argument('-g', '--graph', help='enable graph', action='store_true')
    parser.add_argument('-e', '--epochs', type=int, nargs=1, help='epochs to train', default=None)
    parser.add_argument('-n', '--samples', type=int, nargs=1, help='samples to sim', default=None)
    parser.add_argument('-d', '--duration', type=int, nargs=1, help='sim duration per sample', default=None)
    args = parser.parse_args()

    if args.path is not None:
        output_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), args.path))

    kwargs = {}
    if args.epochs is not None:
        kwargs['epochs'] = args.epochs[0]
    if args.samples is not None:
        kwargs['samples'] = args.samples[0]
    if args.duration is not None:
        kwargs['duration'] = args.duration[0]

    if args.model == 'cnn_mnist':
        x = CNN_Mnist(output_path, **kwargs)
    elif args.model == 'cifar_lenet':
        x = LeNet_Cifar10(output_path, **kwargs)
    elif args.model == 'cifar_lenet100':
        x = LeNet_Cifar100(output_path, **kwargs)
    elif args.model == 'mnist_zambrano':
        x = Zambrano_Mnist(output_path, **kwargs)
    elif args.model == 'mini_network':
        x = Mini_Network(output_path, **kwargs)
    elif args.model == 'mnist_lenet':
        x = LeNet_Mnist(output_path, **kwargs)
    elif args.model == 'mnist_lenet_padded':
        x = LeNet_padded_Mnist(output_path, **kwargs)
    elif args.model == 'rueck_cifar':
        x = Rueckauer_Cifar10(output_path, **kwargs)
    elif args.model == 'alex_mnist':
        x = Alexnet_Mnist224(output_path, **kwargs)
    elif args.model == 'mnist_auto':
        x = Autoencoder_Mnist(output_path, **kwargs)
    elif args.model == 'mlp':
        x = MultiLayerPerceptron_Mnist(output_path, **kwargs)
    elif args.model == 'cnn_catdog':
        x = CNN_CatsVsDogs(output_path, **kwargs)
    elif args.model == 'squeeze_catdog':
        x = SqueezeNet_CatsVsDogs(output_path, **kwargs)
    elif args.model == 'alex_catdog':
        x = AlexNet_CatsVsDogs(output_path, **kwargs)
    elif args.model == 'cnn_fruits':
        x = CNN_Fruits360(output_path, **kwargs)
    elif args.model == 'squeeze_fruits':
        x = SqueezeNet_Fruits360(output_path, **kwargs)
    else:
        print('no model entered')
        exit()

    # no enables were sent, runs all three

    if not args.train and not args.parse and not args.sim and not args.graph:
        x.train()
        x.parse()
        x.sim()
        x.graph()
    else:
        if args.train:
            x.train()
        if args.parse:
            x.parse()
        if args.sim:
            x.sim()
        if args.graph:
            x.graph()

  




########## The model ###################

#mnist_lenet_padded
#cnn_catdog
#squeeze_catdog
#alex_catdog
#cnn_fruits
#squeeze_fruits
