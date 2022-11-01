from snnbuilder import bin
import snnbuilder.bin.utils

import tensorflow as tf

from snnbuilder.bin.pipeline import ModelPipeline
from snnbuilder.bin.utils import summarize_diagnostics

# GPU running out of memory
# https://stackoverflow.com/questions/53698035/failed-to-get-convolution-algorithm-this-is-probably-because-cudnn-failed-to-in
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


if __name__ == "__main__":
    output_path = '/home/patrick/Documents/snntoolbox_outputs'
    data_path = '/media/patrick/HDD/VLSI_VOC/neuro_comp/dev_outputs/datasets'
    from tensorflow.keras.optimizers import Adam, SGD


    from snnbuilder.models.alexnet import AlexNet_CatsVsDogs
    from snnbuilder.models.cifar10 import Rueckauer_Cifar10
    opt = SGD(momentum=0.9, nesterov=True, learning_rate=0.001)
    m = Rueckauer_Cifar10(output_path, optimizer=opt, epochs=150)
    # m.train()
    m.parse()
    m.sim()
    m.graph()

    exit()

    from snnbuilder.models.fruits import CNN_Fruits360
    # opt = Adam(learning_rate=0.0001)
    m = CNN_Fruits360(output_path, data_path + '/fruits-360')
    # m.train()
    m.parse()
    m.sim()
    m.graph()

    exit()


