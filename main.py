from experiments import TMD_MIL, Liang, Tang, Ito
import warnings
import os
import tensorflow as tf

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
        os.environ["CUDA_VISIBLE_DEVICES"]="0"

EXPERIMENT = 'TMD_MIL'


def main(experiment):
    archive = os.path.join("archive", experiment)

    if experiment == 'TMD_MIL':
        TMD_MIL(archive)

    elif experiment == 'Liang':
        Liang(archive)

    elif experiment == 'Tang':
        Tang(archive)

    elif experiment == 'Ito':
        Ito(archive)


if __name__ == '__main__':
    main(EXPERIMENT)