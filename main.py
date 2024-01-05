from experiments import TMD_MIL, Liang, Tang
import warnings
import os
import tensorflow as tf

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

EXPERIMENT = 'Tang'


def main(experiment):
    archive = os.path.join("archive", experiment)

    if experiment == 'TMD_MIL':
        TMD_MIL(archive)

    elif experiment == 'Liang':
        Liang(archive)

    elif experiment == 'Tang':
        Tang(archive)


if __name__ == '__main__':
    main(EXPERIMENT)