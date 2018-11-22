from common import filesystem, utils
from common.logger import logger
from machine.machine_teacher import base_category_training
from machine.training.lib.params import *
import os

logger.initialize()


def common_training(model_name, model_version,
                    image_size, data_root,
                    save_root="feedback",
                    batch_size=32, epoch=80,
                    epoch_factor=1):
    logger.info("Start Test Training")

    _Train_Common_Params = Train_Common_Params()
    _Train_Common_Params.i_Model_Selection_Params.create(model_name, model_version)
    __train_id = _Train_Common_Params.i_Model_Selection_Params.train_id

    DATA_ROOT_DIR = data_root
    logger.debug("Using Dataset : " + str(DATA_ROOT_DIR))

    __training = os.path.join(DATA_ROOT_DIR, "train")
    __validation = os.path.join(DATA_ROOT_DIR, "validation")
    __test = os.path.join(DATA_ROOT_DIR, "test")

    _Train_Common_Params.i_Dataset_Path.create(__training, __validation, __test)

    _Train_Common_Params.i_Generator_Params.create(width=image_size, height=image_size, batch_size=batch_size)
    _Train_Common_Params.i_Training_Params.create(epoch=epoch, workers=8)

    # SAVE_ROOT_DIR = '/home/joosohn/machines/results/' + save_root + "/" + __train_id
    SAVE_ROOT_DIR = '/results/training_model' + save_root + "/" + __train_id
    logger.debug("Save Result : " + str(SAVE_ROOT_DIR))

    filesystem.create_folder(SAVE_ROOT_DIR)

    _top_weight = utils.file_name_generator(__train_id, 'top_weight', 'h5')
    _result_weight = utils.file_name_generator(__train_id, 'result_weight', 'h5')
    _tensor_log = utils.file_name_generator(__train_id, 'tensorlog', 'log')

    _top_weight_path = os.path.join(SAVE_ROOT_DIR, _top_weight)
    _result_weight_path = os.path.join(SAVE_ROOT_DIR, _result_weight)
    _tensor_log_path = os.path.join(SAVE_ROOT_DIR, _tensor_log)

    _Train_Common_Params.i_Results_Path_Params.create(_top_weight_path, _result_weight_path, _tensor_log_path,
                                                      SAVE_ROOT_DIR)
    logger.info("Start Training " + __train_id)
    base_category_training(_Train_Common_Params, epoch_factor)

    logger.info("Finish Test Training")


def settings():
    import tensorflow as tf
    from keras import backend as K

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from keras.backend.tensorflow_backend import set_session
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    set_session(tf.Session(config=config))


def run_main():
    DATA_ROOT_DIR = "database/parsing_image/dataset"
    SAVE_ROOT = "awesome_suplago"

    settings()
    common_training(model_name='Awesome', model_version='Sulpago',
                    image_size=320,
                    data_root=DATA_ROOT_DIR, save_root=SAVE_ROOT,
                    batch_size=16, epoch=10, epoch_factor=1)
