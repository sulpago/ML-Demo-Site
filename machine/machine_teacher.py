from common.logger import logger
from common import filesystem, utils
from machine.training.lib import image_utils
from machine.training.lib.image_generator import image_data_generation
from machine.training.lib.custom_model import custom_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping,TensorBoard



import os
logger.initialize()


def base_category_training(_Train_Common_Params, epoch_factor=50):
    """
    Common categorizing Training
    :param _Train_Common_Params: Class for Model Training, Inlucing sub-clases
    :return:
    """
    logger.info("Training Start")

    _Model_Selection_Params = _Train_Common_Params.i_Model_Selection_Params
    _Dataset_Path = _Train_Common_Params.i_Dataset_Path
    _Generator_Params = _Train_Common_Params.i_Generator_Params
    _Results_Path_Params = _Train_Common_Params.i_Results_Path_Params
    _Training_Params = _Train_Common_Params.i_Training_Params

    logger.info("Check Class Number for categorizing")
    __train_path = _Dataset_Path.training_path
    __folders = filesystem.get_folders(__train_path)
    __class_number = len(__folders)
    logger.debug("Category Number is : " + str(__class_number))

    logger.info("Generating Input shape and train/validation dataset")
    __input_shape = image_utils.image_input_shape(_Generator_Params)
    __train_generator, __validation_generator, __test_generator, __class_generator = image_data_generation(_Dataset_Path, _Generator_Params)
    logger.debug("Check Classes : " + str(__class_generator))

    logger.debug("Check Train data_generator ")
    __base_path = _Results_Path_Params.base_path

    logger.info("Result and tensor log path Setup")
    __top_weight_path = _Results_Path_Params.top_weight_path
    __result_weight_path = _Results_Path_Params.result_weight_path
    __tensor_log_path = _Results_Path_Params.tensor_log_path

    _Model_Selection_Params.set(__input_shape, __class_number, _Generator_Params.width, _Generator_Params.height)

    __model = custom_model(_Model_Selection_Params)
    if __model is None:
        logger.warning("Not available Model options")
        return False

    # plot_model(__model, to_file='model.png')
    __optimizer = optimizers.RMSprop()
    # __optimizer = keras.optimizers.Adam()
    __model.compile(optimizer=__optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    logger.debug("Loaded Model Type")
    logger.debug(str(__model.summary()))

    __model_name = utils.file_name_generator(identification='result', title='model', extension='png')
    __model_image_path = os.path.join(__base_path, __model_name)

    logger.debug("Add CallbackList")
    __callbacks_list = [
        ModelCheckpoint(__top_weight_path, monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=80, verbose=0),
        TensorBoard(log_dir=__tensor_log_path, histogram_freq=0, write_graph=True, write_images=True)
    ]

    __batch_size = _Generator_Params.batch_size

    _Training_Params.set(batch_size=__batch_size, train_data=__train_generator, validation_data=__validation_generator)

    logger.info("Generating and Fitting Model for training")
    __history_transfer = __model.fit_generator(_Training_Params.train_data,
                                               epochs=_Training_Params.epoch,
                                               steps_per_epoch=_Training_Params.steps_per_epoch // epoch_factor,
                                               validation_data=_Training_Params.validation_data,
                                               validation_steps=_Training_Params.validation_steps // epoch_factor,
                                               callbacks=__callbacks_list,
                                               workers=_Training_Params.workers)

    logger.debug("Save Final Training Values")
    __model.save(__result_weight_path)

    # logger.debug("Save History as Plot")
    # __history_name = utils.file_name_generator(identification='result', title='histogram', extension='png')
    # __history_log_path = os.path.join(__base_path, __history_name)
    # history_save(__history_transfer, __history_log_path)

    scores = __model.evaluate_generator(__test_generator,
                                        steps=_Training_Params.validation_steps // epoch_factor,
                                        use_multiprocessing=False)
    logger.info("Accuracy: %.2f%%" % (scores[1] * 100))
    logger.debug("Training End")

    return True
