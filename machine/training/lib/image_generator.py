from keras.preprocessing import image as k_image


def image_data_generation(_Dataset_Path, _Generator_Params):
    """
    Train Data Generator
    :param _Dataset_Path: _Dataset_Path Object
    :param _Generator_Params: _Generator_Params Object, For Training
    :return:
    """
    # https://keras.io/preprocessing/image/

    train_data_gen = k_image.ImageDataGenerator(
        rotation_range=30,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        data_format='channels_last')

    valid_data_gen = k_image.ImageDataGenerator(
        fill_mode='nearest',
        data_format='channels_last')

    test_data_gen = k_image.ImageDataGenerator(
        fill_mode='nearest',
        data_format='channels_last')

    train_generator = train_data_gen.flow_from_directory(
        _Dataset_Path.training_path,
        target_size=_Generator_Params.target_size,
        batch_size=_Generator_Params.batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    validation_generator = valid_data_gen.flow_from_directory(
        _Dataset_Path.validation_path,
        target_size=_Generator_Params.target_size,
        batch_size=_Generator_Params.batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    test_generator = test_data_gen.flow_from_directory(
        _Dataset_Path.test_path,
        target_size=_Generator_Params.target_size,
        batch_size=_Generator_Params.batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    auto_class_generator = train_generator.class_indices

    return (train_generator, validation_generator, test_generator, auto_class_generator)