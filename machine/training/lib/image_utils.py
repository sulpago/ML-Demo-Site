def image_input_shape(_Generator_Params):
    from keras import backend as K
    K.set_image_dim_ordering('tf')

    __format_width = int(_Generator_Params.width)
    __format_height = int(_Generator_Params.height)

    if K.image_data_format() == 'channels_first':
        __input_shape = (3, __format_width, __format_height)
    else:
        __input_shape = (__format_width, __format_height, 3)
    return __input_shape


def image_file(image_file):
    __open_pil = pilImage.open(image_file)
    __numpyed_array = np.asarray(__open_pil)
    return __numpyed_array


def object_image_reshape(file_path, width=256, height=192, dimension=3):
    __input_width = width
    __input_height = height
    __input_dimension = dimension

    __check_shape_type = (__input_height, __input_width, __input_dimension)
    # logger.debug("__check_shape_type " + str(__check_shape_type))

    with pilImage.open(file_path) as __open_pil:
        __resized_img = __open_pil.resize((__input_width, __input_height))
        __convert = __resized_img.convert("RGB")
        __numpyed_array = np.array(__convert)

        __check_shape = __numpyed_array.shape
        if __check_shape != __check_shape_type:
            return None
        else:
            return __numpyed_array


def reshape(target, width=200, height=200, dimension=3):
    __input_width = width
    __input_height = height
    __input_dimension = dimension
    __reshape_img = None

    if (target is None):
        __input_zeros = np.zeros((__input_height * __input_width * __input_dimension))
        __reshape_img = np.reshape(__input_zeros, [1, __input_height, __input_width, __input_dimension])
    else:
        __check_img_len = len(target)
        if __check_img_len < __input_dimension :
            return False, None

        try:
            __p_img = target
            if target.shape[2] > __input_dimension:
                __p_img = target[:, :, :__input_dimension]  # 차원 축소 ;
            __rsz_img = cv2.resize(__p_img, (__input_height, __input_width))
            __reshape_img = np.reshape(__rsz_img, [1, __input_height, __input_width, __input_dimension])
            return True, __reshape_img
        except:
            return False, None

    return  False, None


def image_purity_check(file_name, img_width, img_height, img_dim):
    if file_name is None:
        return False
    __file_path = file_name + ".jpg"

    try:
        __result = object_image_reshape(__file_path, img_width, img_height, img_dim)
        if __result is not None:
            return True
        else:
            return False
    except:
        return False