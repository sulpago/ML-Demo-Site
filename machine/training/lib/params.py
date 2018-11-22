class Datagen_Params:
    def __init__(self):
        self.rescale = 1
        self.rotation_range = 0
        self.zoom_range = 0
        self.cval = 0


class Train_Common_Params:
    def __init__(self):
        self.i_Model_Selection_Params = Model_Selection_Params()
        self.i_Dataset_Path = Dataset_Path()
        self.i_Generator_Params = Generator_Params()
        self.i_Training_Params = Training_Params()
        self.i_Results_Path_Params = Results_Path_Params()


class Model_Selection_Params:
    def __init__(self):
        self.model_name = None
        self.model_version = None
        self.input_shape = None
        self.class_number = None
        self.train_id = None
        self.width = None
        self.height = None

    def create(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version
        self.train_id = model_name + "_" + model_version

    def set(self, input_shape, class_number, width, height):
        self.input_shape = input_shape
        self.class_number = class_number
        self.width = width
        self.height = height


class Dataset_Path:
    def __init__(self, train=None, valid=None, test=None):
        self.training_path = train
        self.validation_path = valid
        self.test_path = test

    def create(self, train, valid, test):
        self.training_path = train
        self.validation_path = valid
        self.test_path = test


class Generator_Params:
    def __init__(self, width=None, height=None, batch_size=None):
        self.target_size = (width, height)
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.shuffle = True
        self.class_mode = 'categorical'

    def create(self, width, height, batch_size):
        self.target_size = (width, height)
        self.width = width
        self.height = height
        self.batch_size = batch_size


class Results_Path_Params:
    def __init__(self):
        self.base_path = None
        self.top_weight_path = None
        self.result_weight_path = None
        self.tensor_log_path = None

    def create(self, top_weight_path, result_weight_path, tensor_log_path, base_path=None):
        self.top_weight_path = top_weight_path
        self.result_weight_path = result_weight_path
        self.tensor_log_path = tensor_log_path
        self.base_path = base_path


class Training_Params:
    def __init__(self, epoch=None, workers=None):
        self.batch_size = None
        self.epoch = epoch
        self.workers = workers
        self.train_data = None
        self.validation_data = None
        self.steps_per_epoch = None
        self.validation_steps = None

    def create(self, epoch, workers):
        self.epoch = epoch
        self.workers = workers

    def set(self, batch_size, train_data, validation_data):
        self.batch_size = batch_size
        self.train_data = train_data
        self.validation_data = validation_data
        self.steps_per_epoch = self.train_data.samples // self.batch_size
        self.validation_steps = self.validation_data.samples // self.batch_size
