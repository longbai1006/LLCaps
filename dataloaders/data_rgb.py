import os
from .dataset_rgb import DataLoaderTrain, DataLoaderTrainPatch, DataLoaderVal, DataLoaderValPatch

def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)

def get_training_data_patch(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrainPatch(rgb_dir, img_options, None)

def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir,img_options, None)

def get_validation_data_patch(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderValPatch(rgb_dir,img_options, None)
