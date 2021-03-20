import numpy as np 
import pandas as pd
import socket
from tensorflow.keras.applications.vgg16 \
    import preprocess_input
from keras_custom.generators.generator_wrapers \
    import generator_for_continuous_master_TEST


"""
E.g. generators used for eval are defined here.
"""

# hostname = socket.gethostname()
# server_num = int(hostname[4:6])
# print(f'server_num = {server_num}')
# if server_num <= 20:
#     imagenet_test = f'/mnt/fast-data{server_num}/datasets/ILSVRC/2012/clsloc/val_white/'
# else:
#     imagenet_test = f'/fast-data{server_num}/datasets/ILSVRC/2012/clsloc/val_white/'


def data_directory(part):
    """
    Check which server we are on and return the corresponding 
    imagenet data directory.
    """
    hostname = socket.gethostname()
    server_num = int(hostname[4:6])
    print(f'server_num = {server_num}')
    if server_num <= 20:
        if part == 'val_white_occluded_v1':  # TODO: this is ad hoc.
            imagenet_test = f'/mnt/fast-data{server_num}/datasets/ken/{part}/'
            #imagenet_test = f'val_white_occuded_v1/'
        else:
            imagenet_test = f'/mnt/fast-data{server_num}/datasets/ILSVRC/2012/clsloc/{part}/'
    else:
        imagenet_test = f'/fast-data{server_num}/datasets/ILSVRC/2012/clsloc/{part}/'
    return imagenet_test


def test_generator_constructor(focus_wnid, nontrg_wnids, usr_focus_index, usr_intensity, part, num_classes=None):
    """
    To avoid repeating generator_for_continuous_master_TEST 
    every time!
    # TODO: num_classes deprecated!

    return:
    ------
        test_gen, test_steps
    """
    # temp
    # focus_wnid = ['n01440764']
    if type(focus_wnid) == type(['123']):
        pass
    else:
        focus_wnid = [focus_wnid]
    
    # use nontrg_wnids as a signal for whether the generator is used for 
    # computing hit rate or false alarm rate.
    if nontrg_wnids is None:
        # means we are at computing hit rate
        classes = focus_wnid
        # and we don't subsample the already very few positive examples
        subsample_rate = 1
    else:
        classes = nontrg_wnids
        subsample_rate = 0.5  # temp 26-8-20, just for speed up.

    imagenet_test = data_directory(part=part)
    print('CHECK: imagenet_test = ', imagenet_test)
    test_gen, test_steps = generator_for_continuous_master_TEST(
                                    directory=imagenet_test,
                                    classes=classes,   # Careful!
                                    batch_size=16,
                                    seed=42,
                                    shuffle=False,
                                    subset=None,
                                    validation_split=0.0,
                                    class_mode='sparse',
                                    target_size=(224, 224),
                                    preprocessing_function=preprocess_input,
                                    horizontal_flip=False,
                                    AlexNetAug=False,
                                    focus_classes=None,
                                    subsample_rate=subsample_rate,
                                    # ----------------------
                                    usr_focus_index=usr_focus_index,
                                    usr_intensity=usr_intensity,
                                    num_of_classes=num_classes,
                                    )
    return test_gen, test_steps


def load_classes(num_classes, df):
    """
    load in all imagenet/imagenetA or other dataframe classes,
    return:
    -------
        n classes of wnids, indices and descriptions
    """
    df = pd.read_csv(f'groupings-csv/{df}_Imagenet.csv',
                     usecols=['wnid', 'idx', 'description'])
    sorted_indices = np.argsort([i for i in df['wnid']])[:num_classes]

    wnids = np.array([i for i in df['wnid']])[sorted_indices]
    indices = np.array([int(i) for i in df['idx']])[sorted_indices]
    descriptions = np.array([i for i in df['description']])[sorted_indices]
    return wnids.tolist(), indices, descriptions