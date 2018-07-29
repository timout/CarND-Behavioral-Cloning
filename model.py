import matplotlib.pyplot as plt

import numpy as np
import cv2
import io, os
import functools
import pandas as pd

import sklearn
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, SpatialDropout2D, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

data_dir = 'data'
try_prefix = 'try'
log_file = 'driving_log.csv'
img_dir = 'IMG'
train_file = os.path.join(data_dir, 'train.csv')
validation_file = os.path.join(data_dir, 'validation.csv')

def find_tryouts(data_dir, prefix):
    """ Find all tryouts dirs under data_dir, all tryout dirs mys start with prefix """
    filenames = os.listdir(data_dir)
    return sorted(list(map(lambda d: os.path.join(data_dir, d), filter(lambda s: s.startswith(prefix), filenames))))

def fix_logs_paths(img_dir, data_frame):
    """ Fix paths to images in a log dataframe """
    pathfn = lambda p: os.path.join(img_dir, p.split('/')[-1])
    data_frame.center = data_frame.center.apply(pathfn)
    data_frame.left = data_frame.left.apply(pathfn)
    data_frame.right = data_frame.right.apply(pathfn)
    
def load_log(log_dir, log_file, img_dir):
    """ Load tryout csv file """
    f = os.path.join(log_dir, log_file)
    df = pd.read_csv(f, header=None, names=['center','left','right', 'angle', 'throttle', 'break', 'speed'])
    i = os.path.join(log_dir, img_dir)
    fix_logs_paths(i, df)
    return df
        
def merge_logs(dfs):
    """ Merge dataframes with train data """
    return pd.concat(dfs, ignore_index=True)

def prepare_log():
    from sklearn.model_selection import train_test_split
    tryouts = find_tryouts(data_dir, try_prefix)
    log_frames = list(map(lambda t: load_log(t, log_file, img_dir), tryouts))
    merged_log = merge_logs(log_frames)
    merged_log = merged_log.reindex(np.random.permutation(merged_log.index))
    train_samples, validation_samples = train_test_split(merged_log, test_size=0.2)
    train_samples = train_samples.reindex(np.random.permutation(train_samples.index))
    validation_samples = validation_samples.reindex(np.random.permutation(validation_samples.index))
    train_samples.to_csv(train_file)
    validation_samples.to_csv(validation_file)
    return len(train_samples), len(validation_samples)

def generator(file_name, batch_size):
    """ Generator to load train and validation data """
    from sklearn.utils import shuffle
    while 1: # Loop forever so the generator never terminates
        chunk_iter = pd.read_csv(file_name, chunksize=batch_size)
        for chunk in chunk_iter:
            images = []
            angles = []
            for row in chunk.itertuples():
                img = cv2.imread(row.center)
                ang = float(row.angle)
                images.append(img)
                angles.append(ang)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def resize_img(img):
    import tensorflow as tf  # drive.py requirement
    return tf.image.resize_images(img, (80, 160))

def create_model():
    """ Create model """
    model = Sequential([
        Lambda(resize_img, input_shape=(160, 320, 3)),
        Lambda(lambda x: x/127.5 - 1.),
        Convolution2D(16, (5, 5), activation='relu', padding="same"),
        MaxPooling2D((2,2)),
        Convolution2D(32, (5, 5), activation='relu', padding="same"),
        MaxPooling2D((2,2)),
        Dropout(0.5),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])   
    model.summary()
    return model

def save_model(model, filename):
    """ Save model """
    model.save(filename)
    print("model saved, filename: {}.".format(filename))

def run_model(model, filename):
    """ Run model """
    BATCH_SIZE = 128
    EPOCH = 32
    train_size, validation_size = prepare_log()
    train_generator = generator(train_file, BATCH_SIZE)
    validation_generator = generator(validation_file, BATCH_SIZE)
    model.fit_generator(train_generator, 
                        steps_per_epoch=train_size/BATCH_SIZE, 
                        validation_data=validation_generator, 
                        validation_steps=validation_size/BATCH_SIZE, 
                        epochs=EPOCH)
    save_model(model, filename)  

def main():
    m_name = 'model.h5'
    m = create_model()
    run_model(m, m_name) 


if __name__ == '__main__':
    main()
