import matplotlib.pyplot as plt

import numpy as np
import cv2
import io, os
import functools
import pickle
import pandas as pd

import sklearn
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, SpatialDropout2D, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

# Training data was recorded in multiple attemps to see the progress.
# Every attempt went under './data' dir as './data/tryNN' where NN is a tryout number

data_dir = 'data'            # Directory with all training data 
try_prefix = 'try'           # Prefix for dirs with traning data
log_file = 'driving_log.csv' # CSV file with training data description, generated by simulator
img_dir = 'IMG'              # Directory with training images for every attempt, like ./data/try1/IMG
train_file = os.path.join(data_dir, 'train.csv') # Generated file with fixed training data split, needed for generator 
validation_file = os.path.join(data_dir, 'validation.csv') # Generated file with fixed validation data split, needed for generator 


def find_tryouts(data_dir, prefix):
    """ 
    Find all tryouts dirs under data_dir, all tryout dirs must start with prefix 
    and returns list of dirs with training data
    """
    filenames = os.listdir(data_dir)
    return sorted(list(map(lambda d: os.path.join(data_dir, d), filter(lambda s: s.startswith(prefix), filenames))))

def fix_logs_paths(img_dir, data_frame):
    """ Fix paths to images in a log dataframe """
    pathfn = lambda p: os.path.join(img_dir, p.split('/')[-1])
    data_frame.center = data_frame.center.apply(pathfn)
    data_frame.left = data_frame.left.apply(pathfn)
    data_frame.right = data_frame.right.apply(pathfn)
    
def load_log(log_dir, log_file, img_dir):
    """ Load tryout csv file (driving_log.csv) """
    f = os.path.join(log_dir, log_file)
    df = pd.read_csv(f, header=None, names=['center','left','right', 'angle', 'throttle', 'break', 'speed'])
    i = os.path.join(log_dir, img_dir)
    fix_logs_paths(i, df)
    return df
        
def merge_logs(dfs):
    """ Merge loaded dataframes for evere 'driving_log.csv' """
    return pd.concat(dfs, ignore_index=True)

def prepare_log():
    """ 
    Load 'driving_log.csv' for every dir merge, shuffle, split by 80/20 for training and validation.
    Save resulted training and validation subsets to separate csv files - for generators
    """ 
    from sklearn.model_selection import train_test_split
    tryouts = find_tryouts(data_dir, try_prefix)
    log_frames = list(map(lambda t: load_log(t, log_file, img_dir), tryouts))

    #add records for flipped images, simple duplication
    log = merge_logs(log_frames)
    rev_log = log.copy()
    log['r'] = False
    rev_log['r'] = True
    merged_log = merge_logs([log, rev_log])

    merged_log = merged_log.reindex(np.random.permutation(merged_log.index))
    train_samples, validation_samples = train_test_split(merged_log, test_size=0.2)
    train_samples = train_samples.reindex(np.random.permutation(train_samples.index))
    validation_samples = validation_samples.reindex(np.random.permutation(validation_samples.index))
    train_samples.to_csv(train_file)
    validation_samples.to_csv(validation_file)
    return len(train_samples), len(validation_samples)

def generator(file_name, batch_size):
    """ 
    Generator to load train and validation data, Loads data in chunks (batch_size) 
    """
    from sklearn.utils import shuffle
    while 1: # Loop forever so the generator never terminates
        chunk_iter = pd.read_csv(file_name, chunksize=batch_size)
        for chunk in chunk_iter:
            images = []
            angles = []
            for row in chunk.itertuples():
                img = cv2.imread(row.center)
                ang = float(row.angle)
                # if reverse flag is true flip the image
                if row.r: 
                    img = np.fliplr(img)
                    ang = -ang
                images.append(img)
                angles.append(ang)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def create_model():
    """ Create model """
    model = Sequential([
        Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)),
        Cropping2D(cropping=((70, 25), (0, 0))),
        Convolution2D(24, (5, 5), activation='relu'),
        MaxPooling2D((2,2)),
        Convolution2D(36, (5, 5), activation='relu'),
        MaxPooling2D((2,2)),
        Convolution2D(48, (5, 5), activation='relu'),
        MaxPooling2D((2,2)),
        Convolution2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dropout(.2),
        Dense(100),
        Dropout(.2),
        Dense(50),
        Dropout(.2),
        Dense(10),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])   
    model.summary()
    return model

def save_model(model, filename):
    """ Save model """
    model.save(filename)
    print("model saved, filename: {}.".format(filename))

def run_model(model):
    """ Run model """
    BATCH_SIZE = 64
    EPOCH = 8
    train_size, validation_size = prepare_log()
    train_generator = generator(train_file, BATCH_SIZE)
    validation_generator = generator(validation_file, BATCH_SIZE)
    return model.fit_generator(train_generator, 
                        steps_per_epoch=train_size/BATCH_SIZE, 
                        validation_data=validation_generator, 
                        validation_steps=validation_size/BATCH_SIZE, 
                        epochs=EPOCH,
                        verbose=1)                   
    
def save_history(history, history_file):
    """ Save model metrics """
    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)      

def main():
    m_name = 'model.h5'
    history_file = 'run_history'
    m = create_model()
    history = run_model(m) 
    save_model(m, m_name)  
    save_history(history, history_file)


if __name__ == '__main__':
    main()
