from skimage import io, transform, morphology, feature, img_as_bool
from os.path import isfile, join, isdir
from random import shuffle
from os import listdir
import numpy as np
import pickle


input = 'data/separate-test/validation-090-test-80-frames.pickle'


with open(input, 'rb') as inp:
    data = pickle.load(inp)

with open('data/separate-test/one-set.pickle', 'wb') as out:
    pickle.dump(data[2], out)
