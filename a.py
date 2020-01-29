from skimage import io, transform, morphology, feature, img_as_bool
from os.path import isfile, join, isdir
from random import shuffle
from os import listdir
import numpy as np
import pickle


input = 'data/separate-test/validation-090-test-80-frames.pickle'


with open(input, 'rb') as inp:
    data = pickle.load(inp)

first = data[0][0]

n = len(first)

io.imsave("90-1.jpg", first[0])
io.imsave("90-2.jpg", first[n//3])
io.imsave("90-3.jpg", first[2*(n//3)])
io.imsave("90-4.jpg", first[n-1])

# with open('data/separate-test/one-set.pickle', 'wb') as out:
#     pickle.dump(data[2], out)
