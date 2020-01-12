from skimage import io, transform, morphology, feature, img_as_bool
from os.path import isfile, join, isdir
from random import shuffle
from os import listdir
import numpy as np
import pickle

OUTPUT_H = 120
OUTPUT_W = 60
OUTPUT_FRAMES = 80


def read_img(filename):
    return io.imread(filename)


def pad_positions(start, end, minimum, maximum, pad_size):
    start = start - pad_size if start - pad_size > minimum else minimum
    end = end + pad_size if end + pad_size < maximum else maximum
    return start, end


def get_start_end_positions(img_sum):
    reverse_sum = img_sum[::-1]
    start = np.argmax(img_sum > 0)
    end = len(reverse_sum) - np.argmax(reverse_sum > 0)
    return pad_positions(start, end, 0, img_sum.shape[0], 0)


def select_silhouette(img):
    sum_of_columns = np.sum(img, 1)
    sum_of_rows = np.sum(img, 0)

    start_col, end_col = get_start_end_positions(sum_of_rows)
    start_row, end_row = get_start_end_positions(sum_of_columns)

    return img[start_row: end_row, start_col:end_col]


def preprocess(img):
    img = select_silhouette(img)
    img = np.array(img)
    # print(img.shape)
    if img.shape[0] > OUTPUT_H:
        img = img[:OUTPUT_H, :]
    if img.shape[1] > OUTPUT_W:
        img = img[:, :OUTPUT_W]
    dh = OUTPUT_H - img.shape[0]
    dw = OUTPUT_W - img.shape[1]
    img = np.pad(img, ((dh//2, dh-(dh//2)), (dw//2, dw-(dw//2))), mode='constant', constant_values=0)
    # img = transform.rescale(img, .4)
    # img = morphology.binary_erosion(img)
    # img = morphology.binary_dilation(img)
    # img = morphology.binary_erosion(img)
    # img = morphology.binary_dilation(img)
    # img = img_as_bool(transform.resize(img, (60, 30)))
    # index = feature.shape_index(img)
    # index[np.isnan(index)] = np.NINF
    # return np.nan_to_num(index)
    return img


def preprocess_folder(path):
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    if len(files) is 0:
        return False
    pics = []
    for f in files:
        img = read_img(f)
        img = preprocess(img)
        pics.append(img)
    pics = np.array(pics)
    if pics.shape[0] < OUTPUT_FRAMES:
        pics = np.tile(pics, (OUTPUT_FRAMES//pics.shape[0] + 1, 1, 1))
    if pics.shape[0] > OUTPUT_FRAMES:
        pics = pics[:OUTPUT_FRAMES]
    return pics


def preprocess_user(path):
    degrees = [join(path, f) for f in listdir(path) if isdir(join(path, f))]
    multiple_user_degrees = []
    for f in degrees:
        user_degree = preprocess_folder(f)
        if user_degree is not False:
            multiple_user_degrees.append(user_degree)
        else:
            print('++++++++', f)
    return multiple_user_degrees


def preprocess_all_users(path):
    dirs = [(join(path, f), f) for f in listdir(path) if isdir(join(path, f))]
    users = []
    for f, name in dirs:
        for user_degree in preprocess_user(f):
            users.append((user_degree, name))
    shuffle(users)
    return np.array(users)


maps = [
    ("data/90-test/train1", f'data/90-test/train1-90-test-{OUTPUT_FRAMES}-frames.pickle'),
    ("data/90-test/train2", f'data/90-test/train2-90-test-{OUTPUT_FRAMES}-frames.pickle'),
    ("data/90-test/train3", f'data/90-test/train3-90-test-{OUTPUT_FRAMES}-frames.pickle'),
    ("data/90-test/train4", f'data/90-test/train4-90-test-{OUTPUT_FRAMES}-frames.pickle'),
    ("data/90-test/test", f'data/90-test/test-90-test-{OUTPUT_FRAMES}-frames.pickle'),
    ("data/90-test/validation", f'data/90-test/validation-90-test-{OUTPUT_FRAMES}-frames.pickle')
]


# pickle_file = open('data/train1.pickle', 'wb')
# aa = preprocess_all_users("data/train1")
# pickle_file = open('data/train2.pickle', 'wb')
# aa = preprocess_all_users("data/train2")
# pickle_file = open('data/train3.pickle', 'wb')
# aa = preprocess_all_users("data/train3")
# pickle_file = open('data/train4.pickle', 'wb')
# aa = preprocess_all_users("data/train4")
# pickle_file = open('data/validation.pickle', 'wb')
# aa = preprocess_all_users("data/validation")
# pickle_file = open('data/test.pickle', 'wb')
# aa = preprocess_all_users("data/test")
# pickle.dump(aa, pickle_file)
# print(aa.shape)


for (src, dst) in maps:
    with open(dst, 'wb') as pickle_file:
        aa = preprocess_all_users(src)
        pickle.dump(aa, pickle_file)
