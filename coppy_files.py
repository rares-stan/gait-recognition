from distutils.file_util import copy_file
from os import listdir, mkdir
from os.path import isdir, join, isfile


def copy_from_dir(src, dst):
    mkdir(dst)
    files = [join(src, f) for f in listdir(src) if isfile(join(src, f))]
    for f in files:
        copy_file(f, dst)


org = "C:\\Facultate\\disertatie\\gait\\DatasetB-1\\silhouettes"
fin = "C:\\Facultate\\disertatie\\test\\data\\train4"

origs = [(join(org, f, 'nm-06', '090'), f) for f in listdir(org) if isdir(join(org, f))]
for img_dir, name in origs:
    copy_from_dir(img_dir, join(fin, name))
