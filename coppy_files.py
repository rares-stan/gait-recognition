from distutils.file_util import copy_file
from os import listdir, mkdir, makedirs
from os.path import isdir, join, isfile

maps = [
    ('nm-01', 'train1'),
    ('nm-02', 'train2'),
    ('nm-03', 'train3'),
    ('nm-04', 'train4'),
    ('nm-05', 'test'),
    ('nm-06', 'validation')
]

degrees = [
    # '000',
    '018',
    # '036',
    # '054',
    # '072',
    # '090',
    # '108',
    # '126',
    # '144',
    # '162',
    # '180'
]


def copy_from_dir(src, dst):
    for degree in degrees:
        degree_dst = join(dst, degree)
        makedirs(degree_dst)
        degree_src = join(src, degree)
        files = [join(degree_src, f) for f in listdir(degree_src) if isfile(join(degree_src, f))]
        for f in files:
            copy_file(f, degree_dst)


org = "C:\\Dizertatie\\DatasetB-1\\silhouettes\\"
fin = "C:\\Dizertatie\\gait-recognition\\data\\00-test\\"

for (src, dst) in maps:
    origs = [(join(org, f, src), f) for f in listdir(org) if isdir(join(org, f))]
    for img_dir, name in origs:
        copy_from_dir(img_dir, join(fin, dst, name))
