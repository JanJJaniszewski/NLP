import os
import shutil
import zipfile
from os.path import join
from time import sleep

import config as cf


def unpack_data():
    print('Unpacking data from input folder and saving it in throughput/Z_A folder')
    with zipfile.ZipFile(cf.Z_input, 'r') as zip_ref:
        zip_ref.extractall(cf.Z_A_throughput)
    print('Finished')


def remove_file(f):
    try:
        shutil.rmtree(join(cf.Z_A_zipfolder, f))
    except PermissionError as e:
        print('Cannot delete some files still')
        sleep(30)
        shutil.rmtree(join(cf.Z_A_zipfolder, f))


def rename_data():
    print('Renaming and reorganizing text files, deleting audio files')
    folders = os.listdir(cf.Z_A_zipfolder)
    [os.rename(join(cf.Z_A_zipfolder, f, 'Text.txt'), join(cf.Z_A_zipfolder, f'{f}.txt')) for f in folders]
    [remove_file(f) for f in folders]

    print('Finished')


if __name__ == '__main__':
    unpack_data()
    rename_data()
