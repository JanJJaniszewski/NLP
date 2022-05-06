import zipfile
import os
from os.path import join
import config as cf
import shutil

def unpack_data():
    print('Unpacking data from input folder and saving it in throughput/Z_A folder')
    with zipfile.ZipFile(cf.Z_input, 'r') as zip_ref:
        zip_ref.extractall(cf.Z_Z_throughput)
    print('Finished unpacking and saving')

    print('Renaming and reorganizing text files, deleting audio files')
    folders = os.listdir(cf.Z_Z_zipfolder)
    [os.rename(join(cf.Z_Z_zipfolder, f, 'Text.txt'), join(cf.Z_Z_zipfolder, f'{f}.txt')) for f in folders]
    [shutil.rmtree(join(cf.Z_Z_zipfolder, f)) for f in folders]

if __name__ == '__main__':
    unpack_data()