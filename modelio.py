import pickle
import os
import getpass
import time
import datetime

__author__ = 'haotian'


def save(model, data_direction, notes=''):
    if not os.path.isfile(data_direction):
        print("invalid training file name, save unsuccessful")
        return None
    if not os.path.exists('model'):
        os.makedirs('model')
    author = getpass.getuser()
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    modeltype = type(model).__name__

    filename_prefix = 'model/{}_{}'.format(author, timestamp)
    filename_ending = 1
    while os.path.isfile('{}_{}'.format(filename_prefix,filename_ending)):
        filename_ending += 1

    filename = '{}_{}'.format(filename_prefix,filename_ending)
    logname = '{}_log.txt'.format(filename)

    file = open(filename, 'wb')
    log = open(logname, 'w')

    bytestream = pickle.dumps(model)
    file.write(bytestream)
    file.close()

    log.write('creator:\t{}\n'.format(author))
    log.write('model type:\t{}\n'.format(modeltype))
    log.write('based on data:\t{}\n'.format(data_direction))
    log.write('created time:\t{}\n'.format(timestamp))
    log.write('notes:\t{}\n\n'.format(notes))
    log.write('binary backup (DO NOT MODIFY!)\n')
    log.write(str(bytestream))
    log.close()

    lut = open('model/all_logs.txt', 'a')
    lut.write('filename:\t{}\n'.format(filename[6:]))
    lut.write('creator:\t{}\n'.format(author))
    lut.write('model type:\t{}\n'.format(modeltype))
    lut.write('based on data:\t{}\n'.format(data_direction))
    lut.write('created time:\t{}\n'.format(timestamp))
    lut.write('notes:\t{}\n\n'.format(notes))

    return filename


def load(filename):
    if os.path.isfile(filename):
        byte_stream = open(filename, 'rb').read()
    elif os.path.isfile('model/'+filename):
        byte_stream = open('model/'+filename, 'rb').read()
    else:
        print("can't find file {}".format(filename))
        return None
    model = pickle.loads(byte_stream)
    return model


