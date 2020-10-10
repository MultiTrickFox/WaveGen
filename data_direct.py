import config
from ext import parallel, pickle_save, pickle_load

from glob import glob
from math import ceil
from copy import deepcopy
from random import shuffle

from librosa.core import load

from numpy import abs, log, power, e, sum, clip, max, argmax, min, argmin

from scipy.io.wavfile import write

##


def data_to_audio(data,meta):

    data.reshape(data.shape[0])

    if config.zscore_scale:

        _, mean, std, scale = meta

        data *= scale
        data *= std
        data += mean

    elif config.minmax_scale:

        _, spec_min, spec_max = meta

        data *= spec_max - spec_min
        data += spec_min

    elif config.log_scale:

        data = power(e,data-1e-10)

    return data


def audio_to_data(signal, song_id):

    meta = [song_id]

    data = deepcopy(signal)
    print('\tmax min initially:', max(data), min(data))

    if config.zscore_scale:

        mean = data.mean()
        std = data.std()
        data -= mean
        data /= std

        print('\tmax min after std:', max(data), min(data))

        scale = max([abs(max(data)),abs(min(data))])
        data /= scale

        meta.extend([mean, std, scale])

    elif config.minmax_scale:

        data_min = min(data)
        data_max = max(data)
        data -= data_min
        data /= data_max - data_min

        print('\tmax min after min/max:', max(data), min(min))

        meta.extend([data_min, data_max])

    elif config.log_scale:

        data = log(data + 1e-10)

        print('\tmax min after log:', max(data), min(data))

    print('\tfinal vector shape:', data.shape)

    return data, meta


def main():

    files = glob(config.data_path+'/*.wav') # + glob('data/*.mp3') # try ffmpeg -i input.mp3 output.wav

    converted = []

    for file_id, file in enumerate(files):

        print(f'reading: {file}')
        song_id = [0 if i == file_id else 1 for i in range(len(files))]

        # analysis
        signal, sample_rate = load(file, config.sample_rate)
        data, meta = audio_to_data(signal, song_id)
        converted.append([data,meta])

        # synthesis
        signal_recons = data_to_audio(data,meta)
        write(f'{file.split("/")[-1]}_{file_id}.wav', config.sample_rate, signal_recons)
        signal_recons, sample_rate = load(f'{file.split("/")[-1]}_{file_id}.wav', config.sample_rate)

    pickle_save(converted, config.data_path+'.pk')
    print('saved data.')


def load_data(with_meta=False):
    from torch import Tensor
    data = pickle_load(config.data_path+'.pk')
    data_tensors = []
    for sequence,meta in data:
        sequence = Tensor(sequence).view(-1,1)
        if config.use_gpu:
            sequence = sequence.cuda()
        data_tensors.append(sequence if not with_meta else [sequence,meta])
    return data_tensors

def split_data(data, dev_ratio=None, do_shuffle=False):
    if not dev_ratio: dev_ratio = config.dev_ratio
    if do_shuffle: shuffle(data)
    if dev_ratio:
        hm_train = int(len(data)*(1-dev_ratio))
        data_dev = data[hm_train:]
        data = data[:hm_train]
        return data, data_dev
    else:
        return data, []

def batchify_data(data, batch_size=None, do_shuffle=True):
    if not batch_size: batch_size = config.batch_size
    if do_shuffle: shuffle(data)
    hm_batches = int(len(data)/batch_size)
    return [data[i*batch_size:(i+1)*batch_size] for i in range(hm_batches)] \
        if hm_batches else [data]





if __name__ == '__main__':
    main()
