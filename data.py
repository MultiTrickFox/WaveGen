import config
from ext import parallel, pickle_save, pickle_load

from glob import glob
from math import ceil
from copy import deepcopy
from random import shuffle

from librosa import fft_frequencies, amplitude_to_db, db_to_amplitude

from librosa.core import load

from librosa.effects import split, trim

from librosa.core.spectrum import stft, istft, griffinlim
from librosa.feature import chroma_stft

from librosa.feature import mfcc
from librosa.feature.inverse import mfcc_to_audio

from numpy import abs, log, power, e, sum, clip, max, argmax, min, argmin
from numpy import zeros_like, zeros, array, tile, concatenate

from librosa.display import specshow
from matplotlib.pyplot import plot, show

from scipy.io.wavfile import write

##


def data_to_audio(data,meta):

    spec = deepcopy(data[:,:len(config.frequencies_range)])
    spec = spec.T

    if config.zscore_scale:

        _, mean, std, scale = meta

        spec *= scale
        spec *= std
        spec += mean

    elif config.minmax_scale:

        _, spec_min, spec_max = meta

        spec *= spec_max - spec_min
        spec += spec_min

    elif config.log_scale:

        spec = power(e,spec-1e-10)

    spec = db_to_amplitude(spec)

    empty_lower_frequencies = zeros((config.frequencies_range[0], spec.shape[1]), spec.dtype)
    empty_higher_frequencies = zeros((len(config.frequencies_of_bins)-config.frequencies_range[-1], spec.shape[1]), spec.dtype)
    spec = concatenate([empty_lower_frequencies, spec], 0)
    spec = concatenate([spec, empty_higher_frequencies], 0)

    signal_recons = griffinlim(spec, hop_length=config.fft_hop_len, win_length=config.fft_window_len)
    # signal_recons2 = mfcc_to_audio(mfccs, config.mel_bins)

    return signal_recons


def audio_to_data(signal, song_id):

    meta = [song_id]

    if config.silence_thr_db:
        signal, _ = trim(signal, config.silence_thr_db, frame_length=config.fft_bins, hop_length=config.fft_hop_len)
        #signal = split(spec, default_db_min)

    spec = abs(stft(signal, config.fft_bins, config.fft_hop_len, config.fft_window_len))
    mfccs = mfcc(signal, config.sample_rate, n_mfcc=config.mfcc_bins)
    chroma = chroma_stft(signal, config.sample_rate, n_fft=config.fft_bins, hop_length=config.fft_hop_len, win_length=config.fft_window_len)

    # rows-frequencies , cols-times
    # show(specshow(spec, sr=default_sample_rate))
    # show(specshow(mfccs, sr=default_sample_rate))
    # show(plot(chroma))

    spec_mod = deepcopy(spec)
    print('max min initially:', max(spec_mod), min(spec_mod))

    spec_mod = spec_mod[config.frequencies_range[0]:config.frequencies_range[-1]+1,]
    print('max min after bandpass:', max(spec_mod), min(spec_mod))

    spec_mod = amplitude_to_db(spec_mod)  # log scale.
    print('max min in db:', max(spec_mod), min(spec_mod))

    #spec_mod = clip(spec_mod, config.amp_min_thr_db, config.amp_max_thr_db)
    print('clipped.')

    # show(specshow(spec_mod, sr=default_sample_rate))

    if config.zscore_scale:

        mean = spec_mod.mean() # 1 # mean.resize(len(mean),1) # tdo: mean and std along an axis?
        std = spec_mod.std() # 1 # std.resize(len(std),1)
        spec_mod -= mean
        spec_mod /= std
        scale = max([abs(max(spec_mod)),abs(min(spec_mod))])
        spec_mod /= scale

        meta.extend([mean, std, scale])
        print('max min after std:', max(spec_mod), min(spec_mod))

    elif config.minmax_scale:

        spec_min = min(spec_mod)
        spec_max = max(spec_mod)
        spec_mod -= spec_min
        spec_mod /= spec_max - spec_min

        meta.extend([spec_min, spec_max])
        print('max min after min/max:', max(spec_mod), min(spec_mod))

    elif config.log_scale:

        spec_mod = log(spec_mod + 1e-10)
        print('max min after log:', max(spec_mod), min(spec_mod))

    vector = spec_mod
    # vector = concatenate([vector, array([song_id]*vector.shape[1]).T], 0)
    # vector = concatenate([vector, chroma], 0)
    vector = vector.T  # now first index time, second index frequency

    # show(plot(vector.sum(1)))
    print('final vector shape:', vector.shape)

    return vector, meta


def main():

    files = glob(config.data_path+'/*.wav') # + glob('data/*.mp3') # try ffmpeg -i input.mp3 output.wav

    converted = []

    for file_id, file in enumerate(files):

        print(f'reading: {file}')

        song_id = [0 if i == file_id else 1 for i in range(len(files))]

        # read file
        signal, sample_rate = load(file, config.sample_rate)
        # show(plot(signal))
        data, meta = audio_to_data(signal, song_id)
        converted.append([data,meta])

        # reconstruct & save
        signal_recons = data_to_audio(data,meta)
        # write output
        out_name = f'{file.split("/")[-1]}_{file_id}.wav'
        write(out_name, config.sample_rate, signal_recons)
        # write('data/recons2.wav', config.sample_rate, signal_recons2)

        # display final
        signal_recons, sample_rate = load(out_name, config.sample_rate)
        # print(signal.shape, signal_recons.shape)
        # for e1,e2 in zip(signal,signal_recons):
        #     input(f'{e1} / {e2}')

        # input("Continue to next..?")

    pickle_save(converted, config.data_path+'.pk')
    print('saved data.')


def load_data(with_meta=False):
    from torch import Tensor
    data = pickle_load(config.data_path+'.pk')
    data_tensors = []
    for sequence,meta in data:
        sequence = Tensor(sequence)
        if config.use_gpu:
            sequence = sequence.cuda()
        data_tensors.append(sequence if not with_meta else [sequence,meta])
    return data_tensors

def split_dataset(data, dev_ratio=None, do_shuffle=False):
    if not dev_ratio: dev_ratio = config.dev_ratio
    if do_shuffle: shuffle(data)
    if dev_ratio:
        hm_train = int(len(data)*(1-dev_ratio))
        data_dev = data[hm_train:]
        data = data[:hm_train]
        return data, data_dev
    else:
        return data, []

def batchify(data, batch_size=None, do_shuffle=True):
    if not batch_size: batch_size = config.batch_size
    if do_shuffle: shuffle(data)
    hm_batches = int(len(data)/batch_size)
    return [data[i*batch_size:(i+1)*batch_size] for i in range(hm_batches)] \
        if hm_batches else [data]





if __name__ == '__main__':
    main()





# Extra(s)

# import scipy
# import numpy as np
#
# def stft(x, fftsize=1024, overlap=4):
#     hop = int(fftsize / overlap)
#     w = scipy.hanning(fftsize+1)[:-1]      # better reconstruction with this trick +1)[:-1]
#     return np.array([np.fft.rfft(w*x[i:i+fftsize]) for i in range(0, len(x)-fftsize, hop)])
#
# def istft(X, overlap=4):
#     fftsize=(X.shape[1]-1)*2
#     hop = int(fftsize / overlap)
#     w = scipy.hanning(fftsize+1)[:-1]
#     x = scipy.zeros(X.shape[0]*hop)
#     wsum = scipy.zeros(X.shape[0]*hop)
#     for n,i in enumerate(range(0, len(x)-fftsize, hop)):
#         x[i:i+fftsize] += scipy.real(np.fft.irfft(X[n])) * w   # overlap-add
#         wsum[i:i+fftsize] += w ** 2.
#     pos = wsum != 0
#     x[pos] /= wsum[pos]
#     return x