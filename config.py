from librosa import fft_frequencies
from librosa.core import frames_to_time

## data params

sample_rate = 8_000 # 22_050 # 44_100

fft_bins = 2048
fft_window_len = fft_bins
fft_hop_len = fft_window_len//4

mfcc_bins = 20
mel_bins = 128

silence_thr_db = 0

band_low_hz = 20
band_high_hz = 20_000

frequencies_of_bins = fft_frequencies(sample_rate, fft_bins).tolist()
frequencies_range = [i for i,f in enumerate(frequencies_of_bins) if band_low_hz <= f <= band_high_hz]
times_of_bins = lambda hm_steps: frames_to_time(range(0,hm_steps), sample_rate, fft_hop_len, fft_bins)

zscore_scale = True
minmax_scale = False
log_scale = False

data_path = 'data'
dev_ratio = 0

## model params

ticket_size = len(frequencies_range)//4   # ?
sub_models = [['l'], ['l'], ['f'], ['f']] # orig, create, attend, keep
init_xavier = True
forget_bias = 0

seq_window_len = 50
seq_stride_len = seq_window_len//2
seq_force_len = seq_window_len//2

loss_squared = False

learning_rate = .01
batch_size = 0
gradient_clip = 0
hm_epochs = 20
optimizer = 'custom'

model_path = 'model'
fresh_model = False
fresh_meta = True

use_gpu = False

## interact params

hm_wav_gen = 1

##

config_to_save = ['sample_rate', 'fft_bins', 'fft_window_len', 'fft_hop_len', 'mfcc_bins', 'mel_bins',
                  'band_low_hz', 'band_high_hz', 'frequencies_of_bins', 'frequencies_range',
                  'zscore_scale', 'minmax_scale', 'log_scale',
                  'ticket_size',
                  'seq_window_len', 'seq_stride_len', 'seq_force_len',]
