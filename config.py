from librosa import fft_frequencies
from librosa.core import frames_to_time

## base params

act_classical_rnn = False

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

hm_steps_back = 0
timestep_size = len(frequencies_range)
in_size = timestep_size*(hm_steps_back+1)
hm_modalities = 1
out_size = timestep_size*hm_modalities*3 if not act_classical_rnn else timestep_size
creation_info = [in_size,'l',512,'f',out_size]

init_xavier = True
forget_bias = 0

## train params

seq_window_len = 3000
seq_stride_len = seq_window_len-1
seq_force_ratio = 1

loss_squared = True

learning_rate = 5e-4
batch_size = 0
gradient_clip = 0
hm_epochs = 500
optimizer = 'custom'

model_path = 'models/model'
fresh_model = True
fresh_meta = True
ckp_per_ep = 50

use_gpu = False

## interact params

hm_extra_steps = seq_window_len

hm_wav_gen = 1

output_file = 'resp'

##

config_to_save = ['sample_rate', 'fft_bins', 'fft_window_len', 'fft_hop_len', 'mfcc_bins', 'mel_bins',
                  'band_low_hz', 'band_high_hz', 'frequencies_of_bins', 'frequencies_range',
                  'zscore_scale', 'minmax_scale', 'log_scale',
                  'hm_steps_back', 'in_size', 'hm_modalities', 'out_size',
                  'seq_window_len', 'seq_stride_len', 'seq_force_len',]
