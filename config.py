from glob import glob
from librosa import fft_frequencies

## data params

sample_rate = 8_000 # 22_050 # 44_100

fft_bins = 2048
window_len = fft_bins
hop_len = window_len//4

mfcc_bins = 20
mel_bins = 128

silence_thr_db = None

band_low_hz = 20
band_high_hz = 20_000

frequencies_of_bins = fft_frequencies(sample_rate, fft_bins)
frequencies_range = [i for i,f in enumerate(frequencies_of_bins) if band_low_hz <= f <= band_high_hz]

zscore_scale = True
minmax_scale = False
log_scale = False

concat_id_info = False
concat_chroma_info = False

## model params

timestep_size = len(frequencies_range) + len(glob('data/*.wav'))*int(concat_id_info) + 12*int(concat_chroma_info)
in_size = timestep_size
out_size = len(frequencies_range)
creation_info = [in_size, 'l', 500, 'f', out_size]

init_xavier = True
forget_bias = 0

loss_squared = False

learning_rate = .2
batch_size = 10
gradient_clip = 0

model_path = 'model.pk'
fresh_meta = True

## interact params
