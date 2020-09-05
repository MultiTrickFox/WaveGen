import config

from model import load_model
model = load_model()

from data import load_data
data = load_data(with_meta=True)

from random import shuffle
shuffle(data)
d = data[:config.hm_wav_gen]

for i,(seq,meta) in enumerate(d):

    from model import respond_to
    seq = respond_to(model, [seq], do_grad=False)[-1]

    from data import data_to_audio
    seq = data_to_audio(seq, meta)

    from data import write
    write(f'resp{i}.wav', config.sample_rate, seq)