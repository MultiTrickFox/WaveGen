import config

from model import load_model
model = load_model()
while not model:
    config.model_path = input('valid model: ')
    model = load_model()

from data import load_data, split_data
data = load_data(with_meta=True)
data, _ = split_data(data)

from random import shuffle
shuffle(data)
d = data[:config.hm_wav_gen]

for i,(seq,meta) in enumerate(d):

    from model import respond_to
    _, seq = respond_to(model, [seq], training_run=False, extra_steps=config.hm_extra_steps)
    seq = seq.detach()
    if config.use_gpu:
        seq = seq.cpu()
    seq = seq.numpy()

    from data import data_to_audio
    seq = data_to_audio(seq, meta)

    from data import write
    write(f'{config.output_file}{i}.wav', config.sample_rate, seq)