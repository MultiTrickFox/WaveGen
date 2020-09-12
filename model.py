import config
from ext import pickle_save, pickle_load, now

from torch import tensor, Tensor, cat, stack
from torch import zeros, ones, eye, randn
from torch import sigmoid, tanh, relu, softmax
from torch import pow, log, exp, sqrt, norm, mean
from torch import float32, no_grad
from torch.nn.init import xavier_normal_
from torch import normal
from torch.nn.functional import elu

from collections import namedtuple
from copy import deepcopy
from math import ceil

from numpy import pi
from numpy import sqrt as nsqrt

##


FF = namedtuple('FF', 'w')
LSTM = namedtuple('LSTM', 'wf bf wk bk wi bi ws bs')


def make_Llayer(in_size, layer_size):

    layer = LSTM(
        randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32),
        zeros(1,                  layer_size, requires_grad=True, dtype=float32),
        randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32),
        zeros(1,                  layer_size, requires_grad=True, dtype=float32),
        randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32),
        zeros(1,                  layer_size, requires_grad=True, dtype=float32),
        randn(in_size+layer_size, layer_size, requires_grad=True, dtype=float32),
        zeros(1,                  layer_size, requires_grad=True, dtype=float32),
    )

    with no_grad():
        for k,v in layer._asdict().items():
            if k == 'bf':
                v += config.forget_bias
        # layer.bf += config.forget_bias

    if config.init_xavier:
        xavier_normal_(layer.wf)
        xavier_normal_(layer.wk)
        xavier_normal_(layer.ws)
        xavier_normal_(layer.wi, gain=5/3)

    return layer

def make_Flayer(in_size, layer_size):

    layer = FF(
        randn(in_size, layer_size, requires_grad=True, dtype=float32),
        # zeros(1, layer_size,       requires_grad=True, dtype=float32),
    )

    # if config.init_xavier:
    #     xavier_normal_(layer.w)

    return layer


make_layer = {
    'l': make_Llayer,
    'f': make_Flayer,
}


def prop_Llayer(layer, state, input):

    layer_size = layer.wf.size(1)
    prev_out = state[:,:layer_size]
    state = state[:,layer_size:]

    inp = cat([input,prev_out],dim=1)

    reset  = sigmoid(inp@layer.wf + layer.bf)
    write  = sigmoid(inp@layer.wk + layer.bk)
    interm = tanh   (inp@layer.wi + layer.bi)
    read   = sigmoid(inp@layer.ws + layer.bs)

    state = reset*state + write*interm
    out = read*tanh(state)

    return out, cat([out,state],dim=1)

def prop_Llayer2(layer, state, input):

    inp = cat([input,state],dim=1)

    forget = sigmoid(inp@layer.wf + layer.bf)
    keep   = sigmoid(inp@layer.wk + layer.bk)
    interm = tanh   (inp@layer.wi + layer.bi)

    state  = forget*state + keep*interm
    inp = cat([input,state],dim=1)

    show   = sigmoid(inp@layer.ws + layer.bs)

    out = show*tanh(state)

    return out, state

def prop_Flayer(layer, inp):

    # return tanh(inp@layer.w) # + layer.b)
    return inp@layer.w



prop_layer = {
    LSTM: prop_Llayer,
    FF: prop_Flayer,
}


def make_model(info=None):

    if not info: info = config.creation_info

    layer_sizes = [e for e in info if type(e)==int]
    layer_types = [e for e in info if type(e)==str]

    return [make_layer[layer_type](layer_sizes[i], layer_sizes[i+1]) for i,layer_type in enumerate(layer_types)]

def prop_model(model, states, inp):
    new_states = []

    out = inp

    state_ctr = 0

    for layer in model:

        if type(layer) != FF:

            out, state = prop_layer[type(layer)](layer, states[state_ctr], out)
            new_states.append(state)
            state_ctr += 1

        else:

            out = prop_Flayer(layer, out)

        # dropout(out, inplace=True)

    means = out[:,:config.out_size//3]
    deviances = out[:,config.out_size//3:config.out_size//3*2]
    multipliers = out[:,-config.out_size//3:]

    deviances = exp(deviances) # deviances = elu(deviances)
    multipliers = multipliers.view(multipliers.size(0), config.hm_modalities, config.timestep_size)
    multipliers = softmax(multipliers, dim=1)
    multipliers = multipliers.view(multipliers.size(0), config.hm_modalities * config.timestep_size)

    return [means, deviances, multipliers], new_states

def distribution_loss(label, out):

    means, deviances, multipliers = out

    means = means.view(means.size(0), config.hm_modalities, config.timestep_size)
    deviances = deviances.view(deviances.size(0), config.hm_modalities, config.timestep_size)
    multipliers = multipliers.view(multipliers.size(0), config.hm_modalities, config.timestep_size)
    label = label.repeat(1, config.hm_modalities, 1)

    loss = 1/nsqrt(2*pi) * exp( -.5 * pow((label-means)/deviances,2) ) /deviances *multipliers

    loss = -log(loss.sum(1) + 1e-10)

    return loss.sum()

def sample_from_out(out):

    means, deviances, multipliers = out

    sample = normal(means, deviances)

    sample = sample.view(sample.size(0), config.hm_modalities, config.timestep_size)
    multipliers = multipliers.view(multipliers.size(0), config.hm_modalities, config.timestep_size)

    sample = (sample*multipliers).sum(1)

    return sample


def respond_to(model, sequences, state=None, training_run=True, extra_steps=0):

    responses = []

    loss = 0
    sequences = deepcopy(sequences)
    if not state:
        state = empty_state(model, len(sequences))

    max_seq_len = max(len(sequence) for sequence in sequences)
    hm_windows = ceil(max_seq_len/config.seq_stride_len)
    has_remaining = list(range(len(sequences)))

    for i in range(hm_windows):

        window_start = i*config.seq_stride_len
        window_end = min(window_start+config.seq_window_len, max_seq_len)

        for window_t in range(window_end-window_start -1):

            t = window_start+window_t

            has_remaining = [i for i in has_remaining if len(sequences[i][t+1:t+2])]

            inp = stack([sequences[i][t] for i in has_remaining],dim=0) *config.seq_force_ratio
            if config.seq_force_ratio != 1:
                if t:
                    inp_plus = stack([responses[t-1][i] for i in has_remaining],dim=0) *(1-config.seq_force_ratio)
                    inp = inp + inp_plus

            lbl = stack([sequences[i][t+1] for i in has_remaining], dim=0)

            partial_state = [stack([layer_state[i] for i in has_remaining], dim=0) for layer_state in state]

            for i in range(1,config.hm_steps_back+1):
                t_prev = t-i
                if t_prev>=0:
                    prev_inp = stack([sequences[i][t_prev] for i in has_remaining],dim=0) *config.seq_force_ratio
                else:
                    prev_inp = zeros(len(has_remaining),config.timestep_size) if not config.use_gpu else zeros(len(has_remaining),config.timestep_size).cuda()
                if config.seq_force_ratio != 1:
                    if t_prev-1>=0:
                        prev_inp_plus = stack([responses[t_prev-1][i] for i in has_remaining], dim=0) *(1-config.seq_force_ratio)
                        prev_inp = prev_inp + prev_inp_plus
                inp = cat([inp,prev_inp],dim=1)

            out, partial_state = prop_model(model, partial_state, inp)

            loss += distribution_loss(lbl, out)
            # loss += sequence_loss(lbl, out)

            out = sample_from_out(out)

            if t >= len(responses):
                responses.append([out[has_remaining.index(i),:] if i in has_remaining else None for i in range(len(sequences))])
            else:
                responses[t] = [out[has_remaining.index(i),:] if i in has_remaining else None for i in range(len(sequences))]

            for s, ps in zip(state, partial_state):
                for ii,i in enumerate(has_remaining):
                    s[i] = ps[ii]

            if window_t+1 == config.seq_stride_len:
                state_to_transfer = [e.detach() for e in state]

        state = state_to_transfer

        responses = [[r.detach() if r is not None else None for r in resp] if t>=window_start else resp for t,resp in enumerate(responses)]

    if training_run:
        loss.backward()
        return float(loss)

    else:

        if len(sequences) == 1:
            for t_extra in range(extra_steps):
                t = max_seq_len+t_extra-1
                prev_responses = [response[0] for response in reversed(responses[-(config.hm_steps_back+1):])]
                for i in range(1, config.hm_steps_back+1):
                    if len(sequences[0][t-1:t]):
                        prev_responses[i-1] = sequences[0][t-1]
                inp = cat([response.view(1,-1) for response in prev_responses],dim=1)
                out, state = prop_model(model, state, inp)
                out = sample_from_out(out)
                responses.append([out.view(-1)])

            responses = stack([ee for e in responses for ee in e], dim=0)

        return float(loss), responses


def sequence_loss(label, out, do_stack=False):

    if do_stack:
        label = stack(label,dim=0)
        out = stack(out, dim=0)

    loss = pow(label-out, 2) if config.loss_squared else (label-out).abs()

    return loss.sum()


def sgd(model, lr=None, batch_size=None):

    if not lr: lr = config.learning_rate
    if not batch_size: batch_size = config.batch_size

    with no_grad():

        for layer in model:
            for param in layer._asdict().values():
                if param.requires_grad:

                    param.grad /=batch_size

                    if config.gradient_clip:
                        param.grad.clamp(min=-config.gradient_clip,max=config.gradient_clip)

                    param -= lr * param.grad
                    param.grad = None


moments, variances, ep_nr = [], [], 0

def adaptive_sgd(model, lr=None, batch_size=None,
                 alpha_moment=0.9,alpha_variance=0.999,epsilon=1e-8,
                 grad_scaling=False):

    if not lr: lr = config.learning_rate
    if not batch_size: batch_size = config.batch_size

    global moments, variances, ep_nr
    if not (moments and variances):
        moments = [[zeros(weight.size()) for weight in layer._asdict().values()] for layer in model]
        variances = [[zeros(weight.size()) for weight in layer._asdict().values()] for layer in model]
        if config.use_gpu:
            moments = [[e2.cuda() for e2 in e1] for e1 in moments]
            variances = [[e2.cuda() for e2 in e1] for e1 in variances]
    ep_nr +=1

    with no_grad():
            for _, layer in enumerate(model):
                for __, weight in enumerate(layer._asdict().values()):
                    if weight.requires_grad:

                        lr_ = lr
                        weight.grad /= batch_size

                        if moments:
                            moments[_][__] = alpha_moment * moments[_][__] + (1-alpha_moment) * weight.grad
                            moment_hat = moments[_][__] / (1-alpha_moment**(ep_nr+1))
                        if variances:
                            variances[_][__] = alpha_variance * variances[_][__] + (1-alpha_variance) * weight.grad**2
                            variance_hat = variances[_][__] / (1-alpha_variance**(ep_nr+1))
                        if grad_scaling:
                            lr_ *= norm(weight)/norm(weight.grad)

                        weight -= lr_ * (moment_hat if moments else weight.grad) / ((sqrt(variance_hat)+epsilon) if variances else 1)
                        weight.grad = None


def load_model(path=None, fresh_meta=None):
    if not path: path = config.model_path
    if not fresh_meta: fresh_meta = config.fresh_meta
    path = path+'.pk'
    obj = pickle_load(path)
    if obj:
        model, meta, configs = obj
        if config.use_gpu:
            TorchModel(model).cuda()
        global moments, variances, ep_nr
        if fresh_meta:
            moments, variances, ep_nr = [], [], 0
        else:
            moments, variances, ep_nr = meta
            if config.use_gpu:
                moments = [[e2.cuda() for e2 in e1] for e1 in moments]
                variances = [[e2.cuda() for e2 in e1] for e1 in variances]
        for k_saved, v_saved in configs:
            v = getattr(config, k_saved)
            if v != v_saved:
                print(f'config conflict resolution: {k_saved} {v} -> {v_saved}')
                setattr(config, k_saved, v_saved)
        return model

def save_model(model, path=None):
    from warnings import filterwarnings
    filterwarnings("ignore")
    if not path: path = config.model_path
    path = path+'.pk'
    if config.use_gpu:
        moments_ = [[e2.detach().cuda() for e2 in e1] for e1 in moments]
        variances_ = [[e2.detach().cuda() for e2 in e1] for e1 in variances]
        meta = [moments_, variances_]
        model = pull_copy_from_gpu(model)
    else:
        meta = [moments, variances]
    meta.append(ep_nr)
    configs = [[field,getattr(config,field)] for field in dir(config) if field in config.config_to_save]
    pickle_save([model,meta,configs],path)


def empty_state(model, batch_size=1):
    states = []
    for layer in model:
        if type(layer) != FF:
            state = zeros(batch_size, getattr(layer,layer._fields[0]).size(1))
            if type(layer) == LSTM and prop_layer[LSTM] == prop_Llayer:
                state = cat([state]*2,dim=1)
            if config.use_gpu: state = state.cuda()
            states.append(state)
    return states


def collect_grads(model):
    grads = [zeros(param.size()) for layer in model for param in layer._asdict().values()]
    if config.use_gpu: grads = [e.cuda() for e in grads]
    ctr = -1
    for layer in model:
        for field in layer._fields:
            ctr += 1
            param = getattr(layer,field)
            if param.requires_grad:
                grads[ctr] += param.grad
                param.grad = None

    return grads

def give_grads(model, grads):
    ctr = -1
    for layer in model:
        for field in layer._fields:
            ctr += 1
            param = getattr(layer,field)
            if param.grad:
                param.grad += grads[ctr]
            else: param.grad = grads[ctr]





from torch.nn import Module, Parameter

class TorchModel(Module):

    def __init__(self, model):
        super(TorchModel, self).__init__()
        for layer_name, layer in enumerate(model):
            for field_name, field in layer._asdict().items():
                if type(field) != Parameter:
                    field = Parameter(field)
                setattr(self,f'layer{layer_name}_field{field_name}',field)
            setattr(self,f'layertype{layer_name}',type(layer))

            model[layer_name] = (getattr(self, f'layertype{layer_name}')) \
                (*[getattr(self, f'layer{layer_name}_field{field_name}') for field_name in getattr(self, f'layertype{layer_name}')._fields])
        self.model = model

    def forward(self, states, inp):
        prop_model(self.model, states, inp)


def pull_copy_from_gpu(model):
    return [type(layer)(*[weight.detach().cpu() for weight in layer._asdict().values()]) for layer in model]
