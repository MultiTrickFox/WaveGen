import config
from ext import pickle_save, pickle_load, now

from torch import tensor, Tensor, cat, stack
from torch import zeros, ones, eye, randn
from torch import sigmoid, tanh, relu, softmax
from torch import pow, log, sqrt, norm
from torch import float32, no_grad
from torch.nn.init import xavier_normal_

from collections import namedtuple
from copy import deepcopy
from math import ceil

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

    forget = sigmoid(inp@layer.wf + layer.bf)
    keep   = sigmoid(inp@layer.wk + layer.bk)
    interm = tanh   (inp@layer.wi + layer.bi)
    show   = sigmoid(inp@layer.ws + layer.bs)

    state = forget*state + keep*interm
    out = show*tanh(state)

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


def make_submodel(info=None):

    if not info: info = [len(config.frequencies_range),'l',len(config.frequencies_range)]

    layer_sizes = [e for e in info if type(e)==int]
    layer_types = [e for e in info if type(e)==str]

    return [make_layer[layer_type](layer_sizes[i], layer_sizes[i+1]) for i,layer_type in enumerate(layer_types)]

def prop_submodel(model, states, inp):
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

    return out, new_states


def make_model(hls=None):
    if not hls:
        hls = [['l'], ['l'], ['f'], ['f']] if not config.sub_models else config.sub_models
    return {
        'originator': make_submodel([len(config.frequencies_range)] + hls[0] + [config.ticket_size]),
        'creator'   : make_submodel([config.ticket_size] + hls[1] + [len(config.frequencies_range)]),
        'attender'  : make_submodel([config.ticket_size*2] + hls[2] + [1]),
        'keeper'    : make_submodel([len(config.frequencies_range)*2] + hls[3] + [1]),
    }

def prop_model(model, states, inp, prev_tickets, prev_outs):

    ticket, originator_states  = prop_submodel(model['originator'], states[0], inp)
    new, creator_states = prop_submodel(model['creator'], states[1], ticket)

    if prev_tickets is not None:
        attended = prop_attender(model['attender'], ticket, prev_tickets, prev_outs)
        keep, keeper_states = prop_submodel(model['keeper'], states[2], cat([new,attended],dim=1))
        keep = sigmoid(keep)
        out = keep*attended + (1-keep)*new
    else:
        keeper_states = states[2]
        out = new

    return out, ticket, [originator_states,creator_states,keeper_states]


def prop_attender(attender, ticket, prev_tickets, prev_outs):

    ticket = ticket.view(1,ticket.size(0),ticket.size(1))
    ticket = ticket.repeat(prev_tickets.size(0), 1,1)
    inp = cat([ticket,prev_tickets], dim=2)

    attentions, _ = prop_submodel(attender, [], inp)
    attended_out = (prev_outs * softmax(attentions,dim=0)).sum(0)

    return attended_out


def respond_to(model, sequences, state=None, do_grad=True):

    responses = []
    tickets = []

    loss = 0
    sequences = deepcopy(sequences)
    if not state:
        state = empty_state(model, len(sequences))

    max_seq_len = max(len(sequence) for sequence in sequences)
    hm_windows = ceil(max_seq_len/config.seq_stride_len)
    has_remaining = list(range(len(sequences)))

    for i in range(hm_windows):

        if (i+1)%50==0:
            print(f'\t\t{i}/{hm_windows} @ {now()}')

        window_start = i*config.seq_stride_len
        window_end = min(window_start+config.seq_window_len, max_seq_len)

        for window_t in range(window_end-window_start -1):

            t = window_start+window_t

            has_remaining = [i for i in has_remaining if len(sequences[i][t+1:t+2])]

            inp = stack([sequences[i][t] if window_t<=config.seq_force_len else responses[t-1][i] for i in has_remaining], dim=0)
            lbl = stack([sequences[i][t+1] for i in has_remaining], dim=0)

            partial_state = [[stack([layer_state[i] for i in has_remaining], dim=0) for layer_state in submodel_state] for submodel_state in state]
            if t:
                partial_tickets = cat([cat([ticket[i].view(1,1,-1) for ticket in tickets], dim=0) for i in has_remaining], dim=1)
                partial_responses = cat([cat([response[i].view(1,1,-1) for response in responses], dim=0) for i in has_remaining], dim=1)
            else:
                partial_tickets, partial_responses = None, None

            out, ticket, partial_state = prop_model(model, partial_state, inp, partial_tickets, partial_responses)

            if t >= len(responses):
                responses.append([out[has_remaining.index(i),:] if i in has_remaining else None for i in range(len(sequences))])
                tickets.append([ticket[has_remaining.index(i),:] if i in has_remaining else None for i in range(len(sequences))])

            loss += sequence_loss(lbl, out, do_grad=do_grad)

            for sub_state, sub_partial_state in zip(state, partial_state):
                for s, ps in zip(sub_state, sub_partial_state):
                    for ii,i in enumerate(has_remaining):
                        s[i] = ps[ii]

            if window_t+1 == config.seq_stride_len:
                state_to_transfer = [[ee.detach() for ee in e] for e in state]

        state = state_to_transfer

        responses = [[ee.detach() for ee in e] for e in responses]
        tickets = [[ee.detach() for ee in e] for e in tickets]

    if do_grad:
        return loss
    else:
        if len(sequences) == 1:
            responses = stack([ee for e in responses for ee in e],dim=0)
        return loss, responses


def sequence_loss(label, output, do_stack=False, do_grad=True, retain=True):

    if do_stack:
        label = stack(label,dim=0)
        output = stack(output,dim=0)

    loss = pow(label-output,2) if config.loss_squared else (label-output).abs()
    loss = loss.sum()

    if do_grad:
        loss.backward(retain_graph=retain)

    return float(loss)


def sgd(model, lr, batch_size):

    with no_grad():

        for layer in model:
            for param in layer._asdict().values():
                if param.requires_grad:

                    param.grad /=batch_size

                    if config.gradient_clip:
                        param.grad.clamp(min=-config.gradient_clip,max=config.gradient_clip)

                    param -= lr * param.grad
                    param.grad = None


moments, variances = [], []

def adaptive_sgd(model, epoch_nr, lr, batch_size,
                 alpha_moment=0.9,alpha_variance=0.999,epsilon=1e-8,
                 grad_scaling=False):

    if not lr: lr = config.learning_rate
    if not batch_size: batch_size = config.batch_size

    global moments, variances
    if not (moments and variances):
        moments = [[[zeros(weight.size()) for weight in layer._asdict.values()] for layer in sub] for sub in model]
        variances = [[[zeros(weight.size()) for weight in layer._asdict.values()] for layer in sub] for sub in model]
        if config.use_gpu:
            moments = [[[e3.cuda() for e3 in e2] for e2 in e1] for e1 in moments]
            variances = [[[e3.cuda() for e3 in e2] for e2 in e1] for e1 in variances]

    with no_grad():

        for _, layer in enumerate(model):
            for __, weight in enumerate(layer._asdict().values()):
                if weight.requires_grad:

                    lr_ = lr

                    weight.grad /= batch_size

                    if moments:
                        moments[_][__] = alpha_moment * moments[_][__] + (1-alpha_moment) * weight.grad
                        moment_hat = moments[_][__] / (1-alpha_moment**(epoch_nr+1))
                    if variances:
                        variances[_][__] = alpha_variance * variances[_][__] + (1-alpha_variance) * weight.grad**2
                        variance_hat = variances[_][__] / (1-alpha_variance**(epoch_nr+1))
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
        global moments, variances
        if fresh_meta:
            moments, variances = [], []
        else:
            moments, variances = meta
            if config.use_gpu:
                moments = [[[e3.cuda() for e3 in e2] for e2 in e1] for e1 in moments]
                variances = [[[e3.cuda() for e3 in e2] for e2 in e1] for e1 in variances]
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
    meta = [moments, variances]
    configs = [[field,getattr(config,field)] for field in dir(config) if field in config.config_to_save]
    pickle_save([model,meta,configs],path)


def empty_state(model, batch_size=1):
    model_states = []
    for k,v in model.items():
        if k != 'attender':
            states = []
            for layer in v:
                if type(layer) != FF:
                    state = zeros(batch_size, getattr(layer,layer._fields[0]).size(1))
                    if type(layer) == LSTM and prop_layer[LSTM] == prop_Llayer:
                        state = cat([state]*2,dim=1)
                    if config.use_gpu: state = state.cuda()
                    states.append(state)
            model_states.append(states)
    return model_states


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
        for sub_name, sub in model.items():
            for layer_name, layer in enumerate(sub):
                for field_name, field in layer._asdict().items():
                    if type(field) != Parameter:
                        field = Parameter(field)
                    setattr(self,f'sub{sub_name}_layer{layer_name}_field{field_name}',field)
                setattr(self,f'sub{sub_name}_layertype{layer_name}',type(layer))

                sub[layer_name] = (getattr(self, f'sub{sub_name}_layertype{layer_name}')) \
                                    (*[getattr(self, f'sub{sub_name}_layer{layer_name}_field{field_name}') for field_name in
                                        getattr(self, f'sub{sub_name}_layertype{layer_name}')._fields])
        self.model = model

    def forward(self, states, inp):
        prop_model(self.model, states, inp)
