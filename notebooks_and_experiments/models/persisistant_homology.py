from pyexpat import model
import ripser
import numpy as np
import torch
import persim

###### lucid rains code #################
from math import ceil, log
from typing import Optional, Union, Tuple, Callable

import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, pack, unpack

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def identity(t, *args, **kwargs):
    return t

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else (t,) * length

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

# for variable lengthed prefixes

def align_right(t, lens, pad_id = 0):
    batch, seq_len, device, dtype = *t.shape, t.device, t.dtype

    assert lens.ndim == 1 and lens.shape[0] == batch
    assert lens.amax() <= seq_len

    pad_lens = seq_len - lens
    max_pad_len = pad_lens.amax()

    batch_arange = torch.arange(batch, device = device, dtype = torch.long)[..., None]
    prompt_len_arange = torch.arange(seq_len, device = device, dtype = torch.long)

    t = F.pad(t, (max_pad_len, 0), value = 0)
    offset = max_pad_len - pad_lens

    aligned = t[batch_arange, prompt_len_arange + offset[..., None]]
    return aligned

# nucleus

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk

def top_k(logits, frac_num_tokens = 0.1, k = None):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# top_a

def top_a(logits, min_p_pow = 2.0, min_p_ratio = 0.02):
    probs = F.softmax(logits, dim = -1)
    max_probs = torch.amax(probs, dim = -1, keepdim = True)
    limit = torch.pow(max_probs, min_p_pow) * min_p_ratio
    return torch.where(probs < limit, float('-inf'), logits)

# contrastive decoding function

def contrastive_decode_fn(
    expert_logits,
    amateur_logits,
    alpha = 0.1,
    beta = 0.5
):
    """
    Appendix A Algorithm 2
    https://arxiv.org/abs/2309.09117
    """

    cutoff = log(alpha) + expert_logits.amax(dim = -1, keepdim = True)
    diffs = (1 + beta) * expert_logits - beta * amateur_logits
    contrastive_decode_logits = diffs.masked_fill(expert_logits < cutoff, -torch.finfo(expert_logits.dtype).max)
    return contrastive_decode_logits

# autoregressive wrapper class

class AutoregressiveWrapper(Module):
    def __init__(
        self,
        net,
        ignore_index = -100,
        pad_value = 0,
        mask_prob = 0.,
        add_attn_z_loss = False
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

        # paper shows masking (MLM) in conjunction with autoregressive decoder-only training leads to big improvements https://arxiv.org/abs/2210.13432
        assert mask_prob < 1.
        self.mask_prob = mask_prob

        # whether to add router z-loss
        self.add_attn_z_loss = add_attn_z_loss

 

    def forward(self, x, return_outputs = False, **kwargs):
        seq, ignore_index, add_attn_z_loss = x.shape[1], self.ignore_index, self.add_attn_z_loss

        inp, target = x[:, :-1], x[:, 1:]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        if self.mask_prob > 0.:
            rand = torch.randn(inp.shape, device = x.device)
            rand[:, 0] = -torch.finfo(rand.dtype).max # first token should not be masked out
            num_mask = min(int(seq * self.mask_prob), seq - 1)
            indices = rand.topk(num_mask, dim = -1).indices
            mask = ~torch.zeros_like(inp).scatter(1, indices, 1.).bool()
            kwargs.update(self_attn_kv_mask = mask)

        logits, cache = self.net(
            inp,
            return_intermediates = True,
            return_attn_z_loss = add_attn_z_loss,
            **kwargs
        )
        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index
        )
        ph_layer = PersistentHomologyLayer()
        model_outputs = rearrange(logits, 'b n c -> b n c')
        # convert to list
        model_outputs = model_outputs.tolist()
        target = target.tolist()
        dgm1 = ph_layer(model_outputs)
        dgm2 = ph_layer(target)
        ph_loss = persim.bottleneck(dgm1, dgm2)
        # convert to tensor
        ph_loss = torch.tensor(ph_loss)

        loss = loss + ph_loss

        if add_attn_z_loss:
            loss = loss + cache.attn_z_loss

        if not return_outputs:
            return loss

        return loss, (logits, cache)
    
##### Personal code ####################
# persistent homology layer
class PersistentHomologyLayer(torch.nn.Module):
    def __init__(self, sample_rate=7):
        super(PersistentHomologyLayer, self).__init__()
        self.sample_rate = sample_rate
        self.nucleotide_mapping = {
            28: np.array([1, 0, 0, 0]),
            29: np.array([0, 1, 0, 0]),
            30: np.array([0, 0, 1, 0]),
            31: np.array([0, 0, 0, 1])
        }

    def encode_nucleotide_to_vector(self, nucleotide):
        return self.nucleotide_mapping.get(nucleotide)

    def chaos_4d_representation(self, dna_sequence):
        points = [self.encode_nucleotide_to_vector(dna_sequence[0])]
        for nucleotide in dna_sequence[1:]:
            vector = self.encode_nucleotide_to_vector(nucleotide)
            if vector is None:
                continue
            next_point = 0.5 * (points[-1] + vector)
            points.append(next_point)
        return np.array(points)

    def forward(self, dna_sequences):
        c4dr_points = []
        points = self.chaos_4d_representation(dna_sequences)
        dgm = ripser.ripser(points, maxdim=2)['dgms']
        return dgm[1]
    
# persistant homology loss
def ph_loss(dgm1, dgm2):
    ph_loss = persim.bottleneck(dgm1, dgm2)
    return ph_loss


dna_sequences = [30, 31, 30, 31, 28, 31, 31, 30, 28, 31, 29, 31, 31, 29, 30, 31, 30,
       31, 31, 29, 31, 30, 30, 28, 28, 29, 31, 29, 28, 30, 30, 31, 28, 31,
       30, 31, 31, 30, 31, 31, 29, 30, 29, 31, 30, 29, 28, 30, 31, 28, 30,
       28, 28, 30, 29, 30, 29, 29, 31, 29, 29, 28, 30, 29]



##### example usage of the persistent homology layer ####
import torch
from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 512,
        depth = 12,
        heads = 8
    )
)

model = AutoregressiveWrapper(
    model
)

# mock data

x = torch.randint(0, 20000, (1, 1024))

# derive cross entropy loss, masking all taken care of

loss = model(x)
loss.backward()