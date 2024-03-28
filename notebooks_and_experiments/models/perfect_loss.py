import logging
from datasets import load_dataset, Dataset
import pandas as pd
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained('Hack90/virus_pythia_31_1024')
df = load_dataset('Hack90/virus_tiny', split='train').to_pandas()

# Count all non-ATGC nucleotides and calculate the ratio
df['unknown_nucleotides'] = df.Sequence.apply(lambda x: sum(i not in 'ATGC' for i in x))
df['unknown_nucleotide_ratio'] = df.unknown_nucleotides / df.Length

# Log the statistics of unknown nucleotides and their ratio
logging.info(f"\nUnknown nucleotides statistics:\n{df[['unknown_nucleotides', 'unknown_nucleotide_ratio']].describe()}")

# Drop sequences with more than 0% unknown nucleotides
df = df[df.unknown_nucleotide_ratio == 0]

# Select ten samples randomly from each family if the family has more than 10 samples, else select all samples
df = df.groupby('Family').apply(lambda x: x.sample(n=min(10, len(x)))).reset_index(drop=True)

# Log the family value counts and sequence statistics
logging.info(f"\nFamily value counts:\n{df.Family.value_counts()}")
logging.info(f"\nSequence statistics:\n{df.Length.describe()}")

# Function to split sequence into batches of max_length
def batched_sequence(seq, max_length):
    return [seq[i:i+max_length] for i in range(0, len(seq), max_length)]

# Function to clean up the sequence
def clean_up(seq):
    return seq.lower()

# Apply the batched_sequence and clean_up functions to the 'Sequence' column
df['text'] = df['Sequence'].apply(lambda x: batched_sequence(clean_up(x), 1024))

# Explode the 'text' column and drop unnecessary columns
df = df.explode('text')
df = df.drop(columns=['Sequence', 'unknown_nucleotides', 'unknown_nucleotide_ratio', '__index_level_0__']).reset_index()
df = df[df.text.str.len() == 1024]

# Log the first five rows of the DataFrame
logging.info(f"\nFirst five rows of the DataFrame:\n{df.head(5)}")

# Log the total number of rows in the DataFrame and the number of nucleotides
logging.info(f"\nTotal number of rows in the DataFrame: {len(df)}")
logging.info(f"\nTotal number of nucleotides: {df.text.str.len()}")

# Create a dataset from the DataFrame
ds = Dataset.from_pandas(df)

# Function to tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024)

# Tokenize the dataset
tokenized_datasets = ds.map(tokenize_function, batched=True, batch_size=100000)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Log the first example of the tokenized dataset
logging.info(f"\nFirst example of the tokenized dataset:\n{tokenized_datasets[0]}")

# Log the shape of the input_ids and attention_mask tensors
logging.info(f"\nShape of input_ids tensor: {tokenized_datasets['input_ids'][0].shape}")
logging.info(f"\nShape of attention_mask tensor: {tokenized_datasets['attention_mask'][0].shape}")

########## Persistant Homology Loss ##########
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
##### Gudhi topoloss ####
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

 
    @staticmethod
    def forward(self, x, return_outputs = False, **kwargs):
        seq, ignore_index, add_attn_z_loss = x.shape[1], self.ignore_index, self.add_attn_z_loss

        inp, target = x[:, :-1], x[:, 1:]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)
        target = target.clone().detach()
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
        # convert the logits to dna sequences by taking the argmax
        model_outputs = torch.argmax(logits, dim=2).clone().detach()
        print('model_outputs_shape',  model_outputs.shape)
        model_outputs = model_outputs.tolist()[0]
       # print(model_outputs)#, model_outputs.shape)
        target = target.tolist()[0]
        nucl_mapping =  {
            3: np.array([1, 0, 0, 0]),
            4: np.array([0, 1, 0, 0]),
            5: np.array([0, 0, 1, 0]),
            6: np.array([0, 0, 0, 1])
        }
        dgm1 , coverage_1 = ph_layer(model_outputs, 7,nucl_mapping)
        dgm2 , coverage_2 = ph_layer(target, 7,nucl_mapping)
        if coverage_1 < 0.5 or coverage_2 < 0.5:
            print("Coverage is less than 0.5")
            return loss#, (logits, cache)
        ph_loss = persim.bottleneck(dgm1, dgm2)
        # convert to tensor
        ph_loss = torch.tensor(ph_loss).clone().detach().requires_grad_(True)
        
        loss = ph_loss #loss + 
       # ph_loss

        if add_attn_z_loss:
            loss = loss + cache.attn_z_loss

        if not return_outputs:
            return loss

        return loss, (logits, cache)
    
##### Personal code ####################
# persistent homology layer
class PersistentHomologyLayer(torch.autograd.Function):
    def __init__(self, sample_rate=10):
        super(PersistentHomologyLayer, self).__init__()
        self.sample_rate = sample_rate
        self.nucleotide_mapping = {
            3: np.array([1, 0, 0, 0]),
            4: np.array([0, 1, 0, 0]),
            5: np.array([0, 0, 1, 0]),
            6: np.array([0, 0, 0, 1])
        }
    
    @staticmethod
    def encode_nucleotide_to_vector(nucleotide, nucleotide_mapping):
        return nucleotide_mapping.get(nucleotide)
     #   return self.nucleotide_mapping.get(nucleotide)

    @staticmethod
    def chaos_4d_representation(dna_sequence, nucleotide_mapping):
        def encode_nucleotide_to_vector(nucleotide):
            return nucleotide_mapping.get(nucleotide)

        points = [encode_nucleotide_to_vector(dna_sequence[0])]
        if points[0] is None:
            points[0] = np.array([0, 0, 0, 0])
        for nucleotide in dna_sequence[1:]:
            vector = encode_nucleotide_to_vector(nucleotide)
            if vector is None:
                continue
            next_point = 0.5 * (points[-1] + vector)
            points.append(next_point)
        return np.array(points)

    @staticmethod
    def forward(dna_sequences, sample_rate, nucleotide_mapping):
        c4dr_points = []
        points = PersistentHomologyLayer.chaos_4d_representation(dna_sequences, nucleotide_mapping)
        coverage = len(points) / len(dna_sequences)
        points = points[::sample_rate]
        dgm = ripser.ripser(points, maxdim=2)['dgms']
        return dgm[0], coverage
    
# persistant homology loss
def ph_loss(dgm1, dgm2):
    ph_loss = persim.bottleneck(dgm1, dgm2)
    return ph_loss


dna_sequences = [30, 31, 30, 31, 28, 31, 31, 30, 28, 31, 29, 31, 31, 29, 30, 31, 30,
       31, 31, 29, 31, 30, 30, 28, 28, 29, 31, 29, 28, 30, 30, 31, 28, 31,
       30, 31, 31, 30, 31, 31, 29, 30, 29, 31, 30, 29, 28, 30, 31, 28, 30,
       28, 28, 30, 29, 30, 29, 29, 31, 29, 29, 28, 30, 29]

# convert to tensor
seqs = []
for seq in tokenized_datasets['input_ids']:
    seq = torch.tensor(seq)
    seq = seq.reshape(1, 1024)
    seqs.append(seq)


##### example usage of the persistent homology layer ####
import torch
from x_transformers import TransformerWrapper, Decoder #,AutoregressiveWrapper

model = TransformerWrapper(
    num_tokens = 7,
    max_seq_len = 1024,
    attn_layers = Decoder(
        dim = 256,
        depth = 6,
        heads = 2
    )
)

model = AutoregressiveWrapper(
    model,  mask_prob = 0.15
)

# mock data


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# derive cross entropy loss, masking all taken care of
for i in range(0,100):
    print(seqs[i], seqs[i].shape)
       # Zero the gradients
    optimizer.zero_grad()
    loss = model(model, x = seqs[i])
    print(loss)
    loss.backward()
    optimizer.step()

