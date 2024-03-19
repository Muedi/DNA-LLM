import ripser
import numpy as np
import torch
import persim

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


import torch
from x_transformers import XTransformer

model = XTransformer(
    dim = 512,
    enc_num_tokens = 256,
    enc_depth = 6,
    enc_heads = 8,
    enc_max_seq_len = 1024,
    dec_num_tokens = 256,
    dec_depth = 6,
    dec_heads = 8,
    dec_max_seq_len = 1024,
    tie_token_emb = True      # tie embeddings of encoder and decoder
)

src = torch.randint(0, 256, (1, 1024))
src_mask = torch.ones_like(src).bool()
tgt = torch.randint(0, 256, (1, 1024))

loss = model(src, tgt, mask = src_mask) # (1, 1024, 512)
loss.backward()
