import ripser
import numpy as np
import torch

class PersistentHomologyLayer():
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
     #   print(points[0])
        dgm = ripser.ripser(points, maxdim=2)['dgms']
        return dgm[1]