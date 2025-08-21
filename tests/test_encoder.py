import unittest

import numpy

from mininear.encoder import Encoder, _ALPHABET

try:
    from importlib.resources import files as resource_files
except ImportError:
    from importlib_resources import files as resource_files

class TestEncoder(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.sequences = []
        g = numpy.random.Generator(numpy.random.PCG64(seed=42))
        sizes = g.poisson(300, 10)
        for i, s in enumerate(sizes):
            seq = ''.join(_ALPHABET[g.choice(len(_ALPHABET))] for _ in range(s))
            cls.sequences.append(seq)
        with resource_files(__package__).joinpath("embeddings.npz").open("rb") as src:
            embeddings = numpy.load(src)
            cls.embeddings = [ embeddings[k] for k in sorted(embeddings, key=int) ]

    def test_embedding(self):
        encoder = Encoder()
        for i, seq in enumerate(self.sequences):
            out = encoder.encode_sequence(seq)
            self.assertTrue(numpy.all(numpy.isclose(out, self.embeddings[i], atol=1e-5)))

