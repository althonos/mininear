from __future__ import annotations

import typing

import numpy

from .layers import ResNet, LinearLayer, ResConvLayer, ConvLayer

try:
    from importlib.resources import files as resource_files
except ImportError:
    from importlib_resources import files as resource_files

if typing.TYPE_CHECKING:
    from .utils import ArrayNx256


_ALPHABET = 'XARNDCEQGHILKMFPSTWYV'
_TABLE = bytes.maketrans(
    bytearray(range(256)),
    bytearray([   
        (chr(c).upper() not in 'XUBJ0') * _ALPHABET.find(chr(c).upper()) & 0xff 
        for c in range(256) 
    ])
)


class Encoder:
    """An encoder for converting a protein sequence to NEAR embedding.
    """

    model: ResNet

    def __init__(self) -> None:
        with resource_files(__package__).joinpath("weights.npz").open("rb") as f:
            params = numpy.load(f)
            self.model = ResNet(
                LinearLayer(params['embedding_layer.weight'].T),
                [
                    ResConvLayer(
                        ConvLayer(
                            params[f'conv_layers.{i}.conv1.weight'],
                            params[f'conv_layers.{i}.conv1.bias'],
                        ),
                        ConvLayer(
                            params[f'conv_layers.{i}.conv2.weight'],
                            params[f'conv_layers.{i}.conv2.bias'],
                        )
                    )
                    for i in range(8)
                ]
            )

    def encode_sequence(self, sequence: str) -> ArrayNxM[numpy.floating]:
        raw = bytearray(sequence, 'ascii')
        encoded = raw.translate(_TABLE)
        array = numpy.asarray(encoded)
        onehot = numpy.eye(len(_ALPHABET) + 1)[array][:, :-1]
        return self.model(onehot)

