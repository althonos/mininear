from __future__ import annotations

import abc
import typing

import numpy

from ._conv import conv1D

if typing.TYPE_CHECKING:
    from .utils import ArrayN, ArrayNxM


class Layer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, X: ArrayNxM[numpy.floating]) -> ArrayNxM[numpy.floating]:
        raise NotImplementedError


class ELU(Layer):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def __call__(self, X: ArrayNxM[numpy.floating]):
        return numpy.where(X > 0, X, self.alpha * (numpy.exp(X) - 1))


class LinearLayer(Layer):
    def __init__(
        self,
        weights: ArrayNxM[numpy.floating], 
        biases: Optional[ArrayN[numpy.floating]] = None,
    ):
        self.weights = weights
        self.biases = biases

    def __call__(self, X: ArrayNxM):
        _X = numpy.asarray(X, dtype=numpy.float32)
        out = _X @ self.weights
        if self.biases is not None:
            out += self.biases
        return out


class EmbeddingLayer(Layer):
    def __init__(
        self,
        weights: ArrayNxM[numpy.floating], 
        biases: Optional[ArrayN[numpy.floating]] = None,
        ignore_index=-1,
    ):
        self.weights = weights
        self.biases = biases
        self.ignore_index = ignore_index

    def __call__(self, X: ArrayN):
        _X = numpy.asarray(X)
        mask = _X == self.ignore_index
        out = numpy.take(self.weights, numpy.where(mask, 0, _X), axis=0)
        out = numpy.where(mask[..., None], 0.0, out)
        return out


class ConvLayer(Layer):

    def __init__(
        self,
        weights: ArrayNxM[numpy.floating], 
        biases: Optional[ArrayN[numpy.floating]] = None,
    ):
        self.weights = weights
        self.biases = biases

    def __call__(self, X: ArrayNxM[numpy.floating]):
        out = conv1D(
            X[numpy.newaxis] if X.ndim == 2 else X,
            self.weights.transpose(2, 1, 0), 
            stride=1, 
            pad=(3, 3)
        )
        if self.biases is not None:
            out += self.biases
        return out[0] if X.ndim == 2 else out


class ResConvLayer(Layer):
    def __init__(
        self,
        conv1: ConvLayer,
        conv2: ConvLayer,
    ):
        super().__init__()
        self.conv1 = conv1
        self.act = ELU()
        self.conv2 = conv2

    def __call__(self, X: ArrayNxM[numpy.floating]):
        X += self.conv2(self.act(self.conv1(self.act(X))))
        return X


class ResNet(Layer):
    def __init__(
        self,
        embedding: Layer,
        convolutions: List[Layer],
    ):
        super().__init__()
        self.embedding_layer = embedding
        self.conv_layers = convolutions
   
    def __call__(self, X: ArrayNxM[numpy.floating]) -> ArrayNxM[numpy.floating]:
        e = self.embedding_layer(X)
        for layer in self.conv_layers:
            e = layer(e)
        return e