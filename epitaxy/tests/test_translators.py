from epitaxy.base import *
from epitaxy.base.translators import Translator
import pytest

from typing import Dict, List, Tuple, Any, Optional, Union
from jaxtyping import Array, Float, Int

import jax
from jax import numpy as jnp
from jax import random as jr

import numpy as np

import equinox as eqx
from equinox import Module

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations


def test_LinearTranslator():
    linear = eqx.nn.Linear(10, 5, use_bias=True, key=jr.PRNGKey(0))
    translator = Translator()
    translation = translator(linear)

    # Check that the translation is a well-formed Dense layer
    assert issubclass(type(translation), layers.Dense)
    assert translation.units == 5
    assert translation.activation == activations.linear
    assert translation.use_bias == True

    # Check that the weights and biases are the same
    assert np.allclose(translation.kernel, np.array(linear.weight.T))
    assert np.allclose(translation.bias, np.array(linear.bias))

    # Check that the output is the same
    x = np.ones((1, 10))
    y = translation(x).numpy().squeeze()
    y_expected = linear(x.squeeze())

    assert np.allclose(y, y_expected)

def test_Conv1DTranslator():
    conv = eqx.nn.Conv1d(3, 5, 3, use_bias=True, key=jr.PRNGKey(0))
    translator = Translator()
    translation = translator(conv)

    # Check that the translation is a well-formed Conv1D layer
    assert issubclass(type(translation), layers.Conv1D)
    assert translation.filters == 5
    assert translation.kernel_size == (3,)
    assert translation.strides == (1,)
    assert translation.activation == activations.linear
    assert translation.use_bias == True

    # build the layer
    translation.build(input_shape=(1, 3, 3))

    # Check that the weights and biases are the same
    assert np.allclose(translation.kernel, np.array(conv.weight).T)
    assert np.allclose(translation.bias, np.array(conv.bias.reshape(5,)))

    # Check that the output is the same
    x = np.ones((1, 3, 3))
    y = translation(x).numpy().squeeze()
    y_expected = conv(x.squeeze()).squeeze()

    assert np.allclose(y, y_expected)

def test_Conv2DTranslator():
    conv = eqx.nn.Conv2d(3, 5, 3, use_bias=True, key=jr.PRNGKey(0))
    translator = Translator()
    translation = translator(conv)

    # Check that the translation is a well-formed Conv2D layer
    assert issubclass(type(translation), layers.Conv2D)
    assert translation.filters == 5
    assert translation.kernel_size == (3, 3)
    assert translation.strides == (1, 1)
    assert translation.activation == activations.linear
    assert translation.use_bias == True

    # build the layer
    translation.build(input_shape=(1, 3, 3, 3))

    # Check that the weights and biases are the same
    assert np.allclose(translation.kernel, np.array(conv.weight).T)
    assert np.allclose(translation.bias, np.array(conv.bias.reshape(5,)))

    # Check that the output is the same
    x = np.ones((1, 3, 3, 3))
    y = translation(x).numpy().squeeze()
    y_expected = conv(x.squeeze()).squeeze()
    
    assert np.allclose(y, y_expected)

def test_Conv3DTranslator():
    conv = eqx.nn.Conv3d(3, 5, 3, use_bias=True, key=jr.PRNGKey(0))
    translator = Translator()
    translation = translator(conv)

    # Check that the translation is a well-formed Conv3D layer
    assert issubclass(type(translation), layers.Conv3D)
    assert translation.filters == 5
    assert translation.kernel_size == (3, 3, 3)
    assert translation.strides == (1, 1, 1)
    assert translation.activation == activations.linear
    assert translation.use_bias == True

    # build the layer
    translation.build(input_shape=(1, 3, 3, 3, 3))

    # Check that the weights and biases are the same
    assert np.allclose(translation.kernel, np.array(conv.weight).T)
    assert np.allclose(translation.bias, np.array(conv.bias.reshape(5,)))

    # Check that the output is the same
    x = np.ones((1, 3, 3, 3, 3))
    y = translation(x).numpy().squeeze()
    y_expected = conv(x.squeeze()).squeeze()
    
    assert np.allclose(y, y_expected)

def test_MaxPool1DTranslator():
    pool = eqx.nn.MaxPool1d(3, 2, padding=0)
    translator = Translator()
    translation = translator(pool)

    # Check that the translation is a well-formed MaxPooling1D layer
    assert issubclass(type(translation), layers.MaxPooling1D)
    assert translation.pool_size == (3,)
    assert translation.strides == (2,)
    assert translation.padding == 'valid'

    # Check that the output is the same
    x = np.ones((1, 10, 1))
    y = translation(x).numpy().squeeze()
    y_expected = pool(x.reshape((1, 10))).squeeze()

    assert np.allclose(y, y_expected)

def test_MaxPool2DTranslator():
    pool = eqx.nn.MaxPool2d(3, 2, padding=0)
    translator = Translator()
    translation = translator(pool)

    # Check that the translation is a well-formed MaxPooling2D layer
    assert issubclass(type(translation), layers.MaxPooling2D)
    assert translation.pool_size == (3, 3)
    assert translation.strides == (2, 2)
    assert translation.padding == 'valid'

    # Check that the output is the same
    x = np.ones((1, 10, 10, 1))
    y = translation(x).numpy().squeeze()
    y_expected = pool(x.reshape((1, 10, 10))).squeeze()

    assert np.allclose(y, y_expected)

def test_MaxPool3DTranslator():
    pool = eqx.nn.MaxPool3d(3, 2, padding=0)
    translator = Translator()
    translation = translator(pool)

    # Check that the translation is a well-formed MaxPooling3D layer
    assert issubclass(type(translation), layers.MaxPooling3D)
    assert translation.pool_size == (3, 3, 3)
    assert translation.strides == (2, 2, 2)
    assert translation.padding == 'valid'

    # Check that the output is the same
    x = np.ones((1, 10, 10, 10, 1))
    y = translation(x).numpy().squeeze()
    y_expected = pool(x.reshape((1, 10, 10, 10))).squeeze()

    assert np.allclose(y, y_expected)

def test_AvgPool1DTranslator():
    pool = eqx.nn.AvgPool1d(3, 2, padding=0)
    translator = Translator()
    translation = translator(pool)

    # Check that the translation is a well-formed AveragePooling1D layer
    assert issubclass(type(translation), layers.AveragePooling1D)
    assert translation.pool_size == (3,)
    assert translation.strides == (2,)
    assert translation.padding == 'valid'

    # Check that the output is the same
    x = np.ones((1, 10, 1))
    y = translation(x).numpy().squeeze()
    y_expected = pool(x.reshape((1, 10))).squeeze()

    assert np.allclose(y, y_expected)

def test_AvgPool2DTranslator():
    pool = eqx.nn.AvgPool2d(3, 2, padding=0)
    translator = Translator()
    translation = translator(pool)

    # Check that the translation is a well-formed AveragePooling2D layer
    assert issubclass(type(translation), layers.AveragePooling2D)
    assert translation.pool_size == (3, 3)
    assert translation.strides == (2, 2)
    assert translation.padding == 'valid'

    # Check that the output is the same
    x = np.ones((1, 10, 10, 1))
    y = translation(x).numpy().squeeze()
    y_expected = pool(x.reshape((1, 10, 10))).squeeze()

    assert np.allclose(y, y_expected)

def test_AvgPool3DTranslator():
    pool = eqx.nn.AvgPool3d(3, 2, padding=0)
    translator = Translator()
    translation = translator(pool)

    # Check that the translation is a well-formed AveragePooling3D layer
    assert issubclass(type(translation), layers.AveragePooling3D)
    assert translation.pool_size == (3, 3, 3)
    assert translation.strides == (2, 2, 2)
    assert translation.padding == 'valid'

    # Check that the output is the same
    x = np.ones((1, 10, 10, 10, 1))
    y = translation(x).numpy().squeeze()
    y_expected = pool(x.reshape((1, 10, 10, 10))).squeeze()

    assert np.allclose(y, y_expected)

