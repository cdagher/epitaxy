from epitaxy.base import *
from epitaxy.base.translators import ModelTranslator, ModelType
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


def test_SequentialTranslator():
    class sequential_model(Module):
        layers: List[Module]
        in_size: int = eqx.field(static=True)
        out_size: int = eqx.field(static=True)

        def __init__(self, in_size, out_size):
            self.in_size = in_size
            self.out_size = out_size
            self.layers = [
                eqx.nn.Linear(in_size, 20, use_bias=True, key=jr.PRNGKey(0)),
                eqx.nn.Linear(20, 10, use_bias=True, key=jr.PRNGKey(0)),
                eqx.nn.Linear(10, out_size, use_bias=False, key=jr.PRNGKey(0))
            ]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    model = sequential_model(10, 5)
    print(f"Model Type: {type(model)}")
    translator = ModelTranslator()
    translation = translator(model, ModelType.SEQUENTIAL)

    # Check that the output is the same
    x = np.ones((1, 10))
    y = translation(x).numpy().squeeze()
    y_expected = model(x.squeeze())

    assert np.allclose(y, y_expected)