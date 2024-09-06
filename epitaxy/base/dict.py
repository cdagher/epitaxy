from typing import Dict

import jax
from jax import numpy as jnp

import equinox as eqx
from equinox import Module

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations

d_layers: Dict[Module, keras.layers.Layer] = {
    eqx.nn.Linear: layers.Dense,
    eqx.nn.Conv: layers.Conv,
    eqx.nn.Conv1d: layers.Conv1D,
    eqx.nn.Conv2d: layers.Conv2D,
    eqx.nn.Conv3d: layers.Conv3D,
    # TODO: Implement ConvTranspose
    eqx.nn.ConvTranspose1d: layers.Conv1DTranspose,
    eqx.nn.ConvTranspose2d: layers.Conv2DTranspose,
    eqx.nn.ConvTranspose3d: layers.Conv3DTranspose,
    eqx.nn.LSTMCell: layers.LSTMCell,
    eqx.nn.GRUCell: layers.GRUCell,
    eqx.nn.Dropout: layers.Dropout,
    eqx.nn.MultiheadAttention: layers.MultiHeadAttention,
    eqx.nn.MaxPool1d: layers.MaxPooling1D,
    eqx.nn.MaxPool2d: layers.MaxPooling2D,
    eqx.nn.MaxPool3d: layers.MaxPooling3D,
    eqx.nn.AvgPool1d: layers.AveragePooling1D,
    eqx.nn.AvgPool2d: layers.AveragePooling2D,
    eqx.nn.AvgPool3d: layers.AveragePooling3D,
    eqx.nn.BatchNorm: layers.BatchNormalization
}

d_activations: Dict[str, keras.layers.Activation] = {
    jax.nn.relu: layers.ReLU,
    jax.nn.leaky_relu: layers.LeakyReLU,
    eqx.nn.PReLU: layers.PReLU,
    jax.nn.sigmoid: activations.Sigmoid,
    jax.nn.tanh: activations.Tanh,
    jax.nn.elu: layers.ELU,
    jax.nn.softplus: activations.Softplus,
    jax.nn.softmax: activations.Softmax,
    jnp.ravel: layers.Flatten
}
