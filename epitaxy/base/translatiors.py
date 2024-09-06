from typing import Dict, List, Tuple, Any, Optional, Union
from jaxtyping import Array, Float, Int

from abc import ABC

import jax
from jax import numpy as jnp

import equinox as eqx
from equinox import Module

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations

from dict import *


class Translator:
    # _module: Module
    # _translation_types: Tuple = (layers.Layer, activations.Activation)
    # _type: Union[layers.Layer, activations.Activation]
    # _translation: Union[layers.Layer, activations.Activation]

    def __call__(self, module: Module) -> Union[layers.Layer, activations.Activation]:
        if type(module) in d_layers.keys():
            # get the weights and biases (if any) from the Module
            weights: Array = module.weight
            biases: Array = module.bias
            use_bias: bool = module.use_bias
            in_features: Int = module.in_features
            out_features: Int = module.out_features
            pass    
        elif type(module) in d_activations.keys():
            # get the activation function from the module
            pass
        else:
            raise ValueError(f"Module {type(module)} not found in translation dictionaries.")
        

    def __LinearTranslator(module: eqx.nn.Linear) -> layers.Dense:
        if not issubclass(type(module), eqx.nn.Linear):
            raise ValueError(f"Module {type(module)} is not a Linear module.")
        
        weights: Array = module.weight
        biases: Array = module.bias
        use_bias: bool = module.use_bias
        in_features: Int = module.in_features
        out_features: Int = module.out_features

        ret = layers.Dense(
            units=out_features,
            activation=None,
            use_bias=use_bias,
            input_dim=(1, in_features),
            kernel_initializer=tf.constant_initializer(weights),
            bias_initializer=tf.constant_initializer(biases) if use_bias else 'zeros'
        )

        return ret
    
    def __ConvTranslator(module: eqx.nn.Conv) -> layers.Conv:
        if not issubclass(type(module), eqx.nn.Conv):
            raise ValueError(f"Module {type(module)} is not a Conv module.")
        
        weights: Array = module.weight
        biases: Array = module.bias
        use_bias: bool = module.use_bias
        in_channels: Int = module.in_channels
        out_channels: Int = module.out_channels
        kernel_size: Tuple[Int, ...] = module.kernel_size
        stride: Tuple[Int, ...] = module.stride
        groups: Int = module.groups
        dilation: Tuple[Int, ...] = module.dilation
        padding_mode: str = module.padding_mode
        num_spatial_dims: Int = module.num_spatial_dims

        if padding_mode == 'ZEROS':
            padding = 'same'
        else:
            padding = 'valid'
            print(f"Warning: Unrecognized padding mode '{padding_mode}'. Using 'valid' padding instead.")
            
        
        ret = layers.Conv(
            rank=num_spatial_dims,
            filters=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding_mode,
            data_format=None,
            dilation_rate=dilation,
            groups=groups,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=tf.constant_initializer(weights),
            bias_initializer=tf.constant_initializer(biases) if use_bias else 'zeros',
            activation=None
        )

        ret.build(input_shape=(1, in_channels, *kernel_size))

        return ret
    
    def __PoolTranslator(module: eqx.nn.Pool) -> layers.Pool:
        if not issubclass(type(module), eqx.nn.Pool):
            raise ValueError(f"Module {type(module)} is not a Pool module.")
