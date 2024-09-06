from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from jaxtyping import Array, Float, Int

import jax
from jax import numpy as jnp

import numpy as np

import equinox as eqx
from equinox import Module

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations

from .dict import *


class Translator:

    def __call__(self, module: Module) -> Union[layers.Layer, Callable]:
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
        
        if issubclass(type(module), eqx.nn.Linear):
            return self.__LinearTranslator(module)

    def __LinearTranslator(self, module: eqx.nn.Linear) -> layers.Dense:
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
            kernel_initializer=tf.constant_initializer(np.array(weights).T),
            bias_initializer=tf.constant_initializer(np.array(biases)) if use_bias else 'zeros'
        )

        ret.build(input_shape=(1, in_features))

        return ret
    
    def __ConvTranslator(module: eqx.nn.Conv) -> layers.Layer:
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
            bias_initializer=tf.constant_initializer(biases) if use_bias else 'zeros'
        )

        ret.build(input_shape=(1, in_channels, *kernel_size))

        return ret
    
    def __PoolTranslator(module: eqx.nn.Pool) -> layers.Layer:
        if not issubclass(type(module), eqx.nn.Pool):
            raise ValueError(f"Module {type(module)} is not a Pool module.")
        
        if issubclass(type(module), eqx.nn.MaxPool):
            pool_type = 'max'
        elif issubclass(type(module), eqx.nn.AvgPool):
            pool_type = 'avg'

        kernel_size: Tuple[Int, ...] = module.kernel_size
        stride: Tuple[Int, ...] = module.stride
        padding: Tuple[Tuple[Int, Int], ...] = module.padding
        use_ceil: bool = module.use_ceil

        spatial_dims = module.num_spatial_dims

        if isinstance(padding, int):
            if padding == kernel_size:
                padding = 'same'
        else:
            padding = 'valid'
            print(f"Warning: Unsupported padding amount '{padding}'. Using 'valid' padding instead.")

        if spatial_dims == 1:
            if pool_type == 'max':
                ret = layers.MaxPooling1D(
                    pool_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    data_format=None,
                    name=None
                )
            else:
                ret = layers.AveragePooling1D(
                    pool_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    data_format=None,
                    name=None
                )
        elif spatial_dims == 2:
            if pool_type == 'max':
                ret = layers.MaxPooling2D(
                    pool_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    data_format=None,
                    name=None
                )
            else:
                ret = layers.AveragePooling2D(
                    pool_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    data_format=None,
                    name=None
                )
        elif spatial_dims == 3:
            if pool_type == 'max':
                ret = layers.MaxPooling3D(
                    pool_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    data_format=None,
                    name=None
                )
            else:
                ret = layers.AveragePooling3D(
                    pool_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    data_format=None,
                    name=None
                )
        else:
            raise ValueError(f"Unsupported number of spatial dimensions: {spatial_dims}.")
        
        return ret
    
    def __BatchNormTranslator(module: eqx.nn.BatchNorm) -> layers.BatchNormalization:
        if not issubclass(type(module), eqx.nn.BatchNorm):
            raise ValueError(f"Module {type(module)} is not a BatchNorm module.")
        
        raise NotImplementedError("BatchNormTranslator not yet implemented.")
           
