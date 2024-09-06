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
        if isinstance(module, eqx.nn.Linear):
            return self.__LinearTranslator(module)
        elif isinstance(module, eqx.nn.Conv):
            return self.__ConvTranslator(module)
        elif isinstance(module, eqx.nn.Pool):
            return self.__PoolTranslator(module)
        elif isinstance(module, eqx.nn.BatchNorm):
            return self.__BatchNormTranslator(module)
        else:
            raise ValueError(f"Module {type(module)} not found in translation dictionaries or is unsupported.")

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
    
    def __ConvTranslator(self, module: eqx.nn.Conv) -> layers.Layer:
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
            padding = 'valid'
        else:
            padding = 'valid'
            print(f"Warning: Unrecognized padding mode '{padding_mode}'. Using 'valid' padding instead.")
            
        kwargs = {
            'filters': out_channels,
            'kernel_size': kernel_size,
            'strides': stride,
            'padding': padding,
            'data_format': None,
            'dilation_rate': dilation,
            'groups': groups,
            'activation': None,
            'use_bias': use_bias,
            'kernel_initializer': tf.constant_initializer(np.array(weights).T),
            'bias_initializer': tf.constant_initializer(np.array(biases)) if use_bias else 'zeros'
        }

        if num_spatial_dims == 1:
            ret = layers.Conv1D(**kwargs)
        elif num_spatial_dims == 2:
            ret = layers.Conv2D(**kwargs)
        elif num_spatial_dims == 3:
            ret = layers.Conv3D(**kwargs)

        # ret.build(input_shape=(in_channels,))

        return ret
    
    def __PoolTranslator(self, module: eqx.nn.Pool) -> layers.Layer:
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

        kwargs = {
            'pool_size': kernel_size,
            'strides': stride,
            'padding': padding,
            'data_format': None,
            'name': None
        }

        if spatial_dims == 1:
            if pool_type == 'max':
                ret = layers.MaxPooling1D(**kwargs)
            else:
                ret = layers.AveragePooling1D(**kwargs)
        elif spatial_dims == 2:
            if pool_type == 'max':
                ret = layers.MaxPooling2D(**kwargs)
            else:
                ret = layers.AveragePooling2D(**kwargs)
        elif spatial_dims == 3:
            if pool_type == 'max':
                ret = layers.MaxPooling3D(**kwargs)
            else:
                ret = layers.AveragePooling3D(**kwargs)
        else:
            raise ValueError(f"Unsupported number of spatial dimensions: {spatial_dims}.")
        
        return ret
    
    def __BatchNormTranslator(self, module: eqx.nn.BatchNorm) -> layers.BatchNormalization:
        if not issubclass(type(module), eqx.nn.BatchNorm):
            raise ValueError(f"Module {type(module)} is not a BatchNorm module.")
        
        raise NotImplementedError("BatchNormTranslator not yet implemented.")
           
