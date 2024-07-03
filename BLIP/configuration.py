import os
import torch
from typing import Union
from typing import Optional, Dict, Any
import torch.nn as nn
from BLIP import logging
import json

logger = logging.get_logger(__name__)
class BlipTextConfig:
    def __init__(
        self,
        attention_probs_dropout_prob=0.0,
        bos_token_id=30522,
        encoder_hidden_size=768,
        eos_token_id=2,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        hidden_size=768,
        initializer_factor=1.0,
        initializer_range=0.02,
        intermediate_size=3072,
        is_decoder=True,
        label_smoothing=0.0,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        model_type="blip_text_model",
        num_attention_heads=12,
        num_hidden_layers=12,
        pad_token_id=0,
        projection_dim=768,
        sep_token_id=102,
        transformers_version="4.41.2",
        use_cache=True,
        vocab_size=30524
    ):
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.bos_token_id = bos_token_id
        self.encoder_hidden_size = encoder_hidden_size
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.is_decoder = is_decoder
        self.label_smoothing = label_smoothing
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.pad_token_id = pad_token_id
        self.projection_dim = projection_dim
        self.sep_token_id = sep_token_id
        self.transformers_version = transformers_version
        self.use_cache = use_cache
        self.vocab_size = vocab_size

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4, sort_keys=True)


class BlipVisionConfig:
    def __init__(
        self,
        attention_dropout=0.0,
        dropout=0.0,
        hidden_act="gelu",
        hidden_size=768,
        image_size=384,
        initializer_factor=1.0,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-5,
        model_type="blip_vision_model",
        num_attention_heads=12,
        num_channels=3,
        num_hidden_layers=12,
        patch_size=16,
        projection_dim=512,
        transformers_version="4.41.2",
    ):
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.num_hidden_layers = num_hidden_layers
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.transformers_version = transformers_version

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4, sort_keys=True)


class BlipConfig(nn.Module):
    r"""
    [`BlipConfig`] is the configuration class to store the configuration of a [`BlipModel`]. It is used to instantiate
    a BLIP model according to the specified arguments, defining the text model and vision model configs. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the BLIP-base
    [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BlipTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BlipVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original BLIP implementation.
        image_text_hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden state of the image-text fusion layer.
        label_smoothing (float, optional, *optional*, defaults to 0.0):
            A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    """

    model_type = "blip"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        image_text_hidden_size=256,
        label_smoothing=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `BlipTextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. Initializing the `BlipVisionConfig` with default values.")

        self.text_config = BlipTextConfig(**text_config)
        self.vision_config = BlipVisionConfig(**vision_config)

        self.text_config.encoder_hidden_size = self.vision_config.hidden_size

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
        self.initializer_range = 0.02
        self.image_text_hidden_size = image_text_hidden_size
        self.label_smoothing = label_smoothing

    @classmethod
    def from_text_vision_configs(cls, text_config: BlipTextConfig, vision_config: BlipVisionConfig, **kwargs):
        r"""
        Instantiate a [`BlipConfig`] (or a derived class) from blip text model configuration and blip vision model
        configuration.

        Returns:
            [`BlipConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)