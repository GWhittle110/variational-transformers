"""
Transformer model architectures for each prior/posterior distribution parametrisation
"""

import torch
from torch import nn
import torch.nn.functional as F

from typing import Union, Callable

from components import Cholesky, PositiveDefinite


class TransformerModel(nn.Module):
    """
    Abstract base class of transformer models
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer model.
        Args:
            x: Input tensor. Can be batched or not.

        Returns:
            Output tensor.

        """


class GMMTransformerModel(TransformerModel):
    def __init__(self, n_components: int, state_size: int, n_observations: int, d_model: int, nhead: int,
                 scale_parametrisation: str = None, num_transformer_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.gelu, **kwargs):
        """
        Transformer model for dealing with Gaussian mixture model priors and posteriors.
        Args:
            n_components: Number of components in Gaussian mixture model.
            state_size: Size of the Gaussian mixture model's state.
            n_observations: Number/size of observations.
            d_model: Number of features in input to transformer layer.
            nhead: Number of self-attention heads in transformer.
            scale_parametrisation: Parametrisation of scale for Gaussian mixture model.
                Defaults to covariance_matrix.
            num_transformer_layers: Number of transformer layers in model.
                Defaults to 6.
            dim_feedforward: Dimensionality of feedforward network model in each transformer layer.
                Defaults to 2048.
            dropout: Level of dropout.
                Defaults to 0.1.
            activation: Activation function of feedforward network model. "relu", "gelu" or a callable.
                Defaults to "gelu".
            **kwargs: Additional keyword arguments for the transformer encoder layer.
        """
        super().__init__()
        self.n_components = n_components
        self.state_size = state_size
        self.n_parameters = n_components * (1 + state_size + state_size ** 2)
        self.n_observations = n_observations
        self.scale_parametrisation = "covariance_matrix" if scale_parametrisation is None else scale_parametrisation
        self.n_in = self.n_parameters + n_observations
        self.n_out = self.n_parameters
        self.d_model = d_model
        self.nhead = nhead
        self.feedforward_in = nn.Linear(self.n_in, d_model * nhead)
        transformer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                                                 activation=activation, **kwargs)
        self.transformer_encoder = nn.TransformerEncoder(transformer, num_transformer_layers)
        self.feedforward_out = nn.Linear(self.d_model, self.n_out)
        match self.scale_parametrisation:
            case "covariance_matrix":
                self.scale_transform = PositiveDefinite(state_size)
            case "precision_matrix":
                self.scale_transform = PositiveDefinite(state_size)
            case "scale_tril":
                self.scale_transform = Cholesky(state_size)
            case _:
                raise AssertionError('scale_parametrisation must be one of "covariance_matrix", "precision_matrix" or '
                                     '"scale_tril"')
        self.init_weights()

    def forward(self, x):
        shape = x.shape
        x = self.feedforward_in(x)
        x = x.reshape(shape[:-1] + (self.nhead, self.d_model))
        x = self.transformer_encoder(x)
        x = x[..., -1, :]
        x = self.feedforward_out(x)
        phi_w = F.softmax(x[..., :self.n_components], dim=-1)
        phi_mu = x[..., self.n_components:self.n_components + self.state_size * self.n_components]
        raw_scale = x[..., self.n_components + self.state_size * self.n_components:]
        scale_shape = raw_scale.shape
        phi_scale = self.scale_transform(raw_scale.reshape(raw_scale.shape[:-1] + (self.n_components,
                                                                                   self.state_size, self.state_size))
                                         ).reshape(scale_shape)
        phi = torch.cat([phi_w, phi_mu, phi_scale], dim=-1)
        return phi

    def init_weights(self):
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
