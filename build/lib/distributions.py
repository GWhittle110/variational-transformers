"""
Custom distributions used by the model for meta-priors, priors, likelihoods and sampling
"""

import torch
from torch.distributions import (Distribution, Wishart, MultivariateNormal, Independent, Categorical,
                                 MixtureSameFamily, Dirichlet, constraints)
from torch.distributions.utils import lazy_property
from torch.types import _size
from typing import Optional


class GaussianMixtureModel(MixtureSameFamily):
    arg_constraints = {
        "weights": constraints.simplex,
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real_vector

    def __init__(self, weights: torch.Tensor, loc: torch.Tensor, covariance_matrix: Optional[torch.Tensor] = None,
                 precision_matrix: Optional[torch.Tensor] = None, scale_tril: Optional[torch.Tensor] = None):
        """
        Gaussian Mixture Model distribution. Supports sampling with different parameters in dimension 0.

        Example - no batching:
            >>> weights = torch.ones(3) / 3
            >>> loc = torch.zeros(3, 2, dtype=torch.float32)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32).broadcast_to(3, 2, 2)
            >>> gmm = GaussianMixtureModel(weights, loc, covariance_matrix)
            >>> print(gmm.sample())

        Example - batching:
            >>> weights = torch.ones(4, 3) / 3
            >>> loc = torch.zeros(4, 3, 2, dtype=torch.float32)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32).broadcast_to(4, 3, 2, 2)
            >>> gmm = GaussianMixtureModel(weights, loc, covariance_matrix)
            >>> print(gmm.sample())

        Args:
            weights: Weighting of each mixture component. Must all be positive and sum to 1.
            loc: torch.Tensor of means of each mixture component.
            covariance_matrix: Tensor of covariance matrices of each mixture component. Must all be positive definite.
            precision_matrix:  Tensor of precision matrices of each mixture component. Must all be positive definite.
            scale_tril: torch.Tensor of lower triangular representation of scale matrix, i.e. Cholesky decomposition of
                covariance matrix. Must have positive diagonal elements,
        """
        super().__init__(Categorical(weights),
                         Independent(MultivariateNormal(loc,
                                                        covariance_matrix=covariance_matrix,
                                                        precision_matrix=precision_matrix,
                                                        scale_tril=scale_tril),
                                     0))
        self.n_components = weights.shape[-1]
        self.state_size = loc.shape[-1]

    @lazy_property
    def weights(self):
        return self.weights

    @lazy_property
    def loc(self):
        return self.loc

    @lazy_property
    def covariance_matrix(self):
        return self.covariance_matrix

    @lazy_property
    def precision_matrix(self):
        return self.precision_matrix

    @lazy_property
    def scale_tril(self):
        return self.scale_tril


class MetaPrior(Distribution):
    """
    Abstract base class of meta-priors p(phi) where phi parametrises a prior over x, that is to say priors over priors.
    """

    def __init__(self, prior: type[Distribution]):
        super().__init__()
        self.prior = prior

    def decode_sample(self, sample: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode tensor of sampled parameters to dictionary of tensors keyed by parameter.

        Args:
            sample: Sampled tensor

        Returns:
            Decoded sample
        """
        raise NotImplementedError

    def encode_sample(self, decoded_sample: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode dictionary of sampled parameters to a singular tensor. Inverse operation of decode_sample

        Args:
            decoded_sample:  Dictionary of decoded sample

        Returns:
            Tensor encoding sample
        """
        raise NotImplementedError


class GaussianMixtureModelConjugateMetaPrior(MetaPrior):
    arg_constraints = {
        "weights_concentration": constraints.simplex,
        "loc_loc": constraints.real_vector,
        "loc_covariance_matrix": constraints.positive_definite,
        "loc_precision_matrix": constraints.positive_definite,
        "loc_scale_tril": constraints.lower_cholesky,
        "scale_df": constraints.greater_than(0),
        "scale_loc": constraints.real_vector,
        "scale_covariance_matrix": constraints.positive_definite,
        "scale_precision_matrix": constraints.positive_definite,
        "scale_scale_tril": constraints.lower_cholesky,
        "scale_eps": constraints.greater_than_eq(0)
    }
    has_rsample = True

    def __init__(self,
                 weights_concentration: Optional[torch.Tensor] = None,
                 loc_loc: Optional[torch.Tensor] = None,
                 loc_covariance_matrix: Optional[torch.Tensor] = None,
                 loc_precision_matrix: Optional[torch.Tensor] = None,
                 loc_scale_tril: Optional[torch.Tensor] = None,
                 scale_parametrisation: Optional[str] = None,
                 scale_df: Optional[torch.Tensor] = None,
                 scale_covariance_matrix: Optional[torch.Tensor] = None,
                 scale_precision_matrix: Optional[torch.Tensor] = None,
                 scale_scale_tril: Optional[torch.Tensor] = None,
                 scale_eps: Optional[float] = None,
                 n_components: Optional[int] = None,
                 state_size: Optional[int] = None,
                 ):
        """
        Meta-prior for a Gaussian Mixture Model. Uses conjugate priors for all parameters. Each parameter is referred to
        as for a single distribution, but must be provided as a tensor of parameters for each component.

        Example - default values:

            >>> gmm_conjugate_meta_prior = GaussianMixtureModelConjugateMetaPrior(n_components=3, state_size=2)
            >>> sample = gmm_conjugate_meta_prior.sample((1,))
            >>> print(gmm_conjugate_meta_prior.decode_sample(sample))

        Args:
            weights_concentration: Concentration parameter for Dirichlet prior over weights.
                Defaults to uniform.
            loc_loc: Loc parameter for Multivariate Normal prior over loc.
                Defaults to zero for all components.
            loc_covariance_matrix: Covariance matrix parameter for Multivariate Normal prior over loc.
                Defaults to identity for all components.
            loc_precision_matrix: Precision matrix parameter for Multivariate Normal prior over loc.
                Defaults to identity for all components.
            loc_scale_tril: Lower triangular scale parameter for Multivariate Normal prior over loc.
                Defaults to identity for all components.
            scale_parametrisation: Parametrisation used for scale parameter, common between all components.
                Defaults to "covariance_matrix".
            scale_df: Degrees of freedom of Wishart prior for precision_matrix.
                Defaults to state_size + 1 for all components.
            scale_covariance_matrix: Covariance matrix of Wishart prior for precision_matrix.
                Defaults to identity for all components.
            scale_precision_matrix: Precision matrix of Wishart prior for precision_matrix.
                Defaults to identity for all components.
            scale_scale_tril: Lower triangular scale matrix of Wishart prior for precision_matrix.
                Defaults to identity for all components.
            scale_eps: Small constant to add to diagonal of sampled scale matrix.
                Defaults to 1e-6.
            n_components: Number of components in mixture model. Only specify if using default value for
                weights_concentration else overridden.
            state_size: Size of state in mixture model. Only specify if using default value for loc_loc else overridden.
        """
        super().__init__(GaussianMixtureModel)

        if n_components is not None or state_size is not None:
            assert n_components is not None and state_size is not None, \
                "both n_components and state_size must be specified if using default values"
        self.n_components = weights_concentration.shape[-1] if weights_concentration is not None else n_components
        self.state_size = loc_loc.shape[-1] if loc_loc is not None else state_size
        self.weights_concentration = torch.ones(self.n_components) / self.n_components \
            if weights_concentration is None else weights_concentration
        self.loc_loc = torch.zeros((self.n_components, self.state_size), dtype=torch.float32) \
            if loc_loc is None else loc_loc
        if loc_covariance_matrix is None and loc_precision_matrix is None and loc_scale_tril is None:
            self.loc_covariance_matrix = torch.eye(self.state_size, dtype=torch.float32
                                                   ).broadcast_to(self.n_components, self.state_size, self.state_size)
            self.loc_precision_matrix = loc_precision_matrix
            self.loc_scale_tril = loc_scale_tril
        else:
            self.loc_covariance_matrix = loc_covariance_matrix
            self.loc_precision_matrix = loc_precision_matrix
            self.loc_scale_tril = loc_scale_tril
        self.scale_parametrisation = "covariance_matrix" if scale_parametrisation is None else scale_parametrisation
        self.scale_df = self.state_size + 1 if scale_df is None else scale_df
        if scale_covariance_matrix is None and scale_precision_matrix is None and scale_scale_tril is None:
            self.scale_covariance_matrix = torch.eye(self.state_size, dtype=torch.float32
                                                     ).broadcast_to(self.n_components, self.state_size, self.state_size)
            self.scale_precision_matrix = scale_precision_matrix
            self.scale_scale_tril = scale_scale_tril
        else:
            self.scale_covariance_matrix = scale_covariance_matrix
            self.scale_precision_matrix = scale_precision_matrix
            self.scale_scale_tril = scale_scale_tril
        self.scale_eps = 1e-6 if scale_eps is None else scale_eps

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        weights = Dirichlet(self.weights_concentration).sample(sample_shape)
        loc = Independent(MultivariateNormal(self.loc_loc,
                                             covariance_matrix=self.loc_covariance_matrix,
                                             precision_matrix=self.loc_precision_matrix,
                                             scale_tril=self.loc_scale_tril
                                             ), 0).sample(sample_shape)
        precision_matrix = Independent(Wishart(self.scale_df,
                                               covariance_matrix=self.scale_covariance_matrix,
                                               precision_matrix=self.scale_precision_matrix,
                                               scale_tril=self.scale_scale_tril
                                               ), 0).sample(sample_shape)
        # Add jitter of magnitude scale_eps to diagonal
        precision_matrix += self.scale_eps * torch.eye(self.state_size)
        match self.scale_parametrisation:
            case "covariance_matrix":
                scale = torch.linalg.inv(precision_matrix)
            case "precision_matrix":
                scale = precision_matrix
            case "scale_tril":
                scale = torch.linalg.cholesky(torch.linalg.inv(precision_matrix))
            case _:
                raise AssertionError('scale_parametrisation must be one of "covariance_matrix", "precision_matrix" or '
                                     '"scale_tril"')
        return torch.cat([weights, loc.flatten(start_dim=-2), scale.flatten(start_dim=-3)], dim=-1)

    def decode_sample(self, sample: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode tensor of sampled parameters to dictionary of tensors keyed by GMM parameter.

        Args:
            sample: Sampled tensor

        Returns:
            Decoded sample
        """
        sample_shape = sample.shape[:-1]
        weights = sample[..., :self.n_components]
        loc = sample[..., self.n_components:self.n_components*(1+self.state_size)].reshape(*sample_shape,
                                                                                           self.n_components,
                                                                                           self.state_size)
        scale = sample[..., self.n_components*(1+self.state_size):].reshape(*sample_shape,
                                                                            self.n_components,
                                                                            self.state_size,
                                                                            self.state_size)
        return {"weights": weights,
                "loc": loc,
                self.scale_parametrisation: scale}

    def encode_sample(self, decoded_sample: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode dictionary of sampled parameters to a singular tensor. Inverse operation of decode_sample

        Args:
            decoded_sample:  Dictionary of decoded sample

        Returns:
            Tensor encoding sample
        """
        weights = decoded_sample["weights"]
        loc = decoded_sample["loc"]
        scale = decoded_sample[self.scale_parametrisation]
        return torch.cat([weights, loc.flatten(start_dim=-2), scale.flatten(start_dim=-3)], dim=-1)


    @lazy_property
    def weights_concentration(self):
        return self.weights_concentration

    @lazy_property
    def loc_loc(self):
        return self.loc_loc

    @lazy_property
    def loc_covariance_matrix(self):
        return self.loc_covariance_matrix

    @lazy_property
    def loc_precision_matrix(self):
        return self.loc_precision_matrix

    @lazy_property
    def loc_scale_tril(self):
        return self.loc_scale_tril

    @lazy_property
    def scale_df(self):
        return self.scale_df

    @lazy_property
    def scale_loc(self):
        return self.scale_loc

    @lazy_property
    def scale_covariance_matrix(self):
        return self.scale_covariance_matrix

    @lazy_property
    def scale_precision_matrix(self):
        return self.scale_precision_matrix

    @lazy_property
    def scale_scale_tril(self):
        return self.scale_scale_tril

    @lazy_property
    def scale_eps(self):
        return self.scale_eps


class ObservationModel(Distribution):
    """
    Abstract base class of observation models p(z|x).
    """


class CompleteDistribution(Distribution):
    has_rsample = True

    def __init__(self, meta_prior: MetaPrior, observation_model: ObservationModel):
        """
        Complete distribution over prior, state and observation, p(phi, x, z). We implicitly decompose this
        hierarchically as p(phi) p(x|phi) p(z|x).

        Args:
            meta_prior: Meta-prior distribution, p(phi)
            observation_model: Observation model, p(z|x)
        """
        super().__init__()
        self.meta_prior = meta_prior
        self.prior = meta_prior.prior
        self.observation_model = observation_model

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        phi = self.meta_prior.sample(sample_shape)
        phi_decoded = self.meta_prior.decode_sample(phi)
        x = self.prior(**phi_decoded).sample()
        z = self.observation_model
        return torch.cat([phi, x, z])
