import torch

from distributions import GaussianMixtureModelConjugateMetaPrior, MappedGaussianObservationModel, CompleteDistribution
from transformers import GMMTransformerModel
from train import train


n_components = 4
state_size = 2


def mapping(x: torch.Tensor) -> torch.Tensor:
    ret = torch.einsum("...i,...i->...", x, x)
    return ret.reshape(ret.shape + (1,))


conjugate_meta_prior = GaussianMixtureModelConjugateMetaPrior(n_components=n_components, state_size=state_size,
                                                              scale_parametrisation="precision_matrix")
observation_model = MappedGaussianObservationModel(covariance_matrix=0.1 * torch.eye(1), mapping=mapping)
complete_distribution = CompleteDistribution(conjugate_meta_prior, observation_model)

model = GMMTransformerModel(n_components=n_components, state_size=state_size,
                            n_observations=observation_model.n_observations, d_model=32, nhead=8,
                            scale_parametrisation="precision_matrix")

model = train(model, complete_distribution, compute_prior_loss=True, batch_size=1000, warmup_epochs=2,
              progress_bar=True, verbose=True)
