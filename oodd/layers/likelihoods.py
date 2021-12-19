import math
from collections import namedtuple, Callable
from typing import Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from .activations import get_activation
from .convolutions import NormedSameConv2d
from .linear import NormedDense, NormedLinear


fields = ["likelihood", "samples", "mean", "mode", "variance", "distribution", "distribution_kwargs"]
LikelihoodData = namedtuple(typename="LikelihoodData", field_names=fields, defaults=[torch.empty(0)] * len(fields))


class LikelihoodModule(nn.Module):
    """General module for parameterizing likelihoods.

    Can change the number of channels/features in the input but not the spatial shape, should it have any.
    """

    def __init__(self, input_shape, out_shape, distribution, activation="LeakyReLU"):
        super().__init__()
        self.input_shape = input_shape
        self.out_shape = out_shape
        self.activation = get_activation(activation)()
        self.distribution = distribution

    def get_distribution_kwargs(self, x):
        raise NotImplementedError

    def forward(self, x_p, x=None):
        """
        Forward pass the input to the likelihood transformation, x_p, to obtain x_hat.
        Optionally compute the likelihood of x_hat given the original x.
        """
        x_p = self.activation(x_p)
        distr_kwargs = self.get_distribution_kwargs(x_p)
        distribution = self.distribution(**distr_kwargs)

        if x is None:
            likelihood = None
        else:
            likelihood = self.log_likelihood(distribution, x)

        if distribution.has_rsample:
            samples = distribution.rsample()
        else:
            samples = distribution.sample()

        data = LikelihoodData(
            likelihood=likelihood,
            samples=samples,
            mean=distribution.mean,
            mode=self.mode(distribution),
            variance=distribution.variance,
            distribution=None,
            distribution_kwargs=distr_kwargs,
        )
        return likelihood, data

    @staticmethod
    def mode(distribution):
        return None


class GaussianLikelihoodConv2d(LikelihoodModule):
    def __init__(self, input_shape, out_shape, kernel_size=3, activation="LeakyReLU", weightnorm=True):
        super().__init__(input_shape, out_shape, distribution=D.Normal, activation=activation)
        self.parameter_net = NormedSameConv2d(input_shape, 2 * out_shape[0], kernel_size, weightnorm=weightnorm)
        self.std_activation = nn.Softplus(beta=np.log(2))

    def get_distribution_kwargs(self, x):
        x = self.parameter_net(x)
        mu, lv = x.chunk(2, dim=1)
        scale = self.std_activation(lv)
        params = {
            "loc": mu,
            "scale": scale,
        }
        return params

    @staticmethod
    def log_likelihood(distribution, x):
        return distribution.log_prob(x).sum((1, 2, 3))

    @staticmethod
    def mode(distribution):
        return distribution.mean


class GaussianLikelihoodDense(LikelihoodModule):
    def __init__(self, input_shape, out_shape, activation="LeakyReLU", weightnorm=True):
        super().__init__(input_shape, out_shape, distribution=D.Normal, activation=activation)
        self.parameter_net = NormedDense(input_shape, 2 * np.prod(out_shape), weightnorm=weightnorm)
        self.std_activation = nn.Softplus(beta=np.log(2))

    def get_distribution_kwargs(self, x):
        x = self.parameter_net(x)
        mu, lv = x.chunk(2, dim=1)
        mu, lv = mu.view(-1, *self.out_shape), lv.view(-1, *self.out_shape)
        scale = self.std_activation(lv)
        params = {
            "loc": mu,
            "scale": scale,
        }
        return params

    @staticmethod
    def log_likelihood(distribution, x):
        return distribution.log_prob(x).sum(1)

    @staticmethod
    def mode(distribution):
        return distribution.mean


class BernoulliLikelihoodConv2d(LikelihoodModule):
    def __init__(self, input_shape, out_shape, kernel_size=3, activation="LeakyReLU", weightnorm=True):
        super().__init__(input_shape, out_shape, distribution=D.Bernoulli, activation=activation)
        self.parameter_net = NormedSameConv2d(input_shape, out_shape[0], kernel_size, weightnorm=weightnorm)

    def get_distribution_kwargs(self, x):
        return {"logits": self.parameter_net(x)}

    @staticmethod
    def log_likelihood(distribution, x):
        return distribution.log_prob(x).sum((1, 2, 3))

    @staticmethod
    def mode(distribution):
        return torch.round(distribution.probs)


class BernoulliLikelihoodDense(LikelihoodModule):
    def __init__(self, input_shape, out_shape, bias=True, activation="LeakyReLU", weightnorm=True):
        super().__init__(input_shape, out_shape, distribution=D.Bernoulli, activation=activation)
        self.parameter_net = NormedDense(input_shape, np.prod(out_shape), bias=bias, weightnorm=weightnorm)

    def get_distribution_kwargs(self, x):
        logits = self.parameter_net(x)
        logits = logits.view(-1, *self.out_shape)
        return {"logits": logits}

    @staticmethod
    def log_likelihood(distribution, x):
        all_dims_but_first = list(range(x.ndim))[1:]
        return distribution.log_prob(x).sum(all_dims_but_first)

    @staticmethod
    def mode(distribution):
        return BernoulliLikelihoodConv2d.mode(distribution)


class BernoulliLikelihoodIdentity(LikelihoodModule):
    def __init__(self, input_shape, out_shape, activation="Identity"):
        super().__init__(input_shape, out_shape, distribution=D.Bernoulli, activation="Identity")

    def get_distribution_kwargs(self, x):
        logits = x.view(-1, *self.out_shape)
        return {"logits": logits}

    @staticmethod
    def log_likelihood(distribution, x):
        all_dims_but_first = list(range(x.ndim))[1:]
        return distribution.log_prob(x).sum(all_dims_but_first)

    @staticmethod
    def mode(distribution):
        return BernoulliLikelihoodConv2d.mode(distribution)


class BetaLikelihoodConv2d(LikelihoodModule):
    def __init__(self, input_shape, out_shape, kernel_size=3, activation="LeakyReLU", weightnorm=True):
        super().__init__(input_shape, out_shape, distribution=D.Beta, activation=activation)
        self.parameter_net = NormedSameConv2d(input_shape, 2 * out_shape[0], kernel_size, weightnorm=weightnorm)
        self.concentration_activation = nn.Softplus()

    def get_distribution_kwargs(self, x):
        x = self.parameter_net(x)
        x = self.concentration_activation(x)
        alpha, beta = x.chunk(2, dim=1)
        params = {
            "concentration1": alpha,
            "concentration0": beta,
        }
        return params

    @staticmethod
    def log_likelihood(distribution, x):
        return distribution.log_prob(x).sum((1, 2, 3))

    @staticmethod
    def mode(distribution):
        return torch.round(distribution.mean)


class DiscretizedLogisticLikelihoodConv2d(LikelihoodModule):
    """
    Assume input data to be originally uint8 (0, ..., 255) and then rescaled
    by 1/255: discrete values in {0, 1/255, ..., 255/255}.

    If using the discretize logistic logprob implementation here, this should
    be rescaled by 255/256 and shifted by <1/256 in this class. So the data is
    inside 256 bins between 0 and 1.

    Note that mean and logscale are parameters of the underlying continuous
    logistic distribution, not of its discretization.

    From "Improved Variational Inference with Inverse Autoregressive Flow" paper:
        The ﬁrst layer of the encoder, and the last layer of the decoder, consist of convolutions that project from/to
        input space. The pixel data is scaled to the range [0, 1], and the data likelihood of pixel values in the
        generative model is the probability mass of the pixel value under the logistic distribution. Noting that the
        CDF of the standard logistic distribution is simply the sigmoid function, we simply compute the probability
        mass per input pixel using
            p(x_i | µ_i, s_i ) = CDF(x_i + 1/256 | µ_i, s_i) − CDF(x_i | µ_i, s_i ),
        where the locations µ_i are output of the decoder, and the log-scales log(s_i) are learned scalar
        parameter per input channel abd
            CDF(x|µ,s) = 1 / (1 + exp(-(x-µ)/s))
        is the cumulative distribution function for the logistic distribution.
    """

    log_scale_bias = -1.0

    def __init__(self, input_shape, out_shape, n_bins=256, activation="LeakyReLU", weightnorm=True, double=False):
        super().__init__(input_shape, out_shape, distribution=None, activation=activation)
        self.n_bins = n_bins
        self.double_precision = double
        self.out_channels = out_shape[0]
        self.parameter_net = NormedSameConv2d(input_shape, 2 * out_shape[0], kernel_size=3, weightnorm=weightnorm)

    def get_distribution_kwargs(self, x):
        x = self.parameter_net(x)
        mean, ls = x.chunk(2, dim=1)
        ls = ls + self.log_scale_bias
        ls = ls.clamp(min=-7.0)
        mean = mean + 0.5  # initialize to mid interval
        params = {
            "mean": mean,
            "logscale": ls,
        }
        return params

    @staticmethod
    def mean(params):
        return params["mean"]

    @staticmethod
    def mode(params):
        return params["mean"]

    @staticmethod
    def sample(params):
        # We're not quantizing 8bit, but it doesn't matter
        sample = logistic_rsample((params["mean"], params["logscale"]))
        sample = sample.clamp(min=0.0, max=1.0)
        return sample

    def log_likelihood(self, x, params):
        """Input data x should be inside (not at the edge) of n_bins equally-sized
        bins between 0 and 1. E.g. if n_bins=256 the 257 bin edges are:

            0, 1/256, ..., 255/256, 1.
        """
        x = x * (255 / 256) + 1 / 512

        logprob = log_discretized_logistic(
            x, params["mean"], params["logscale"], n_bins=self.n_bins, reduce="none", double=self.double_precision
        )
        return logprob

    def forward(self, x_p, x=None):
        distr_kwargs = self.get_distribution_kwargs(x_p)
        mean = self.mean(distr_kwargs)
        mode = self.mode(distr_kwargs)
        sample = self.sample(distr_kwargs)
        if x is None:
            likelihood = None
        else:
            likelihood = self.log_likelihood(x, distr_kwargs)

        data = LikelihoodData(
            likelihood=likelihood,
            distribution=None,
            mean=mean.clamp(0, 1),
            mode=mode.clamp(0, 1),
            variance=None,
            samples=sample,
            distribution_kwargs=distr_kwargs,
        )
        return likelihood, data


class DiscretizedLogisticMixLikelihoodConv2d(LikelihoodModule):
    """
    Sampling and loss computation are based on the original tf code.

    Assume input data to be originally uint8 (0, ..., 255) and then rescaled
    by 1/255: discrete values in {0, 1/255, ..., 255/255}.

    When using the original discretize logistic mixture logprob implementation,
    this data should be rescaled to be in [-1, 1] which is done in this module.

    Mean and mode are not implemented for now.

    Output channels for now is fixed to 3 and n_bins to 256.
    """

    def __init__(self, input_shape, out_shape, nr_mix=10, kernel_size=1, activation="LeakyReLU", weightnorm=True):
        """Discretized Logistic Mixture distribution

        Args:
            ch_in (int): Number of input channels
            nr_mix (int, optional): Number of components. Defaults to 10.
        """
        if out_shape[0] != 3:
            raise NotImplementedError("Currently does not support other than 3 color channels in output")
        
        out_channels = out_shape[0]
        out_features = (out_channels * 3 + 1) * nr_mix  # mean, variance and mixture coeff per channel plus logits

        super().__init__(input_shape, out_shape, distribution=None, activation=activation)
        self.parameter_net = NormedSameConv2d(input_shape, out_features, kernel_size=kernel_size, weightnorm=weightnorm)

    def get_distribution_kwargs(self, x):
        l = self.parameter_net(x)
        # mean, log_scale, coeff = discretized_mix_logistic_split_kwargs(l)
        params = {"mean": None, "all_params": l}
        return params

    @staticmethod
    def mean(params):
        return params["mean"]

    @staticmethod
    def mode(params):
        return params["mean"]

    @staticmethod
    def sample(params):
        samples = discretized_mix_logistic_rsample(params["all_params"])
        samples = (samples + 1) / 2  # Transform from [-1, 1] to [0, 1]
        samples = samples.clamp(min=0.0, max=1.0)
        return samples

    def log_likelihood(self, x, params):
        x = x * 2 - 1  # Transform from [0, 1] to [-1, 1]
        logprob = log_discretized_mix_logistic(x, params["all_params"])
        return logprob

    def forward(self, x_p, x=None):
        distr_kwargs = self.get_distribution_kwargs(x_p)
        mean = self.mean(distr_kwargs)
        mode = self.mode(distr_kwargs)
        samples = self.sample(distr_kwargs)
        if x is None:
            likelihood = None
        else:
            likelihood = self.log_likelihood(x, distr_kwargs)

        data = LikelihoodData(
            likelihood=likelihood,
            distribution=None,
            mean=samples,
            mode=samples,
            variance=None,
            samples=samples,
            distribution_kwargs=distr_kwargs,
        )
        return likelihood, data


class DiscretizedLogisticMixLikelihoodDense(LikelihoodModule):
    """
    Sampling and loss computation are based on the original tf code.

    Assume input data to be originally uint8 (0, ..., 255) and then rescaled
    by 1/255: discrete values in {0, 1/255, ..., 255/255}.

    When using the original discretize logistic mixture logprob implementation,
    this data should be rescaled to be in [-1, 1] which is done in this module.

    Mean and mode are not implemented for now.

    Output channels for now is fixed to 3 and n_bins to 256.
    """

    def __init__(self, input_shape, out_shape, nr_mix=10, kernel_size=1, activation="LeakyReLU", weightnorm=True):
        """Discretized Logistic Mixture distribution

        Args:
            ch_in (int): Number of input channels
            nr_mix (int, optional): Number of components. Defaults to 10.
        """
        if out_shape[0] != 3:
            raise NotImplementedError("Currently does not support other than 3 color channels in output")

        out_channels = out_shape[0]
        out_features = (out_channels * 3 + 1) * nr_mix  # mean, variance and mixture coeff per channel plus logits
        in_features = input_shape[0]  # Channel dimension (or simply feature dimension for dense)
        super().__init__(input_shape, out_shape, distribution=None, activation=activation)
        self.parameter_net = NormedLinear(in_features, out_features, dim=1, weightnorm=weightnorm)

    def get_distribution_kwargs(self, x):
        l = self.parameter_net(x)
        # mean, log_scale, coeff = discretized_mix_logistic_split_kwargs(l)
        params = {"mean": None, "all_params": l}
        return params

    @staticmethod
    def mean(params):
        return params["mean"]

    @staticmethod
    def mode(params):
        return params["mean"]

    @staticmethod
    def sample(params):
        samples = discretized_mix_logistic_rsample(params["all_params"])
        samples = (samples + 1) / 2  # Transform from [-1, 1] to [0, 1]
        samples = samples.clamp(min=0.0, max=1.0)
        return samples

    def log_likelihood(self, x, params):
        x = x * 2 - 1  # Transform from [0, 1] to [-1, 1]
        logprob = log_discretized_mix_logistic(x, params["all_params"])
        return logprob

    def forward(self, x_p, x=None):
        distr_kwargs = self.get_distribution_kwargs(x_p)
        mean = self.mean(distr_kwargs)
        mode = self.mode(distr_kwargs)
        samples = self.sample(distr_kwargs)
        if x is None:
            likelihood = None
        else:
            likelihood = self.log_likelihood(x, distr_kwargs)

        data = LikelihoodData(
            likelihood=likelihood,
            distribution=None,
            mean=samples,  # TODO We need the mean and mode here
            mode=samples,  # TODO We need the mean and mode here
            variance=None,
            samples=samples,
            distribution_kwargs=distr_kwargs,
        )
        return likelihood, data


def logistic_rsample(mu_ls):
    """
    Returns a sample from Logistic with specified mean and log scale.

    :param mu_ls: a tensor containing mean and log scale along dim=1,
            or a tuple (mean, log scale)
    :return: a reparameterized sample with the same size as the input
            mean and log scale
    """
    # Get parameters
    try:
        mu, log_scale = torch.chunk(mu_ls, 2, dim=1)
    except TypeError:
        mu, log_scale = mu_ls
    scale = log_scale.exp()

    # Get uniform sample in open interval (0, 1)
    u = torch.zeros_like(mu)
    u.uniform_(1e-7, 1 - 1e-7)

    # Transform into logistic sample
    sample = mu + scale * (torch.log(u) - torch.log(1 - u))

    return sample


def discretized_mix_logistic_rsample(l):
    """
    Code taken from pytorch adaptation of original PixelCNN++ tf implementation
    https://github.com/pclucas14/pixel-cnn-pp
    """

    def to_one_hot(tensor, n):
        one_hot = torch.zeros(tensor.size() + (n,))
        one_hot = one_hot.to(tensor.device)
        one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), 1.0)
        return one_hot

    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)  # "channels first" to "channels last"
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])

    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda:
        temp = temp.cuda()
    temp.uniform_(1e-5, 1.0 - 1e-5)
    temp = logit_probs.data - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(l[:, :, :, :, nr_mix : 2 * nr_mix] * sel, dim=4), min=-7.0)
    coeffs = torch.sum(torch.tanh(l[:, :, :, :, 2 * nr_mix : 3 * nr_mix]) * sel, dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.FloatTensor(means.size())
    if l.is_cuda:
        u = u.cuda()
    u.uniform_(1e-5, 1.0 - 1e-5)
    u = nn.Parameter(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.0), max=1.0)
    x1 = torch.clamp(torch.clamp(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.0), max=1.0)
    x2 = torch.clamp(torch.clamp(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.0), max=1.0)

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out


def log_discretized_logistic(x, mean, log_scale, n_bins=256, reduce="mean", double=False):
    """
    Log of the probability mass of the values x under the logistic distribution
    with parameters mean and scale. The sum is taken over all dimensions except
    for the first one (assumed to be batch). Reduction is applied at the end.

    Assume input data to be inside (not at the edge) of n_bins equally-sized
    bins between 0 and 1. E.g. if n_bins=256 the 257 bin edges are:
    0, 1/256, ..., 255/256, 1.
    If values are at the left edge it's also ok, but let's be on the safe side

    Variance of logistic distribution is
        var = scale^2 * pi^2 / 3

    :param x: tensor with shape (batch, channels, dim1, dim2)
    :param mean: tensor with mean of distribution, shape
                 (batch, channels, dim1, dim2)
    :param log_scale: tensor with log scale of distribution, shape has to be either
                  scalar or broadcastable
    :param n_bins: bin size (default: 256)
    :param reduce: reduction over batch: 'mean' | 'sum' | 'none'
    :param double: whether double precision should be used for computations
    :return:
    """
    log_scale = _input_check(x, mean, log_scale, reduce)
    if double:
        log_scale = log_scale.double()
        x = x.double()
        mean = mean.double()
        eps = 1e-14
    else:
        eps = 1e-7

    scale = log_scale.exp()

    # Set values to the left of each bin
    x = torch.floor(x * n_bins) / n_bins

    cdf_plus = torch.ones_like(x)
    idx = x < (n_bins - 1) / n_bins
    cdf_plus[idx] = torch.sigmoid((x[idx] + 1 / n_bins - mean[idx]) / scale[idx])

    cdf_minus = torch.zeros_like(x)
    idx = x >= 1 / n_bins
    cdf_minus[idx] = torch.sigmoid((x[idx] - mean[idx]) / scale[idx])

    log_prob = torch.log(cdf_plus - cdf_minus + eps)

    log_prob = log_prob.sum((1, 2, 3))
    log_prob = _reduce(log_prob, reduce)
    if double:
        log_prob = log_prob.float()
    return log_prob


def log_discretized_mix_logistic(x, l):
    """Log-likelihood for mixture of discretized logistics

    Assumes the data has been rescaled to [-1, 1] interval and that the input is
    colour images with 3 channels (channels first)

    Code taken from pytorch adaptation of original PixelCNN++ tf implementation
    https://github.com/pclucas14/pixel-cnn-pp

    Args:
        x (torch.Tensor): Original input image (the true distribution) as (B, C, H, W)
        l (torch.Tensor): Predicted distribution over the image space as (B, C * N_components, H, W)

    Returns:
        torch.Tensor: Likelihood
    """

    # channels last
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)

    # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    xs = [int(y) for y in x.size()]
    # predicted distribution, e.g. (B,32,32,100)
    ls = [int(y) for y in l.size()]

    assert xs[-1] == 3, "Discretized Logistic Mixture likelihood is only applicable to RGB images (not gray-scale)"
    assert -1.0 <= x.min() and x.max() <= 1.0

    # Unpack the parameters of the mixture of logistics.
    # We need four quantities: logit probs, means, logs_scales and coeffs
    # - logit_probs: (B, H, W, nr_mix)
    # - means: (B, H, W, C, nr_mix)
    # - log_scales: (B, H, W, C, nr_mix)
    # - coeffs: (B, H, W, C, nr_mix)
    # This gives in total a number of parameters of: nr_mix * C * 3 + nr_mix
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])  # 3 for mean, scale, coef for each mixture component
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix : 2 * nr_mix], min=-7.0)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix : 3 * nr_mix])

    # Get the means and adjust them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + torch.zeros(xs + [nr_mix]).to(x.device)
    # x = x.unsqueeze(-1) + nn.Parameter(torch.zeros(xs + [nr_mix]).to(x.device), requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (
        means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]
    ).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below
    # for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999,
    # log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which
    # never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero
    # instead of selecting: this requires use to use some ugly tricks to avoid
    # potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as
    # output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1.0 - inner_inner_cond) * (
        log_pdf_mid - np.log(127.5)
    )
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + torch.log_softmax(logit_probs, dim=-1)
    log_probs = torch.logsumexp(log_probs, dim=-1)

    # return -torch.sum(log_probs)
    log_prob = log_probs.sum((1, 2))  # keep batch dimension
    return log_prob


def _reduce(x, reduce):
    if reduce == "mean":
        x = x.mean()
    elif reduce == "sum":
        x = x.sum()
    return x


def _input_check(x, mean, scale_param, reduce):
    assert x.dim() == 4
    assert x.size() == mean.size()
    if scale_param.numel() == 1:
        scale_param = scale_param.view(1, 1, 1, 1)
    if reduce not in ["mean", "sum", "none"]:
        msg = "unrecognized reduction method '{}'".format(reduce)
        raise RuntimeError(msg)
    return scale_param


def get_likelihood(name):
    try:
        klass = globals()[name]
    except KeyError:
        raise KeyError(f"Likelihood layer `{name}` not recognized")
    return klass

# -------------------------------------



class DiscretizedLogisticMixLikelihoodConv2dMono(LikelihoodModule):
    """
    Sampling and loss computation are based on the original tf code.

    Assume input data to be originally uint8 (0, ..., 255) and then rescaled
    by 1/255: discrete values in {0, 1/255, ..., 255/255}.

    When using the original discretize logistic mixture logprob implementation,
    this data should be rescaled to be in [-1, 1] which is done in this module.

    Mean and mode are not implemented for now.

    Output channels is fixed to 1 and n_bins to 256.
    """

    def __init__(self, input_shape, out_shape, nr_mix=10, kernel_size=1, activation="LeakyReLU", weightnorm=True):
        """Discretized Logistic Mixture distribution

        Args:
            ch_in (int): Number of input channels
            nr_mix (int, optional): Number of components. Defaults to 10.
        """
        if out_shape[0] != 1:
            raise NotImplementedError("Does not support other than 1 color channels in output")

        out_channels = out_shape[0]
        out_features = (out_channels * 3 + 1) * nr_mix  # mean, variance and mixture coeff per channel plus logits

        super().__init__(input_shape, out_shape, distribution=None, activation=activation)
        self.parameter_net = NormedSameConv2d(input_shape, out_features, kernel_size=kernel_size, weightnorm=weightnorm)

    def get_distribution_kwargs(self, x):
        l = self.parameter_net(x)
        # mean, log_scale, coeff = discretized_mix_logistic_split_kwargs(l)
        params = {"mean": None, "all_params": l}
        return params

    @staticmethod
    def mean(params):
        return params["mean"]

    @staticmethod
    def mode(params):
        return params["mean"]

    @staticmethod
    def sample(params):
        # rsample_discretized_logistic_mixture
        samples = discretized_mix_logistic_rsample(params["all_params"])

        # rsample_discretized_logistic_mixture(
        #     logit_probs: torch.Tensor,
        #     locs: torch.Tensor,
        #     log_scales: torch.Tensor,
        #     eps: float = 1e-5,
        #              rsample_categorical: bool = False,
        #                                          tau: float = 1.0,
        # ):
        samples = (samples + 1) / 2  # Transform from [-1, 1] to [0, 1]
        samples = samples.clamp(min=0.0, max=1.0)
        return samples

    def log_likelihood(self, x, params):
        x = x * 2 - 1  # Transform from [0, 1] to [-1, 1]
        logprob = log_discretized_mix_logistic(x, params["all_params"])
        # discretized_logistic_mixture_ll(
        #     y: torch.Tensor,
            # logit_probs: torch.Tensor,
            # locs: torch.Tensor,
            # log_scales: torch.Tensor,
            # num_bins: int = 256,
        #                 reduce_dim: int = -1,
        # ):
        return logprob

    def forward(self, x_p, x=None):
        distr_kwargs = self.get_distribution_kwargs(x_p)
        mean = self.mean(distr_kwargs)
        mode = self.mode(distr_kwargs)
        samples = self.sample(distr_kwargs)
        if x is None:
            likelihood = None
        else:
            likelihood = self.log_likelihood(x, distr_kwargs)

        data = LikelihoodData(
            likelihood=likelihood,
            distribution=None,
            mean=samples,
            mode=samples,
            variance=None,
            samples=samples,
            distribution_kwargs=distr_kwargs,
        )
        return likelihood, data

# @jit.script
def rsample_discretized_logistic_mixture(
    logit_probs: torch.Tensor,
    locs: torch.Tensor,
    log_scales: torch.Tensor,
    eps: float = 1e-5,
    rsample_categorical: bool = False,
    tau: float = 1.0,
):
    """Return a reparameterized sample from a given Discretized Logistic Mixture distribution.
    Code taken from PyTorch adaptation of original PixelCNN++ TensorFlow implementation:
    https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py but does not include the channel specific conditional modelling.
    Args:
        logit_probs (torch.Tensor): (*, num_mix)
        locs (torch.Tensor): (*, D, num_mix)
        log_scales (torch.Tensor): (*, D, num_mix)
        num_mix (int): Number of mixture components
        eps (float): Bounds [eps, 1-eps] on the uniform rv used to sample the mixture coefficients and the logistic.
        tau (float): Temperature for Gumbel sampling
    Returns:
        torch.Tensor: Sample from the DLM `(*, D)`
    """
    # sample mixture indicator from categorical
    if rsample_categorical:
        argmax = rsample_gumbel_softmax(logit_probs, hard=True, tau=tau, return_argmax=True)
    else:
        gumbel = -torch.log(-torch.log(torch.empty_like(logit_probs).uniform_(eps, 1.0 - eps)))  # (*, num_mix)
        argmax = torch.argmax(logit_probs + gumbel, dim=-1, keepdim=True)  # (*, 1)

    # broadcast argmax onto mu and sd shapes (including D)
    argmax = argmax.expand(*argmax.shape[:-1], locs.size(-2)).unsqueeze(-1)  # (*, D, 1)

    # select component and remove mixture dimension
    locs = torch.gather(locs, index=argmax, dim=-1).squeeze(-1)
    log_scales = torch.gather(log_scales, index=argmax, dim=-1).squeeze(-1)

    # sample from logistic (we don't actually round to the nearest 8bit value)
    x = rsample_discretized_logistic(locs, log_scales)
    return x

# @jit.script
def rsample_discretized_logistic(mu: torch.Tensor, log_scale: torch.Tensor, eps: float = 1e-8):
    """Return a sample from a discretized logistic with values standardized to be in [-1, 1]
    This is done by sampling the corresponding continuous logistic and clamping values outside
    the interval to the endpoints.
    We do not further quantize the samples here.
    """
    return rsample_logistic(mu, log_scale, eps).clamp(-1, 1)


# @jit.script
def rsample_logistic(mu: torch.Tensor, log_scale: torch.Tensor, eps: float = 1e-8):
    """
    Returns a sample from Logistic with specified mean and log scale.
    :param mu: a tensor containing the mean.
    :param log_scale: a tensor containing the log scale.
    :return: a reparameterized sample with the same size as the input mean and log scale.
    """
    u = torch.empty_like(mu).uniform_(eps, 1 - eps)  # uniform sample in the interval (eps, 1 - eps)
    sample = mu + torch.exp(log_scale) * (torch.log(u) - torch.log(1 - u))  # transform to logistic
    return sample

# @jit.script
def rsample_gumbel_softmax(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
    return_argmax: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
):
    """Returns a sample from the Gumbel-Softmax distribution and optionally discretizes it [1, 2].
    As the softmax temperature τ approaches 0, samples from the Gumbel-Softmax distribution become one-hot
    and the Gumbel-Softmax distribution becomes identical to the categorical distribution Cat(logits).
    Args:
        logits (torch.Tensor): `[..., num_features]` unnormalized log probabilities
        tau (float): non-negative scalar temperature
        hard (bool): If True, the returned samples will be discretized as one-hot vectors,
                     but will be differentiated as if it is the soft sample in autograd.
        return_argmax (bool): If True, return the hard argmax indices instead of one_hot vectors. Defaults to False.
                              Has not effect when `hard=False`.
        dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
        Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
        If ``hard=True``, the returned samples will be one-hot, otherwise they will
        be probability distributions that sum to 1 across `dim`.
    Note:
        The main trick for `hard` is to do  `y_hard + (y_soft - y_soft.detach())`
        This achieves two things:
         1. makes the output value exactly `y_hard` and hence one-hot (since we add then subtract y_soft value)
         2. makes the (undefined) gradient of `y_hard` equal to y_soft gradient (since we strip all other gradients)
        The introduced gradient bias between `y_soft` and `y_hard` is reduced as `tau` is decreased towards zero.
    Examples:
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> rsample_gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> rsample_gumbel_softmax(logits, tau=1, hard=True)
    [1] https://arxiv.org/abs/1611.00712
    [2] https://arxiv.org/abs/1611.01144
    """
    gumbels = -torch.log(-torch.log(torch.empty_like(logits).uniform_(eps, 1.0 - eps)))  # ~Gumbel(0,1)
    logits_sampled = (logits + gumbels) / tau
    y_soft = logits_sampled.softmax(dim)

    if not hard:
        # Reparametrization trick with Gumbel Softmax (bias -> 0 as tau -> 0)
        return y_soft

    # Straight through estimator (always biased)
    index = y_soft.max(dim, keepdim=True)[1]
    if return_argmax:
        return index

    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    return y_hard + (y_soft - y_soft.detach())


def reduce(tensor: torch.Tensor, dim: Union[Tuple[int], int], operation: Callable = torch.sum):
    return operation(tensor, dim)

# @torch.jit.script
def discretized_logistic_mixture_ll(
    y: torch.Tensor,
    logit_probs: torch.Tensor,
    locs: torch.Tensor,
    log_scales: torch.Tensor,
    num_bins: int = 256,
    reduce_dim: int = -1,
):
    """Compute log-likelihood for a mixture of discretized logistics.
    The implementation is partially as in https://arxiv.org/abs/1701.05517 but does not assume
    three RGB colour channels nor does it condition them on each other (as described in Section 2.2).
    Hence, the channels, and all other dimensions, are regarded as independent.
    For more details, refer to documentation for `discretized_logistic_ll`.
    Args:
        y (torch.Tensor): (*, D)
        logit_probs (torch.Tensor): (*, num_mix)
        locs (torch.Tensor): (*, D, num_mix)
        log_scales (torch.Tensor): (*, D, num_mix)
        num_bins (int): Quantization level
        reduce_dim (int, optional): Dimension in y to reduce over. Defaults to -1 corresponding to D.
    """
    # check input
    assert torch.max(y) <= 1.0 and torch.min(y) >= -1.0

    # repeat y for broadcasting to mixture dim
    num_mix = logit_probs.size(-1)
    y = y.unsqueeze(-1).expand(*[-1] * y.ndim, num_mix)  # (*, D, num_mix)

    # compute y-µ and 1/s
    centered_y = y - locs
    inv_stdv = torch.exp(-log_scales)


    # TODO USE discretized_logistic_ll method here

    # compute CDF at left and right "bin edge" (floating) to compute total mass in between (cdf_delta)
    plus_in = inv_stdv * (centered_y + 1.0 / (num_bins - 1))
    cdf_plus = torch.sigmoid(plus_in)
    minus_in = inv_stdv * (centered_y - 1.0 / (num_bins - 1))
    cdf_minus = torch.sigmoid(minus_in)
    cdf_delta = cdf_plus - cdf_minus

    # log probability for edge case of 0 (mass from 0 to 0.5)
    log_cdf_plus = plus_in - F.softplus(plus_in)  # = log CDF(y+0.5) via softplus(y) = y - softplus(-y)

    # log probability for edge case of 255 (mass from 254.5 to 255)
    log_one_minus_cdf_minus = -F.softplus(minus_in)  # = log 1 - CDF(y-0.5)

    # log probability in the center of the bin, to be used in extreme cases where cdf_delta is extremely small
    mid_in = inv_stdv * centered_y
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
    log_prob_mid_safe = torch.where(
        cdf_delta > 1e-5, torch.log(torch.clamp(cdf_delta, min=1e-10)), log_pdf_mid - math.log(num_bins / 2)
    )

    # handle edge cases
    log_prob = torch.where(y < 2 / num_bins - 1, log_cdf_plus, log_prob_mid_safe)  # edge case 0, y < -254/256
    log_prob = torch.where(y > 1 - 2 / num_bins, log_one_minus_cdf_minus, log_prob)  # edge case 255, y > 254/256

    # TODO USE discretized_logistic_ll method here


    # TODO First sum over data dimension?
    log_prob = reduce(log_prob, reduce_dim - 1)  # Reduce data dimension per component
    log_prob = log_prob + torch.log_softmax(logit_probs, dim=-1)  #
    return torch.logsumexp(log_prob, dim=-1)  # Normalize over mixture components (in log-prob space)


