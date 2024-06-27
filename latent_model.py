import os
import sys
import numpy as np
from numbers import Number
from functools import partial
from scipy.linalg import toeplitz
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from pycave.bayes import GaussianMixture as GMM
from pycave.clustering import KMeans
import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from data import Data, DATA_PATH, DATA_FOLDER, TData
from tools import progress

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class no_prints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class ModifiedGammaDistribution(torch.distributions.ExponentialFamily):
    """
    X ~ MGD(alpha, beta, gamma)
    For x in (0, inf):
    p(x) = [gamma * b^(alpha/gamma) / Gamma(alpha/gamma)] * x^(alpha - 1) * exp(- beta x^gamma)
    un-normalized via
    n(x) = N * x^(alpha - 1) * exp(- beta x^gamma)
    with alpha, beta, gamma > 0

    Equivalent to:
    n(x) = N * x^mu * exp(- Lambda x^gamma)
    where mu = alpha - 1, Lambda = beta
    n(x) = N * x^mu * exp(- mu / gamma * (x / c)^gamma)
         = N * exp(- mu / (gamma * c^gamma)) * x^alpha' * exp(- alpha' / gamma * x^gamma)
    where mu = alpha - 1, c = ((alpha - 1) / (beta * gammma))^(1/gamma)

    Note:
    E[X^k] = beta^(-k/gamma) * Gamma((alpha + k) / gamma) / Gamma(alpha/gamma)
    E[X^(k+1)] / E[X^k] = beta^(-1/gamma) * Gamma((alpha + k + 1) / gamma) / Gamma((alpha + k) / gamma)
    X ~ Gamma(alpha, beta) for gamma = 1

    References:
    https://journals.ametsoc.org/view/journals/atsc/68/7/2011jas3645.1.xml?tab_body=pdf
    """

    arg_constraints = {
        'alpha': constraints.positive,
        'beta': constraints.positive,
        'gamma': constraints.positive,
    }
    support = constraints.positive

    def __init__(self, alpha, beta, gamma, validate_args=None):
        self.alpha, self.beta, self.gamma = broadcast_all(alpha, beta, gamma)
        if isinstance(alpha, Number) and isinstance(beta, Number) and isinstance(gamma, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.alpha.size()
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.beta ** (-1 / self.gamma) * torch.exp(
            torch.lgamma((self.alpha + 1) / self.gamma) - torch.lgamma(self.alpha / self.gamma)
        )

    def moment(self, k):
        return self.beta ** (-k / self.gamma) * torch.exp(
            torch.lgamma((self.alpha + k) / self.gamma) - torch.lgamma(self.alpha / self.gamma)
        )

    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.alpha.dtype, device=self.alpha.device)
        if self._validate_args:
            self._validate_sample(value)
        return (
            torch.log(self.gamma)
            + torch.xlogy(self.alpha / self.gamma, self.beta)
            + torch.xlogy(self.alpha - 1, value)
            - self.beta * value ** self.gamma
            - torch.lgamma(self.alpha / self.gamma)
        )


class MomentAutoEncoder(torch.nn.Module):
    def __init__(self, family=ModifiedGammaDistribution):
        super().__init__()
        self.family = family

    def forward(self, x, return_latent=False, from_latent=False):
        if from_latent:
            z, self.c = x
        else:
            self.c = x.sum(axis=1, keepdims=True)
            x /= self.c
            # assume bin counts for [i, i+1] are all from i and estimate via MOM
            t = torch.arange(x.shape[1]).to(x.device)
            z = torch.stack([((x * t ** k) ** (1/k)).sum(axis=1) for k in range(1, 4)], axis=1)
        if return_latent:
            self.c = None
            return z
        else:
            # matching moments is ugly, so do it numerically
            def moments_func(p):
                tuple([p[1] ** (-k / p[2]) * torch.exp(
                    torch.lgamma((p[0] + k) / p[2]) - torch.lgamma(p[0] / p[2])
                ) for k in range(1, 4)])
            p = fsolve(moments_func, z[0])
            self.c = None
            return None


class VariationalEncoder(torch.nn.Module):
    def __init__(self, in_dims=33, hidden_dims=512, latent_dims=3):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_dims, hidden_dims)
        self.linear_mean = torch.nn.Linear(hidden_dims, latent_dims)
        self.linear_logstd = torch.nn.Linear(hidden_dims, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x, return_latent=False):
        x = torch.flatten(x, start_dim=1)
        x = torch.nn.functional.relu(self.linear1(x))
        mu = self.linear_mean(x)
        if return_latent:
            return mu
        else:
            sigma = torch.exp(self.linear_logstd(x))
            z = mu + sigma * self.N.sample(mu.shape)
            self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
            return z


class Decoder(torch.nn.Module):
    def __init__(self, out_dims=33, hidden_dims=512, latent_dims=3):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_dims, hidden_dims)
        self.linear2 = torch.nn.Linear(hidden_dims, out_dims, bias=False)

    def forward(self, z):
        z = torch.nn.functional.relu(self.linear1(z))
        return self.linear2(z)


class VariationalAutoencoder(torch.nn.Module):
    def __init__(self, beta=0.001, data_dims=33, hidden_dims=512, latent_dims=3):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(data_dims, hidden_dims, latent_dims)
        self.decoder = Decoder(data_dims, hidden_dims, latent_dims)
        self.beta = beta
        L = 5
        noise = 0.01
        ker = torch.exp(-0.5 * (torch.arange(0, data_dims) / L) ** 2)
        sig = torch.tensor(toeplitz(ker, ker)) + noise * torch.eye(data_dims)
        self.sig_inv = torch.linalg.inv(sig).to(device)

    def forward(self, x, return_latent=False, from_latent=False):
        if from_latent:
            z = x
        else:
            z = self.encoder(x, return_latent)
        if return_latent:
            return z
        else:
            return torch.exp(self.decoder(z))

    def trainer(self, data, epochs=20, save="models/vae.cp", plot=True, loss_type='mse'):
        losses = []
        opt = torch.optim.Adam(self.parameters())
        for epoch, batch in progress(range(epochs), inner=data, text='Training',
                                     timed=[(1200, lambda: torch.save(self.state_dict(), save))]):
            x = batch['x'].to(device)
            base_sums = x.sum(axis=1)
            x /= base_sums[:, None]

            opt.zero_grad()
            x_hat = self(x)
            if loss_type == 'mse':
                # iid gaussians -> mse
                loss = ((x - x_hat) ** 2).sum() + self.beta * self.encoder.kl
            elif loss_type == 'cor':
                # non-id w Gaussian kernel -> mahalanobis
                loss = torch.sqrt(torch.linalg.vecdot((x - x_hat).T, self.sig_inv @ (x - x_hat).T, dim=0)).sum() +\
                       self.beta * self.encoder.kl
            elif loss_type == 'max':
                # -> smooth max
                alpha = 1
                loss = ((x - x_hat)**2 * torch.nn.functional.softmax(alpha * (x - x_hat), dim=1)).sum() * 1024 +\
                    self.beta * self.encoder.kl
            else:
                raise ValueError('Unknown loss')

            loss.backward()
            losses += [loss.item()]
            opt.step()
        print('Last-epoch loss: %.2f' % sum(losses[-len(data):-1]))
        print('Finished Training')

        if plot:
            plt.plot(np.array(losses))
            plt.savefig('results/tmp_loss.png')
            plt.figure()
            plt.plot(x[0:10].detach().cpu().numpy().T, c="C0")
            plt.plot(x_hat[0:10].detach().cpu().numpy().T, c='C1', ls='--')
            plt.show()


# def get_full_latent(data, model, gmm=None, k=5, d=3, shape=(512, 512, 150)):
#     l = []
#     lab = []
#     print('Visualizing latent space')
#     with torch.no_grad():
#         for i, x in enumerate(data):
#             sys.stdout.write("\r[%.1f %%] - %d / %d" % (100 * (i + 1) / len(data), i + 1, len(data)))
#             sys.stdout.flush()
#             ths_l = torch.full((x.shape[0], d), float('nan'))
#             base_sums = x.sum(axis=1)
#             mask = base_sums > 10 ** (-4)
#             xm = x[mask] / base_sums[mask][:, None]
#             ths_l[mask] = model(xm.to(device)).to("cpu")
#             l += [ths_l]
#             if gmm is not None:
#                 # ths_lab = torch.full((x.shape[0], k), float('nan'))
#                 ths_lab = torch.full((x.shape[0],), float('nan'))
#                 with no_prints():
#                     # ths_lab[mask] = gmm.predict_proba(TData(ths_l[mask]))
#                     ths_lab[mask] = gmm.predict(TData(ths_l[mask])).to(torch.float32)
#                 lab += [ths_lab]
#     print()
#     latent = torch.cat(l).reshape(2, *shape, -1)
#     labels = torch.cat(lab).reshape(2, *shape, -1) if gmm is not None else None
#     return latent, labels


def get_full_latent_by_time(data, model, gmm=None):
    lat, lab, x, y, z, m = [], [], [], [], [], []
    t = 0
    with torch.no_grad():
        for batch in progress(data, 'Visualizing latent space'):
            if batch['t'] != t:
                lat, x, y, z, m = list(map(torch.cat, [lat, x, y, z, m]))
                if gmm is not None:
                    lab = torch.cat(lab)
                yield lat, lab, x, y, z, t, m
                lat, lab, x, y, z, m = [], [], [], [], [], []
                t = batch['t']
            d = batch['x']
            m += [d.sum(axis=1)]
            d /= m[-1][:, None]
            lat += [model(d.to(device)).to("cpu")]
            if gmm is not None:
                with no_prints():
                    # lab += [gmm.predict_proba(TData(lat[-1]))]
                    lab += [gmm.predict(TData(lat[-1])).to(torch.float32)]
            x += [batch['loc'][0]]
            y += [batch['loc'][1]]
            z += [batch['loc'][2]]
    lat, x, y, z, m = list(map(torch.cat, [lat, x, y, z, m]))
    if gmm is not None:
        lab = torch.cat(lab)
    yield lat, lab, x, y, z, t, m


def latent_cluster(data=None, latent_dims=3, hidden_dims=1024, beta=0.001, loss_type='mse', epochs=1,
                   load_path="models/atex_mse.cp", save_path="models/atex_mse.cp"):

    # Train latent encoding
    vae = VariationalAutoencoder(latent_dims=latent_dims, hidden_dims=hidden_dims, beta=beta).to(device)
    if load_path is not None:
        vae.load_state_dict(torch.load(load_path))
    if data is not None:
        vae.trainer(data, epochs=epochs, save=save_path, loss_type=loss_type)

    # vae.trainer(data, epochs=1, save="models/vae_fullest_mse.cp", loss_type='mse')
    # vae.trainer(data, epochs=1, save="models/vae_fullest_cor.cp", loss_type='cor')
    # vae.trainer(data, epochs=1, save="models/vae_fullest_max.cp", loss_type='max')

    # Cluster latent space
    # def extract(x):
    #     return vae((x / x.sum(axis=-1, keepdims=True)).to(device), return_latent=True)
    # data.dataset.transform = extract

    gmm = GMM(num_components=5, batch_size=25000, init_strategy='kmeans++',
              trainer_params=dict(accelerator='gpu', devices=1))
    # km = KMeans(num_clusters=5, batch_size=2048, trainer_params=dict(accelerator='gpu', devices=1))
    # gmm.fit(data)
    # gmm.save("models/gmm")
    # gmm.load("models/gmm")

    return partial(vae, return_latent=True), partial(vae, from_latent=True), gmm

    # ----- VAE -----
    # full: 5-0.0001
    # full_2: 5-0.1
    # full_3: 5-1
    # fullest_mse: 5-0.001
    # fullest_cor: 5-0.001
    # fullest_max: 5-0.001
    # ae_fullest: 5-0


def latent_moments():
    moments = MomentAutoEncoder()
    return partial(moments, return_latent=True), partial(moments, from_latent=True)
