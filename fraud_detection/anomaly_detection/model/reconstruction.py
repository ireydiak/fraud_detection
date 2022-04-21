import numpy as np
import torch
from anomaly_detection.model.base import BaseModel
from anomaly_detection.model.GMM import GMM
from torch import nn
from typing import Tuple, List


class AutoEncoder(nn.Module):
    """
    Implements a basic Deep Auto Encoder
    """

    def __init__(self, enc_layers: list, dec_layers: list, **kwargs):
        super(AutoEncoder, self).__init__()
        self.latent_dim = dec_layers[0][0]
        self.in_features = enc_layers[-1][1]
        self.encoder = self._make_linear(enc_layers)
        self.decoder = self._make_linear(dec_layers)
        self.name = "AutoEncoder"

    @staticmethod
    def from_dataset(in_features, dataset_name: str):
        if dataset_name == "Arrhythmia":
            enc_layers = [
                (in_features, 10, nn.Tanh()),
                (10, 1, None)
            ]
            dec_layers = [
                (1, 10, nn.Tanh()),
                (10, in_features, None)
            ]
        elif dataset_name == "Thyroid":
            enc_layers = [
                (in_features, 12, nn.Tanh()),
                (12, 4, nn.Tanh()),
                (4, 1, None)
            ]
            dec_layers = [
                (1, 4, nn.Tanh()),
                (4, 12, nn.Tanh()),
                (12, in_features, None)
            ]
        else:
            enc_layers = [
                (in_features, 60, nn.Tanh()),
                (60, 30, nn.Tanh()),
                (30, 10, nn.Tanh()),
                (10, 1, None)
            ]
            dec_layers = [
                (1, 10, nn.Tanh()),
                (10, 30, nn.Tanh()),
                (30, 60, nn.Tanh()),
                (60, in_features, None)]
        return AutoEncoder(enc_layers, dec_layers)

    def _make_linear(self, layers: List[Tuple]):
        """
        This function builds a linear model whose units and layers depend on
        the passed @layers argument
        :param layers: a list of tuples indicating the layers architecture (in_neuron, out_neuron, activation_function)
        :return: a fully connected neural net (Sequentiel object)
        """
        net_layers = []
        for in_neuron, out_neuron, act_fn in layers:
            net_layers.append(nn.Linear(in_neuron, out_neuron))
            if act_fn:
                net_layers.append(act_fn)
        return nn.Sequential(*net_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        """
        This function compute the output of the network in the forward pass
        :param x: input
        :return: output of the model
        """
        output = self.encoder(x)
        output = self.decoder(output)
        return x, output

    def get_params(self) -> dict:
        return {
            "latent_dim": self.latent_dim
        }


class DAGMM(BaseModel):
    """
    This class proposes an unofficial implementation of the DAGMM architecture proposed in
    https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf.
    Simply put, it's an end-to-end trained auto-encoder network complemented by a distinct gaussian mixture network.
    """

    def __init__(self, lambda_1=0.005, lambda_2=0.1, reg_covar=1e-12, **kwargs):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.reg_covar = reg_covar
        self.ae = None
        self.gmm = None
        self.K = None
        self.latent_dim = None
        self.name = "DAGMM"
        super(DAGMM, self).__init__(**kwargs)
        self.cosim = nn.CosineSimilarity()
        self.softmax = nn.Softmax(dim=-1)

    def resolve_params(self, dataset_name: str):
        # defaults to parameters described in section 4.3 of the paper
        # https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf.
        latent_dim = self.latent_dim or 1
        if dataset_name == 'Arrhythmia':
            K = 2
            gmm_layers = [
                (latent_dim + 2, 10, nn.Tanh()),
                (None, None, nn.Dropout(0.5)),
                (10, K, nn.Softmax(dim=-1))
            ]
        elif dataset_name == "Thyroid":
            K = 2
            gmm_layers = [
                (latent_dim + 2, 10, nn.Tanh()),
                (None, None, nn.Dropout(0.5)),
                (10, K, nn.Softmax(dim=-1))
            ]
        else:
            K = 4
            gmm_layers = [
                (latent_dim + 2, 10, nn.Tanh()),
                (None, None, nn.Dropout(0.5)),
                (10, K, nn.Softmax(dim=-1))
            ]
        self.latent_dim = latent_dim
        self.K = K
        self.ae = AutoEncoder.from_dataset(self.in_features, dataset_name)
        self.gmm = GMM(gmm_layers)

    def forward(self, x: torch.Tensor):
        """
        This function compute the output of the network in the forward pass
        :param x: input
        :return: output of the model
        """

        # computes the z vector of the original paper (p.4), that is
        # :math:`z = [z_c, z_r]` with
        #   - :math:`z_c = h(x; \theta_e)`
        #   - :math:`z_r = f(x, x')`
        code = self.ae.encoder(x)
        x_prime = self.ae.decoder(code)
        rel_euc_dist = self.relative_euclidean_dist(x, x_prime)
        cosim = self.cosim(x, x_prime)
        z_r = torch.cat([code, rel_euc_dist.unsqueeze(-1), cosim.unsqueeze(-1)], dim=1)

        # compute gmm net output, that is
        #   - p = MLN(z, \theta_m) and
        #   - \gamma_hat = softmax(p)
        gamma_hat = self.gmm.forward(z_r)
        # gamma = self.softmax(output)

        return code, x_prime, cosim, z_r, gamma_hat

    def forward_end_dec(self, x: torch.Tensor):
        """
        This function compute the output of the network in the forward pass
        :param x: input
        :return: output of the model
        """

        # computes the z vector of the original paper (p.4), that is
        # :math:`z = [z_c, z_r]` with
        #   - :math:`z_c = h(x; \theta_e)`
        #   - :math:`z_r = f(x, x')`
        code = self.ae.encoder(x)
        x_prime = self.ae.decoder(code)
        rel_euc_dist = self.relative_euclidean_dist(x, x_prime)
        cosim = self.cosim(x, x_prime)
        z_r = torch.cat([code, rel_euc_dist.unsqueeze(-1), cosim.unsqueeze(-1)], dim=1)

        return code, x_prime, cosim, z_r

    def forward_estimation_net(self, z_r: torch.Tensor):
        """
        This function compute the output of the network in the forward pass
        :param z_r: input
        :return: output of the model
        """

        # compute gmm net output, that is
        #   - p = MLN(z, \theta_m) and
        #   - \gamma_hat = softmax(p)
        gamma_hat = self.gmm.forward(z_r)

        return gamma_hat

    def relative_euclidean_dist(self, x, x_prime):
        return (x - x_prime).norm(2, dim=1) / x.norm(2, dim=1)

    def compute_params(self, z: torch.Tensor, gamma: torch.Tensor):
        r"""
        Estimates the parameters of the GMM.
        Implements the following formulas (p.5):
            :math:`\hat{\phi_k} = \sum_{i=1}^N \frac{\hat{\gamma_{ik}}}{N}`
            :math:`\hat{\mu}_k = \frac{\sum{i=1}^N \hat{\gamma_{ik} z_i}}{\sum{i=1}^N \hat{\gamma_{ik}}}`
            :math:`\hat{\Sigma_k} = \frac{
                \sum{i=1}^N \hat{\gamma_{ik}} (z_i - \hat{\mu_k}) (z_i - \hat{\mu_k})^T}
                {\sum{i=1}^N \hat{\gamma_{ik}}
            }`

        The second formula was modified to use matrices instead:
            :math:`\hat{\mu}_k = (I * \Gamma)^{-1} (\gamma^T z)`

        Parameters
        ----------
        z: N x D matrix (n_samples, n_features)
        gamma: N x K matrix (n_samples, n_mixtures)


        Returns
        -------

        """
        N = z.shape[0]
        K = gamma.shape[1]

        # K
        gamma_sum = torch.sum(gamma, dim=0)
        phi = gamma_sum / N

        # phi = torch.mean(gamma, dim=0)

        # K x D
        # :math: `\mu = (I * gamma_sum)^{-1} * (\gamma^T * z)`
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
        # mu = torch.linalg.inv(torch.diag(gamma_sum)) @ (gamma.T @ z)

        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_mat = mu_z.unsqueeze(-1) @ mu_z.unsqueeze(-2)
        cov_mat = gamma.unsqueeze(-1).unsqueeze(-1) * cov_mat
        cov_mat = torch.sum(cov_mat, dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        # ==============
        K, N, D = gamma.shape[1], z.shape[0], z.shape[1]
        # (K,)
        gamma_sum = torch.sum(gamma, dim=0)
        # prob de x_i pour chaque cluster k
        phi_ = gamma_sum / N

        # K x D
        # :math: `\mu = (I * gamma_sum)^{-1} * (\gamma^T * z)`
        mu_ = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
        # Covariance (K x D x D)

        self.phi = phi.data
        self.mu = mu.data
        self.cov_mat = cov_mat
        # self.covs = covs
        # self.cov_mat = covs

        return phi, mu, cov_mat

    def estimate_sample_energy(self, z, phi=None, mu=None, cov_mat=None, average_energy=True):
        if phi is None:
            phi = self.phi
        if mu is None:
            mu = self.mu
        if cov_mat is None:
            cov_mat = self.cov_mat

        # jc_res = self.estimate_sample_energy_js(z, phi, mu)

        # Avoid non-invertible covariance matrix by adding small values (eps)
        d = z.shape[1]
        eps = self.reg_covar
        cov_mat = cov_mat + (torch.eye(d)).to(self.device) * eps
        # N x K x D
        mu_z = z.unsqueeze(1) - mu.unsqueeze(0)

        # scaler
        inv_cov_mat = torch.cholesky_inverse(torch.linalg.cholesky(cov_mat))
        # inv_cov_mat = torch.linalg.inv(cov_mat)
        det_cov_mat = torch.linalg.cholesky(2 * np.pi * cov_mat)
        det_cov_mat = torch.diagonal(det_cov_mat, dim1=1, dim2=2)
        det_cov_mat = torch.prod(det_cov_mat, dim=1)

        exp_term = torch.matmul(mu_z.unsqueeze(-2), inv_cov_mat)
        exp_term = torch.matmul(exp_term, mu_z.unsqueeze(-1))
        exp_term = - 0.5 * exp_term.squeeze()

        # Applying log-sum-exp stability trick
        # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        max_val = torch.max(exp_term.clamp(min=0), dim=1, keepdim=True)[0]
        exp_result = torch.exp(exp_term - max_val)

        log_term = phi * exp_result
        log_term /= det_cov_mat
        log_term = log_term.sum(axis=-1)

        # energy computation
        energy_result = - max_val.squeeze() - torch.log(log_term + eps)

        if average_energy:
            energy_result = energy_result.mean()

        # penalty term
        cov_diag = torch.diagonal(cov_mat, dim1=1, dim2=2)
        pen_cov_mat = (1 / cov_diag).sum()

        return energy_result, pen_cov_mat

    def compute_loss(self, x, x_prime, energy, pen_cov_mat):
        """

        """
        rec_err = ((x - x_prime) ** 2).mean()
        loss = rec_err + self.lambda_1 * energy + self.lambda_2 * pen_cov_mat

        return loss

    def get_params(self) -> dict:
        return {
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "latent_dim": self.ae.latent_dim,
            "K": self.gmm.K
        }

