import numpy as np
import torch
from collections import defaultdict

# moving average, W = (1-alpha) * W + alpha * new_W
# alpha: staleness
class FedAsync():
    def __init__(self, param_keys, total_clients, alpha=0.6, strategy="polynomial", a=0.5):
        self.param_keys = param_keys
        self.alpha = alpha
        self.a = a
        self.total_clients = total_clients
        self.delay = defaultdict(int)
        self.strategy = strategy

    def update(self, cid, orig_model_params, client_params):
        # orig_model_params: global model weights
        if sum(self.delay.values()) == 0:
            alpha = 1
        elif self.strategy == "constant":
            alpha = self.alpha
        elif self.strategy == "hinge":
            raise NotImplementedError
        elif self.strategy == "polynomial":
            alpha = self.alpha * (self.delay[cid] + 1) ** (-self.a)

        updated_model_params = {}
        for k, new_p in client_params.items():
            updated_model_params[k] = (1 - alpha) * orig_model_params[k] + alpha * new_p

        self.delay[cid] = 0
        for cc in range(self.total_clients):
            if cc != cid:
                self.delay[cc] += 1

        return updated_model_params


# only orthogonal to gradients within delay window
class OrthoFL():
    def __init__(self, param_keys):
        self.param_keys = param_keys
        self.g = defaultdict(dict)
        self.bn = defaultdict(dict)

    def update(self, cid, inital_model_params, curr_global_params, full_params):
        # orig_model_params: global model weights
        # client_agg_weights: number of data samples
        updated_model_params = {}
        for k, p in full_params.items():
            # batch normalization
            if k.find('bn') >= 0:
                updated_model_params[k] = p
            else:
                # the delta weight given by other clients that change during the delay
                delay_p = curr_global_params[k].to(p.device) - inital_model_params[k].to(p.device)
                delta_p = p - inital_model_params[k].to(p.device)
                # orthogonal to the gradients within delay window
                if torch.sum(delay_p).item() != 0:
                    # project gradient
                    new_delta_params = self.project_to_orthogonal(delay_p.to(p.device), delta_p.to(p.device))
                    updated_model_params[k] = p + new_delta_params
                else: # no delay
                    updated_model_params[k] = p

                self.g[cid][k] = delta_p.view(1, -1)

        return updated_model_params

    def project_to_orthogonal(self, orig_grad, other_grad):
        # remove projection on the direction that needs to be orthogonal
        dot_product = torch.dot(orig_grad.flatten(), other_grad.flatten())
        norm_o_squared = torch.dot(other_grad.flatten(), other_grad.flatten()) + 1e-19

        return orig_grad - (other_grad * dot_product / norm_o_squared).view(*orig_grad.size())

    def update_k(self, k, orig_p, client_delta_params, client_agg_weights):
        k_agg_weights = []
        stacked_delta_params = []
        for c, model_params in client_delta_params.items():
            if k not in model_params:
                continue
            k_agg_weights.append(client_agg_weights[c])
            stacked_delta_params.append(model_params[k].cpu())

        if not len(k_agg_weights):
            return

        k_agg_weights = torch.FloatTensor(k_agg_weights)
        k_agg_weights /= k_agg_weights.sum()
        for _ in orig_p.shape:
            k_agg_weights = k_agg_weights.unsqueeze(-1)

        if k.find('running_mean') != -1 or k.find('running_var') != -1:
            new_p = 0.5 * (orig_p + (k_agg_weights * torch.stack(stacked_delta_params, 0)).sum(0).to(orig_p.device))
        elif k.find('num_batches_tracked') != -1:
            new_p = orig_p - torch.stack(stacked_delta_params, 0).sum(0).to(orig_p.device)
        else:
            new_p = orig_p - (k_agg_weights * torch.stack(stacked_delta_params, 0)).sum(0).to(orig_p.device)

        return new_p

