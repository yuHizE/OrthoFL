
def convert_model_key_to_idx(global_key_to_idx, model_params):
    converted_model_params = {}
    for k, p in model_params.items():
        if k in global_key_to_idx:
            global_k_idx = global_key_to_idx[k]
        else:
             continue

        converted_model_params[global_k_idx] = p.cpu()

    return converted_model_params


def prepare_client_params(model, recv_weight):
    new_model_params = {}
    for k, p in model.state_dict().items():
        if k in recv_weight:
            new_model_params[k] = recv_weight[k].cpu()
        else:
            raise KeyError(f'Not found {k} ({p.size()}) in recv_weight')

    return new_model_params