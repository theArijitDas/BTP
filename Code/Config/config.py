import itertools

hyperparams = {
    "eta"               : [0.001, 0.01, 0.1],  # Learning rate
    "feature_extraction": ["sparce_pca", "pca"],
    "sample_gamma_once" : [True, False]
}

def get_params_list(model_name: str, **kwargs) -> list[dict]:
    
    model_params = {
        "model_name"        : [model_name],
        "eta"               : kwargs.get("eta"                  , hyperparams["eta"]),
        "feature_extraction": kwargs.get("feature_extraction"   , hyperparams["feature_extraction"]),
        "sample_gamma_once" : kwargs.get("sample_gamma_once"    , hyperparams["sample_gamma_once"])
    }

    params_list = []
    for value_combination in itertools.product(*model_params.values()):
        params_list.append({key: value for key, value in zip(model_params.keys(), value_combination)})
    
    return params_list