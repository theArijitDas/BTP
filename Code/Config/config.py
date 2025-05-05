import itertools
from Utils.utils import safe_get

hyperparams = {
    "eta"               : [0.001, 0.01],
    "feature_extraction": ["modified_spca", "sparse_pca", "pca"],
    "sampling"          : ["madow", "top_k"],
    "sample_gamma_once" : [True, False]
}

def get_params_list(model_name: str, **kwargs) -> list[dict]:
    model_params = {
        "model_name"        : [model_name],
        "eta"               : safe_get(kwargs, "eta"                  , hyperparams["eta"]               ),
        "feature_extraction": safe_get(kwargs, "feature_extraction"   , hyperparams["feature_extraction"]),
        "sampling"          : safe_get(kwargs, "sampling"             , hyperparams["sampling"]          ),
        "sample_gamma_once" : safe_get(kwargs, "sample_gamma_once"    , hyperparams["sample_gamma_once"] ),
    }

    params_list = []
    for value_combination in itertools.product(*model_params.values()):
        params_list.append({key: value for key, value in zip(model_params.keys(), value_combination)})
    
    return params_list