from Models.model_utils import RewardMatrixHandler, FTPLModel, BestStationaryOptimal

def get_model(N, k, **params):
    if params["model_name"] == "RewardMatrixHandler":
        return RewardMatrixHandler(N)

    if params["model_name"] == "FTPL":
        return FTPLModel(N=N, k=k, eta=params["eta"],
                         feature_extraction=params["feature_extraction"],
                         sample_gamma_once=params["sample_gamma_once"])
    
    if params["model_name"] == "SpikedCovariance":
        pass
    
    if params["model_name"] == "BestStationaryOptimal":
        return BestStationaryOptimal(N, k)