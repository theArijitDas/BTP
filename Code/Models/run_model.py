from typing import Union, List

import numpy as np
from tqdm import tqdm

from Utils.utils import seed_everything, pretty_print_dict
from Utils.utils import dict_to_csv_strings, save_csv
from DataCode.data_load import RandomDataLoader
from Models.model_factory import get_model



def run_model(N: int, k: int, data:str, dist: Union[str, List[str]], params_list: dict, T: int, num_runs: int, result_addr: str):
    print()
    print("#"*50)
    print(f"N               : {N}")
    print(f"k               : {k}")
    print(f"Data            : {data}")
    print(f"Distribution(s) : {dist}")
    print(f"Num. Runs       : {num_runs}")
    print()
    for i, params in enumerate(params_list):
        print("#"*50)
        print(f"Parameter combination: {i+1}/{len(params_list)}\n")
        pretty_print_dict(params)
        print()

        regret_list = []
        for j in range(num_runs):
            seed_everything(j)

            if data=="synthetic":
                data_loader = RandomDataLoader(N, dist)
            
            reward_matrix_handler   = get_model(N, k, model_name="RewardMatrixHandler")
            model                   = get_model(N, k, **params)
            best_stationary_optimal = get_model(N, k, model_name = "BestStationaryOptimal")

            hit_list    = [0 for _ in range(T)]
            regret      = [None for _ in range(T)]

            for t in tqdm(range(T)):
                y               = model(reward_matrix_handler.R)
                y_best_stat     = best_stationary_optimal(reward_matrix_handler.R)

                x_t1, x_t2  = data_loader()

                hit_list[t] = np.dot(y, x_t1) * np.dot(y, x_t2)
                regret[t]   = (y_best_stat @ reward_matrix_handler.R @ y_best_stat) - sum(hit_list)

                reward_matrix_handler.update(x_t1, x_t2)
            
            regret_list.append(regret)
            
        #------------------  Save Output ------------------#
