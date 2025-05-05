import argparse
import ast

from Models.run_model import run_model
from Config.config import get_params_list
from Utils.utils import seed_everything

path_to_results = "./Results"

if __name__ == "__main__":

    #--------------- Parse Arguments ---------------#
    parser = argparse.ArgumentParser()

    # Environment Variables
    parser.add_argument("--seed", default=2025, type=int, help="Seeding Number")
    
    parser.add_argument("--suffix", default="", type=str,
                        help="Suffix to add to the end or result file name")
    
    # Data Variables
    parser.add_argument("--data", default="synthetic", type=str,
                        help="Choose which data to experiment on. \
                            Choose 'synthetic' to load random data based on distribution(s)")
    
    parser.add_argument("--dist", nargs='+', default = "uniform", type = str,
                        choices=["uniform", "gaussian", "poisson", 
                                 "exponential", "beta", "binomial", "lognormal"],
                        help="The name of the disrtibution(s)")
    
    parser.add_argument("-N", default=100, type=int,
                        help="Denote the total number of files that can be demanded")
    
    parser.add_argument("-k", default=10, type=int,
                        help="Size of cache")

    # Method Variables
    parser.add_argument("--model_name", default="FTPL", type=str, choices=["FTPL"],  #Spiked Cov to be added later
                        help="Model to use for prediction")

    parser.add_argument("-T", default=1000, type =  int,
                        help = "The number of iterations an epoch should run.")
    
    parser.add_argument("--num_runs", default=5, type =  int,
                        help = "The number of times a method should run.")
    
    # Hyperparameters (optional)
    parser.add_argument("--eta", nargs='+', default=None, type=float,
                        help="Learning rate (perturbation factor)")
    
    parser.add_argument("--feature_extraction", nargs='+', default=None, type=str, choices=["modified_spca", "sparse_pca", "pca"],
                        help="Which method to use to get probability vector")
    
    parser.add_argument("--sampling", nargs='+', default=None, type=str, choices=["madow", "top_k"],
                        help="Which method to use to sample cache vector from probability vector")
    
    parser.add_argument("--sample_gamma_once", nargs='+', default=None, type=bool,
                        help="Whether to sample gamma once before each epoch or at every instance")
    

    args = parser.parse_args()

    # Environment Variables
    seed    = args.seed
    suffix  = "_" + args.suffix if args.suffix != "" else ""
    
    # Data Variables
    data    = args.data
    dist    = args.dist
    N       = args.N
    k       = args.k

    assert k<=N, f"Size of cache (k={k}) must be less that total number of files (N={N})"

    # Method Variables
    model_name  = args.model_name
    T           = args.T
    num_runs    = args.num_runs

    # Hyperparameters (optional)
    eta                 = args.eta                if args.eta                 is not None else None
    feature_extraction  = args.feature_extraction if args.feature_extraction  is not None else None
    sampling            = args.sampling           if args.sampling            is not None else None
    sample_gamma_once   = args.sample_gamma_once  if args.sample_gamma_once   is not None else None
    

    #--------------- Seed Everything ---------------#
    seed_everything(seed=seed)
    
    #--------------- Load Data Config ---------------#
    params_list = get_params_list(model_name            = model_name,
                                  eta                   = eta,
                                  feature_extraction    = feature_extraction,
                                  sampling              = sampling,
                                  sample_gamma_once     = sample_gamma_once)


    #------------------  Run Model ------------------#
    result_addr = f"{path_to_results}/{data}_{model_name}{suffix}.csv" 
    run_model(N=N, k=k, data=data, dist=dist, params_list=params_list, T=T, num_runs=num_runs, result_addr=result_addr)