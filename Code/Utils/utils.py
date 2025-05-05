import random
import os
import ast
import numpy as np
import json

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def one_hot_encode(value: int, N: int) -> np.ndarray:
    one_hot = np.zeros(N, dtype=int)
    one_hot[value] = 1
    return one_hot


def safe_get(d: dict, key, default):
    """
    Safer dict.get() that also treats None values as missing.
    Returns the value from dict if present and not None, else default.
    """
    val = d.get(key, default)
    return val if val is not None else default



def pretty_print_dict(d: dict):
    max_key_len = max(len(str(k)) for k in d.keys())
    for key, value in d.items():
        print(f"{str(key):<{max_key_len}} : {value}")


def dict_to_csv_strings(data: dict) -> tuple[str, str]:
    headers = []
    values = []

    for key, val in data.items():
        headers.append(key)

        # Convert to JSON if complex, else string
        if isinstance(val, (list, dict)):
            val_str = json.dumps(val)
        else:
            val_str = str(val)
        
        # Escape double quotes inside the value
        val_str = val_str.replace('"', '""')

        # Wrap value in double quotes to ensure it's treated as a single CSV cell
        values.append(f'"{val_str}"')

    header_str = ",".join(headers)
    value_str = ",".join(values)
    
    return header_str, value_str


def save_csv(addr, header_str, value_str):
    directory, filename = os.path.split(addr)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if not os.path.exists(addr):
        with open(addr, 'w') as f:
            f.write(header_str)
    
    with open(addr, 'a') as f:
        f.write('\n'+value_str)