import torch
import pandas


"""Read data (train, valid, test) from the local data directory

Parameters
----------
name : str
    directory name, for example, "mtcars", "gp_matern15"
sep : char, optional
    character use to separate entries
type: str, optional
    indicate whether to read from the "train", "valid", or "test" sub-folder

Returns
-------
tuple
    a tuple of X and y, features and labels
"""
def read_data(name, sep=",", type="train", floattype=torch.float32):
    file_path_X = f"./data/{name}/{type}/X.csv"
    file_path_y = f"./data/{name}/{type}/y.csv"

    try:
        X = torch.tensor(pandas.read_csv(file_path_X, sep=sep, header=None).values,
                         dtype=floattype)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path_X}")
        return None
    except pandas.errors.EmptyDataError:
        print(f"Error: CSV file is empty: {file_path_X}")
        return None
    except pandas.errors.ParserError:
         print(f"Error: Failed to parse CSV file: {file_path_X}. "
               "Check the delimiter and file format.")
         return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
    try:
        y = torch.tensor(pandas.read_csv(file_path_y, sep=sep, header=None).values,
                         dtype=floattype)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path_y}")
        return None
    except pandas.errors.EmptyDataError:
        print(f"Error: CSV file is empty: {file_path_y}")
        return None
    except pandas.errors.ParserError:
         print(f"Error: Failed to parse CSV file: {file_path_y}. "
               "Check the delimiter and file format.")
         return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
    return X, y 