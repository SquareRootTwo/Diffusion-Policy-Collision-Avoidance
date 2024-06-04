import torch

def normalize_data(data, stats):
    """
    Normalize data to [-1,1] range
    """
    # stats.shape = (Data_dimension, 2)
    # data.shape = (B, Timestep, Data_dimension)
    # nomalize to [0,1]
    data = (data - stats[:, 0]) / (stats[:, 1] - stats[:, 0] + 1e-6) 

    # normalize to [-1,1]
    data = data * 2 - 1

    return data


def unnormalize_data(data, stats):
    """
    Unnormalize data from [-1,1] range
    """

    # stats.shape = (Data_dimension, 2)
    # data.shape = (B, Timestep, Data_dimension)
    # unnormalize from [-1,1]
    data = (data + 1) / 2

    # write this for loop without for loop
    data = data * (stats[:, 1] - stats[:, 0] + 1e-6) + stats[:, 0]
    
    return data
