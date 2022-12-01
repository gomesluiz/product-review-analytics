import os
import pandas as pd

def download_dataset(source) -> None:
    """Download data from a url.
    
    Args:
        source (str): source data file
        
    Returns:
        None
    """
       
    return pd.read_csv(source)