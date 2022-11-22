import pandas as pd


def load_dataset(path, stratify=False):
    """Get the data from csv file

    Args:
        path(str): the file complete path.

    Returns:
        dataframe: A pandas dataframe.
    """
    dataset = pd.read_csv(path)

    if stratify:
        dataset = dataset.groupby("polarity", group_keys=False).apply(
            lambda x: x.sample(frac=0.4)
        )
        dataset.reset_index(drop=True, inplace=True)

    return dataset
