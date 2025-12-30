"""
Data loading module.

This module loads the breast cancer dataset and
converts it into a Pandas DataFrame.
"""
from sklearn.datasets import load_breast_cancer
import pandas as pd

def load_data():
    """
    Load the breast cancer dataset.

    Returns:
        pd.DataFrame: DataFrame containing features and target column.
    """
    # pylint: disable=E1101
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df
