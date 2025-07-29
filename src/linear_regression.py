import numpy as np
import pandas as pd

def mean_normalization(X) -> pd.Series:
    return (X - X.mean()) / (X.max() - X.min())


def z_score_normalization(X) -> pd.Series:
    return (X - X.mean()) / X.std()

