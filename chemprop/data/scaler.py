from typing import Any, List, Optional

import numpy as np


class StandardScaler:

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None,
                 replace_nan_token: Any = None, atomwise: bool = False,
                 no_scale : bool = False):
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token
        self.atomwise = atomwise
        self.no_scale = no_scale

    def fit(self, X: List[List[float]], atomlens : Optional[int] = None) -> 'StandardScaler':
        X = np.array(X).astype(float)
        if atomlens is not None and self.atomwise:
            atomlens = np.array(atomlens).astype(int).reshape(-1,1)
            X = X / atomlens

        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        if self.no_scale: self.stds = np.ones(self.means.shape)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X: List[List[float]], atomlens : Optional[int] = None):
        X = np.array(X).astype(float)
        transform_factor = self.means

        # If atomistic setting, multiple by number of atoms 
        if atomlens is not None and self.atomwise: 
            atomlens = np.array(atomlens).astype(int).reshape(-1,1)
            transform_factor = transform_factor * atomlens

        transformed_with_nan = (X - transform_factor) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X: List[List[float]], 
                          atomlens : Optional[int] = None):
        X = np.array(X).astype(float)

        transform_factor = self.means
        # If atomistic setting, multiple by number of atoms 
        if atomlens is not None and self.atomwise: 
            atomlens = np.array(atomlens).astype(int).reshape(-1, 1)
            transform_factor = transform_factor * atomlens

        transformed_with_nan = X * self.stds + transform_factor
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none
