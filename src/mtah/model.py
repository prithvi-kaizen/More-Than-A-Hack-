"""Separate boosted alive/dead and live-ownership emission models."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class PossessionModel:
    def __init__(self, max_iter=120):
        kwargs = dict(max_iter=max_iter, max_leaf_nodes=15, min_samples_leaf=80,
                      l2_regularization=1, early_stopping=False, random_state=2026)
        self.alive_model = HistGradientBoostingClassifier(**kwargs)
        self.owner_model = HistGradientBoostingClassifier(**kwargs)

    def fit(self, alive_features, owner_features, labels, alive):
        alive_features, owner_features = np.asarray(alive_features), np.asarray(owner_features)
        labels, alive = np.asarray(labels), np.asarray(alive)
        valid = np.isin(alive, [0, 1])
        live = (alive == 1) & np.isin(labels, ['H', 'A'])
        if set(alive[valid]) != {0, 1} or set(labels[live]) != {'H', 'A'}:
            raise ValueError('TRAINING_CLASSES_MISSING')
        self.alive_model.fit(alive_features[valid], alive[valid])
        x, y = owner_features[live], (labels[live] == 'H').astype(int)
        self.owner_model.fit(np.concatenate((x, -x)), np.concatenate((y, 1-y)))
        return self

    def predict_proba(self, alive_features, owner_features):
        live = self.alive_model.predict_proba(alive_features)[:, 1]
        home = 0.5*(self.owner_model.predict_proba(owner_features)[:, 1] +
                    1-self.owner_model.predict_proba(-owner_features)[:, 1])
        return np.column_stack((live*home, live*(1-home), 1-live))
