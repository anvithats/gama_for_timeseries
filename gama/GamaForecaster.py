import inspect
import pandas as pd
from .gama import Gama
from gama.configuration.timeseries import timeseries_config
from gama.utilities.metrics import scoring_to_metric
from sktime.base import BaseEstimator


class GamaForecaster(Gama):

    def __init__(self, config=None, scoring="neg_log_loss", *args, **kwargs):
        if not config:
            config = timeseries_config

        self._metrics = scoring_to_metric(scoring)
        if any(metric.requires_probabilities for metric in self._metrics):
            config = {
                alg: hp
                for (alg, hp) in config.items()
                if not (
                        inspect.isclass(alg)
                        and issubclass(alg, BaseEstimator)
                        and not hasattr(alg(), "predict_proba")
                )
            }
        super().__init__(*args, **kwargs, config=config, scoring=scoring)

    def _predict(self, fh=None, x=None):
        y = self.model.predict(fh)
        return y

    def fit(self, y: pd.DataFrame, x, fh=None, *args, **kwargs):
        super().fit(x, y, fh, *args, **kwargs)

