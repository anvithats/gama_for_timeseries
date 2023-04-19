import pandas as pd

from .gama import Gama
from gama.configuration.timeseries import timeseries_config
from gama.utilities.metrics import scoring_to_metric


class GamaForecaster(Gama):

    def __init__(self, config=None, scoring="neg_log_loss", *args, **kwargs):  # have to change the scoring
        if not config:
            config = timeseries_config

        self._metrics = scoring_to_metric(scoring)
        super().__init__(*args, **kwargs, config=config, scoring=scoring)

    def _predict(self, fh=None, x=None):
        y = self.predict(fh)
        return y

    def fit(self, y: pd.DataFrame, fh=None, x=None, *args, **kwargs):
        super().fit(y, fh, x, *args, **kwargs)
        # return self
