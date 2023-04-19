from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from gama import GamaForecaster


if __name__ == "__main__":
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=36)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    automl = GamaForecaster(max_total_time=180, store="nothing", n_jobs=1)
    print("Starting `fit`")
    automl.fit(y_train)
    y_pred = automl.predict(fh)

    # mape = MeanAbsolutePercentageError(symmetric=False)
    # print("forecast performance:", mape(y_test, y_pred))
