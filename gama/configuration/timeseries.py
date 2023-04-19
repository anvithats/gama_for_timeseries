# preprocessing
from sklearn.preprocessing import PowerTransformer, RobustScaler, MinMaxScaler

# forecasters
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import STLForecaster, TrendForecaster, PolynomialTrendForecaster
from sktime.forecasting.naive import NaiveForecaster


timeseries_config = {
                    MinMaxScaler: {},
                    RobustScaler: {},
                    PowerTransformer: {},
                    NaiveForecaster: {
                                    "forecaster_strategy": ["drift", "last", "mean"],
                                    "forecaster_sp": [4, 6, 12],
                                    },
                    STLForecaster: {
                                    "forecaster_sp": [4, 6, 12],
                                    },
                    ThetaForecaster: {
                                    "forecaster_sp": [4, 6, 12],
                                    },
                    TrendForecaster: {},

                    PolynomialTrendForecaster: {},

                    }

