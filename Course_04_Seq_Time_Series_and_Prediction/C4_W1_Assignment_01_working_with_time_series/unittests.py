import numpy as np
from dlai_grader.grading import test_case, print_feedback
from types import FunctionType
import tensorflow as tf
from dlai_grader.io import suppress_stdout_stderr
import re

SPLIT_TIME = 1100
WINDOW_SIZE = 50


def test_train_val_split(learner_func):
    def g():
        cases = []
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "train_val_split has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        dummy_times = np.arange(1200)
        dummy_series = np.arange(1200)

        sol_time_train = sol_series_train = np.arange(1100)
        sol_time_valid = sol_series_valid = np.arange(1100, 1200)

        t = test_case()
        try:
            time_train, series_train, time_valid, series_valid = learner_func(
                dummy_times, dummy_series
            )
        except Exception as e:
            t.failed = True
            t.msg = "There was an error evaluating the train_val_split function"
            t.want = "No exceptions"
            t.got = f"{str(e)}"
            return [t]

        t = test_case()
        if not type(time_train) == np.ndarray:
            t.failed = True
            t.msg = "Output time_train has a wrong type"
            t.want = np.ndarray
            t.got = type(time_train)
            return [t]

        t = test_case()
        if not type(series_train) == np.ndarray:
            t.failed = True
            t.msg = "Output series_train has a wrong type"
            t.want = np.ndarray
            t.got = type(series_train)
            return [t]

        t = test_case()
        if not type(time_valid) == np.ndarray:
            t.failed = True
            t.msg = "Output time_valid has a wrong type"
            t.want = np.ndarray
            t.got = type(time_valid)
            return [t]

        t = test_case()
        if not type(series_valid) == np.ndarray:
            t.failed = True
            t.msg = "Output series_valid has a wrong type"
            t.want = np.ndarray
            t.got = type(series_valid)
            return [t]

        t = test_case()
        if not len(time_train) == 1100:
            t.failed = True
            t.msg = "Output time_train has wrong length"
            t.want = 1100
            t.got = len(time_train)
        else:
            if not np.allclose(time_train, sol_time_train):
                t.failed = True
                t.msg = f"Got incorrect values for time_train when using times={dummy_times} and a split at time_step={2}"
                t.want = sol_time_train
                t.got = time_train
        cases.append(t)

        t = test_case()
        if not len(series_train) == 1100:
            t.failed = True
            t.msg = "Output series_train has wrong length"
            t.want = 1100
            t.got = len(series_train)
        else:
            if not np.allclose(series_train, sol_series_train):
                t.failed = True
                t.msg = f"Got incorrect values for series_train when using series={dummy_series} and a split at time_step={2}"
                t.want = sol_series_train
                t.got = series_train
        cases.append(t)

        t = test_case()
        if not len(time_valid) == len(dummy_times) - 1100:
            t.failed = True
            t.msg = "Output time_valid has wrong length"
            t.want = len(dummy_times) - 1100
            t.got = len(time_valid)
        else:
            if not np.allclose(time_valid, sol_time_valid):
                t.failed = True
                t.msg = f"Got incorrect values for time_valid when using times={dummy_times} and a split at time_step={2}"
                t.want = sol_time_valid
                t.got = time_valid
        cases.append(t)

        t = test_case()
        if not len(series_valid) == len(dummy_times) - 1100:
            t.failed = True
            t.msg = "Output series_valid has wrong length"
            t.want = len(dummy_times) - 1100
            t.got = len(series_valid)
        else:
            if not np.allclose(series_valid, sol_series_valid):
                t.failed = True
                t.msg = f"Got incorrect values fo series_valid when using series={dummy_series} and a split at time_step={2}"
                t.want = sol_series_valid
                t.got = series_valid
        cases.append(t)
        return cases

    cases = g()
    print_feedback(cases)


def test_compute_metrics(learner_func):
    def g():
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "compute_metrics has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        zeros = np.zeros(5)
        ones = np.ones(5)
        sol_mse, sol_mae = 1.0, 1.0

        t = test_case()
        try:
            mse, mae = learner_func(zeros, ones)
        except Exception as e:
            t.failed = True
            t.msg = "There was an error evaluating the compute_metrics function"
            t.want = "No exceptions"
            t.got = f"{str(e)}"
            return [t]

        t = test_case()
        if not np.issubdtype(type(mse), np.number):
            t.failed = True
            t.msg = f"mse has incorrect type"
            t.want = "a subdtype of np.number"
            t.got = type(mse)
            return [t]

        t = test_case()
        if not np.issubdtype(type(mae), np.number):
            t.failed = True
            t.msg = f"mae has incorrect type"
            t.want = "a subdtype of np.number"
            t.got = type(mae)
            return [t]

        cases = []
        t = test_case()
        if mse != sol_mse:
            t.failed = True
            t.msg = f"incorrect mse for series of zeros and forecasts of ones"
            t.want = sol_mse
            t.got = mse
        cases.append(t)

        t = test_case()
        if mae != sol_mae:
            t.failed = True
            t.msg = f"incorrect mae for series of zeros and forecasts of ones"
            t.want = sol_mae
            t.got = mae
        cases.append(t)

        t = test_case
        mse, mae = learner_func(ones, ones)
        sol_mse, sol_mae = 0.0, 0.0

        t = test_case()
        if mse != sol_mse:
            t.failed = True
            t.msg = f"incorrect mse for series of ones and forecasts of ones"
            t.want = sol_mse
            t.got = mse
        cases.append(t)

        t = test_case()
        if mae != sol_mae:
            t.failed = True
            t.msg = f"incorrect mae for series of ones and forecasts of ones"
            t.want = sol_mae
            t.got = mae
        cases.append(t)

        dummy_series = np.array([1, 2, 3, 4, 5])
        dummy_forecast = np.array([6, 7, 8, 9, 10])

        mse, mae = learner_func(dummy_series, dummy_forecast)
        sol_mse, sol_mae = 25.0, 5.0

        t = test_case()
        if mse != sol_mse:
            t.failed = True
            t.msg = f"incorrect mse for series={dummy_series} and forecasts={dummy_forecast}"
            t.want = sol_mse
            t.got = mse
        cases.append(t)

        t = test_case()
        if mae != sol_mae:
            t.failed = True
            t.msg = f"incorrect mae for series={dummy_series} and forecasts={dummy_forecast}"
            t.want = sol_mae
            t.got = mae
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def test_naive_forecast(learner_var):
    def g():
        t = test_case()
        if not isinstance(learner_var, np.ndarray):
            t.failed = True
            t.msg = "naive_forecast has incorrect type"
            t.want = np.ndarray
            t.got = type(learner_var)
            return [t]

        cases = []

        t = test_case()

        if learner_var.shape != (361,):
            t.failed = True
            t.msg = f"naive_forecast has incorrect shape"
            t.want = (361,)
            t.got = learner_var.shape
        cases.append(t)

        t = test_case()
        first_solution_forecast = [60.526978, 61.19469, 52.63221, 61.69174, 56.193645]
        if not np.allclose(first_solution_forecast, learner_var[:5]):
            t.failed = True
            t.msg = f"naive_forecast has incorrect first 5 values"
            t.want = first_solution_forecast
            t.got = learner_var[:5]
        cases.append(t)

        t = test_case()
        last_solution_forecast = np.array(
            [21.317099, 26.351381, 25.613142, 27.43651, 26.209599]
        )
        if not np.allclose(last_solution_forecast, learner_var[-5:]):
            t.failed = True
            t.msg = f"naive_forecast has incorrect last 5 values"
            t.want = last_solution_forecast
            t.got = learner_var[-5:]
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def test_moving_average_forecast(learner_func):
    def g():
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "naive_forecast has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        dummy_series = np.arange(0, 30)
        t = test_case()
        try:
            learner_mvg_avg = learner_func(dummy_series, 5)
        except Exception as e:
            t.failed = True
            t.msg = "There was an error evaluating the moving_average_forecast function"
            t.want = "No exceptions"
            t.got = f"{str(e)}"
            return [t]

        solution_mvg_avg = np.array(
            [
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
                25.0,
                26.0,
            ]
        )

        t = test_case()
        if not isinstance(learner_mvg_avg, np.ndarray):
            t.failed = True
            t.msg = f"moving_average_forecast has incorrect return type"
            t.want = np.ndarray
            t.got = type(learner_mvg_avg)
            return [t]

        t = test_case()
        if not learner_mvg_avg.shape == solution_mvg_avg.shape:
            t.failed = True
            t.msg = "np_forecast has a wrong shape"
            t.want = solution_mvg_avg.shape
            t.got = learner_mvg_avg.shape
            return [t]

        cases = []
        t = test_case()
        if not np.allclose(learner_mvg_avg, solution_mvg_avg):
            t.failed = True
            t.msg = f"moving_average_forecast returned incorrect values for series: {dummy_series} and window_size={5}"
            t.want = solution_mvg_avg
            t.got = learner_mvg_avg
        cases.append(t)

        t = test_case()
        learner_mvg_avg = learner_func(dummy_series, 15)
        solution_mvg_avg = np.array(
            [
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
                21.0,
            ]
        )
        if not np.allclose(learner_mvg_avg, solution_mvg_avg):
            t.failed = True
            t.msg = f"moving_average_forecast returned incorrect values for series: {dummy_series} and window_size={15}"
            t.want = solution_mvg_avg
            t.got = learner_mvg_avg
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def test_diff_series(learner_var):
    def g():
        t = test_case()
        if not isinstance(learner_var, np.ndarray):
            t.failed = True
            t.msg = "diff_series has incorrect type"
            t.want = np.ndarray
            t.got = type(learner_var)
            return [t]

        t = test_case()

        if not learner_var.shape == (1096,):
            t.failed = True
            t.msg = f"diff_series has incorrect shape"
            t.want = (1096,)
            t.got = learner_var.shape
            return [t]

        cases = []
        t = test_case()
        sol_diff_series_init = np.array(
            [1.8541336, 4.37471, 2.3798103, 0.7992935, 2.5722847]
        )
        if not np.allclose(learner_var[:5], sol_diff_series_init):
            t.failed = True
            t.msg = f"diff_series has incorrect first 5 values"
            t.want = sol_diff_series_init
            t.got = learner_var[:5]
        cases.append(t)

        sol_diff_series_end = np.array(
            [5.231127, 2.957058, 6.125614, 3.531084, 3.534523]
        )
        t = test_case()
        if not np.allclose(learner_var[-5:], sol_diff_series_end):
            t.failed = True
            t.msg = f"diff_series has incorrect last 5 values"
            t.want = sol_diff_series_end
            t.got = learner_var[-5:]
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def test_diff_moving_avg(learner_var):
    def g():
        t = test_case()
        if not isinstance(learner_var, np.ndarray):
            t.failed = True
            t.msg = "diff_moving_avg has incorrect type"
            t.want = np.ndarray
            t.got = type(learner_var)
            return [t]

        t = test_case()
        if not learner_var.shape == (361,):
            t.failed = True
            t.msg = f"diff_moving_avg has incorrect shape"
            t.want = (361,)
            t.got = learner_var.shape
            return [t]

        cases = []

        sol_diff_ma_start = np.array(
            [3.7926393, 3.8671985, 3.7156622, 3.725135, 3.6709442]
        )
        t = test_case()
        if not np.allclose(learner_var[:5], sol_diff_ma_start):
            t.failed = True
            t.msg = f"diff_moving_avg has incorrect first 5 values"
            t.want = sol_diff_ma_start
            t.got = learner_var[:5]
        cases.append(t)

        sol_diff_ma_end = np.array([4.1262302, 4.19637, 4.165605, 4.2264843, 4.2272344])
        t = test_case()
        if not np.allclose(learner_var[-5:], sol_diff_ma_end):
            t.failed = True
            t.msg = f"diff_moving_avg has incorrect last 5 values"
            t.want = sol_diff_ma_end
            t.got = learner_var[-5:]
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def test_diff_moving_avg_plus_past(learner_var):
    def g():
        t = test_case()
        if not isinstance(learner_var, np.ndarray):
            t.failed = True
            t.msg = "diff_moving_avg_plus_past has incorrect type"
            t.want = np.ndarray
            t.got = type(learner_var)
            return [t]

        cases = []

        t = test_case()
        if not learner_var.shape == (361,):
            t.failed = True
            t.msg = f"diff_moving_avg_plus_past has incorrect shape"
            t.want = (361,)
            t.got = learner_var.shape
            return [t]

        t = test_case()
        sol_start = np.array([60.286503, 59.153976, 59.546032, 59.878906, 57.351723])
        if not np.allclose(learner_var[:5], sol_start):
            t.failed = True
            t.msg = f"diff_moving_avg_plus_past has incorrect first 5 values"
            t.want = sol_start
            t.got = learner_var[:5]
        cases.append(t)

        t = test_case()
        sol_end = np.array([25.246485, 26.852455, 25.476501, 26.904999, 65.0158])
        if not np.allclose(learner_var[-5:], sol_end):
            t.failed = True
            t.msg = f"diff_moving_avg_plus_past has incorrect last 5 values"
            t.want = sol_end
            t.got = learner_var[-5:]
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)


def test_smooth_past_series(learner_var):
    def g():
        t = test_case()
        if not isinstance(learner_var, np.ndarray):
            t.failed = True
            t.msg = "smooth_past_series has incorrect type"
            t.want = np.ndarray
            t.got = type(learner_var)
            return [t]

        cases = []

        t = test_case()
        if not learner_var.shape == (361,):
            t.failed = True
            t.msg = f"diff_moving_avg_plus_smooth_past has incorrect shape"
            t.want = (361,)
            t.got = learner_var.shape
            return [t]

        t = test_case()
        sol_smooth_past_start = np.array(
            [55.13228, 54.10422, 53.275375, 52.182724, 51.436935]
        )
        if not np.allclose(learner_var[:5], sol_smooth_past_start):
            t.failed = True
            t.msg = f"diff_moving_avg_plus_smooth_past has incorrect first 5 values"
            t.want = sol_smooth_past_start
            t.got = learner_var[:5]
        cases.append(t)

        t = test_case()
        sol_smooth_past_end = np.array(
            [28.81515, 31.856792, 35.53977, 39.16417, 42.837337]
        )
        if not np.allclose(learner_var[-5:], sol_smooth_past_end):
            t.failed = True
            t.msg = f"diff_moving_avg_plus_smooth_past has incorrect last 5 values"
            t.want = sol_smooth_past_end
            t.got = learner_var[-5:]
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)
