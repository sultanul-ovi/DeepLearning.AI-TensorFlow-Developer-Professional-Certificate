import numpy as np
import tensorflow as tf

from dlai_grader.grading import test_case, print_feedback
from types import FunctionType
from dlai_grader.io import suppress_stdout_stderr

SPLIT_TIME = 2500
WINDOW_SIZE = 64
BATCH_SIZE = 256
SHUFFLE_BUFFER_SIZE = 1000

def test_parse_data_from_file(learner_func):
    def g():
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "parse_data_from_file has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        t = test_case()
        try:
            time, temp = learner_func('data/daily-min-temperatures.csv')
        except Exception as e:
            t.failed = True
            t.msg = "There was an error evaluating parse_data_from_file function"
            t.want = "No exceptions"
            t.got = f"{str(e)}"
            return [t]
        
        t = test_case()
        if not isinstance(time, np.ndarray):
            t.failed = True
            t.msg = "Output variable 'time' has an incorrect type"
            t.want = np.ndarray
            t.got = type(time)
            return [t]
        
        t = test_case()
        if not isinstance(temp, np.ndarray):
            t.failed = True
            t.msg = "Output variable 'temperatures' has an incorrect type"
            t.want = np.ndarray
            t.got = type(temp)
            return [t]
        
        t = test_case()
        if not temp.dtype == float:
            t.failed = True
            t.msg = "Output 'temperatures' has a wrong data type"
            t.want = np.float
            t.got = float
            return [t]
        
        cases = []

        t = test_case()
        if not len(temp) == 3650: 
            t.failed = True
            t.msg =  "Output 'temperatures' has a wrong shape"
            t.want = 3650
            t.got = f"{len(temp)}"
        cases.append(t)
        
        t = test_case()	
        if not len(time) == 3650: 
            t.failed = True
            t.msg =  "Output 'times' has a wrong shape"
            t.want = 3650
            t.got = f"{len(time)}"
        cases.append(t)		
        
        t = test_case()
        if not all(temp[:5]==[20.7, 17.9, 18.8, 14.6, 15.8]):
            t.failed = True
            t.msg = "First 5 values of output 'temperatures' are incorrect"
            t.want = [20.7, 17.9, 18.8, 14.6, 15.8]
            t.got = temp[:5]
        cases.append(t)


        t = test_case()
        if not all(time[:5]==[0, 1, 2, 3, 4]):
            t.failed = True
            t.msg = "First 5 values of output 'times' are incorrect"
            t.want = [0, 1, 2, 3, 4]
            t.got = time[:5]
        cases.append(t)

        return cases 
    cases = g()
    print_feedback(cases)

def test_create_uncompiled_model(learner_func):
    def g():
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_uncompiled_model has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        
        t = test_case()
        try: 
            model = learner_func()
        except Exception as e:
            t.failed = True
            t.msg = "There was an error evaluating your function"
            t.want = "No exceptions"
            t.got = f"{str(e)}"
            return [t]
        
        t = test_case()
        if not isinstance(model, tf.keras.Model):
            t.failed = True
            t.msg = "create_uncompiled_model has a wrong output type"
            t.want = tf.keras.Model
            t.got = type(model)
            return [t]
        
        t = test_case()
        try:
            model_input = model.inputs[0]
        except Exception as e:
            t.failed = True
            t.msg = "your model is missing the Input"
            t.want = "a model with a defined Input"
            t.got = str(e)
            return [t]
        t = test_case()
        if not isinstance(model_input, tf.keras.KerasTensor):
            t.failed = True
            t.msg = "the input of your model has incorrect type"
            t.want = "a tf.keras.KerasTensor defined via tf.keras.Input"
            t.got = model_input
            return [t]
            
        input_shape = model.input_shape
        t = test_case()
        if input_shape not in [(None, None, 1), (None, WINDOW_SIZE, 1)]:
            t.failed = True
            t.msg = "model has incorrect input_shape"
            t.want = f"either (None, None, 1) or (None, {WINDOW_SIZE}, 1)"
            t.got = input_shape
            return [t]

        cases = []
        t = test_case()
        if not model.output_shape == (None, 1):
            t.failed = True
            t.msg = "The model has a wrong output shape"
            t.want = (None, 1)
            t.got = model.output_shape
        cases.append(t)

        t = test_case()
        if not model.loss == None:
            t.failed = True
            t.msg = "Seems like your model is already compiled"
            t.want = "An uncompiled model with no loss defined"
            t.got = f"loss: {model.loss}"
        cases.append(t)

        return cases 
    cases = g()
    print_feedback(cases)



def test_create_model(learner_func):
    def g():
        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "create_model has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        
        t = test_case()
        try: 
            model = learner_func()
        except Exception as e:
            t.failed = True
            t.msg = "There was en error evaluating create_model"
            t.want = "No exceptions"
            t.got = f"{str(e)}"
            return [t]
        
        t = test_case()
        if not isinstance(model, tf.keras.Model):
            t.failed = True
            t.msg = "create_model has a wrong output type"
            t.want = tf.keras.Model
            t.got = type(model)
            return [t]

        cases = []

        t = test_case()
        if isinstance(model.loss, tf.keras.losses.Loss):
            if not isinstance(model.loss, (tf.keras.losses.MeanSquaredError, tf.keras.losses.Huber)):
                t.failed = True
                t.msg = "incorrect loss function used for model"
                t.want = "and instance of tf.keras.losses.MeanSquaredError or tf.keras.losses.Huber"
                t.got = model.loss

        elif isinstance(model.loss, str):
            if not ("mse" in model.loss)|("huber" in model.loss):
                t.failed = True
                t.msg = "incorrect loss function used for model"
                t.want = "mse or huber"
                t.got = model.loss
        else:
            t.failed = True
            t.msg = "Wrong type for loss function"
            t.want = "a string or a class from tf.keras.losses"
            t.got = type(model.loss)
        cases.append(t)

        t = test_case()
        if not isinstance(model.optimizer, (tf.keras.optimizers.SGD, tf.keras.optimizers.Adam)):
            t.failed = True
            t.msg = "Got a wrong optimizer"
            t.want = f"{tf.keras.optimizers.SGD} or {tf.keras.optimizers.Adam}"
            t.got = model.optimizer
        cases.append(t)

        return cases 
    cases = g()
    print_feedback(cases)


def parameter_count(model):
    total_params_solution, train_params_solution = 60_000, 60_000
    total_params = model.count_params()
    num_trainable_params = sum(
        [w.shape.num_elements() for w in model.trainable_weights]
    )
    total_msg = f"\033[92mYour model has {total_params:,} total parameters and the reference is {total_params_solution:,}"
    train_msg = f"\033[92mYour model has {num_trainable_params:,} trainable parameters and the reference is {train_params_solution:,}"
    if total_params > total_params_solution:
        total_msg += f"\n\033[91mWarning! this exceeds the reference which is {total_params_solution:,}. If the kernel crashes while training, switch to a simpler architecture."
    else:
        total_msg += "\033[92m. You are good to go!"
    if num_trainable_params > train_params_solution:
        train_msg += f"\n\033[91mWarning! this exceeds the reference which is {train_params_solution:,}. If the kernel crashes while training, switch to a simpler architecture."
    else:
        train_msg += "\033[92m. You are good to go!"
    print(total_msg)
    print()
    print(train_msg)
