import numpy as np
import tensorflow as tf

from dlai_grader.grading import test_case, print_feedback
from types import FunctionType
from dlai_grader.io import suppress_stdout_stderr

SPLIT_TIME = 1100
WINDOW_SIZE = 20
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

def test_windowed_dataset(learner_func):
    def g():
        function_name = "windowed_dataset"
        
        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        
        series_train = np.linspace(0,5,100)
        t = test_case()
        try:
            dataset = learner_func(series_train, window_size=WINDOW_SIZE)
        except Exception as e:
            t.failed = True
            t.msg = "There was an error evaluating the train_val_datasets function."
            t.want = "No exceptions"
            t.got = f"{str(e)}"
            return [t]

        t = test_case()
        if not isinstance(dataset, tf.data.Dataset):
            t.failed = True
            t.msg = f"windowed_dataset's return value has incorrect type"
            t.want = tf.data.Dataset
            t.got = type(dataset)
            return [t]       

        batch_of_features, batch_of_labels = next((iter(dataset)))
        t = test_case()
        if not isinstance(batch_of_features, tf.Tensor):
            t.failed =True
            t.msg =  f"batch_of_features has incorrect type"
            t.want = tf.Tensor
            t.got = type(batch_of_features)
            return [t]

        t = test_case()
        if not isinstance(batch_of_labels, tf.Tensor):
            t.failed = True
            t.msg =  f"batch_of_labels has incorrect type"
            t.want = tf.Tensor
            t.got = type(batch_of_labels)
            return [t]
		
        t = test_case()
        if not batch_of_features.dtype in (tf.float32, tf.float64):
            t.failed =True
            t.msg =  f"batch_of_features has incorrect data  type"
            t.want = f"{tf.float32} or {tf.float64}"
            t.got = batch_of_features.dtype
        cases.append(t)

        t = test_case()
        if not batch_of_labels.dtype in (tf.float32, tf.float64):
            t.failed = True
            t.msg =  f"batch_of_labels has incorrect data type"
            t.want =  f"{tf.float32} or {tf.float64}"
            t.got = batch_of_labels.dtype
        cases.append(t)
		
        t = test_case()
        if batch_of_features.shape[0] != 32:
            t.failed = True
            t.msg =  f"batch_of_features has incorrect batch_size when using window_size=1"
            t.want = (32)
            t.got = batch_of_features.shape[0]
        cases.append(t)

        t = test_case()
        if batch_of_features.shape[1:] != (WINDOW_SIZE):
            t.failed = True
            t.msg =  f"batch_of_features has incorrect length when using window_size=1"
            t.want = (WINDOW_SIZE)
            t.got = batch_of_features.shape[1:]
        cases.append(t)
		
        t = test_case()
        if batch_of_labels.shape != (32,):
            t.failed = True
            t.msg =  f"batch_of_labels has incorrect shape when using window_size=1"
            t.want = (32,)
            t.got = batch_of_labels.shape
        cases.append(t)
        
        return cases
    
    cases = g()
    print_feedback(cases)


def test_create_model(learner_func, windowed_dataset):
    def g():
        function_name = "create_model"

        cases = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = f"{function_name} has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        
        t = test_case()
        try:
            model = learner_func(window_size=1)
        except Exception as e:
            t.failed = True
            t.msg = "There was an error evaluating the create_model function."
            t.want = "No exceptions"
            t.got = f"{str(e)}"
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


        t = test_case()
        if not model.input_shape[1] == 1:
            t.failed = True
            t.msg = "Got a wrong input shape for window_size=1"
            t.want = 1
            t.got = model.input_shape[1]
            return [t]
        cases.append(t)

        t = test_case()
        if not model.output_shape[1] == 1:
            t.failed = True
            t.msg = "Got a wrong output shape for the model"
            t.want = 1
            t.got = model.output_shape[1]
            return [t]
        cases.append(t)

              
        t = test_case()
        try: 
            data = windowed_dataset(range(32),1)
        except Exception as e:
            t.failed = True
            t.msg = "There was an error evaluating the windowed_dataset function. Please check that that function passes all the tests and come back"
            t.want = "No exceptions"
            t.got = f"{str(e)}"
            return [t]
        
        t = test_case()
        try:
            with suppress_stdout_stderr():
                model.fit(data)
        except Exception as e:
            t.failed = True
            t.msg = "There was an error training your model"
            t.want = "No exceptions"
            t.got = f"{str(e)}"
            return [t]	

        t = test_case()
        if isinstance(model.loss, tf.keras.losses.Loss):
            if not isinstance(model.loss, tf.keras.losses.MeanSquaredError):
                t.failed = True
                t.msg = "incorrect loss function used for model"
                t.want = "and instance of tf.keras.losses.MeanSquaredError"
                t.got = model.loss

        elif isinstance(model.loss, str):
            if "mse" not in model.loss:
                t.failed = True
                t.msg = "incorrect loss function used for model"
                t.want = "mse"
                t.got = model.loss
        else:
            t.failed = True
            t.msg = "Wrong type for loss function"
            t.want = "a string or a class from tf.keras.losses"
            t.got = type(model.loss)
        cases.append(t)

        return cases

    cases = g()
    print_feedback(cases)

def parameter_count(model):
    total_params_solution, train_params_solution = 3_200, 3_200
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