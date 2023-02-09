r"""
Train and test the inverse problem, i.e., given mean and variance
predict the values of the parameters. However, this is a harder
problem with respect to the previous one, due to the presence of
outliners. For this reason more robust approach is used: RandomForest
"""

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from typing import Iterable, Tuple, List, Dict, Any
from fsml.utils import read_csv_content, evaluate, print_dict
import fsml.learn.config as config
import os.path as opath
import os
import pickle


def split(columns: Iterable[str]) -> Tuple[List[str], List[str]]:
    """ Split the input columns to find the parameters and the variables names """
    # I have structured the data such that all the output variables
    # are written in uppercase, and the parameters in lowercase.
    parameters = [col for col in columns if col.islower() and col != "time"]
    outputs    = [col for col in columns if col.isupper()]
    return parameters, outputs


def get_ranges(in_number: int) -> List[int]:
    """ Given an integer it returns [i / 2, i, i + i / 2] """
    return [in_number // 2, in_number, in_number + in_number // 2]


def save_model(estimator: RandomForestRegressor, filepath: str) -> None:
    """ Save the input model into the input file path """
    with open(filepath, mode='rb') as iostream:
        pickle.dump(estimator, iostream)


def perform_random_search(
    train_x: List[List[float]], train_y: List[List[float]], 
    test_x : List[List[float]], test_y : List[List[float]]
) -> Tuple[float, Dict[str, Any]]:
    r""" 
    Perform Random Search

    :param train_x: the input features of the train set
    :param train_y: the ground truth of the train set
    :param test_x: the input features of the test set
    :param test_y: the ground truth of the test set
    :return: The accuracy 
    """
    print("[*] Creating the Random Grid for a first Hyper-parameter Tuning")
    random_grid = {
        'n_estimators'     : config.RAND_SEARCH_N_ESTIMATORS,
        'max_features'     : config.RAND_SEARCH_MAX_FEATURES,
        'max_depth'        : config.RAND_SEARCH_MAX_DEPTH,
        'min_samples_split': config.RAND_SEARCH_MIN_SAMPLES_SPLIT,
        'min_samples_leaf' : config.RAND_SEARCH_MIN_SAMPLES_LEAF,
        'bootstrap'        : config.RAND_SEARCH_BOOSTRAP
    }

    print_dict(random_grid)

    rf_random_search = RandomForestRegressor()
    random_search = RandomizedSearchCV(estimator=rf_random_search,
                                       param_distributions=random_grid,
                                       n_iter=config.RAND_SEARCH_NUM_ITERATIONS,
                                       cv=config.RAND_SEARCH_NUM_CROSS_VALIDATION,
                                       verbose=1,
                                       random_state=42,
                                       n_jobs=-1)
    
    print("[*] Fitting the Random Search")
    random_search.fit(train_x, train_y)

    print("Random Search Result -- Best Parameters")
    print_dict(random_search.best_params_)

    print("Random Search Result -- Best Estimator")
    best_random_estimator = random_search.best_estimator_
    print(best_random_estimator)
    random_accuracy = evaluate(best_random_estimator, test_x, test_y)

    return random_accuracy, random_search.best_params_


def perform_grid_search(
    params : Dict[str, Any],     train_x: List[List[float]], 
    train_y: List[List[float]],  test_x : List[List[float]], 
    test_y : List[List[float]]
) -> Tuple[float, RandomForestRegressor]:
    r"""
    Perform Grid search from results obtained from RandomSearch
    and compare the obtained accuracy with the one given as input.

    :param params: The parameters from RandomSearch
    :param random_acc: the input accuracy
    :return: The best estimator
    """
    print("[*] Setting up GridSearch with parameters")
    params_grid = {
        'bootstrap'         : [True],
        'n_estimators'      : get_ranges(params['n_estimators']),
        'max_features'      : get_ranges(params['max_features']),
        'max_depth'         : get_ranges(params['max_depth']),
        'min_samples_leaf'  : get_ranges(params['min_samples_leaf']),
        'min_samples_split' : get_ranges(params['min_samples_split'])
    }
    print_dict(params_grid)

    # Create the base random forest and the GridSearch
    base_rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=base_rf, 
                               param_grid=params_grid,
                               cv=config.GRID_SEARCH_NUM_CROSS_VALIDATION,
                               n_jobs=-1,
                               verbose=1)
    
    # Fit the Grid Search
    print("[*] Fitting the GridSearch")
    grid_search.fit(train_x, train_y)

    print("Grid Search Results --- Best Parameters")
    print_dict(grid_search.best_params_)

    print("Grid Search Results --- Best Estimator")
    best_estimator = grid_search.best_estimator_
    print(best_estimator)
    grid_acc = evaluate(best_estimator, test_x, test_y)

    return grid_acc, best_estimator


def train_and_test(data_file: str, random_search: bool=False, grid_search: bool=False) -> None:
    r"""
    Train and test a RandomForest Regressor. The regressor can either be a
    default one, defined in the config file, or the best estimator
    obtained by running first RandomForest and then Grid Search or just
    Grid Search with a set of default possibilities. Finally it saves
    the model to the default model path

    :param data_file: the file with the dataset
    :param random_search: True to apply random search
    :param grid_search: True to apply Grid Search
    :return:
    """
    _, points = read_csv_content(data_file)
    params, output = split(points)

    input_data = points.loc[:, output].values.tolist()
    output_data = points.loc[:, params].values.tolist()

    scaler = StandardScaler()
    scaler.fit(input_data)
    input_data = scaler.transform(input_data)

    train_x_data, test_x_data, train_y_data, test_y_data = train_test_split(
        input_data, output_data, train_size=0.75, test_size=0.25, random_state=42
    )

    print("[*] Creating the Base RandomForest Regressor")
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    print(regressor)
    print("[*] Fitting the Base Regressor")
    regressor.fit(train_x_data, train_y_data)
    base_accuracy = evaluate(regressor, test_x_data, test_y_data)
    
    print("[*] Currently used parameters")
    print_dict(regressor.get_params())

    random_params = config.GRID_SEARCH_BASE_PARAMETERS
    if random_search:
        random_accuracy, random_params = perform_random_search(train_x_data, train_y_data,
                                                test_x_data,  test_y_data)
        
        print('Improvement of {:0.2f}% with Random Search.'.format( 
            100 * (random_accuracy - base_accuracy) / base_accuracy
        ))

    filepath_linux_format = opath.basename(data_file).replace('\\', '/')
    csv_filename = opath.basename(filepath_linux_format)
    model_path = opath.join(
        config.MODEL_PATH, 
        f"{csv_filename}_RandomForest_KFoldCrossValidation" + \
            f"{config.GRID_SEARCH_NUM_CROSS_VALIDATION}.pth"
    )

    if grid_search:
        grid_acc, estimator = perform_grid_search(
            random_params, train_x_data, train_y_data,
            test_x_data,  test_y_data
        )

        print('Improvement of {:0.2f}% with Grid Search.'.format( 
            100 * (grid_acc - base_accuracy) / base_accuracy
        ))

        save_model(estimator, model_path)

        return model_path
    
    # If no grid search is done use a default regressor
    regressor = RandomForestRegressor(max_depth=100, 
                                      max_features='sqrt', 
                                      min_samples_leaf=4,
                                      min_samples_split=10, 
                                      n_estimators=800,
                                      verbose=1)
    
    regressor.fit(train_x_data, train_y_data)
    predictions = regressor.predict(test_x_data)
    evaluate(predictions, test_y_data)
    save_model(regressor, model_path)

    return model_path