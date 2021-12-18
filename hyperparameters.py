
import math

import numpy as np
from sklearn.model_selection import RandomizedSearchCV


def get_descicion_tree_grid():
    global dtc_grid
    # grid for descision tree classifier
    rf_max_depth = [int(x) for x in np.linspace(5, 55, 11)]
    # Add the default as a possible value
    rf_max_depth.append(None)
    max_f = ['log2', 'sqrt', 'auto']
    max_f.append(None)
    dtc_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': rf_max_depth,
        'min_samples_split': [int(x) for x in np.linspace(2, 10, 9)],
        'min_samples_leaf': [int(x) for x in np.linspace(1, 10, 5)],
        'max_features': max_f,
        'random_state': [None, 1, 0, 42],
        'min_impurity_decrease': [0.0, 0.05, 0.1]
    }


get_descicion_tree_grid()


# grid for RandomForestClassifier
def create_random_forest_hyperparam_grid(X):
    # grid for descision tree classifier
    rf_max_depth = [int(x) for x in np.linspace(5, 55, 11)]
    # Add the default as a possible value
    rf_max_depth.append(None)
    max_leaf_nodes_arr = [int(x) for x in np.linspace(1, 10, 5)]
    max_leaf_nodes_arr.append(None)
    rfc_grid = {
        'n_estimators': [x for x in range(int(math.sqrt(X.size)))],
        'criterion': ['gini', 'entropy'],
        'max_depth': rf_max_depth,
        'min_samples_split': [int(x) for x in np.linspace(2, 10, 9)],
        'max_features': ['log2', 'sqrt', 'auto'],
        'max_leaf_nodes': max_leaf_nodes_arr,
        'bootstrap': [True, False],
        'random_state': [None, 1, 0, 42],
        'verbose': [int(x) for x in np.linspace(0, 10, 7)]
    }
    return rfc_grid


def get_best_params(X, Y, base):
    # rf_base = DecisionTreeClassifier()
    base_type = type(base).__name__
    if base_type == 'DecisionTreeClassifier':
        rf_grid = dtc_grid
    else:
        rf_grid = create_random_forest_hyperparam_grid(X)
    rf_random = RandomizedSearchCV(estimator=base, param_distributions=rf_grid,
                                   n_iter=200, cv=5, verbose=2, random_state=42,
                                   n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X, Y)

    # best parameters from the random search
    bp = rf_random.best_params_

    print(bp)
    return bp
