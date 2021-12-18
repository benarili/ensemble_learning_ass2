import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from rotation_forest import RotationForestClassifier
from hyperparameters import get_best_params


def create_decision_tree(X_train, y_train, optimize):
    clf = DecisionTreeClassifier()
    if optimize:
        params = get_best_params(X_train,y_train,clf)
        clf = DecisionTreeClassifier(**params)
    clf.fit(X_train, y_train)
    return clf


def create_random_forest(X_train, y_train, optimize):
    rfc = RandomForestClassifier()
    if optimize:
        params = get_best_params(X_train,y_train,rfc)
        rfc = RandomForestClassifier(**params)
    rfc.fit(X_train, y_train)
    return rfc


def create_rotation_forest(X_train, y_train, optimize):
    rf = RotationForestClassifier()
    if optimize:
        params = get_best_params(X_train,y_train,rf)
        rf = RotationForestClassifier(**params)
    rf.fit(X_train, y_train)
    return rf


def create_test_train_sets(data_info):
    df = pd.read_csv(data_info.get_csv_path())

    X = df[data_info.get_feature_cols()]
    y = df[data_info.get_target_col()]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    return X_test, X_train, y_test, y_train


def create_all_trees(metric, X_test, X_train, y_test, y_train, optimize_hyper_params=False):
    #create models
    clf = create_decision_tree(X_train, y_train,optimize_hyper_params)
    rfc = create_random_forest(X_train, y_train,optimize_hyper_params)
    rf = create_rotation_forest(X_train, y_train,optimize_hyper_params)

    #predict
    clf_pred = clf.predict(X_test)
    rfc_pred = rfc.predict(X_test)
    rf_pred = rf.predict(X_test)

    clf_acc, rf_acc, rfc_acc = get_metric(clf_pred, metric, rf_pred, rfc_pred, y_test)
    print(
        'decision tree score: {0}, random forest score {1}, rotation forest score {2}'.format(clf_acc, rfc_acc, rf_acc))


def get_metric(clf_pred, metric, rf_pred, rfc_pred, y_test):
    try:
        clf_acc = metric(clf_pred, y_test)
        rfc_acc = metric(rfc_pred, y_test)
        rf_acc = metric(rf_pred, y_test)
    except:
        clf_acc = metric(clf_pred, y_test, pos_label='positive', average='micro')
        rfc_acc = metric(rfc_pred, y_test, pos_label='positive', average='micro')
        rf_acc = metric(rf_pred, y_test, pos_label='positive', average='micro')
    return clf_acc, rf_acc, rfc_acc
