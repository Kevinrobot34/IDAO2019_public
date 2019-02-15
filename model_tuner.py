import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
# from xgboost import DMatrix, train
from optuna import create_study
from scoring import rejection90


def cat_study(x_trn, x_test, y_trn, y_test, study_weights=None, test_weights=None,
                cat_params=None, fit_params=None, target_params=None, n_trials=100):

    def objective(trial):

        def relu(x):
            return (x>0) * abs(x)

        if len(study_weights) > 1:

            train_data = Pool(
                data = x_trn.values,
                label = y_trn.values,
                weight = study_weights[0].values)

            eval_data = Pool(
                data = x_test.values,
                label = y_test.values,
                weight = study_weights[1].values)
        else:

            train_data = Pool(
                data = x_trn.values,
                label = y_trn.values,
                weight = study_weights.loc[x_trn.index].values)

            eval_data = Pool(
                data = x_test.values,
                label = y_test.values,
                weight = study_weights.loc[x_test.index].values)

        search_params = { }
        for k, v in target_params.items():
            if k in ['iterations', 'depth', 'random_strength', 'od_wait']:
                search_params[k] = trial.suggest_int(k, v[0], v[1])
            elif k in ['bagging_temperature', 'l2_leaf_reg']:
                search_params[k] = trial.suggest_loguniform(k, v[0], v[1])
            elif k in ['od_type', 'loss_function', 'eval_metric', 'learning_rate']:
                search_params[k] = trial.suggest_categorical(k, v)


        cat_params.update(search_params)

        cat = CatBoostRegressor(**cat_params)
        cat.fit(X=train_data, eval_set=eval_data, **fit_params)

        y_pred = cat.predict(x_test)
        score = rejection90(y_test, y_pred, sample_weight=test_weights.loc[y_test.index].values)

        return 1.0 - score

    study = create_study()
    study.optimize(objective, n_trials=n_trials)

    return study


def lgbm_study(x_trn, x_test, y_trn, y_test, study_weights=None, test_weights=None,
                lgbm_params=None, fit_params=None, target_params=None, n_trials=100):

    def objective(trial):



        search_params = { }
        for k, v in target_params.items():
            if k in ['n_estimators', 'max_depth', 'num_leaves', 'min_child_samples', 'num_boost_round', 'min_data_in_leaf', 'min_child_weight']:
                search_params[k] = trial.suggest_int(k, v[0], v[1])
            elif k in ['reg_alpha', 'reg_lambda']:
                search_params[k] = trial.suggest_loguniform(k, v[0], v[1])
            elif k in ['subsample', 'drop_rate', 'feature_fraction', 'learning_rate']:
                search_params[k] = trial.suggest_uniform(k, v[0], v[1])


        lgbm_params.update(search_params)

        lgbm = LGBMRegressor(**lgbm_params)
        lgbm.fit(x_trn, y_trn, sample_weight=study_weights.loc[y_trn.index].values, eval_set=(x_test, y_test),
        eval_sample_weight=[study_weights.loc[y_test.index].values], **fit_params)

        y_pred = lgbm.predict(x_test)
        score = rejection90(y_test, y_pred, sample_weight=test_weights.loc[y_test.index].values)

        return 1.0 - score

    study = create_study()
    study.optimize(objective, n_trials=n_trials)

    return study



def xgb_study(x_trn, x_test, y_trn, y_test, study_weights=None, test_weights=None,
                lgbm_params=None, fit_params=None, target_params=None, n_trials=100):

    def objective(trial):

        trainDMatrix = DMatrix(x_trn.values, label=y_trn, weight=study_weights.loc[y_trn.index])
        validDMatrix = DMatrix(x_test.values, label=y_tets, weight=study_weights.loc[y_test.index])

        search_params = { }
        for k, v in target_params.items():
            if k in ['max_depth', 'num_leaves', 'min_child_samples', 'min_data_in_leaf', 'min_child_weight']:
                search_params[k] = trial.suggest_int(k, v[0], v[1])
            elif k in ['reg_alpha', 'reg_lambda']:
                search_params[k] = trial.suggest_loguniform(k, v[0], v[1])
            elif k in ['subsample', 'drop_rate', 'feature_fraction', 'learning_rate']:
                search_params[k] = trial.suggest_uniform(k, v[0], v[1])


        xgb_params.update(search_params)

        xgb = train(xgb_params,
           trainDMatrix, evals=[(validDMatrix, 'valid')], **fit_params)
        y_pred = xgb.predict(DMatrix(x_test.values))
        score = rejection90(y_test, y_pred, sample_weight=test_weights.loc[y_test.index].values)

        return 1.0 - score

    study = create_study()
    study.optimize(objective, n_trials=n_trials)

    return study
