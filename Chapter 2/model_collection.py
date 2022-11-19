import os
import time
import pickle
from joblib import dump, load
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet, BayesianRidge, LinearRegression
from sklearn.svm import LinearSVR
from sklearn.kernel_approximation import Nystroem

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

# on mac, dask xgboost relies on your hostname resolving to an address
# you may have to edit your /etc/hosts file: https://apple.stackexchange.com/questions/253817/cannot-ping-my-local-machine
# related information here: https://bugs.python.org/issue35164
# from dask_ml.xgboost import XGBRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import cross_validate

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

class ModelCollection:
    """
    A class which helps to train and score a subset of sklearn algorithms.
    
    This class is not meant to do all of the work in terms of hyperparameter tuning,
    merely it serves to provide convenience for fitting and collecting cross-validated scores
    of several model in a few lines of code. Optimizing parameters should be done by hand and the
    model in question should be replaced in the dictionary. Note that any model can be added to the
    models attribute so long as the model provides the regular sklearn api functions of fit() and score()
    
    Below is a listing of class attributes which can be modified directly but are not 
    changeable via the constructor. See the constructor docstring for more variables
    that can be set.
    
    Attributes
    ----------
    scoring : array of strings
        passed to sklearn.model_selection.cross_validate's scoring parameter
    models : dictionary of strings and models
        models presenting an sklearn-like API
    scores_ : dictionary
        keys are strings of the model names with values being a dictionary
        indicating the r2 score, mean absoulte error, and root mean squared error
        this attribute is set after calling compute_scores()
    cross_val_scores_ : dictionary
        keys are strings of the model names with values being the 
        output of sklearn.model_selection.cross_validate
        this attribute is set after calling compute_cross_validation_scores()
    """
    scoring = ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
    
    def __init__(self, x, y, cv=10, stratify=False):
        """
        Initialize the necessary data for this class
        
        Parameters
        ----------
        x : pandas.DataFrame
            the covariates used to predict x
        y : pandas.Series
            the dependant variable being predicted
        cv : int, or CV Splitter, default 10
            the number of folds to use for cross validation. If stratify is True, this will also determine
            the number of bins to use to stratify y
        stratify : boolean, optional default False
        """
        
        self.x = x.copy()
        self.y = y.copy()
        
        self.scores_ = {}
        self.cross_val_scores_ = {}
        
        self.cv = cv
        self.stratified_kfold = stratify
        
        self._setup_models()
    
    def fit(self):
        """
        Fit each model with the data passed into the constructor
        
        Returns
        ----------
        fit_times: dictionary
            keys are strings of the model names, values are floats representing the time taken
            to fit the model in seconds
        """
        fit_times = {}
        
        pbar = tqdm(self.models.keys())
        for name in pbar:
            pbar.set_description(f'Fitting {name}')
            dt = self.fit_model(name)
            fit_times[name] = dt
        
        return fit_times
    
    def fit_model(self, model_name):
        """
        Fit the model
        
        Parameters
        ----------
        model_name: str
            The name of the model to be fit. This name must be a key in self.models
            
        Returns
        ----------
        fit_times: dictionary
            keys are strings of the model names, values are floats representing the time taken
            to fit the model in seconds
        """
        
        _x, _y = self.x.copy(), self.y.copy()
        
        start = time.time()
            
        model = self.models[model_name]
        self.models[model_name] = model.fit(_x, _y)

        # in seconds
        return time.time() - start
    
    def compute_cross_validation_scores(self):
        """
        Compute the cross validated scores using sklearn.model_selection.cross_validate for
        each model.
        
        This method sets the cross_val_scores_ attribute on this object.
        
        Returns
        ----------
        cross_val_scores_ : dictionary
            keys are strings of the model names with values being the output 
            of sklearn.model_selection.cross_validate
        """
        self.cross_val_scores_ = {}
        pbar = tqdm(self.models.items())
        for name, _ in pbar:
            pbar.set_description(f'Cross-validating {name}')
            try:
                self.compute_cross_validation_for_model(name)
            except KeyboardInterrupt:
                continue
            
        return self.cross_val_scores_

    def compute_cross_validation_for_model(self, model_name):
        """
        Compute the cross validated scores using sklearn.model_selection.cross_validate for
        each model.
        
        If this object has previously had a cross-validation score made for `model_name`,
        then the cross_val_scores_ dictionary is updated
        
        Parameters
        ----------
        model_name: str
            The name of the model to be fit. This name must be a key in self.models
            
        Returns
        ----------
        cross_val_scores_ : dictionary
            keys are strings of the model names with values being the output 
            of sklearn.model_selection.cross_validate
        """

        # model collections made with ModelCollection.__new__() will not have been initizlied
        # this checks for that case and mostly happens when a new model is made solely to read
        # scores from a previous run
        if not 'cross_val_scores_' in self.__dict__:
            self.cross_val_scores_ = {}
        else:
            if model_name in self.cross_val_scores_:
                del self.cross_val_scores_[model_name]

        model = self.models[model_name]
        # only run cross validation in parallel for models that train in serial
        # otherwise, run cross validation in serial so that models that can train in parallel are able to do so
        cross_val_parallel = int(os.cpu_count() / 2)
        if getattr(model, 'n_jobs', None) is not None:
            # if this model is setup to run in parallel, let it do so and run cross validation in serial
            if model.n_jobs is not None:
                cross_val_parallel = None

        _x, _y = self.x.copy(), self.y.copy()
        self.cross_val_scores_[model_name] = cross_validate(model, 
                                  _x, _y, 
                                  cv = self.cv, 
                                  scoring = self.scoring,
                                  n_jobs=cross_val_parallel,
                                  error_score='raise')

        return self.cross_val_scores_
          
    def compute_scores(self, x, y):
        """
        Compute the r2, mae, and rmse scores for each model using x and y.
        
        This method sets the cross_val_scores_ attribute on this object.
        
        Parameters
        ----------
        x : pandas DataFrame
            the covariates used to predict x
        y : array_like
            the dependant variable being predicted
            
        Returns
        ----------
        scores_ : dictionary
            keys are strings of the model names with values being a dictionary
            indicating the r2 score, mean absoulte error, and root mean squared error
            this attribute is set after calling compute_scores()
        """
        self.scores_ = {}
        for name in self.models.keys():
            r2, rmse, mae = self.compute_scores_for_model(name, x, y)

            self.scores_[name] = {
                'r2' : r2,
                'rmse' : rmse,
                'mae' : mae
            }
            
        return self.scores_
    
    def compute_scores_for_model(self, model_name, x, y):
        """
        Compute the r2, mae, and rmse scores for each model using x and y.
        
        This method sets the cross_val_scores_ attribute on this object.
        
        Parameters
        ----------
        model_name : string
            Which model to score
        x : pandas DataFrame
            the covariates used to predict x
        y : array_like
            the dependant variable being predicted
            
        Returns
        ----------
        scores : tuple
            the r2 score, root mean squared error, and mean absoulte error
        """
        model = self.models[model_name]
        
        y_pred = model.predict(x)
        scores = (
            r2_score(y, y_pred),
            mean_squared_error(y, y_pred, squared=False),
            mean_absolute_error(y, y_pred)
        )
            
        return scores
    
    def predict(self, x):
        predicitons = {}
        
        _x = None 
        for model_name in self.models.keys():
            predicitons[model_name] = self.predict_model(model_name, x)

        return predicitons
    
    def predict_model(self, model_name, x):
        model = self.models[model_name]
        return model.predict(x)
        
    def save(self, filepath):
        """
        Save all of the models to the disk using joblib as recommended 
        by sklearn.
        
        Read more here: https://sklearn.org/modules/model_persistence.html
        
        Parameters
        ----------
        filename : string, optional default ''
            The full path and filename to save the model to. This savs with joblib,
            so it would be smart to include the extension of .joblib.
        """
        
        ar = [(name, model) for name, model in self.models.items()]
        dump(ar, filepath)
        
    def load(self, filename):
        """
        Load all of the models from the disk using joblib as recommended 
        by sklearn.
        
        Read more here: https://sklearn.org/modules/model_persistence.html
        
        Parameters
        ----------
        filename : string, optional default ''
            The full path and filename to load the model from.
        """
        ar = load(filename)
        names_models = ar
        self.models = {name: model for name, model in names_models}
            
    def save_model_cross_val_scores(self, filename=''):
        """
        Save all of the model scores to the disk.
        
        Parameters
        ----------
        filename : string, optional default ''
            The full path and filename to save the scores to.
        """

        with open(filename, 'wb') as f: 
            pickle.dump(self.cross_val_scores_, f)

    def load_model_cross_val_scores(self, filename=''):
        """
        Load all of the model scores from the disk.
        
        Parameters
        ----------
        filename : string, optional default ''
            The full path and filename to load the scores from.
        """
        
        with open(filename, 'rb') as f: 
            self.cross_val_scores_ = pickle.load(f)
    
    def tune_hyperparameters(self, hyperparameters = None):
        """
        Choose the best set of model parameters based off of self.x and self.y

        Parameters
        ----------
        hyperparameters : dict, default None
            the string keys identify the model and must match the keys in self.models. These 
            parameters are used in hyperparameter tuning. The default search space can be found
            by reading this method's code. Model hyperparamters are preferentially chosen from
            this argument, otherwise the defaults are used.
            
        Returns
        ----------
        hyperparameters_ : dictionary
            keys are strings of the model names with values being a dictionary
            indicating what values best fit this dataset. Note: if hyperparameters_ exist
            when computing cross validation or training, the hyperparameters from the best model
            are used.
        """

        parameters = {
#             'Linear Regression': None,
#             'Elastic Net' : dict(
#                 elastic_net__alpha=np.arange(0.05, 1.05, .05),
#                 elastic_net__l1_ratio=np.arange(0, 1, 0.01),
#                 elastic_net__selection = ['cyclic', 'random'],
#             ),
#             'Polynomial' : None,
#             'Bayesian Ridge' : dict(
#                 bayesian_ridge__alpha_1=[1e-6, 1e-5, 1e-4],
#                 bayesian_ridge__alpha_2=[1e-6, 1e-5, 1e-4],
#                 bayesian_ridge__lambda_1=[1e-6, 1e-5, 1e-4],
#                 bayesian_ridge__lambda_2=[1e-6, 1e-5, 1e-4],
#             ),
#             'SVR' : dict(
#                 nystroem__gamma=np.geomspace(1e-3, 1e3, 10),
#                 nystroem__n_components=[100, 200, 300],
#                 svr__C=np.geomspace(1e-3, 1e3, 10)
#             ),
#             'MLP' : dict(
#                 mlp__hidden_layer_sizes = [
#                     (10), (50), (100),
#                     (100, 50, 10), 
#                     (10, 100, 50), 
#                     (10, 50, 100), 
#                 ]
#             ),
#             'Ada Boost' : dict(
#                 ada_boost__n_estimators = [100, 500, 1000],
#                 ada_boost__learning_rate = np.arange(0, 1.1, 0.1)
#             ),
            'Random Forest' : dict(
                random_forest__n_estimators = [200, 300, 400, 500],
                random_forest__min_samples_split = [2],
                random_forest__min_samples_leaf = [1], 
                random_forest__max_features = [4],
                random_forest__max_depth = [20, 30, 40],
            ),
            'Extra Trees' : dict(
                extra_trees__n_estimators = [200, 300, 400, 500],
                extra_trees__min_samples_split = [2],
                extra_trees__min_samples_leaf = [1], 
                extra_trees__max_features = [4],
                extra_trees__max_depth = [20, 30, 40, 50],
            ),
#             'Gradient Boost' : dict(
#                 gradient_boost__learning_rate = np.arange(0, 1.1, 0.1),
#                 gradient_boost__n_estimators = [100, 500, 1000],
#                 gradient_boost__min_samples_split = [2, 3, 4],
#                 gradient_boost__min_samples_leaf = [1, 2, 3], 
#             ),
#             'XGBoost' : dict(
#                 xgboost__learning_rate = np.arange(0, 1.1, 0.1),
#                 xgboost__n_estimators = [100, 500, 1000],
#             )
        }

        if hyperparameters:
            parameters.update(hyperparameters)

        self.hyperparameters_ = {}

        pbar = tqdm(parameters.items())
        for model, params in pbar:
            pbar.set_description(f'Tuning {model}')
            try:
                if params and (model in self.models):
                    clf = GridSearchCV(self.models[model], params, n_jobs=4, cv=self.cv)
                    clf.fit(self.x, self.y)
                    self.hyperparameters_[model] = clf
            except KeyboardInterrupt:
                continue

        return self.hyperparameters_
    
    def update_models_from_hyperparamters(self, hyperparamters = None):
        """Update self.models from a set of hyperparameters

        Parameters
        ----------
        hyperparameters : dict, default None
            the string keys identify the model and must match the keys in self.models. The value is a GridSearchCV object.
            At minimum, the value must support the .best_estimator_ attribute and be able to replace a model in self.models
            If none are provided, this method attempts to use self.hyperparameters_. If self.hyperparamters_
            do not exist, no change happens
        """

        if hyperparamters is None:
            hyperparamters = self.hyperparameters_
        
        for model, params in hyperparamters.items():
            if params:
                self.models[model] = params.best_estimator_
    
    def get_named_pipeline_keys(self):
        """Return the names used to identify each model in the pipelines
        
        Returns
        ----------
        keys : list
            A list of strings identifying the names of each model in the pipeline
        """

        keys = []

        for _, pipeline in self.models.items():
            print(pipeline.named_steps[-1])

        return keys

    def _setup_models(self):
        """
        Initialize the dictionary of models
        
        
        The individual models can be modified after this class is constructed by replacing
        values in self.models
        """
        max_iter = 50_000
        mlp = MLPRegressor(max_iter=max_iter, early_stopping=True)

        self.models = {
            'Linear Regression': Pipeline([('linear_regression', LinearRegression())]),
            'Polynomial' : Pipeline([('poly', PolynomialFeatures()), ('linear_poly', LinearRegression())]),
            'Bayesian Ridge' : Pipeline([('bayesian_ridge', BayesianRidge())]),
            'MLP' : Pipeline([('scale', StandardScaler()), ('mlp', mlp)]),
            'Random Forest' : Pipeline([('random_forest', RandomForestRegressor(n_jobs=None))]),
            'Extra Trees' : Pipeline([('extra_trees', ExtraTreesRegressor(n_jobs=None))]),
            'XGBoost' : Pipeline([('xgboost', XGBRegressor(n_jobs=-1, subsample=0.9, colsample_bytree=0.5))])
        }
    
class ModelStack():
    """
    A class which will stack the "best" models in a ModelCollection.
    
    To better estimate the dependent variable, the fitted models can be stacked.
    The function that will do the stacking can be any function that follows the sklearn API.
    By default, the stacking function is a multiple linear regressor and method 1 below is used.

    Stacking can be done in 2 ways:
        1. The predicted values from the selected models will be run through the stacking model
            and regressed against the predicted values.
        2. The residuals of the best models will be fitted to the stacking model.
    
    The linear mixed effect models are not considered as part of the ModelStack. This is because the data needd for a linear mixed effect
    is different than data for other models. The VotingRegressor cannot handle that situation.
    
    Attributes
    ----------
    mc : ModelCollection
        The fitted ModelCollection passed to the constructor
    stacker : object
        An estimator with an sklearn-like API
    estimation : string, one of ["predicted", "residuals"]
        "predicted" means that the stacker will be fitted with the predictions of each model, regressing
        against the actual value
        "residuals" means that the stacker will be regressed against the residuals
    """
    
    def __init__(self, mc, x, y, n = 3, criterion='rmse'):
        """
        Initialize the necessary data for this class

        Parameters
        ----------
        mc : ModelCollection
            A scored and fitted ModelCollection
        x : pandas DataFrame
            the covariates used to predict x
        y : array_like
            the dependant variable being predicted
        n : integer, default 3
            How many regressors to choose to form a stack
        criterion : string, optional default 'rmse'
            One of ['rmse', 'r2', 'mae']. This is the criterion used to rank
            the error of the models in the model collection and the top n will be used
            to fit an sklearn.ensemble.VotingRegressor
        """
        self.mc = mc
        self.x = x.copy()
        self.y = y.copy()

        self.n = n
        self.criterion = criterion
        
        self.voting_regressor_ = None

    def fit(self):
        """ Choose the model with the top self.n scores according to self.how and fit a voting
            regressor with them.
        """
        
        if getattr(self.mc, 'scores_', None) is None or not self.mc.scores_:
            scores = self.mc.compute_scores(self.x, self.y)
        else:
            scores = self.mc.scores_

        # split the model names and scores into two lists
        names, scores = map(np.array, zip(*[(name, vals[self.criterion]) for name, vals in scores.items()]))
        # sort them
        sort = np.argsort(scores)
        
        # r2 is a number between -inf and 1, but usually between 0 and 1, so reverse
        # the sort so that the best scores are first
        # for mae and rmse, smaller numbers are better models so the sort is already in the correct order
        if self.criterion == 'r2':
            sort = sort[::-1]
        
        # now, sort the models and scores by the criterion and pick off the number we need
        # to create the voting regressor
        names, scores = names[sort][:self.n], scores[sort][:self.n]
        
        models = [(name, self.mc.models[name]) for name in names]
        
        self.voting_regressor_ = VotingRegressor(models, n_jobs=-1)
        
        self.voting_regressor_.fit(self.x, self.y)
        
    def compute_scores(self, x, y):
        """
        Compute the r2, mae, and rmse scores for each model using x and y.
        
        This method sets the cross_val_scores_ attribute on this object.
        
        Parameters
        ----------
        x : pandas DataFrame
            the covariates used to predict x
        y : array_like
            the dependant variable being predicted
            
        Returns
        ----------
        scores_ : dictionary
            keys are strings of the model names with values being a dictionary
            indicating the r2 score, mean absoulte error, and root mean squared error
            this attribute is set after calling compute_scores()
        """
        
        y_pred = self.voting_regressor_.predict(self.x)
        
        self.scores_ = {
            'r2' : r2_score(y, y_pred),
            'rmse' : mean_squared_error(y, y_pred, squared=False),
            'mae' : mean_absolute_error(y, y_pred)
        }
            
        return self.scores_
    
    def predict(self, x):
        """Predict the input data
        Parameters
        ----------
        x : pandas DataFrame
            the covariates used to predict x
        Returns
        ----------
        prediction : array_like
            The results of the prediction made by the voting regressor
        """
        
        prediction = self.voting_regressor_.predict(self.x)
        return prediction

    def save(self, filepath):
        """
        Save the vorint regressor to the disk using joblib as recommended 
        by sklearn.
        
        Read more here: https://sklearn.org/modules/model_persistence.html
        
        Parameters
        ----------
        filename : string, optional default ''
            The full path and filename to save the model to. This savs with joblib,
            so it would be smart to include the extension of .joblib.
        """
        
        ar = [self.voting_regressor_]
        dump(ar, filepath)
        
    def load(self, filename):
        """
        Load the voting regressor from the disk using joblib as recommended 
        by sklearn.
        
        Read more here: https://sklearn.org/modules/model_persistence.html
        
        Parameters
        ----------
        filename : string, optional default ''
            The full path and filename to load the model from.

        """

        ar = load(filename)
        self.voting_regressor_ = ar[0]