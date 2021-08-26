# from dask_ml.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV, BayesianRidge, LinearRegression
from sklearn.svm import LinearSVR
from sklearn import linear_model

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# on mac, dask xgboost relies on your hostname resolving to an address
# you may have to edit your /etc/hosts file: https://apple.stackexchange.com/questions/253817/cannot-ping-my-local-machine
# related information here: https://bugs.python.org/issue35164
# from dask_ml.xgboost import XGBRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

from sklearn.pipeline import Pipeline

from dask_ml.preprocessing import StandardScaler
from dask_ml.preprocessing import PolynomialFeatures
import dask.array as da

import time

import pickle
from joblib import dump, load
import joblib

from statsmodels.api import MixedLM
from sklearn.base import BaseEstimator, RegressorMixin

from pymer4 import Lmer

from tqdm.notebook import tqdm
import glob

from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import numpy as np
import pandas as pd

class MixedLMWrapper(BaseEstimator, RegressorMixin):
    """ 
    An sklearn-style wrapper for the statsmodels MixedLM regressor

    MixedLM documentation: https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.html?highlight=mixedlm#statsmodels.regression.mixed_linear_model.MixedLM
    Code taken and modified from here: https://stackoverflow.com/a/48949667/5217293
    
    Other useful resources:
      - https://www.statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html
      - https://www.statsmodels.org/stable/mixed_linear.html
      - https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.fit_regularized.html#statsmodels.regression.mixed_linear_model.MixedLM.fit_regularized
      - https://nbviewer.jupyter.org/urls/umich.box.com/shared/static/lc6uf6dmabmitjbup3yt.ipynb
      - https://nbviewer.jupyter.org/urls/umich.box.com/shared/static/6tfc1e0q6jincsv5pgfa.ipynb
    
    Attributes
    ----------
    formula : string
        The formula used by the MixedLM model. See the constructor for more details on this
    dependent_label : string
        The label of the dependent variable
    random_effect_variable : string
        The variable that will group the data to control for its random effect
    model_ : statsmodels.regression.mixed_linear_model.MixedLM
        The mixed effect model
        Set after calling fit()
    results_ : statsmodels.regression.mixed_linear_model.MixedLMResults
        The results of the mixed effect model fit.
        Set after calling fit()
    """
    def __init__(self, formula, dependent_label, groups, vc_formula = None, re_formula = None):
        """
        Initialize the MixedLM model

        Parameters
        ----------
        formula : pandas DataFrame
            A formula used to initialize the MixedLM using MixedLM.from_formula(): https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.from_formula.html#statsmodels.regression.mixed_linear_model.MixedLM.from_formula
            The formula is an R-style formula. Read more here: https://www.statsmodels.org/stable/examples/notebooks/generated/formulas.html
        dependent_label : string
            The label of the dependent variable
        groups : string
            The variable that will group the data to control for its random effect
        vc_formula : dictionary, default None
            The variance component (vc) formula that indicates what subgroups within the group should be fit
        re_formula : string
            One-sided random effect formula
        """
        self.formula = formula
        self.dependent_label = dependent_label
        self.groups = groups
        self.vc_formula = vc_formula
        self.re_formula = re_formula

    def fit(self, X, y):
        """
        Fit the MixedLM model

        Parameters
        ----------
        X : array_like
            The covariates used to predict `y`
        y : array_like
            The values that should be predicted
        """
        data = pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)
        data.columns = X.columns.tolist() + [self.dependent_label]

        self.model_ = MixedLM.from_formula(self.formula, 
                                           data=data, 
                                           groups=self.groups,
                                           vc_formula=self.vc_formula, 
                                           re_formula=self.re_formula)
        self.results_ = self.model_.fit(method="lbfgs")
        
        return self
        
    def predict(self, X):
        """
        Make a prediction with the fitted model

        Parameters
        ----------
        X : array_like
            The covariates used to make a prediction
        """
        return self.results_.predict(X)
    
    def summary(self):
        """
        Print a summary of the fitted results 
        
        See the MixedLMResults.summary documentation for more information: https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLMResults.summary.html#statsmodels.regression.mixed_linear_model.MixedLMResults.summary
        """
        return self.results_.summary()
    
    def compute_scores(self, X, y):
        """
        Predict the values and return three scores
        
        Parameters
        ----------
        X : array_like
            The covariates used to predict `y`
        y : array_like
            The values that should be predicted
            
        Returns
        ----------
            scores: tuple
                r2_score, root mean squared error, mean absolute error
        """
        y_pred = self.results_.predict(X)
        
        r2 = r2_score(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        mae = mean_absolute_error(y, y_pred)
        scores = r2, rmse, mae
        return scores

class Pymer4Wrapper(BaseEstimator, RegressorMixin):
    """ 
    An sklearn-style wrapper for the Pymer4 Lmer model, which calls the lmer model from R
    
    https://eshinjolly.com/pymer4/api.html
    
    Attributes
    ----------
    """
    def __init__(self, formula):
        """
        Parameters
        ----------
        """
        self.formula = formula

    def fit(self, X, y):
        """
        Fit the Pymer4 model

        Parameters
        ----------
        X : array_like
            The covariates used to predict `y`
        y : array_like
            The values that should be predicted
        """
        data = pd.concat([pd.DataFrame(X), pd.Series(y)], axis=1)
        data.columns = X.columns.tolist() + [y.name]

        self.model_ = Lmer(self.formula, data=data)
        
        self.model_.fit()
        
        return self
        
    def predict(self, X):
        """
        Make a prediction with the fitted model

        Parameters
        ----------
        X : array_like
            The covariates used to make a prediction
        """
        return self.model_.predict(X)
    
    def summary(self):
        """
        Print a summary of the fitted results 
        
        See the MixedLMResults.summary documentation for more information: https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLMResults.summary.html#statsmodels.regression.mixed_linear_model.MixedLMResults.summary
        """
        return self.model_.summary()
    
    def compute_scores(self, X, y):
        """
        Predict the values and return three scores
        
        Parameters
        ----------
        X : array_like
            The covariates used to predict `y`
        y : array_like
            The values that should be predicted
            
        Returns
        ----------
            scores: tuple
                r2_score, root mean squared error, mean absolute error
        """
        y_pred = self.model_.predict(X)
        
        r2 = r2_score(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        mae = mean_absolute_error(y, y_pred)
        scores = r2, rmse, mae
        return scores    
    
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
    
    def __init__(self, x, y, cv=10, stratify=False, lme_dependent = '', lme_independents = '', lme_group = 'DoY', lme_formula='', lme_locations=None):
        """
        Initialize the necessary data for this class
        
        x will be copied twice, once for all models except linear mixed effect, and again for mixed effect.
        Mixed effect needs a grouping parameter. This parameter, lme_group can either be 'DoY' (day of the year)
        'Month', or 'Season'. The copied x with have this added as a column
        
        y will be copied once. In this copy, the name of the series will be replaced with `lme_dependent`

        Parameters
        ----------
        x : pandas.DataFrame
            the covariates used to predict x
        y : pandas.Series
            the dependant variable being predicted
        cv : int, optional default 10
            the number of folds to use for cross validation. If stratify is True, this will also determine
            the number of bins to use to stratify y
        stratify : boolean, optional default False
            if True, y will be stratified with pandas.cut using the cv as the number of bins so that
            stratified cross validation will be done. 
            Read more here: https://stackoverflow.com/a/54946733/5217293
        lme_dependent: string, default ''
            The dependent variable that linear mixed effect will predict. This string will replace the name of the
            `y` series for use by the linear mixed effect model
        lme_independents: string, default ''
            The independents variables used to predict `lme_dependent`. This will be used to construct an R-style formula
        lme_group: string, default 'DoY'
            One of 'DoY' (day of the year), 'Month', 'Season', or 'Location'. Whatever value is chosen, a column will be added
            to a copied version of x and that column will be used to do the grouping. If 'Location' is specified, an array 
            of locations must be passed to lme_location
        lme_locations: pd.Series default None
            A pandas Series with the same index as x whose column values should indicate a locaitonal grouping of each row in x
        """
        
        self.x = x.copy()
        self.y = y
        
        self.scaler = None
        self.scaled_x = None
        
        self.lme_dependent = lme_dependent
        self.lme_independents = lme_independents
        self.lme_group = lme_group
        self.lme_formula = lme_formula
        self.lme_locations = lme_locations
        
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
        
        _x, _y = self._get_xy(model_name)
        
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
            self._compute_scross_validation_for_model(name)
            
        return self.cross_val_scores_

    def compute_cross_validation_scores_for_model(self, model_name):
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
        if not 'cross_val_scores_' in self.__dict__:
            self.cross_val_scores_ = {}
        else:
            if model_name in self.cross_val_scores_:
                del self.cross_val_scores_[model_name]

        self._compute_scross_validation_for_model(model_name)
        
        return self.cross_val_scores_
    
    def _compute_scross_validation_for_model(self, model_name):
        model = self.models[model_name]
        # for models that can be computed in parallel, let them be
        # if the models are being computed in parallel, don't cross validate in parallel. That would request too many cores
        # if the models are not computed in parallel, run cross validation in parallel
        cross_val_parallel = -1
        if getattr(model, 'n_jobs', None) is not None:
            model.n_jobs = -1
            cross_val_parallel = None

        _x, _y = self._get_xy(model_name)
        self.cross_val_scores_[model_name] = cross_validate(model, 
                                  _x, _y, 
                                  cv = self._get_cross_validation(), 
                                  scoring = self.scoring,
                                  n_jobs=cross_val_parallel)
          
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
        
        _x = self._transform_x_for_prediction(model_name, x)

        y_pred = model.predict(_x)
        scores = (
            r2_score(y, y_pred),
            mean_squared_error(y, y_pred, squared=False),
            mean_absolute_error(y, y_pred)
        )
            
        return scores
    
    def predict(self, x):
        predicitons = {}
        
        for model_name in self.models.keys():
            predicitons[model_name] = self.predict_model(model_name, x)
    
    def predict_model(self, model_name, x):
        
        model = self.models[model_name]
        x = self._transform_x_for_prediction(model_name, x)
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
        ar.append(('scaler', self.scaler))
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
        names_models, self.scaler = ar[:-1], ar[-1][0]
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

    def load_model_cross_val_scores(self, directory='', prefix=''):
        """
        Load all of the model scores from the disk.
        
        Parameters
        ----------
        filename : string, optional default ''
            The full path and filename to load the scores from.
        """
        
        with open(filename, 'rb') as f: 
            self.cross_val_scores_ = pickle.load(f)
    
    def _get_cross_validation(self):
        """
        Return a cross validation depending on the value of self.stratified_kfold
        
        Returns
        ----------
        cv : int or an sklearn StratifiedKFold
        """
        if self.stratified_kfold:
            return self._stratified_kfold()
        return self.cv
    
    def _setup_models(self):
        """
        Initialize the dictionary of models
        
        
        The individual models can be modified after this class is constructed by replacing
        values in self.models
        """
        #  models that will be placed into a pipeline
        elastic = ElasticNetCV(cv=10, 
                       l1_ratio=np.arange(0.1, 1.1, .1),
                       max_iter=10000
                      )
        mlp = MLPRegressor(max_iter=2000, early_stopping=True)
        mlp1 = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=100000, early_stopping=True)
        mlp2 = MLPRegressor(hidden_layer_sizes=(100, 50, 50, 50, 50), max_iter=100000, early_stopping=True)
        mlp3 = MLPRegressor(hidden_layer_sizes=(100, 50, 50, 50, 50, 100), max_iter=100000, early_stopping=True)

        self.models = {
            'Linear Regression': LinearRegression(),
            'Elastic Net' : elastic,
            'Polynomial' : Pipeline([('poly', PolynomialFeatures()), ('linear', linear_model.LinearRegression())]),
            'Bayesian Ridge' : BayesianRidge(),
            'SVR' :  LinearSVR(max_iter=20000),
            'Linear Mixed Effect': Pymer4Wrapper(formula=self.lme_formula),
            'MLP' :  mlp,
            'MLP1' : mlp1,
            'MLP2' : mlp2,
            'MLP3' : mlp3,
            'Ada Boost' : AdaBoostRegressor(),
            'Random Forest' : RandomForestRegressor(n_jobs=-1),
            'Extra Trees' : ExtraTreesRegressor(n_jobs=-1),
            'Gradient Boost' : GradientBoostingRegressor(),
            'XGBoost' : XGBRegressor(n_jobs=-1, subsample=0.9, colsample_bytree=0.5)
        }

    def _stratified_kfold(self):
        # stratified k-fold cross validation: https://stackoverflow.com/a/54946733/5217293
        y_cat = pd.cut(self.y, self.cv, labels=range(self.cv))
        return StratifiedKFold(self.cv).split(self.x, y_cat)
    
    def _get_xy(self, model_name):
        if self.scaler is None:
            self.scaler = StandardScaler().fit(self.x)
            
        if self.scaled_x is None:
            scaled_values = self.scaler.transform(self.x)
            self.scaled_x = pd.DataFrame(scaled_values, index=self.x.index, columns=self.x.columns)
        
        if model_name == 'Linear Mixed Effect':
            _x = self._transform_x_for_lme(self.scaled_x)
            _y = self.y.copy().rename(self.lme_dependent)
        else:
            _x = self.scaled_x
            _y = self.y
        
        return _x, _y
    
    def _transform_x_for_prediction(self, model_name, x):
        scaled_values = self.scaler.transform(x)
        scaled_x = pd.DataFrame(scaled_values, index=x.index, columns=x.columns)
        
        if model_name == 'Linear Mixed Effect':
            _x = self._transform_x_for_lme(scaled_x)
        else:
            _x = scaled_x
        
        return _x
    
    def _transform_x_for_lme(self, x):
        if not x.empty:
            x_lme = x.copy()
            if self.lme_group == 'DoY':
                x_lme['DoY'] = x_lme.index.dayofyear
            elif self.lme_group == 'Month':
                x_lme['Month'] = x_lme.index.month
            elif self.lme_group == 'Season':
                x_lme['Season'] = (x_lme.index.month%12 // 3 + 1)
            elif self.lme_group == 'Location':
                x_lme = pd.concat([x_lme, self.lme_locations], axis=1)
        else:
            x_lme = []
        
        return x_lme   
    
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
        self.y = y
        
        self.n = n
        self.criterion = criterion
        
        self.voting_regressor_ = None
        
    def fit(self):
        """ Choose the model with the top self.n scores according to self.how and fit a voting
            regressor with them.
        """
        
        if self.mc.scores_:
            scores = self.mc.scores_
        else:
            _x = self.x.copy()
            scores = self.mc.compute_scores(_x, self.y)
        
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
        
        _x = self.x.copy()
        self.voting_regressor_.fit(_x, self.y)
        
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
        
        _x = x.copy()
        
        y_pred = self.voting_regressor_.predict(_x)
        
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
        
        _x = x.copy()
        prediction = self.voting_regressor_.predict(_x)
        return prediction
    
    def save(self, directory='', prefix=''):
        """
        Save this model to the disk using joblib as recommended 
        by sklearn.
        
        Read more here: https://sklearn.org/modules/model_persistence.html
        
        Parameters
        ----------
        directory : string, optional default ''
            The directory that all of the models will be saved to.
            This can be a file system path
        prefix : string, optional default ''
            The prefix to prepend to each model name
        """
        
        dump(self.voting_regressor_, f'{directory}/{prefix}stacked.joblib')
        
    def load(self, directory='', prefix=''):
        """
        Load a stacking model from the disk using joblib as recommended 
        by sklearn.
        
        Read more here: https://sklearn.org/modules/model_persistence.html
        
        Parameters
        ----------
        directory : string, optional default ''
            The directory that all of the models were saved to.
            This can be a file system path
        prefix : string, optional default ''
            The prefix to prepended to each model name
        """
        
        self.voting_regressor_ = load(f'{directory}/{prefix}stacked.joblib')