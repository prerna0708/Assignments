import pandas as pd
import numpy as np

from joblib import dump, load
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report,r2_score, accuracy_score, roc_auc_score
from sklearn.feature_selection import f_regression, f_classif, chi2
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

svc_param_list = {
    'kernel': ['rbf'],
    'gamma': [1e-3, 1e-4, 1, 10, 100],
    'C': [1, 10, 100, 1000]
}

rf_param_list = {
    'max_depth' : range(3, 7),
    'min_samples_split' : range(1,3),
    'n_estimators':  [1000, 3000],
    'max_features': range(5, 12)
}


class HeatwaveBinaryModel(object):

    def __init__(self, model_type):

        self.model_type = model_type
        self.load_model()

    def refit_procedure(self, dataset, random_state, tune_hyperparameters = False):

        self.random_state = random_state

        self.dataset = dataset

        self.__data_preprocess(refit = True)

        self.__fit_model(tune_hyperparameters = False)

        self.__summary_model_fit()

    def predict_procedure(self, new_dataset):

        self.dataset = new_dataset

        self.__data_preprocess(refit = False)

        return self.model.predict(self.X)

    def dump_model(self):
        '''
            After training a scikit-learn model,
            we can persist the model by this function for future use without having to retrain.
        '''
        dump(self.model, 'app/%s.joblib' %self.model_type)

        print('%s model is saved in %s.joblib !! \n' %(self.model_type, self.model_type))

    def load_model(self):

        self.model = load('app/%s.joblib' %self.model_type)

        print('%s model is loaded from %s.joblib !! \n' %(self.model_type, self.model_type))


    def __data_preprocess(self, refit = True):

        # remove rows with no information of co2_emission_kt or temp_mean
        country = self.dataset['country.1'].dropna().unique()
        self.dataset = self.dataset[
            self.dataset['country.1'].isin(
                country[
                    (self.dataset.groupby('country.1')['co2_emission_kt'].count() > 0) &
                    (self.dataset.groupby('country.1')['temp_mean'].count() > 0)
                ]
            )]

        # add new lagged variable : the mean tempeature of that that country in last year
        self.dataset['temp_mean_l1'] = self.dataset.groupby(['country.1']).temp_mean.shift(1)
        # add new lagged variable : the median tempeature of that that country in last year
        self.dataset['temp_median_l1'] = self.dataset.groupby(['country.1']).temp_median.shift(1)
        # add new lagged variable : 1 : heat wave happened in the last year, 0 :not happened in the last year
        self.dataset['is_hw_happend_l1'] = self.dataset.groupby(['country.1']).is_hw_happend.shift(1) \
            .apply(lambda r: 1 if r else 0)
        # add new lagged variable : 1 : heat wave happened four years ago, 0 :not happened four years ago
        self.dataset['is_hw_happend_l4'] = self.dataset.groupby(['country.1']).is_hw_happend.shift(4) \
            .apply(lambda r: 1 if r else 0)

        self.dataset['log_co2_emission_kt'] =  self.dataset['co2_emission_kt'].apply(
            lambda r : np.log(r) if r > 0 else 0
        )

        self.dataset['log_methane_emission_kt'] =  self.dataset['methane_emission_kt'].apply(
            lambda r : np.log(r) if r > 0 else 0
        )

        self.dataset['log_land_area_sq_km'] =  self.dataset['land_area_sq_km'].apply(
            lambda r : np.log(r) if r > 0 else 0
        )

        self.dataset['log_total_population'] = self.dataset['total_population'].apply(
            lambda r : np.log(r) if r > 0 else 0
        )

        self.dataset['year'] =self.dataset['year.1']
        # extract the subset of feature space.
        self.features = [
            'year', 'latitude', 'longitude','forest_area_ratio', 'urban_pop_ratio', 'log_total_population', 'agri_land_ratio',
            'log_co2_emission_kt', 'log_methane_emission_kt', 'log_land_area_sq_km','livestock_prod_ind', 'food_prod_ind',
            'temp_mean','temp_median_l1', 'is_hw_happend_l4','region'
        ]

        if refit:

            self.column_trans = ColumnTransformer(
                [
                    ('incomeLevel_binned_numeric', OneHotEncoder(dtype= 'int'), ['region']),
                    ('Impute',SimpleImputer(strategy ='mean'), self.features[:-1])
                ],
                remainder = 'passthrough'
            )


            self.X = self.column_trans.fit_transform(self.dataset[self.features])

            dump(self.column_trans, 'app/column_transformer.joblib')

            # target variable
            self.y = self.dataset.is_hw_happend.values

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y, test_size=0.3, random_state= self.random_state)
        else:

            self.column_trans = load('app/column_transformer.joblib')

            self.X = self.column_trans.transform(self.dataset[self.features])

    def __fit_model(self, tune_hyperparameters = False, tune_score = None):

        if tune_hyperparameters:

            if tune_score is None:
                'recall'

            if model_type == 'svc':
                self.model = tune_hyperparameters(svc_param_list, tune_score)
            elif model_type == 'rf':
                self.model = tune_hyperparameters(rf_param_list, tune_score)


        # fit the train data to the three models
        self.model.fit(self.X_train, self.y_train)

    def __summary_model_fit(self):

        self.accuracy_score = accuracy_score(self.y_test, self.model.predict(self.X_test))
        self.roc_auc_score = roc_auc_score(self.y_test, self.model.predict(self.X_test))

        print('accuracy_score: %f' %self.accuracy_score)

        print('roc_auc_score: %f' %self.roc_auc_score)

        self.classification_report = classification_report(
            self.y_test, self.model.predict(self.X_test),
            target_names = ['Non Heat Wave', 'Heat Wave'],
            output_dict = True
            )

        print(
                classification_report(
                    self.y_test, self.model.predict(self.X_test),
                    target_names = ['Non Heat Wave', 'Heat Wave'],
                    output_dict = False
                    )
            )

    def __tune_hyperparameters(self, param_list, scoring = 'f1_macro'):

        cv_search = GridSearchCV(self.model, param_list, cv=5, scoring = scoring, n_jobs = -1)

        mdl_cv = cv_search.fit(self.X_train, self.y_train)
        # summarize results
        print("Best: %f using %s" % (mdl_cv.best_score_, mdl_cv.best_params_))

        return mdl_cv.best_estimator_
