"""
ML Engine
By: Mansour Ahmadi (mansourweb@gmail.com)
    Yaohui Chen    (yaohway@gmail.com)
Created Date: 3 Jun 2019
Last Modified Date: 16 June 2019
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
import sklearn
import os
import tempfile
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import GridSearchCV


class MLEngine:

    def __init__(self, model_dir=None, classifier='rf', dataset_path=None,
                 columns=list()):
        model_file_name = 'reachability_model.pkl'
        self.classifier = classifier
        if len(columns) == 0:
            self.columns = ['reachable label', 'path length', 'undiscovered neighbours', 'new cov', 'size']
        else:
            self.columns = columns
        self.best_model_params = {}

        self.features = pd.DataFrame(columns=self.columns)
        self.labels = []
        if self.classifier == 'rf':
            if dataset_path is None:
                self.clf = RandomForestRegressor(n_estimators=10, max_depth=4)
                self.dataset_path = ''
            else:
                self.dataset_path = dataset_path
                print 'Initializing model from: ', self.dataset_path
                self.build_model()
        elif self.classifier == 'svr':
            if dataset_path == '':
                self.clf = SVR(kernel='linear', gamma='scale', C=1.0)
            else:
                self.dataset_path = dataset_path
                print 'Initializing model from: ', self.dataset_path
                self.build_model()
        else:
            print 'Classifier is not supported'

        if model_dir is not None:
            self.model_file_path = os.path.join(model_dir, self.classifier + '_' + model_file_name)
        else:
            model_dir = tempfile.mkdtemp()
            self.model_file_path = os.path.join(model_dir, model_file_name)
        print 'Model is saved here: {}'.format(self.model_file_path)

        # if self.classifier == 'sgd':
        #     self.clf = SGDRegressor(max_iter=1000, alpha=1, penalty='l1')

    def load_model(self):
        pass

    def build_model(self):
        if not os.path.exists(self.dataset_path):
            print 'dataset does not exist'
            exit(1)

        self.find_optimal_param()
        self.model_construction()

    def model_construction(self):
        self.clf = RandomForestRegressor(n_estimators=self.best_model_params['n_estimators'],
                                         max_depth=self.best_model_params['max_depth'])
        self.update_model(features=[], labels=[])

    def predict(self, features):
        try:
            return self.clf.predict(np.array(features).reshape(1, -1))[0]
        except sklearn.exceptions.NotFittedError:
            print 'The model is not fitted yet.'
            return sum(features[0])

    def update_model(self, features, labels):
        features_dict_list = list()
        if len(self.labels) == 0:
            self.labels = np.array(labels).reshape(-1, 1)
        else:
            if len(labels) > 0:
                self.labels = np.concatenate((self.labels, np.array(labels).reshape(-1, 1)))
            else:
                self.labels = np.array(self.labels).reshape(-1, 1)

        for feature in features:
            features_dict_list.append(dict(zip(self.columns, feature)))
        if self.features.shape[0] == 0:
            self.features = pd.DataFrame(features_dict_list)
        else:
            self.features = pd.concat([self.features, pd.DataFrame(features_dict_list)])
        # self.labels = np.array(self.labels).reshape(-1, 1)
        if self.classifier == 'rf' or self.classifier == 'svr':
            self.clf.fit(X=self.features, y=self.labels.ravel())

    def remove_model(self, model_file_path=''):
        if model_file_path == '':
            os.remove(self.model_file_path)
        else:
            os.remove(model_file_path)

    def save_model(self):
        pickle.dump(self.clf, open(self.model_file_path, 'wb'))

    def find_optimal_param(self):
        dataset = pd.read_csv(self.dataset_path)
        self.labels = dataset.label
        dataset.drop('label', axis=1, inplace=True)
        dataset.drop('id', axis=1, inplace=True)
        self.features = dataset
        if self.classifier == 'rf':
            grid = self.get_rfregressor_params()
        elif self.classifier == 'svr':
            grid = self.get_svregressor_params()

        gd_sr = GridSearchCV(estimator=grid['clf'],
                             param_grid=grid['grid_param'],
                             scoring='neg_mean_squared_error',
                             cv=5,
                             n_jobs=-1)
        gd_sr.fit(self.features, self.labels)
        print grid['name'], gd_sr.best_params_, 'MSE: ', gd_sr.best_score_
        self.best_model_params = gd_sr.best_params_

    def get_rfregressor_params(self):
        grid_param = {
            'n_estimators': [10, 20, 50, 70],
            'max_depth': [3, 4, 5, 6]
        }
        clf = RandomForestRegressor()
        return {'clf': clf, 'grid_param': grid_param, 'name': 'rfreg'}

    def get_svregressor_params(self):
        grid_param = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.01, 0.1, 1, 10, 100]
        }
        clf = SVR()
        return {'clf': clf, 'grid_param': grid_param, 'name': 'svreg'}

    def get_corr(self, output='/tmp/corr_mat.pdf'):
        dataset = pd.read_csv(self.dataset_path)
        plt.figure(figsize=(25, 20))
        sb.heatmap(dataset.corr(), annot=True, cmap=sb.diverging_palette(20, 220, n=200))
        plt.savefig(output, pad_inches=0)


def test():
    mlengine= MLEngine()
    predicted_value = mlengine.predict([[0,2,1]])
    print "predicted value0 : ", predicted_value
    mlengine.update_model([[0,2,1]], [[200]])
    predicted_value = mlengine.predict([0,2,1])
    print "predicted value1 : ", predicted_value
    mlengine.update_model([[1,3,2]], [[143]])
    predicted_value = mlengine.predict([2,3,2])
    print "predicted value2: ", predicted_value


def test1():
    mlengine = MLEngine(dataset_path='/home/eric/work/savior/meuzz_data_norm_short.csv')
    predicted_value = mlengine.predict([[0,2,1,3,2]])
    print "predicted value0 : ", predicted_value
    mlengine.update_model([[2,2,1,6,8]], [[200]])
    predicted_value = mlengine.predict([0,2,1,7,8])
    print "predicted value1 : ", predicted_value
    mlengine.update_model([[1,3,2,6,4]], [[143]])
    predicted_value = mlengine.predict([5,6,2,3,2])
    print "predicted value2: ", predicted_value

def test2():
    mlengine= MLEngine(classifier='svr')
    predicted_value = mlengine.predict([[0,2,1]])
    print "predicted value0 : ", predicted_value
    mlengine.update_model([[0,2,1]], [[200]])
    predicted_value = mlengine.predict([0,2,1])
    print "predicted value1 : ", predicted_value
    mlengine.update_model([[1,3,2]], [[143]])
    predicted_value = mlengine.predict([2,3,2])
    print "predicted value2: ", predicted_value
    
def test_corr():
    mlengine = MLEngine(dataset_path='/tmp/meuzz_data.csv')
    mlengine.get_corr()


if __name__ == "__main__":
    test()
    test1()
    test2()
    test_corr()
