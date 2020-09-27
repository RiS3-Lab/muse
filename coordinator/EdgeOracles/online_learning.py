"""
Online learning module
By: Boyu Wang (boywang@cs.stonybrook.edu)
    Yaohui Chen (yaohway@gmail.com)
Created Date: 2 Jun 2019
Last Modified Date: 17 June 2019
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model.ridge import Ridge
from tempfile import TemporaryFile

# utility functions
# append 1 to the features, act as a bias term
def utl_add_bias(features):
    features = np.array(features)
    features = np.concatenate([features, np.ones([features.shape[0], 1])], axis=1)
    return features

# for features and scores, it would be better if you could normalize them first before fitting into the model
class OnlineLearningModule:
    def __init__(self, save_model_file = None, dataset_path=None):
        #load Cn_inv and W from file
        try:
            self.save_model_file = save_model_file
            self.load_model()
            self.is_init = True
        except Exception:
            self.is_init = False

        if dataset_path is not None:
            print "reading data from: ",dataset_path
            dataset = pd.read_csv(dataset_path, delimiter=',')
            print "dataset shape: ",dataset.shape
            labels = dataset.label
            print "labels shape: ",labels.shape
            dataset.drop('window', axis=1, inplace=True)
            dataset.drop('label', axis=1, inplace=True)
            dataset.drop('id', axis=1, inplace=True)
            features = dataset
            print "features shape: ",features.shape
            # train model with the initial data if no model is provided
            if save_model_file is not None:
                self.update_model(features, labels)


    def load_model(self):
        f = self.save_model_file
        model = pickle.load(open(f,'rb'))
        print("Loading model from {0}".format(f))
        self.W = model['W']
        self.Cn_inv = model['Cn_inv']

    def save_model(self):
        print "[Online learning] Saving model"	
        if not self.save_model_file is None:
            f = self.save_model_file
            print("Saving model to {0}".format(f))
            model = {'W':self.W, 'Cn_inv':self.Cn_inv}
            pickle.dump(model, open(f,'wb'))

    # features: numpy array of size [n, fea_dim]
    # scores: numpy array of size [n]
    # alpha is the weight for l2 regularization, can be tuned
    def first_update(self, features, scores, alpha):
        self.alpha = alpha

        # conversion from list to numpy array
        features = np.array(features)
        scores = np.array(scores)

        features = utl_add_bias(features)

        init_clf = Ridge(alpha=self.alpha, fit_intercept=False, random_state=0)
        init_clf.fit(features, scores)

        # W: numpy matrix of size [fea_dim+1, 1]
        init_W = np.matrix(init_clf.coef_).T
        features = np.matrix(features)
        Cn = features.T * features + self.alpha * np.eye(features.shape[1])
        Cn_inv = np.linalg.inv(Cn)
        XiYi = np.expand_dims(np.sum(np.array(features) * np.expand_dims(scores, 1), axis=0), 1) # [d, 1]

        # self.Cn = Cn
        self.Cn_inv = Cn_inv
        self.W = init_W
        print "Weight: ", self.W.shape
        print "Cn_inv: ", self.Cn_inv.shape
        # print(np.sum(np.abs(Cn_inv * XiYi - self.W)))
        self.is_init = True

    # update the classifier with new batch of data
    # features: numpy array of size [n, fea_dim]
    # scores: numpy array of size [n]
    def update_model(self, features, scores, alpha=1.0):
        if not self.is_init:
            self.first_update(features, scores, alpha)
            # print "after first update W: ", self.W
            return
        # print "prev W: ", self.W

        # conversion from list to numpy array
        features = np.array(features)
        scores = np.array(scores)

        features = utl_add_bias(features)
        [n, d] = features.shape
        features = np.matrix(features) # [n, d]
        scores = np.matrix(scores).T # [n, 1]
        print features.shape
        print self.Cn_inv.shape
        denom = features * self.Cn_inv * features.T + np.eye(n)
        # denom = np.matmul(np.matmul(features, self.Cn_inv), features.transpose()) + np.eye(n)
        denom = np.linalg.inv(denom)
        # update Cn_inv
        Cn_inv = self.Cn_inv - self.Cn_inv * features.T * denom *features * self.Cn_inv
        self.W = self.W + Cn_inv * (features.T * scores - features.T * features * self.W)
        # Cn_inv = self.Cn_inv - np.matmul(np.matmul(np.matmul(np.matmul(self.Cn_inv, features.transpose()), denom), features), self.Cn_inv)
        # self.W = self.W + np.matmul(Cn_inv, np.matmul(features.transpose(), scores) - np.matmul(np.matmul(Xk, Xk.transpose()), last_coef))
        # print "after W: ", self.W

    # get the predicted scores for the input features
    # features: numpy array of size [n, fea_dim]
    # return: scores: [n]
    def predict(self, features):
        try:
            # conversion from list to numpy array
            features = np.array(features)

            # append 1 to the features
            features = utl_add_bias(features)
            scores = (features * self.W).A1
        except Exception:
            # conversion from list to numpy array
            features = np.array(features)
            features = utl_add_bias(features)
            # print len(features), len(features[0])
            # print features
            self.W = np.matrix(np.ones([len(features[0]), 1]))
            # print "score: ", features * self.W
            scores = (features * self.W).A1

        return scores[0]

# usage
def test():
    print ""
    print("Test1")
    d = 5
    n = 101
    alpha = 1.0
    features = np.random.rand(n, d)
    scores = np.random.rand(n)
    print "labels shape: ",scores.shape
    print "features shape: ",features.shape

    # initialize the model
    print('initialize the model')
    model = OnlineLearningModule()
    model.update_model(features, scores, alpha)

    # test the model
    pred_scores = model.predict(features)
    err1 = np.linalg.norm(pred_scores-scores) / n
    print('average prediction error on the first batch:{}'.format(err1))

    # update the model with new data
    features2 = np.random.rand(n, d)
    scores2 = np.random.rand(n)
    print('update model with new data')
    model.update_model(features2, scores2)

    pred_scores2 = model.predict(features2)
    err2 = np.linalg.norm(pred_scores2-scores2) / n
    print('average prediction error on the second batch:{}'.format(err2))
    pred_scores = model.predict(features)
    err1 = np.linalg.norm(pred_scores-scores) / n
    print('after update, average prediction error on the first batch:{}'.format(err1))

    # compare online learning with offline learning, make sure they have same result
    print('compare online learning with offline learning')
    features_all = np.concatenate([features, features2], axis=0)
    scores_all = np.concatenate([scores, scores2])
    model2 = OnlineLearningModule()
    model2.update_model(features_all, scores_all, alpha)
    # test the model
    pred_scores = model2.predict(features)
    err1 = np.linalg.norm(pred_scores-scores) / n
    print('offline learning, average prediction error on the first batch:{}'.format(err1))
    pred_scores2 = model2.predict(features2)
    err2 = np.linalg.norm(pred_scores2-scores2) / n
    print('offline learning, average prediction error on the second batch:{}'.format(err2))

def test2():
    print ""
    print("Test2")
    mlengine= OnlineLearningModule()
    predicted_value = mlengine.predict([[0,2,1]])
    print("predicted value0 : ", predicted_value)
    mlengine.update_model([[0,2,1], [2,3,1]], [400, 32])
    predicted_value = mlengine.predict([[0,2,1]])
    print("predicted value1 : ", predicted_value)
    mlengine.update_model([[1,3,2]], [24])
    predicted_value = mlengine.predict([[2,3,2]])
    print("predicted value2: ", predicted_value)

def test3():
    print ""
    print("Test3")
    outf = "/tmp/.testdump"
    mlengine = OnlineLearningModule(outf)
    predicted_value = mlengine.predict([[0,2,1]])
    print("predicted value0 : ", predicted_value)
    mlengine.update_model([[0,2,1], [2,3,1]], [400, 32])
    predicted_value = mlengine.predict([[0,2,1]])
    print("predicted value1 : ", predicted_value)
    mlengine.update_model([[1,3,2]], [24])
    mlengine.save_model()

    mlengine2 = OnlineLearningModule(outf)
    predicted_value = mlengine.predict([[2,3,2]])
    print("predicted value2: ", predicted_value)

def test4():
    print ""
    print("TestInitialDataSet")
    dataset_path = '/home/eric/work/savior/newtcpdump_data.csv'
    mlengine = OnlineLearningModule(dataset_path=dataset_path)
    predicted_value = mlengine.predict([[0,2,1,4,5,5,6,2,1,6,6,6,6]])
    print("predicted value0 : ", predicted_value)
    mlengine.update_model([[0,2,1,6,6,8,9,0,4,2,2,2,5], [1,3,2,3,1,8,5,6,4,3,6,6,6]], [400, 32])
    predicted_value = mlengine.predict([[0,2,1,6,7,2,5,3,5,6,5,5,5]])
    print("predicted value1 : ", predicted_value)


if __name__ == '__main__':
    test()
    test2()
    test3()
    test4()

