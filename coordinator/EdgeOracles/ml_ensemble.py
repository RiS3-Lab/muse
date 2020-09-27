from ml_engine import *
from online_learning import *

class EnsembleLearning:
    def __init__(self, save_model_file = None, dataset_path=None):
        if save_model_file != None:
            self.rf_model = save_model_file + ".ensemble.rf"
            self.ol_model = save_model_file + ".ensemble.ol"
            print 'Model is saved here {}'.format(self.rf_model)
            print 'Model is saved here {}'.format(self.ol_model)
        else:
            self.rf_model = None
            self.ol_model = None
        self.online_learning = OnlineLearningModule(self.ol_model, dataset_path=dataset_path)
        self.rf_engine = MLEngine(self.rf_model, dataset_path=dataset_path)

    def predict(self, feature):
        return (self.rf_engine.predict(feature) + self.online_learning.predict(feature))/2.0

    def update_model(self, features, labels):
        self.rf_engine.update_model(features, labels)
        self.online_learning.update_model(features, labels)

    def save_model(self):
        self.rf_engine.save_model()
        self.online_learning.save_model()


def testEnsemble():
    print "TEST Ensemble"
    mlengine= EnsembleLearning()
    predicted_value = mlengine.predict([[0,2,1]])
    print "predicted value0 : ", predicted_value
    mlengine.update_model([[0,2,1]], [[200]])
    predicted_value = mlengine.predict([[0,2,1]])
    print "predicted value1 : ", predicted_value
    mlengine.update_model([[1,3,2]], [[143]])
    predicted_value = mlengine.predict([[2,3,2]])
    print "predicted value2: ", predicted_value

def testEnsembleInit():
    print ""
    print("TestInitialDataSet")
    dataset_path = '/home/eric/work/savior/newtcpdump_data.csv'
    mlengine = EnsembleLearning(dataset_path=dataset_path)
    predicted_value = mlengine.predict([[0,2,1,4,5,6,4,5,2,5,5,5]])
    print("predicted value0 : ", predicted_value)
    mlengine.update_model([[0,2,1,6,6,4,2,4,2,5,5,5], [1,3,2,3,1,7,8,4,2,3,4,4]], [400, 32])
    predicted_value = mlengine.predict([[0,2,1,6,7,5,6,4,7,6,6,6]])
    print("predicted value1 : ", predicted_value)

if __name__ == "__main__":
    testEnsemble()
    testEnsembleInit()
