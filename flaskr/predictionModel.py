from joblib import load
from sklearn.metrics import r2_score


def __init__(self):
    self.model = load("assets/pipeline_proyecto2.joblib")


def make_predictions(predictionModel, data):
    result = predictionModel.model.predict(data)
    if result == 1:
        result = "Yes"
    else:
        result = "No"
    return result


def calcuate_r2(self, data_x, data_y):
    data_y = data_y.str.extract('(\d)', expand=True).astype(int)
    y_pred = self.model.predict(data_x)
    return r2_score(data_y, y_pred)
