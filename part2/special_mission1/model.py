import os
from catboost import CatBoostClassifier

class CatBoostModel:
    def __init__(self, model_dir="./model"):
        self.model = CatBoostClassifier().load_model(model_dir)
    
    def inference(self, X):
        return self.model.predict_proba(X)[:, 1]
