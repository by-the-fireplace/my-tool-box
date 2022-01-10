import pandas as pd


class AutoML:
    def __init__(self, data: pd.DataFrame, train_model: bool = True):
        self.data = data

    def train(self, configs):
        pass
