from abc import ABC, abstractmethod


class GeneralModel(ABC):
    """General class of models."""

    def __init__(self, param_files=None):
        super().__init__()

    @abstractmethod
    def fit(self, data_train, model_folder=None):
        """
        Fits the model.
        """

    @abstractmethod
    def load_trained_model(self, model_folder):
        pass

    @abstractmethod
    def save_model(self, model_folder):
        pass

    @abstractmethod
    def score(self, X):
        pass
