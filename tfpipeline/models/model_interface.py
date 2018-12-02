from abc import ABCMeta, abstractmethod

class ModelInterface(metaclass=ABCMeta):

    @abstractmethod
    def map_input(self, dataset, num_parallel_calls):
        pass

    @abstractmethod
    def inference(self, batch):
        pass

    @abstractmethod
    def loss(self, logit, label):
        pass

    @abstractmethod
    def predict(self, logit):
        pass
