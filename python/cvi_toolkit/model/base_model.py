import abc

class model_base(abc.ABC):
    @abc.abstractmethod
    def load_model(self):
        return NotImplemented

    @abc.abstractmethod
    def inference(self):
        return NotImplemented
