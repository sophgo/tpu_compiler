import abc

class model_base(abc.ABC):
    @abc.abstractmethod
    def load_model(self):
        return NotImplemented

    @abc.abstractmethod
    def inference(self):
        return NotImplemented

    @abc.abstractmethod
    def get_all_tensor(self):
        return NotImplemented

    @abc.abstractmethod
    def get_op_info(self):
        return NotImplemented