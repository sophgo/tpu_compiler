import os
import sys
from .mlir_parser import MlirParser
from cvi_toolkit.utils.log_setting import setup_logger

logger = setup_logger('root', log_level="INFO")

class IntermediateFile:
    __files__ = []

    def __init__(self, prefix, name, keep=True):
        if prefix:
            self.name = '{}_{}'.format(prefix, name)
        else:
            self.name = name
        self.extension = self.name.split('.')[-1]
        self._keep = keep
        if os.path.exists(self.name):
            self.discard(force=True)
        self.__files__.append(self)

    def __str__(self):
        return self.name

    @classmethod
    def cleanup(cls):
        for file in cls.__files__:
            file.discard()
        del cls.__files__

    def keep(self):
        self._keep = _keep

    def discard(self, force=False):
        if not os.path.exists(self.name):
            return
        if not force and self._keep:
            return
        if self.extension == 'mlir':
            try:
                parser = MlirParser(self.name)
                weight_npz = parser.get_weight_file_name()
                if os.path.exists(weight_npz):
                    logger.debug("remove:", weight_npz)
                    os.remove(weight_npz)
            except:
                pass
        logger.debug("remove:", self.name)
        os.remove(self.name)
