import sys
IS_PY3 = sys.version_info >= (3,0)
if IS_PY3:
    from .cvinn import cvinn
    from .data.cvi_data import cvi_data
    from .data.preprocess import preprocess