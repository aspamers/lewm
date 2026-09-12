from lewm.jepa import JEPA
from lewm.jepa_fm import FlowJEPA
from lewm.module import SIGReg, ARPredictor, Transformer, Embedder, MLP
from lewm.decoder import CLSDecoder


def __getattr__(name):
    # Core models can be used without the optional training/environment stack.
    if name in {"get_column_normalizer", "get_img_preprocessor", "SaveCkptCallback", "EMACallback"}:
        from lewm import utils
        return getattr(utils, name)
    if name == "GridSaveCallback":
        from lewm.grid_viz import GridSaveCallback
        return GridSaveCallback
    raise AttributeError(name)

__all__ = [
    "JEPA",
    "FlowJEPA",
    "SIGReg",
    "ARPredictor",
    "Transformer",
    "Embedder",
    "MLP",
    "CLSDecoder",
    "get_column_normalizer",
    "get_img_preprocessor",
    "SaveCkptCallback",
    "EMACallback",
    "GridSaveCallback",
]
