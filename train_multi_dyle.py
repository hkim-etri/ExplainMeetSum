from MultiDyle.model import ExperimentMultiDyle
from config import Config

config = Config()


def train():
    # Building the wrapper
    wrapper = ExperimentMultiDyle()
    wrapper.train()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("once")
    train()
