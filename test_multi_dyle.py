from MultiDyle.model import ExperimentMultiDyle
from config import Config

config = Config()


def test():

    # Building the wrapper
    wrapper = ExperimentMultiDyle(load_train=False)

    wrapper.test()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    test()
