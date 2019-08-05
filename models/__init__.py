from .model import Model


def get_model(num_class, config):
    return Model(num_class, config)