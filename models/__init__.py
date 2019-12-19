from .model import Model


def get_model(num_class,ctx, config):
    return Model(num_class,ctx, config)