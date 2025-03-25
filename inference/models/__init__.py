def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'grconvnet':
        from .grconvnet import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'fcgnet':
        from .fcgnet import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'ggcnn':
        from .ggcnn import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'senet':
        from .senetgrasp import SEResUNet
        return SEResUNet
    elif network_name == 'ggcnn2':
        from .ggcnn2 import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'tfgraspnet':
        from .TFgraspnet import SwinTransformerSys
        return SwinTransformerSys
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
