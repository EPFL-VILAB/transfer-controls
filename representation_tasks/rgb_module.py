from . import RepresentationModule

class RGBModule(RepresentationModule):
    '''
    Module that does nothing: RGB -> RGB
    '''
    def __init__(self):
        super(RGBModule, self).__init__()

    def get_representation(self, x, layer_ids=None):
        return x

    def _shared_step(self, batch, is_train):
        # Not used, since this is not a trainable module
        pass