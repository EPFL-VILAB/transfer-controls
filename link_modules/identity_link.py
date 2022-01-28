from . import LinkModule


class IdentityLink(LinkModule):
    '''
    Link module that does nothing: x -> x
    '''
    def __init__(self):
        super(IdentityLink, self).__init__()

    def forward(self, x):
        return x