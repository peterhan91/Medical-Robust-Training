from torch import nn

class _DomainSpecificBatchNorm(nn.Module):
    _version = 2

    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.01, affine=True,
                 track_running_stats=True):
        super(_DomainSpecificBatchNorm, self).__init__()
        # ModuleList can be indexed like a regular Python list, 
        # but modules it contains are properly registered, 
        # and will be visible by all Module methods.
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label[0]] # might means the domain label should be [0, 1]
        return bn(x), domain_label


class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))