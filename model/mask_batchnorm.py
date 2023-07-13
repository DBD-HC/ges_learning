import torch.nn as nn
import torch


class MaskedBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input, padded_length=None, valid_length=None, mask=None):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            if mask is None:
                mask = torch.arange(padded_length, dtype=torch.float32, device=input.device)
                # valid_length = valid_length.to(input.device)
                mask = mask[None, :] < valid_length[:, None]
                mask = mask.reshape(-1)

            mean = input[mask].mean([0, 2, 3])
            # use biased var in train
            var = input[mask].var([0, 2, 3], unbiased=False)
            n = input[mask].numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input[mask] = (input[mask] - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input[mask] = input[mask] * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


class MyLayerNorm(torch.nn.Module):
    def __init__(self, size=None, eps=1e-6):
        super(MyLayerNorm, self).__init__()
        # self.gamma = torch.nn.Parameter(torch.ones(size))
        # self.beta = torch.nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x, batch_size=-1, padding_len=None):
        input_shape = x.shape

        # x = x.view(batch_size, padding_len, -1)
        mean = x.mean([1, 2])
        var = x.var([1, 2], unbiased=False)
        norm = (x - mean[:, None, None]) / (torch.sqrt(var[:, None, None] + self.eps))
        # output = self.gamma * norm + self.beta
        # return torch.reshape(norm, shape=input_shape)
        return norm


class MaskedBatchNorm1d(MaskedBatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input, padded_length=None, valid_length=None, mask=None):
        if input.dim() == 2:
            input = input[:, :, None, None]
        elif input.dim() == 3:
            input = input[:, :, :, None]
        else:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )

        res = super().forward(input, padded_length, valid_length, mask=mask)
        res = torch.squeeze(res)
        return res