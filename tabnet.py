import torch
import torch.nn as nn
import torch.nn.functional as F


class _Sparsemax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        Args:
            ctx: autograd context
            input (torch.Tensor): 2-D tensor, (N, C).
        Returns:
            torch.Tensor: (N, C).
        """
        dim = 1
        # translate input by max for numerical stability.
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)
        z_sorted = torch.sort(input, dim=dim, descending=True)[0]
        input_size = input.size()[dim]
        range_values = torch.arange(1, input_size + 1).to(input.device)
        range_values = range_values.expand_as(z_sorted)

        # Determine sparsity of projection
        range_ = torch.arange(
            1, input.size(dim) + 1, dtype=input.dtype, device=input.device
        )
        bound = 1.0 + range_ * z_sorted
        cumsum_zs = torch.cumsum(z_sorted, dim)
        is_gt = torch.gt(bound, cumsum_zs)
        k = torch.max(is_gt * range_, dim=dim, keepdim=True)[0]

        zs_sparse = is_gt * z_sorted

        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)
        output = (input - taus).clamp(min=0.0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        dim = 1

        nonzeros = output != 0.0
        sum_grad = torch.sum(grad_output * nonzeros, dim=dim, keepdim=True) / torch.sum(nonzeros, dim=dim, keepdim=True)
        return nonzeros * (grad_output - sum_grad.expand_as(grad_output))


sparsemax = _Sparsemax.apply


class Sparsemax(nn.Module):
    def forward(self, input):
        return sparsemax(input)
