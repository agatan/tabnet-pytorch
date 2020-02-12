import math
from typing import List

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
        sum_grad = torch.sum(grad_output * nonzeros, dim=dim, keepdim=True) / torch.sum(
            nonzeros, dim=dim, keepdim=True
        )
        return nonzeros * (grad_output - sum_grad.expand_as(grad_output))


sparsemax = _Sparsemax.apply


class GLU(nn.Module):
    def forward(self, input):
        return F.glu(input)


class GhostBatchNorm(nn.Module):
    def __init__(
        self, num_features: int, momentum: float, ghost_batch_size: int
    ) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)
        self.ghost_batch_size = ghost_batch_size

    def forward(self, input):
        batch_size = input.size(0)
        chunks = input.chunk((batch_size - 1) // self.ghost_batch_size + 1, dim=0)
        normalized_chunks = [self.bn(chunk) for chunk in chunks]
        return torch.cat(normalized_chunks, dim=0)


class SharedFeatureTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        bn_momentum: float,
        ghost_batch_size: int,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, hidden_size * 2, bias=False),
            GhostBatchNorm(
                hidden_size * 2, momentum=bn_momentum, ghost_batch_size=ghost_batch_size
            ),
            GLU(),
        )
        self.residual_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=False),
            GhostBatchNorm(
                hidden_size * 2, momentum=bn_momentum, ghost_batch_size=ghost_batch_size
            ),
            GLU(),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): (N, C)
        Returns:
            torch.Tensor: (N, C)
        """
        x = self.block(input)
        return (x + self.residual_block(x)) * math.sqrt(0.5)


class FeatureTransformer(nn.Module):
    def __init__(
        self, in_channels: int, bn_momentum: float, ghost_batch_size: int
    ) -> None:
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2, bias=False),
            GhostBatchNorm(
                in_channels * 2, momentum=bn_momentum, ghost_batch_size=ghost_batch_size
            ),
            GLU(),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): (N, C)
        Returns:
            torch.Tensor: (N, C)
        """
        return (input + self.residual_block(input)) * math.sqrt(0.5)


class TabNet(nn.Module):
    def __init__(
        self,
        dense_channels: int,
        cat_cardinalities: List[int],
        out_channels: int,
        n_decision_steps: int,
        cat_emb_dim: int = 1,
        bn_momentum: float = 0.1,
        n_d: int = 16,
        n_a: int = 16,
        relaxation_factor: float = 2.0,
        ghost_batch_size: int = 256,
    ):
        """
        Args:
            dense_channels: number of dense features.
            cat_cardinalities: categorical feature cardinalities.
            out_channels: number of output channels.
            n_decision_steps: number of decision step layers.
            cat_emb_dim: categorical feature embedding size.
            bn_momentum: batch normalization momentum.
            n_d: hidden size of decision output.
            n_a: hidden size of attentive transformer.
            relaxation_factor: relaxation parameter of feature selection regularization.
            ghost_batch_size: ghost batch size for GhostBatchNorm.
        """
        super().__init__()
        self.n_decision_steps = n_decision_steps
        self.n_d = n_d
        self.n_a = n_a
        self.relaxation_factor = relaxation_factor

        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, cat_emb_dim)
                for cardinality in cat_cardinalities
            ]
        )

        feature_channels = dense_channels + cat_emb_dim * len(cat_cardinalities)
        self.dense_bn = nn.BatchNorm1d(feature_channels, momentum=bn_momentum)

        hidden_size = n_d + n_a

        shared_feature_transformer = SharedFeatureTransformer(
            feature_channels, hidden_size, bn_momentum, ghost_batch_size
        )
        self.feature_transformers = nn.ModuleList(
            [
                nn.Sequential(
                    shared_feature_transformer,
                    FeatureTransformer(hidden_size, bn_momentum, ghost_batch_size),
                    FeatureTransformer(hidden_size, bn_momentum, ghost_batch_size),
                )
                for _ in range(n_decision_steps)
            ]
        )
        self.attentive_transformers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_a, feature_channels, bias=False),
                    GhostBatchNorm(
                        feature_channels,
                        momentum=bn_momentum,
                        ghost_batch_size=ghost_batch_size,
                    ),
                )
                for _ in range(n_decision_steps - 1)
            ]
        )
        self.fc = nn.Linear(n_d, out_channels, bias=False)

    def forward(self, dense_features=None, cat_features=None):
        """
        Args:
            dense_features (Optional[torch.Tensor]): numerical dense features. shape: (N, dense_channels), dtype: float
            cat_features (Optional[torch.Tensor]): categorical features. shape: (N, len(cat_cardinalities)), dtype: long
        Returns:
            logits (torch.Tensor): shape: (N, out_channels), dtype: float
            masks (List[torch.Tensor]): masks for each decision step. `n_decision_steps` tensors of shape (N, feature_size)
            sparsity_regularization (torch.Tensor): feature selection sparsity regularization loss
        """
        assert dense_features is not None or cat_features is not None

        batch_size = (
            dense_features.size(0)
            if dense_features is not None
            else cat_features.size(0)
        )
        dtype = torch.float
        device = (
            dense_features.device if dense_features is not None else cat_features.device
        )

        feature = dense_features
        if cat_features is not None:
            # Convert cateogirical features into dense values.
            assert cat_features.size(1) == len(self.cat_embeddings)
            cat_embs = [
                self.cat_embeddings[i](cat_features[:, i])
                for i in range(len(self.cat_embeddings))
            ]
            if feature is None:
                feature = torch.cat(cat_embs, dim=-1)
            else:
                # Concat dense and categoical features.
                feature = torch.cat([dense_features] + cat_embs, dim=-1)

        aggregated_output = torch.zeros(
            batch_size, self.n_d, dtype=dtype, device=device
        )
        masked_feature = feature
        prior_scale_term = torch.ones(
            batch_size, feature.size(1), dtype=dtype, device=device
        )
        mask = torch.zeros_like(prior_scale_term)
        masks: List[torch.Tensor] = []
        aggregated_masks = torch.zeros_like(prior_scale_term)
        sparsity_regularization = torch.tensor(0.0).to(dtype=dtype, device=device)

        for step in range(self.n_decision_steps):
            x = self.feature_transformers[step](masked_feature)  # (N, hidden_size)
            decision_out, coef_out = x.split(self.n_d, dim=1)  # (N, n_d), (N, n_a)

            if step != 0:
                decision_out = F.relu(decision_out)
                aggregated_output += decision_out
                # For visualization and interpretability, aggregate feature mask values for all steps.
                scale = decision_out.sum(1, keepdim=True) / (self.n_decision_steps - 1)
                aggregated_masks += scale * mask

            if step != self.n_decision_steps - 1:
                # Prepare mask values for the next decision step.
                mask = self.attentive_transformers[step](coef_out)
                mask = mask * prior_scale_term
                mask = sparsemax(mask)
                # Update prior scale term to regulate feature selection
                prior_scale_term = prior_scale_term * (self.relaxation_factor - mask)
                # Update sparsity regularization
                sparsity_regularization += (mask * (mask + 1e-5).log()).sum(1).mean(
                    0
                ) / (self.n_decision_steps - 1)
                masked_feature = mask * feature
                masks.append(mask)
        logits = self.fc(aggregated_output)
        return logits, masks, sparsity_regularization
