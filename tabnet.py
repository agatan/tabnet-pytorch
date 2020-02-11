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


class SharedFeatureTransformer(nn.Module):
    def __init__(
        self, in_channels: int, hidden_size: int, bn_momentum: float = 0.1
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, hidden_size * 2, bias=False),
            # TODO: Ghost Batch Normalization
            nn.BatchNorm1d(hidden_size * 2, momentum=bn_momentum),
            GLU(),
        )
        self.residual_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=False),
            # TODO: Ghost Batch Normalization
            nn.BatchNorm1d(hidden_size * 2, momentum=bn_momentum),
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
    def __init__(self, in_channels: int, bn_momentum: float = 0.1) -> None:
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2, bias=False),
            # TODO: Ghost Batch Normalization
            nn.BatchNorm1d(in_channels * 2, momentum=bn_momentum),
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
        out_channels: int,
        n_decision_steps: int,
        cat_cardinalities: List[int] = [],
        cat_emb_dim: int = 1,
        bn_momentum: float = 0.1,
        decision_hidden_size: int = 16,
        attention_hidden_size: int = 16,
        relaxation_factor: float = 2.0,
    ):
        super().__init__()
        self.n_decision_steps = n_decision_steps
        self.decision_hidden_size = decision_hidden_size
        self.attention_hidden_size = attention_hidden_size
        self.relaxation_factor = relaxation_factor

        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, cat_emb_dim)
                for cardinality in cat_cardinalities
            ]
        )

        feature_channels = dense_channels + cat_emb_dim * len(cat_cardinalities)
        self.dense_bn = nn.BatchNorm1d(feature_channels, momentum=bn_momentum)

        hidden_size = decision_hidden_size + attention_hidden_size

        shared_feature_transformer = SharedFeatureTransformer(
            feature_channels, hidden_size, bn_momentum
        )
        self.feature_transformers = nn.ModuleList(
            [
                nn.Sequential(
                    shared_feature_transformer,
                    FeatureTransformer(hidden_size, bn_momentum),
                    FeatureTransformer(hidden_size, bn_momentum),
                )
                for _ in range(n_decision_steps)
            ]
        )
        self.attentive_transformers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(attention_hidden_size, feature_channels, bias=False),
                    # TODO: Ghost Batch Normalization
                    nn.BatchNorm1d(feature_channels, momentum=bn_momentum),
                )
                for _ in range(n_decision_steps - 1)
            ]
        )
        self.fc = nn.Linear(decision_hidden_size, out_channels, bias=False)

    def forward(self, dense_features, cat_features=None):
        """
        Args:
            dense_features (torch.Tensor): numerical dense features. shape: (N, dense_channels), dtype: float
            cat_features (Optional[torch.Tensor]): categorical features. shape: (N, len(cat_cardinalities)), dtype: long
        Returns:
            logits (torch.Tensor): shape: (N, out_channels), dtype: float
            masks (List[torch.Tensor]): masks for each decision step. `n_decision_steps` tensors of shape (N, feature_size)
            sparsity_regularization (torch.Tensor): feature selection sparsity regularization loss
        """
        batch_size = dense_features.size(0)
        dtype = dense_features.dtype
        device = dense_features.device

        feature = dense_features
        if cat_features is not None:
            # Convert cateogirical features into dense values.
            assert cat_features.size(1) == len(self.cat_embeddings)
            cat_embs = [
                self.cat_embeddings[i](cat_features[:, i])
                for i in range(len(self.cat_embeddings))
            ]
            # Concat dense and categoical features.
            feature = torch.cat([dense_features] + cat_embs, dim=-1)

        aggregated_output = torch.zeros(
            batch_size, self.decision_hidden_size, dtype=dtype, device=device
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
            decision_out, coef_out = x.split(
                self.decision_hidden_size, dim=1
            )  # (N, decision_hidden_size), (N, attention_hidden_size)

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
