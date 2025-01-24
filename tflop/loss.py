"""
 Following code is referenced from the pytorch implementation of the Supervised Contrastive Loss:
 https://github.com/HobbitLong/SupContrast
"""

import torch
import torch.nn as nn


class TableCL(nn.Module):
    """
    A contrastive learning model for table data with masks.

    Attributes:
        temperature (float): Temperature scaling factor for the loss function.
        sup_con_loss (SupConLoss): Instance of the supervised contrastive loss.
    """

    def __init__(self, temperature=0.1):
        """
        Initialize the TableCL module.

        Args:
            temperature (float): Temperature scaling factor for the contrastive loss.
        """
        super(TableCL, self).__init__()
        self.temperature = temperature
        self.sup_con_loss = SupConLoss(temperature)

    def forward(self, features, masks, input_coords_length):
        """
        Compute the batch loss for the given features and masks.

        Args:
            features (torch.Tensor): Feature representations, shape [batch_size, bbox_token_length, d_model].
            masks (torch.Tensor): Masks, shape [batch_size, num_layers, bbox_token_length, bbox_token_length].
            input_coords_length (torch.Tensor): Lengths of input coordinates, shape [batch_size].

        Returns:
            torch.Tensor: Average batch loss.
        """
        batch_loss, valid_batch_size = 0, 0
        batch_size, _, _, _ = masks.shape
        for data_i in range(batch_size):
            selected_mask = masks[data_i][
                :,
                : (input_coords_length[data_i] + 1),
                : (input_coords_length[data_i] + 1),
            ]  # [1, bbox_tok_cnt + 1, bbox_tok_cnt + 1]
            assert selected_mask.shape[0] == 1
            selected_feature = features[data_i][
                : (input_coords_length[data_i] + 1)
            ]  # [bbox_tok_cnt + 1, d_model]
            selected_feature = selected_feature.unsqueeze(
                0
            )  # [1, bbox_tok_cnt + 1, d_model]

            batch_loss += self.sup_con_loss(selected_feature, mask=selected_mask)

            # check if the data is valid
            float_selected_mask = selected_mask[0].to(
                torch.float
            )  # [bbox_tok_cnt + 1, bbox_tok_cnt + 1]
            sanity_tensor = torch.eye(
                float_selected_mask.shape[0],
                dtype=float_selected_mask.dtype,
                device=float_selected_mask.device,
            )
            sanity_tensor[0, 0] = 0
            if torch.sum(float_selected_mask != sanity_tensor) != 0:
                valid_batch_size += 1

        valid_batch_size = max(valid_batch_size, 1)
        batch_loss = batch_loss / valid_batch_size

        return batch_loss


class SupConLoss(nn.Module):
    """
    A PyTorch implementation of a modified version of Supervised Contrastive Loss.

    Args:
        temperature (float): Temperature scaling factor for contrastive loss. Default is 0.1.

    Methods:
        forward(features, mask):
            Computes the modified supervised contrastive loss for the given features and mask.
    """

    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, mask):
        """
        Forward pass to compute the supervised contrastive loss.

        Args:
            features (torch.Tensor): Feature representations, shape [batch_size, bbox_token_length, d_model].
            masks (torch.Tensor): Masks, shape [batch_size, num_layers, bbox_token_length, bbox_token_length].

        Returns:
            torch.Tensor: A scalar tensor representing the computed contrastive loss.
        """
        batch_size, bbox_token_length, d_model = features.shape

        # compute logits
        dot_contrast = torch.div(
            torch.matmul(features, features.transpose(1, 2)), self.temperature
        )  # [batch_size, bbox_token_length, bbox_token_length]

        # for numerical stability
        logits_max, _ = torch.max(
            dot_contrast, dim=-1, keepdim=True
        )  # [batch_size, bbox_token_length, 1]
        logits = (
            dot_contrast - logits_max.detach()
        )  # [batch_size, bbox_token_length, bbox_token_length]

        # logits mask (diagonal)
        logits_mask = 1 - torch.eye(
            bbox_token_length, dtype=logits.dtype, device=logits.device
        ).unsqueeze(
            0
        )  # [1, bbox_token_length, bbox_token_length]

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        negative_mask = 1 - mask
        negative_mask[negative_mask < 1] = 0
        negative_denom = torch.sum(
            exp_logits * negative_mask, dim=-1, keepdim=True
        )  # [batch_size, bbox_token_length, 1]

        positive_mask = mask.clone()
        positive_mask[positive_mask > 0] = 1
        positive_denom = torch.sum(
            exp_logits * positive_mask, dim=-1, keepdim=True
        )  # [batch_size, bbox_token_length, 1]

        denominator = negative_denom + positive_denom.detach() + 1e-6
        log_prob = logits - torch.log(denominator)

        # compute mean of log-likelihood over positive
        mask = mask * logits_mask
        mean_log_prob_pos = (mask * log_prob).sum(-1) / (
            mask.sum(-1) + 1e-6
        )  # [batch_size, bbox_token_length]

        # loss
        loss = -1 * mean_log_prob_pos.mean()

        return loss
