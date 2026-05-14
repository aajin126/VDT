import torch
import torch.nn as nn


class AELoss(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=0.001):

        super().__init__()
        self.kl_weight = kl_weight  
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight


    def forward(self, inputs, reconstructions, posteriors, optimizer_idx=None, global_step=None, split="train"):
        """
        inputs, reconstructions: (B, T, C, H, W)
        posteriors: distribution object with .kl()
        """

        seq_len = 10
        B = inputs.shape[0]

        ce_loss = 0

        for k in range(seq_len):

            ce_loss = ce_loss +  torch.nn.functional.binary_cross_entropy(
                reconstructions[:, k],
                inputs[:, k],
                reduction="sum"
            ).div(B)

        ce_loss = ce_loss / seq_len

        kl_loss = posteriors.kl().mean()

        loss = ce_loss + self.kl_weight * kl_loss

        log = {
            f"{split}/total_loss": loss.detach(),
            f"{split}/ce_loss": ce_loss.detach(),
            f"{split}/kl_loss": kl_loss.detach(),
        }

        return loss, log
