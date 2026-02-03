import torch


class NegationMask(torch.nn.Module):
    def __init__(self, negation_mask_size):
        super().__init__()
        # self.negation_mask = torch.ones(negation_mask_size)
        self.register_buffer("negation_mask", torch.ones(negation_mask_size))

    def update_mask(self, new_mask):
        self.negation_mask = new_mask

    def forward(self, x):
        return x * self.negation_mask
