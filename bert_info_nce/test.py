# import torch
# from torch import nn
# from typing import Optional

# class FocalLoss(nn.Module):
#     def __init__(self,
#                  alpha: Optional[torch.Tensor] = None,
#                  gamma: float = 2.0,
#                  reduction: str = 'mean',
#                  ignore_index: int = -100):
#         super().__init__()
#         if reduction not in ('mean', 'sum', 'none'):
#             raise ValueError('Reduction must be one of: "mean", "sum", "none".')
#         self.alpha = alpha
#         self.gamma = gamma
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.nll_loss = nn.NLLLoss(weight=alpha, reduction='none', ignore_index=ignore_index)

#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         if x.ndim > 2:
#             c = x.shape[1]
#             x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
#             y = y.view(-1)

#         unignored_mask = y != self.ignore_index
#         y = y[unignored_mask]
#         if len(y) == 0:
#             return torch.tensor(0., device=x.device, dtype=x.dtype)
#         x = x[unignored_mask]

#         log_p = nn.functional.log_softmax(x, dim=-1)
#         ce = self.nll_loss(log_p, y)

#         all_rows = torch.arange(len(x))
#         log_pt = log_p[all_rows, y]

#         pt = log_pt.exp()
#         focal_term = (1 - pt)**self.gamma

#         loss = focal_term * ce

#         if self.reduction == 'mean':
#             loss = loss.mean()
#         elif self.reduction == 'sum':
#             loss = loss.sum()

#         return loss


# # Function to test FocalLoss on different cases
# def test_focal_loss():
#     # Test case 1: Simple multiclass classification
#     print("Test Case 1: Simple multiclass classification")
#     x = torch.tensor([[0.1, 0.5, 0.4], [0.3, 0.2, 0.5], [0.2, 0.6, 0.2]], dtype=torch.float32)  # logits
#     y = torch.tensor([1, 2, 0])  # ground truth labels
#     focal_loss_fn = FocalLoss(gamma=2.0)
#     loss = focal_loss_fn(x, y)
#     print(f"Loss: {loss.item()}")

#     # Test case 2: Input with ignore_index
#     print("\nTest Case 2: Input with ignore_index")
#     x_ignore = torch.tensor([[0.1, 0.5, 0.4], [0.3, 0.2, 0.5], [0.2, 0.6, 0.2]], dtype=torch.float32)  # logits
#     y_ignore = torch.tensor([1, -100, 0])  # ground truth with ignored label
#     loss_ignore = focal_loss_fn(x_ignore, y_ignore)
#     print(f"Loss with ignored label: {loss_ignore.item()}")

#     # Test case 3: All labels ignored
#     print("\nTest Case 3: All labels ignored")
#     x_all_ignore = torch.tensor([[0.1, 0.5, 0.4], [0.3, 0.2, 0.5], [0.2, 0.6, 0.2]], dtype=torch.float32)
#     y_all_ignore = torch.tensor([-100, -100, -100])  # all labels are ignored
#     loss_all_ignore = focal_loss_fn(x_all_ignore, y_all_ignore)
#     print(f"Loss with all labels ignored: {loss_all_ignore.item()}")


# # Run the test cases
# if __name__ == "__main__":
#     test_focal_loss()

clusters = [5, 4, 3, 2, 1, 0]

if len(clusters) == max(clusters) + 1:
    print(1)

