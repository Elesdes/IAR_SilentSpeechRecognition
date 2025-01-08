import torch


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))


def criterion_choice(criterion_name: str) -> torch.nn.Module:
    match criterion_name:
        case "mse":
            return torch.nn.MSELoss()
        case "mae":
            return torch.nn.L1Loss()
        case "smooth_l1":
            return torch.nn.SmoothL1Loss()
        case "rmse":
            return RMSELoss()
        case _:
            raise ValueError(f"Criterion {criterion_name} not recognized")
