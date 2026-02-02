import torch
from torch import nn

from typing import Literal
NoneOrType = lambda T: type[T] | None

class HAMLoss(nn.Module):

    # predictions: (Batchsize, HorizonSize, Variates)
    # num_variates: None if uniform mask is applied over all variates or number of variates
    # ~ 2.5x speedup without CPU-GPU transfer with batch size = 1
    def __init__(self, 
                 horizon_size: int, 
                 device: torch.device,
                 loss_fn: Literal["mse", "mae", "smape", "mape", "mase"] = "mse", 
                 num_variates: NoneOrType(int) = None, 
                 variate_indices: NoneOrType(list[int]) = None,
                 cpu_gpu_transfer: bool = False):

        super().__init__()

        if loss_fn == "mse":
            self.loss_fn = self.MSE_per_timestep
        elif loss_fn == "mae":
            self.loss_fn = self.MAE_per_timestep
        elif loss_fn == "smape":
            self.loss_fn = self.sMAPE_per_timestep
        elif loss_fn == "mape":
            self.loss_fn = self.MAPE_per_timestep
        elif loss_fn == "mase":
            self.loss_fn = self.MASE_per_timestep
        else:
            raise NotImplementedError

        assert variate_indices is None or all([x < num_variates for x in variate_indices]), "Invalid variate dimensions!"

        if num_variates is None:
            self.mask = torch.zeros(horizon_size).to(device)
            self.dims = (0, 2)
        else:
            assert isinstance(num_variates, int), "Integer size only!"
            self.mask = torch.zeros((horizon_size, num_variates)).to(device)

            self.variates_mask = torch.zeros(num_variates).to(device)
            if variate_indices is None:
                self.variate_indices = torch.tensor(list(range(num_variates))).to(device)
            else:
                self.variate_indices = torch.tensor(variate_indices).to(device)

            self.dims = (0,)

        if not cpu_gpu_transfer:
            self.indices_tensors = [None] + [torch.tensor(range(0, h)).to(device) for h in range(1, horizon_size+1)]
        else:
            self.device = device

        self.cpu_gpu_transfer = cpu_gpu_transfer

    def assign_mask(self, timestep: int, mode: Literal["causal", "anti-causal"]):

        if not mode in ["causal", "anti-causal"]: 
            raise NotImplementedError

        assert timestep <= self.mask.shape[0], \
                "Timestep only includes 0 (all zeroes) to H (the entire horizon)"

        self.mask.fill_(0)
        
        if (timestep > 0 and mode == "causal") or \
                (timestep < self.mask.shape[0] and mode == "anti-causal"):
            if len(self.dims) == 2:
                if self.cpu_gpu_transfer:
                    ones_idx = torch.tensor(range(0, timestep)).to(self.device) if mode == "causal" \
                                    else torch.tensor(range(timestep, self.mask.shape[0])).to(self.device)
                else:
                    ones_idx = self.indices_tensors[timestep if mode == "causal" else self.mask.shape[0]-timestep]

                self.mask.index_fill_(0, ones_idx, 1)
            else:
                self.mask.index_fill_(1, self.variate_indices, 1) 
                
                if self.cpu_gpu_transfer:
                    zeroes_idx = torch.tensor(range(0, timestep)).to(self.device) if mode == "anti-causal" \
                                    else torch.tensor(range(timestep, self.mask.shape[0])).to(self.device)
                else:
                    zeroes_idx = self.indices_tensors[self.mask.shape[0]-timestep if mode == "causal" else timestep]
                if not zeroes_idx is None and len(zeroes_idx) > 0:
                    self.mask.index_fill_(0, zeroes_idx, 0) 
    
    def MSE_per_timestep(self, pred: torch.Tensor, gt: torch.Tensor):

        se = (pred - gt)**2
        se = se.mean(self.dims)
        se *= self.mask

        return se.mean()

    def sMAPE_per_timestep(self, pred: torch.Tensor, gt: torch.Tensor):

        ae = torch.abs(pred - gt)
        factor = torch.abs(pred) + torch.abs(gt)
        N = pred.shape[1]
        
        smape = (ae / factor) * 2. / N
        smape[smape != smape] = 0

        smape = smape.mean(self.dims)
        smape *= self.mask

        return smape.mean()

    def MAPE_per_timestep(self, pred: torch.Tensor, gt: torch.Tensor):
        
        N = pred.shape[1]
        ape = (100. / N) * torch.abs((gt - pred) / gt)

        ape = ape.mean(self.dims)
        ape *= self.mask

        return ape.mean()

    def MASE_per_timestep(self, pred: torch.Tensor, gt: torch.Tensor):

        # Forecast window taken as entire dataset for scale for now (similar to instance norm vs norm)
        
        N = pred.shape[1]
        scale = torch.abs(gt[:, 1:, :] - gt[:, :-1, :]) / (N - 1.)
        ase = torch.abs(pred - gt) / N

        ase = ase.mean(self.dims)
        ase *= self.mask

        return ase.mean()

    def MAE_per_timestep(self, pred: torch.Tensor, gt: torch.Tensor):

        ae = torch.abs(pred - gt)
        ae = ae.mean(self.dims)
        ae *= self.mask

        return ae.mean()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):

        return self.loss_fn(pred, gt)
