import torch
from torch import nn

from typing import Literal
from . import NoneOrType

class HAMLoss(nn.Module):
    def __init__(self, 
                 horizon_size: int, 
                 device: torch.device,
                 loss_fn: Literal["mse", "mae", "smape", "mape", "mase"] = "mse", 
                 num_variates: NoneOrType(int) = None,
                 variate_indices: NoneOrType(list[int]) = None,
                 cpu_gpu_transfer: bool = False) -> None:
        """
        predictions: (Batchsize, HorizonSize, Variates)
        num_variates: None if all variates are included or number of variates
        variate_indices: None if all variates need to be considered one by one or joint distribution indices
                         Only relevant if num_variates isn't None

        ~ 2.5x speedup without CPU-GPU transfer with batch size = 1
        """

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

            if variate_indices is None:
                self.mask = torch.zeros((horizon_size, num_variates, num_variates)).to(device)
                for idx in range(num_variates):
                    self.mask[:,idx,idx] = 1
            else:
                self.mask = torch.zeros((horizon_size, num_variates)).to(device)
                self.variate_indices = torch.tensor(variate_indices).to(device)

            self.dims = (0,)

        if not cpu_gpu_transfer:
            if variate_indices is None:
                self.indices_tensors = [None] + [torch.tensor(range(0, h)).to(device) for h in range(1, horizon_size+1)]

            else:
                self.indices_tensors = [None] + [torch.tensor(range(0, h)).to(device) for h in range(1, horizon_size+1)]
        else:
            self.device = device

        self.cpu_gpu_transfer = cpu_gpu_transfer

    def assign_mask(self, timestep: int, mode: Literal["causal", "anticausal"]) -> None:

        self.mask.fill_(0)

        if (timestep > 0 and mode == "causal") or \
                (timestep < self.mask.shape[0] and mode == "anticausal"):
            if len(self.dims) == 2:
                if self.cpu_gpu_transfer:
                    ones_idx = torch.tensor(range(0, timestep)).to(self.device) if mode == "causal" \
                                    else torch.tensor(range(timestep, self.mask.shape[0])).to(self.device)
                else:
                    ones_idx = self.indices_tensors[timestep if mode == "causal" else self.mask.shape[0]-timestep]

                self.mask.index_fill_(0, ones_idx, 1)
            else:

                if len(self.mask.shape) != 3:
                    self.mask.index_fill_(1, self.variate_indices, 1) 

                if self.cpu_gpu_transfer:
                    zeroes_idx = torch.tensor(range(0, timestep)).to(self.device) if mode == "anticausal" \
                                    else torch.tensor(range(timestep, self.mask.shape[0])).to(self.device)
                else:
                    zeroes_idx = self.indices_tensors[self.mask.shape[0]-timestep if mode == "causal" else timestep]
                if not zeroes_idx is None and len(zeroes_idx) > 0:
                    self.mask.index_fill_(0, zeroes_idx, 0) 

    def MSE_per_timestep(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:

        se = (pred - gt)**2
        se = se.mean(self.dims)
        
        if len(self.mask.shape) == 3:
            se = torch.einsum("...hv,...hwv->...hw", se, self.mask)
            loss = se.mean(dim=0)
        else:
            se = se * self.mask
            loss = se.mean()
        
        return loss

    def sMAPE_per_timestep(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:

        ae = torch.abs(pred - gt)
        factor = torch.abs(pred) + torch.abs(gt)
        N = pred.shape[1]

        smape = (ae / factor) * 2. / N
        smape[smape != smape] = 0

        smape = smape.mean(self.dims)
        
        if len(self.mask.shape) == 3:
            smape = torch.einsum("...hv,...hwv->...hw", smape, self.mask)
            loss = smape.mean(dim=0)
        else:
            smape = smape * self.mask
            loss = smape.mean()

        return loss

    def MAPE_per_timestep(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:

        N = pred.shape[1]
        ape = (100. / N) * torch.abs((gt - pred) / gt)

        ape = ape.mean(self.dims)
        
        if len(self.mask.shape) == 3:
            ape = torch.einsum("...hv,...hwv->...hw", ape, self.mask)
            loss = ape.mean(dim=0)
        else:
            ape = ape * self.mask
            loss = ape.mean()

        return loss

    def MASE_per_timestep(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:

        # Forecast window taken as entire dataset for scale for now (similar to instance norm vs norm)

        N = pred.shape[1]
        scale = torch.abs(gt[:, 1:, :] - gt[:, :-1, :]) / (N - 1.)
        ase = torch.abs(pred - gt) / N

        ase = ase.mean(self.dims)
        
        if len(self.mask.shape) == 3:
            ase = torch.einsum("...hv,...hwv->...hw", ase, self.mask)
            loss = ase.mean(dim=0)
        else:
            ase = ase * self.mask
            loss = ase.mean()

        return loss

    def MAE_per_timestep(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:

        ae = torch.abs(pred - gt)
        ae = ae.mean(self.dims)
        
        if len(self.mask.shape) == 3:
            ae = torch.einsum("...hv,...hwv->...hw", ae, self.mask)
            loss = ae.mean(dim=0)
        else:
            ase = ae * self.mask
            loss = ae.mean()

        return loss

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:

        return self.loss_fn(pred, gt)
