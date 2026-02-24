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

        self.horizon_size = horizon_size

        assert variate_indices is None or all([x < num_variates for x in variate_indices]), "Invalid variate dimensions!"

        if num_variates is None:
            self.mask = torch.zeros(horizon_size).to(device)
            self.dims = (0, 2)
            self.working_mask = self.mask.clone()

        else:
            assert isinstance(num_variates, int), "Integer size only!"

            if variate_indices is None:
                self.mask = torch.zeros((horizon_size, num_variates, num_variates)).to(device)
                for idx in range(num_variates):
                    self.mask[:,idx,idx] = 1
                self.working_mask = self.mask.clone()
            else:
                self.mask = torch.zeros((horizon_size, num_variates)).to(device)
                self.variate_indices = torch.tensor(variate_indices).to(device)
                self.working_mask = self.mask.clone()

            self.dims = (0,)

        if not cpu_gpu_transfer:
            self.indices_tensors = [None] + [torch.tensor(range(0, h)).long().to(device) for h in range(1, horizon_size+1)]
            self.indices_tensors_rev = [torch.tensor(range(h, horizon_size)).long().to(device) for h in range(1, horizon_size+1)] + [None]
        else:
            self.device = device

        self.cpu_gpu_transfer = cpu_gpu_transfer

    def assign_mask(self, timestep: int, mode: Literal["causal", "anticausal"]) -> None:
        
        if (timestep == 0 and mode == "causal") or (timestep == self.horizon_size and mode == "anticausal"):
            self.working_mask.fill_(0)
        else:
            self.working_mask.copy_(self.mask)

        if len(self.dims) == 2:
            self.mask.fill_(0)

            if self.cpu_gpu_transfer:
                ones_idx = torch.tensor(range(0, timestep)).to(self.device) if mode == "causal" \
                                else torch.tensor(range(timestep, self.mask.shape[0])).to(self.device)
            else:
                ones_idx = self.indices_tensors[timestep if mode == "causal" else self.mask.shape[0]-timestep]

            if not ones_idx is None:
                self.working_mask.index_fill_(0, ones_idx, 1)
        else:

            if len(self.mask.shape) != 3:
                self.mask.index_fill_(1, self.variate_indices, 1) 

            if self.cpu_gpu_transfer:
                zeroes_idx = torch.tensor(range(0, timestep)).to(self.device) if mode == "anticausal" \
                                else torch.tensor(range(timestep, self.mask.shape[0])).to(self.device)
            else:
                zeroes_idx = self.indices_tensors[timestep] if mode == "anticausal" else self.indices_tensors_rev[timestep]
                if not zeroes_idx is None:
                    self.working_mask.index_fill_(0, zeroes_idx, 0)

    def MSE_per_timestep(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # B x V x H

        # Ignore Transformer model outputs
        pred = pred[:, :, -self.horizon_size:]
        gt = gt[:, :, -self.horizon_size:]

        se = (pred - gt)**2
        se = se.mean(self.dims)

        if len(self.working_mask.shape) == 3:
            se = torch.einsum("...vh,...hvw->...hw", se, self.working_mask)
            loss = se.mean(dim=0)
        else:
            se = se * self.working_mask
            loss = se.mean()

        return loss

    def sMAPE_per_timestep(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # B x V x H

        # Ignore Transformer model outputs
        pred = pred[:, :, -self.horizon_size:]
        gt = gt[:, :, -self.horizon_size:]

        ae = torch.abs(pred - gt)
        factor = torch.abs(pred) + torch.abs(gt)
        N = pred.shape[1]

        smape = (ae / factor) * 2. / N
        smape[smape != smape] = 0

        smape = smape.mean(self.dims)

        if len(self.working_mask.shape) == 3:
            smape = torch.einsum("...vh,...hvw->...hw", smape, self.working_mask)
            loss = smape.mean(dim=0)
        else:
            smape = smape * self.working_mask
            loss = smape.mean()

        return loss

    def MAPE_per_timestep(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # B x V x H

        # Ignore Transformer model outputs
        pred = pred[:, :, -self.horizon_size:]
        gt = gt[:, :, -self.horizon_size:]

        N = pred.shape[-1]
        ape = (100. / N) * torch.abs((gt - pred) / gt)

        ape = ape.mean(self.dims)

        if len(self.working_mask.shape) == 3:
            ape = torch.einsum("...vh,...hvw->...hw", ape, self.working_mask)
            loss = ape.mean(dim=0)
        else:
            ape = ape * self.working_mask
            loss = ape.mean()

        return loss

    def MASE_per_timestep(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # B x V x H
        # Forecast window taken as entire dataset for scale for now (similar to instance norm vs norm)

        # Ignore Transformer model outputs
        pred = pred[:, :, -self.horizon_size:]
        gt = gt[:, :, -self.horizon_size:]

        N = pred.shape[-1]
        scale = torch.abs(gt[:, 1:, :] - gt[:, :-1, :]) / (N - 1.)
        ase = torch.abs(pred - gt) / N

        ase = ase.mean(self.dims)

        if len(self.working_mask.shape) == 3:
            ase = torch.einsum("...vh,...hvw->...hw", ase, self.working_mask)
            loss = ase.mean(dim=0)
        else:
            ase = ase * self.working_mask
            loss = ase.mean()

        return loss

    def MAE_per_timestep(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # B x V x H

        # Ignore Transformer model outputs
        pred = pred[:, :, -self.horizon_size:]
        gt = gt[:, :, -self.horizon_size:]

        ae = torch.abs(pred - gt)
        ae = ae.mean(self.dims)

        if len(self.working_mask.shape) == 3:
            ae = torch.einsum("...vh,...hvw->...hw", ae, self.working_mask)
            loss = ae.mean(dim=0)
        else:
            ae = ae * self.working_mask.transpose(-2, -1)
            loss = ae.mean()

        return loss

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:

        return self.loss_fn(pred, gt)
