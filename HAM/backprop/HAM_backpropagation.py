import torch
from torch import nn

import os

from . import HAMLoss
from . import NoneOrType
from collections.abc import Callable
from typing import Literal

from decimal import Decimal

from tqdm import tqdm

class HAMTrain:

    #MODES = ["causal", "anticausal"]
    MODES = ["causal", "anticausal"]
    SERIALIZATION_DIR = "gradnorms_by_batch/"

    INTERPOLATION_SUBSERIES_MODELS = ["former", "SpaceTime", "DDPM"]
    INTERPOLATION_SUBSERIES_LENGTH = 50

    def __init__(self, 
                 model: nn.Module,
                 load_model_checkpoint: NoneOrType(str),
                 horizon_size: int,
                 dataset_iterator: iter,
                 device: torch.device,
                 num_variates: NoneOrType(int),
                 log_folder: str,
                 model_name: str,
                 loss_fn: Literal["mse", "mae", "smape", "mape", "mase"] = "mae",
                 variate_indices: NoneOrType(list[list[int]]) = None, 
                 layers: NoneOrType(list[str]) | NoneOrType(tuple[str]) = None,
                 cpu_gpu_transfer: bool = False,
                 quantifier_fn: Callable[[torch.Tensor], float] = lambda x: x.norm(),
                 variate_names: NoneOrType(list[str] | tuple[str]) = None) -> None:

        """
        model: Takes as input iter(dataset_iterator)[0].to(device) and returns tensor with the same shape as iter(dataset_iterator)[1]
        load_model_checkpoint: Path to trained model to load
        horizon_size: Per horizon size model's HAM training
        dataset_iterator: Does all the processing towards model.forward's inputs except transfer to device
        num_variates: Total number of multivariate features in the model
        log_folder: Folder to store gradient activities as txt files
        model_name: Name of model to store gradient activities
        variate_indices: Independently every variate if None else specific variate distributions by lists of indices
        layers: All the model's layers if None else specific layer name(s)
        cpu_gpu_transfer: Enable mask tensors from CPU to GPU during gradient calculations using True
        quantifier_fn: Choice of quantifier per weight update tensor typically needs to be N-dim --> value, defaults to 2-norm
        device: CPU or PUs
        """

        model_layers = [p[0] for p in model.named_parameters()]
        assert layers is None or all([l in model_layers for l in layers]), "Unknown layer name in %s!" % ','.join(layers)
        if layers is None:
            layers = model_layers

        assert variate_indices is None or all([v_idx < num_variates for v in variate_indices for v_idx in v]), "Use variate indices within range!"
        assert (not num_variates is None and variate_indices is None and not variate_names is None) or \
                num_variates is None or not variate_indices is None, "Variate names missing for log files"

        self.device = device

        if not load_model_checkpoint is None:
            model.load_state_dict(torch.load(load_model_checkpoint))
        self.model = model.to(device)

        self.dataset = dataset_iterator

        self.log_folder = log_folder
        if not os.path.isdir(log_folder):
            os.makedirs(log_folder)

        self.model_name = model_name
        self.horizon_size = horizon_size
        self.variate_names = variate_names

        self.num_variates = num_variates
        self.variate_indices = variate_indices
        self.layers = layers

        if variate_indices is None:
            self.loss_fns = [HAMLoss(horizon_size = horizon_size, 
                                     device = device,
                                     loss_fn = loss_fn, 
                                     num_variates = num_variates,
                                     variate_indices = variate_indices,
                                     cpu_gpu_transfer = cpu_gpu_transfer)]
        else:
            self.loss_fns = [HAMLoss(horizon_size = horizon_size, 
                                     device = device,
                                     loss_fn = loss_fn, 
                                     num_variates = num_variates,
                                     variate_indices = v,
                                     cpu_gpu_transfer = cpu_gpu_transfer)
                                for v in variate_indices]

        if not layers is None:
            self.filename = lambda x: os.path.join(self.log_folder, "%s%s_%d_gradnorms.txt" % (
                                        self.model_name, os.path.commonprefix(self.layers), self.horizon_size)) \
                                if not isinstance(x, int) else \
                                os.path.join(self.log_folder, "%s%s:%s_%d_gradnorms.txt" % (
                                        self.model_name, os.path.commonprefix(self.layers), self.variate_names[x], self.horizon_size))
        else:
            self.filename = lambda x: os.path.join(self.log_folder, "%s_%d_gradnorms.txt" % (self.model_name, self.horizon_size)) \
                                if not isinstance(x, int) else \
                                os.path.join(self.log_folder, "%s:%s_%d_gradnorms.txt" % (self.model_name, self.variate_names[x], self.horizon_size))

        for mode in self.MODES:
            if os.path.exists(self.filename(mode)):
                raise FileExistsError("Quantifier files exist for this model: %s!" % self.filename(mode))

        if num_variates is None:
            dims = (len(dataset_iterator), len(layers)) 
        else:
            if variate_indices is None:
                dims = (len(dataset_iterator), len(layers), num_variates)
            else:
                dims = (len(dataset_iterator), len(layers), len(variate_indices))

        if not os.path.isdir(self.SERIALIZATION_DIR):
            os.makedirs(self.SERIALIZATION_DIR)

        try:
            self.batch_start = self.deserialize_quantifier_tensor()

        except Exception:
            self.batch_start = 0

            # Keeping dataset's overall tensor in the CPU to enable the same batch sizes as in the training
            self.quantifier_per_timestep = {}
            for mode in self.MODES:
                self.quantifier_per_timestep[mode] = [torch.zeros(dims) for _ in range(horizon_size+1)]

    def serialize_quantifier_tensor(self, batch_idx: int) -> None:

        save_dict = {"batch": batch_idx, "quantifier": self.quantifier_per_timestep}
        save_path = os.path.join(self.SERIALIZATION_DIR, "%s_%d.pth" % (self.model_name, self.horizon_size))
        torch.save(save_dict, save_path)

    def deserialize_quantifier_tensor(self) -> int:

        path = os.path.join(self.SERIALIZATION_DIR, "%s_%d.pth" % (self.model_name, self.horizon_size))

        load_dict = torch.load(path)
        batch_start = load_dict["batch"] + 1

        if batch_start == len(self.dataset):
            print ("Serialization path %s has calculated gradient quantifier values! Please remove to re-run!" % path)
            exit()

        self.quantifier_per_timestep = load_dict["quantifier"]

        return batch_start

    def write_log_file(self, fname, mode, variate_idx=None):

        with open(fname, 'w') as f:
            for h in range(self.horizon_size+1):
                start_end = (0, h) if mode == "causal" else (h, self.horizon_size)
                write_str = "Grad norms for subseries %d->%d: " % (start_end[0], start_end[1])

                for idx, layer in enumerate(self.model.named_parameters()):
                    if layer[0] in self.layers:
                        if not variate_idx is None:
                            write_str += "%s=%.16E " % (layer[0], self.quantifier_per_timestep[mode][h][:, idx, variate_idx].mean())
                        else:
                            write_str += "%s=%.16E " % (layer[0], self.quantifier_per_timestep[mode][h][:, idx].mean())
                write_str += "\n"
                f.write(write_str)

    def backward(self):

        for mode in self.MODES:
            print (mode)

            for batch_idx, inputs in enumerate(tqdm(self.dataset)):

                if batch_idx < self.batch_start:
                    continue

                inputs = [x.to(self.device) for x in inputs]
                input_window, horizon_gt, *metadata = inputs

                horizon_pred = self.model(*inputs)

                for h in range(self.horizon_size + 1):

                    for param in self.model.parameters():
                        if not param.grad is None:
                            param.grad.fill_(0)

                    if any([x in self.model_name for x in self.INTERPOLATION_SUBSERIES_MODELS]):
                        if idx % self.INTERPOLATION_SUBSERIES_LENGTH != 0:
                            continue

                    for loss_idx, loss_fn in enumerate(self.loss_fns):
                        loss_fn.assign_mask(h, mode)
                        loss = loss_fn(horizon_pred, horizon_gt)

                        # If multivariate or one of the joint distributions of variates
                        if len(loss.shape) == 0:

                            loss.backward(retain_graph=True)

                            for idx, (n, param) in enumerate(self.model.named_parameters()):
                                if not n in self.layers:
                                    continue
                                if not param.grad is None:
                                    if len(self.loss_fns) == 1:
                                        self.quantifier_per_timestep[mode][h][batch_idx][idx] = param.grad.norm()
                                    else:
                                        self.quantifier_per_timestep[mode][h][batch_idx][idx][loss_idx] = param.grad.norm()

                            if len(self.loss_fns) > 1:
                                for param in self.model.parameters():
                                    if not param.grad is None:
                                        param.grad.fill_(0)

                        # If independently distributed across variates one by one
                        else:

                            for v_idx in range(self.num_variates):
                                v_loss = loss[v_idx:v_idx+1].mean()
                                v_loss.backward(retain_graph=True)

                                for idx, (n, param) in enumerate(self.model.named_parameters()):
                                    if not n in self.layers:
                                        continue
                                    if not param.grad is None:
                                        self.quantifier_per_timestep[mode][h][batch_idx][idx][v_idx] = param.grad.norm()

                                for param in self.model.parameters():
                                    if not param.grad is None:
                                        param.grad.fill_(0)

                            loss = v_loss

                self.serialize_quantifier_tensor(batch_idx)

                loss.backward(retain_graph=False)

                # No optim.step() between batches with detach()
                for param in self.model.parameters():
                    if not param.grad is None:
                        param.grad.detach()

        # Save gradient quantifier to corresponding files!
        for mode in self.MODES:
            self.quantifier_per_timestep[mode] = [x.cpu().numpy() for x in self.quantifier_per_timestep[mode]]
            if len(self.quantifier_per_timestep[mode][0].shape) == 3:
                for v_idx in range(self.quantifier_per_timestep[mode][0].shape[-1]):
                    fname = self.filename(v_idx)
                    self.write_log_file(fname, mode, v_idx)
            else:
                fname = self.filename(None)
                self.write_log_file(fname, mode)
