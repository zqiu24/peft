# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import time
import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from .matmul_qoft import matmul
from .skew_symmetric import SkewSymmetric

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class MultiplicativeDropoutLayer(nn.Module):
    """
    Implements the multiplicative dropout layer for OFT.
    """

    def __init__(self, p=0.0):
        """
        Initializes the multiplicative dropout layer.

        Parameters:
        p (float): The probability of dropping out a block. Defaults to 0.0.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Applies multiplicative dropout to the input tensor.

        Parameters:
        x (Tensor): The input tensor of shape (D, H, H), where `D` represents
                    the number of OFT blocks, and `H` is the size of the square blocks along the last two dimensions,
                    the block size in OFT.
        """
        if self.training:
            # Ensure the last two dimensions are the same
            if x.shape[-1] != x.shape[-2]:
                raise ValueError("The last two dimensions of input should be the same!")

            D, H, _ = x.shape

            # If block share, skip the multiplicative dropout
            if D == 1:
                return x

            num_to_replace = int(self.p * D)
            num_zeros = D - num_to_replace
            mask = torch.cat([torch.ones(num_to_replace, device=x.device), torch.zeros(num_zeros, device=x.device)])
            mask = mask[torch.randperm(D)].view(D, 1, 1)
            eye_matrix = torch.eye(H, device=x.device).repeat(D, 1, 1)
            x = (1 - mask) * x + mask * eye_matrix
        return x


class OFTRotationModule(nn.Module):
    def __init__(self, r, n_elements, block_size):
        super().__init__()
        self.r = r
        self.n_elements = n_elements
        self.block_size = block_size
        self.weight = nn.Parameter(torch.empty(r, n_elements))
        self.in_features = r * block_size

    def pytorch_skew_symmetric(self, vec, block_size):
        batch_size = vec.shape[0]
        matrix = torch.zeros(batch_size, block_size, block_size, 
                            device=vec.device, dtype=vec.dtype)
        
        # Create indices for upper triangle (excluding diagonal)
        rows, cols = torch.triu_indices(block_size, block_size, 1, device=vec.device)
        matrix[:, rows, cols] = vec
        matrix = matrix - matrix.transpose(-2, -1)
        return matrix
        
    def _cayley_batch(self, Q: torch.Tensor, block_size: int, num_neumann_terms: int = 5) -> torch.Tensor:
        """
        Perform the Cayley parametrization on a batch of skew-symmetric matrices.

        Args:
            data: A batch of skew-symmetric matrices of shape (b, r, c).
        """

        b, _ = Q.shape

        #Q_skew_start = time.time()
        # Q_skew = 0.5 * (Q - Q.transpose(1, 2))
        Q_skew = SkewSymmetric.apply(Q, block_size)
        # Q_skew = self.pytorch_skew_symmetric(Q, block_size)
        #timings["Q_skew"] = time.time() - Q_skew_start

        #R_start = time.time()
        R = torch.eye(block_size, device=Q.device, dtype=Q.dtype).repeat(b, 1, 1)
        if num_neumann_terms > 1:
            R.add_(Q_skew, alpha=2.0)
            # R = R + 2.0 * Q_skew
            if num_neumann_terms > 2:
                Q_squared = torch.bmm(Q_skew, Q_skew)
                R.add_(Q_squared, alpha=2.0)
                # R = R + 2.0 * Q_squared

                Q_power = Q_squared
                for i in range(3, num_neumann_terms):
                    Q_power = torch.bmm(Q_power, Q_skew)
                    R.add_(Q_power, alpha=2.0)
                    # R = R + 2.0 * Q_power
        
        #R_end = time.time()
        #timings["R"] = R_end - R_start

        #timings["total"] = time.time() - total_start_time
        #breakpoint()
        return R #.to(previous_dtype)
    
    def forward(self, x):
        # This module doesn't need to implement the orthogonal transform
        # It's primarily a container for the parameter
        # The actual transformation logic stays in your OFTLayer

        oft_rotation = self._cayley_batch(self.weight, self.block_size)

        orig_shape = x.shape
        x_reshaped = x.reshape(orig_shape[0], orig_shape[1], self.r, self.block_size)
        x_rotated_reshaped = torch.einsum('...rk,rkc->...rc', x_reshaped, oft_rotation)
        x_rotated = x_rotated_reshaped.reshape(*orig_shape)

        # batch_dims = x.shape[:-1]
        # x_reshaped = x.view(*batch_dims, self.r, -1)
        # x_rotated_reshaped = torch.einsum('...rk,rkc->...rc', x_reshaped, oft_rotation)
        # x_rotated = x_rotated_reshaped.reshape(*batch_dims, self.in_features)

        return x_rotated

class OFTLayer(BaseTunerLayer):
    """
    Implements the OFT layer.
    """

    # All names of layers that may contain adapter weights
    adapter_layer_names = ("oft_r", "oft_s")
    # other_param_names is defined on parent class
    other_param_names = ("r", "oft_block_size", "oft_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        """
        Initializes the OFT layer.

        Note, currently only support linear layer and convolutional layer, with further support for other layers to be
        added soon.

        Parameters:
        base_layer: the pretrained model layer
        """
        self.base_layer = base_layer
        # OFT info
        self.oft_r = nn.ParameterDict({})
        self.oft_block_size = {}
        self.oft_s = nn.ParameterDict({})
        self.r = {}
        self.oft_block_size = {}
        self.oft_dropout = nn.ModuleDict({})
        self.coft = {}
        self.eps = {}
        self.block_share = {}
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.MultiheadAttention):
            if not base_layer._qkv_same_embed_dim:
                raise ValueError(f"Only same dim for query/key/value is supported as of now for {self.__class__}.")
            in_features, out_features = base_layer.embed_dim, 3 * base_layer.embed_dim
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    @property
    def _available_adapters(self) -> set[str]:
        return {*self.oft_r}

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return

        warnings.warn("Scaling operation for OFT not supported! Automatically set scale to 1.")

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.oft_r.keys():
                continue

            warnings.warn("Scaling operation for OFT not supported! Automatically set scale to 1.")

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.oft_r.keys():
                continue

            warnings.warn("Unscaling operation for OFT not supported! Keeping scale to 1.")

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

    def update_layer(self, adapter_name, r, oft_block_size, module_dropout, coft, eps, block_share, init_weights):
        """
        Update the linear layer with trainable OFT weights. Override for other layer types.
        """
        """Internal function to create oft adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            oft_block_size (`int`): The block size for added adapter.
            module_dropout (`float`):
                The multiplicative dropout probability for disabling adapter blocks during training.
            coft (`bool`): Whether to use the constrained variant of OFT or not.
            eps (`float`):
                The control strength of COFT. The freedom of rotation. Only has an effect if `coft` is set to True.
            block_share (`bool`): Whether to share the OFT parameters between blocks or not.
            init_weights (`bool`): Whether to initialize weights.
        """
        # Initialize the MultiplicativeDropoutLayer for module_dropout > 0.0.
        if module_dropout > 0.0:
            # oft_dropout_layer = MultiplicativeDropoutLayer(p=module_dropout)
            oft_dropout_layer = nn.Dropout(p=module_dropout)
        else:
            oft_dropout_layer = nn.Identity()
        self.oft_dropout.update(nn.ModuleDict({adapter_name: oft_dropout_layer}))

        if r == 0 and oft_block_size != 0:
            if self.in_features % oft_block_size != 0 or oft_block_size > self.in_features:
                old_oft_block_size = oft_block_size
                oft_block_size = self.adjust_oft_parameters(self.in_features, oft_block_size)
                warnings.warn(
                    f"Invalid `oft_block_size` ({old_oft_block_size})! Adjusted `oft_block_size` to ({oft_block_size})."
                )
            r = int(self.in_features // oft_block_size)
        elif r != 0 and oft_block_size == 0:
            if self.in_features % r != 0 or r > self.in_features:
                old_r = r
                r = self.adjust_oft_parameters(self.in_features, r)
                warnings.warn(f"Invalid `r` ({old_r})! Adjusted `r` to ({r}).")
            oft_block_size = int(self.in_features // r)
        else:
            raise ValueError(
                "Something went wrong, please report this error: https://github.com/huggingface/peft/issues"
            )

        self.coft[adapter_name] = coft
        self.block_share[adapter_name] = block_share
        self.eps[adapter_name] = eps * math.ceil(self.out_features / r) * math.ceil(self.out_features / r)

        # Create weights with provided shape
        if block_share:
            self.oft_r[adapter_name] = nn.Parameter(
                torch.empty(1, math.ceil(self.in_features / r), math.ceil(self.in_features / r))
            )
        else:     
            # self.oft_r[adapter_name] = nn.Linear(self.in_features, self.in_features, bias=False)

            def is_power_of_2(n):
                return n>0 and (n & (n-1)) == 0
            if not is_power_of_2(oft_block_size):
                breakpoint()

            n_elements = oft_block_size * (oft_block_size - 1) // 2
            # self.oft_r[adapter_name] = nn.Parameter(
            #     torch.empty(r, n_elements, dtype=self.get_base_layer().weight.data.dtype)
            # )
            self.oft_r[adapter_name] = OFTRotationModule(r, n_elements, oft_block_size) #, self.get_base_layer().weight.data.dtype, self.get_base_layer().weight.data.device)
            
        # self.oft_s[adapter_name] = nn.Parameter(torch.empty(int(self.out_features), 1))
        # self.oft_s[adapter_name] = nn.Parameter(torch.ones(1))

        if hasattr(self.get_base_layer(), 'weight'):
            device = self.get_base_layer().weight.device
        elif hasattr(self.get_base_layer(), 'qweight'):
            device = self.get_base_layer().qweight.device
        else:
            # Handle the case where neither attribute exists, if necessary
            # This might indicate an unexpected layer type.
            raise AttributeError("The base layer has neither 'weight' nor 'qweight' attribute.")

        # self.layout = torch.eye(r).unsqueeze(0).to(torch.int64)
        # self.oft_matmul = matmul(self.layout, oft_block_size, 'qoft_dsd', trans_a=False, trans_b=False, device=self.get_base_layer().weight.device)
        
        # Initialize weights
        self.reset_oft_parameters(adapter_name, init_weights)

        # set oft r and block size
        self.r[adapter_name] = r
        self.oft_block_size[adapter_name] = oft_block_size

        # TODO: tmp store r and oft_block_size
        self.rank = r
        self.bs = oft_block_size

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)


    def reset_oft_parameters(self, adapter_name, init_weights):
        """
        Reset the OFT parameters.
        """
        if init_weights is False:
            nn.init.normal_(self.oft_r[adapter_name], mean=0.0, std=0.1)
            # nn.init.normal_(self.oft_s[adapter_name], mean=1.0, std=0.1)
            return

        if adapter_name in self.oft_r.keys():
            if init_weights is True:
                # initialize oft_r to zero
                nn.init.zeros_(self.oft_r[adapter_name].weight)
                # nn.init.ones_(self.oft_s[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_weights=}")

    def pytorch_skew_symmetric(self, vec, block_size):
        batch_size = vec.shape[0]
        matrix = torch.zeros(batch_size, block_size, block_size, 
                            device=vec.device, dtype=vec.dtype)
        
        # Create indices for upper triangle (excluding diagonal)
        rows, cols = torch.triu_indices(block_size, block_size, 1, device=vec.device)
        
        # Fill upper triangle
        matrix[:, rows, cols] = vec
        
        # Make skew-symmetric
        matrix = matrix - matrix.transpose(-2, -1)
        
        return matrix 
    
    
    def _cayley_batch(self, Q: torch.Tensor, block_size: int, num_neumann_terms: int = 5) -> torch.Tensor:
        """
        Perform the Cayley parametrization on a batch of skew-symmetric matrices.

        Args:
            data: A batch of skew-symmetric matrices of shape (b, r, c).
        """
        # previous_dtype = Q.dtype

        # total_start_time = time.time()
        # timings = {}

       # Q = Q.to(torch.float32)
        b, _ = Q.shape

        # Q_skew_start = time.time()
        # Q_skew = 0.5 * (Q - Q.transpose(1, 2))
        # Q_skew = SkewSymmetric.apply(Q, block_size)
        Q_skew = self.pytorch_skew_symmetric(Q, block_size)
        #timings["Q_skew"] = time.time() - Q_skew_start

        #R_start = time.time()
        R = torch.eye(block_size, device=Q.device, dtype=Q.dtype).repeat(b, 1, 1)
        if num_neumann_terms > 1:
            # R.add_(Q_skew, alpha=2.0)
            R = R + 2.0 * Q_skew
            if num_neumann_terms > 2:
                Q_squared = torch.bmm(Q_skew, Q_skew)
                # R.add_(Q_squared, alpha=2.0)
                R = R + 2.0 * Q_squared

                Q_power = Q_squared
                for i in range(3, num_neumann_terms):
                    Q_power = torch.bmm(Q_power, Q_skew)
                    # R.add_(Q_power, alpha=2.0)
                    R = R + 2.0 * Q_power
        
        #R_end = time.time()
        #timings["R"] = R_end - R_start

        #timings["total"] = time.time() - total_start_time
        #breakpoint()
        return R #.to(previous_dtype)
    
    def _cayley_batch_inv(self, Q: torch.Tensor, block_size: int, num_neumann_terms: int = 5) -> torch.Tensor:
        b, _ = Q.shape
        Q_skew = SkewSymmetric.apply(Q, block_size)
        previous_dtype = Q_skew.dtype
        if previous_dtype != torch.float32:
            Q_skew = Q_skew.to(torch.float32)
        I = torch.eye(block_size, device=Q.device, dtype=Q.dtype).repeat(b, 1, 1)
        R = torch.linalg.solve(I + Q_skew, I - Q_skew, left=False)
        if previous_dtype != torch.float32:
            R = R.to(previous_dtype)
        return R

    # Copied from https://github.com/Zeju1997/oft/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft-control/oft.py#L155
    def _block_diagonal(self, oft_r: torch.Tensor, rank: int) -> torch.Tensor:
        if oft_r.shape[0] == 1:
            # block share
            blocks = [oft_r[0, ...] for i in range(rank)]
        else:
            blocks = [oft_r[i, ...] for i in range(rank)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    # Copied from https://github.com/Zeju1997/oft/blob/84cebb965df69781e3d9c3c875f5980b421eaf24/oft-control/oft.py#L52
    def _project_batch(self, oft_r, eps=1e-5):
        # scaling factor for each of the smaller block matrix
        eps = eps * 1 / torch.sqrt(torch.tensor(oft_r.shape[0]))
        I = (  # noqa: E741
            torch.zeros((oft_r.size(1), oft_r.size(1)), device=oft_r.device, dtype=oft_r.dtype)
            .unsqueeze(0)
            .expand_as(oft_r)
        )
        diff = oft_r - I
        norm_diff = torch.norm(oft_r - I, dim=(1, 2), keepdim=True)
        mask = (norm_diff <= eps).bool()
        out = torch.where(mask, oft_r, I + eps * (diff / norm_diff))
        return out

    def adjust_oft_parameters(self, in_features, params):
        """
        Adjust the OFT parameters to be divisible by the in_features dimension.
        """
        if params < in_features:
            higher_params = params
            while higher_params <= in_features and in_features % higher_params != 0:
                higher_params += 1
        else:
            return in_features

        lower_params = params
        while lower_params > 1 and in_features % lower_params != 0:
            lower_params -= 1

        if (params - lower_params) <= (higher_params - params):
            return lower_params
        else:
            return higher_params


class Linear(nn.Module, OFTLayer):
    """OFT implemented in Linear layer"""

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 8,
        oft_block_size: int = 0,
        module_dropout: float = 0.0,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_weights: Union[bool, str] = True,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        OFTLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name

        self.update_layer(adapter_name, r, oft_block_size, module_dropout, coft, eps, block_share, init_weights)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer


        self.counter = 0

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        breakpoint()
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self._available_adapters:
                base_layer = self.get_base_layer()
                orig_dtype = base_layer.weight.dtype
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data
                    oft_mat, oft_s = self.get_delta_weight(active_adapter)
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = torch.mm(oft_mat, orig_weights.to(oft_mat.dtype))
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = orig_weights * oft_s

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights.contiguous().to(orig_dtype)
                else:
                    oft_mat, oft_s = self.get_delta_weight(active_adapter)
                    orig_weights = base_layer.weight.data
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = torch.mm(oft_mat, orig_weights.to(oft_mat.dtype))
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = orig_weights * oft_s

                    base_layer.weight.data = orig_weights.contiguous().to(orig_dtype)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        breakpoint()
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        base_layer = self.get_base_layer()
        orig_dtype = base_layer.weight.dtype
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.oft_r.keys():
                oft_mat, oft_s = self.get_delta_weight(active_adapter)

                orig_weights = self.get_base_layer().weight.data
                orig_weights = torch.transpose(orig_weights, 0, 1)
                orig_weights = torch.mm(oft_mat.t(), orig_weights.to(oft_mat.dtype))
                orig_weights = torch.transpose(orig_weights, 0, 1)

                base_layer.weight.data = (orig_weights * (1 / oft_s)).to(orig_dtype)

    def get_delta_weight(self, adapter_name) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        breakpoint()
        oft_r = self.oft_r[adapter_name]
        oft_s = self.oft_s[adapter_name]

        rank = self.r[adapter_name]
        coft = self.coft[adapter_name]
        eps = self.eps[adapter_name]

        if coft:
            with torch.no_grad():
                oft_r.copy_(self._project_batch(oft_r, eps=eps))

        orth_rotate = self._cayley_batch(oft_r)
        weight = self._block_diagonal(orth_rotate, rank)

        return weight, oft_s

    def forward_benchmark_memory(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                with record_function("Linear::unmerge_in_forward"): # Record if unmerge happens here
                    self.unmerge()
            with record_function("Linear::base_layer_forward (disabled)"):
                result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            with record_function("Linear::base_layer_forward (merged)"):
                result = self.base_layer(x, *args, **kwargs)
        else:
            # --- Record OFT-specific forward path ---
            with record_function("Linear::oft_forward_path"):
                # Record initialization
                with record_function("Linear::oft_init_rot_scale"):
                    oft_rotation = torch.eye(self.in_features, device=x.device)
                    oft_scale = torch.ones((int(self.out_features), 1), device=x.device)

                # The loop itself isn't easily recordable as one block,
                # but the operations inside are recorded via _cayley_batch etc.
                # Or we can record the whole loop calculation part
                with record_function("Linear::oft_adapter_loop"):
                    for active_adapter in self.active_adapters:
                        if active_adapter not in self.oft_r.keys():
                            continue
                        # Retrieve parameters (fast, likely negligible memory)
                        oft_r = self.oft_r[active_adapter]
                        oft_block_size = self.oft_block_size[active_adapter]
                        # oft_s = self.oft_s[active_adapter]
                        dropout = self.oft_dropout[active_adapter]
                        rank = self.r[active_adapter]
                        coft = self.coft[active_adapter]
                        eps = self.eps[active_adapter]

                        # COFT projection (uses recorded _project_batch)
                        if coft:
                            with torch.no_grad():
                                oft_r.copy_(self._project_batch(oft_r, eps=eps))

                        # Cayley transform (uses recorded _cayley_batch)
                        with record_function("Linear::adapter_loop_cayley"):
                             orth_rotate = self._cayley_batch(oft_r, oft_block_size)

                        # Dropout
                        with record_function("Linear::adapter_loop_dropout"):
                            orth_rotate = dropout(orth_rotate)

                        # Matmul/Block Diagonal Step
                        with record_function("Linear::adapter_loop_matmul"):
                            current_oft_rot_dtype = oft_rotation.dtype
                            if orth_rotate.dtype != current_oft_rot_dtype:
                                orth_rotate = orth_rotate.to(current_oft_rot_dtype)
                            # Assuming oft_matmul calls the custom kernel
                            # The profiler might show the kernel name or just this block
                            oft_rotation = self.oft_matmul(orth_rotate, oft_rotation.unsqueeze(0)).squeeze(0)
                            oft_rotation = oft_rotation.to(current_oft_rot_dtype) # Cast back

                        # Scaling (small tensor usually)
                        with record_function("Linear::adapter_loop_scaling"):
                             oft_scale = oft_s.to(oft_scale.dtype) * oft_scale # Ensure dtype match before multiply

                # Cast input dtype (usually small memory impact unless input is huge)
                with record_function("Linear::cast_input_dtype"):
                     x = x.to(self.get_base_layer().weight.data.dtype)

                # Weight transformation
                with record_function("Linear::oft_weight_transform"):
                    orig_weight = self.get_base_layer().weight.data
                    orig_weight = torch.transpose(orig_weight, 0, 1)
                    rotated_weight = torch.mm(oft_rotation, orig_weight.to(oft_rotation.dtype))
                    rotated_weight = torch.transpose(rotated_weight, 0, 1)
                    scaled_rotated_weight = rotated_weight * oft_scale.to(rotated_weight.dtype)

                # Final Linear layer
                with record_function("Linear::final_F.linear"):
                    # Cast bias if needed
                    final_weight_dtype = scaled_rotated_weight.dtype
                    x = self._cast_input_dtype(x, final_weight_dtype)
                    bias = self.get_base_layer().bias
                    if bias is not None:
                        bias = self._cast_input_dtype(bias, final_weight_dtype)

                    result = F.linear(input=x, weight=scaled_rotated_weight, bias=bias)

        # Cast result back to original dtype
        with record_function("Linear::cast_output_dtype"):
            result = result.to(previous_dtype)
        return result

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        # previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            # oft_rotation = torch.eye(self.in_features, device=x.device)
            # oft_rotation = torch.eye(self.bs, device=x.device, dtype=x.dtype).repeat(self.rank, 1, 1)

            # oft_scale = torch.ones((int(self.out_features), 1), device=x.device)
            # oft_scale = torch.ones(1, device=x.device)

            for active_adapter in self.active_adapters:
                if active_adapter not in self.oft_r.keys():
                    continue
                oft_r = self.oft_r[active_adapter].weight
                oft_block_size = self.oft_block_size[active_adapter]
                # oft_s = self.oft_s[active_adapter]
                # dropout = self.oft_dropout[active_adapter]

                rank = self.r[active_adapter]
                coft = self.coft[active_adapter]
                eps = self.eps[active_adapter]

                '''
                if coft:
                    with torch.no_grad():
                        oft_r.copy_(self._project_batch(oft_r, eps=eps))
                orth_rotate = self._cayley_batch(oft_r, oft_block_size)

                current_oft_rot_dtype = oft_rotation.dtype
                if orth_rotate.dtype != current_oft_rot_dtype:
                    orth_rotate = orth_rotate.to(current_oft_rot_dtype)
                # oft_rotation = self.oft_matmul(orth_rotate, oft_rotation.unsqueeze(0)).squeeze(0)
                oft_rotation = torch.bmm(orth_rotate, oft_rotation)
                oft_rotation = oft_rotation.to(current_oft_rot_dtype)

                # oft_scale = oft_s * oft_scale


            # x = self._cast_input_dtype(x, oft_r.dtype)
            x = x.to(self.get_base_layer().weight.data.dtype)
            # x = dropout(x) # * oft_scale

            batch_dims = x.shape[:-1]
            x_reshaped = x.view(*batch_dims, rank, -1)
            x_rotated_reshaped = torch.einsum('...rk,rkc->...rc', x_reshaped, oft_rotation)
            x_rotated = x_rotated_reshaped.reshape(*batch_dims, self.in_features)
            x_rotated = x_rotated.to(self.get_base_layer().weight.dtype)
            '''
            x_rotated = oft_r[active_adapter](x)
            
            result = self.base_layer(x_rotated, *args, **kwargs)

        # result = result.to(previous_dtype)
        return result

    def forward_working_backup(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            oft_rotation = torch.eye(self.in_features, device=x.device)
            oft_scale = torch.ones((int(self.out_features), 1), device=x.device)

            for active_adapter in self.active_adapters:
                if active_adapter not in self.oft_r.keys():
                    continue
                oft_r = self.oft_r[active_adapter]
                oft_block_size = self.oft_block_size[active_adapter]
                oft_s = self.oft_s[active_adapter]
                dropout = self.oft_dropout[active_adapter]

                rank = self.r[active_adapter]
                coft = self.coft[active_adapter]
                eps = self.eps[active_adapter]

                if coft:
                    with torch.no_grad():
                        oft_r.copy_(self._project_batch(oft_r, eps=eps))

                orth_rotate = self._cayley_batch(oft_r, oft_block_size)
                orth_rotate = dropout(orth_rotate)


                current_oft_rot_dtype = oft_rotation.dtype
                if orth_rotate.dtype != current_oft_rot_dtype:
                    orth_rotate = orth_rotate.to(current_oft_rot_dtype)
                oft_rotation = self.oft_matmul(orth_rotate, oft_rotation.unsqueeze(0)).squeeze(0)
                oft_rotation = oft_rotation.to(current_oft_rot_dtype)

                '''
                oft_mat = self._block_diagonal(orth_rotate, rank)
                oft_rotation = oft_mat @ oft_rotation
                oft_scale = oft_s * oft_scale
                '''
            x = x.to(self.get_base_layer().weight.data.dtype)

            orig_weight = self.get_base_layer().weight.data
            orig_weight = torch.transpose(orig_weight, 0, 1)
            rotated_weight = torch.mm(oft_rotation, orig_weight.to(oft_rotation.dtype))
            rotated_weight = torch.transpose(rotated_weight, 0, 1)
            scaled_rotated_weight = rotated_weight * oft_scale

            x = self._cast_input_dtype(x, scaled_rotated_weight.dtype)
            bias = self._cast_input_dtype(self.get_base_layer().bias, scaled_rotated_weight.dtype)
            result = F.linear(input=x, weight=scaled_rotated_weight, bias=bias)

        result = result.to(previous_dtype)
        return result

    def forward_benchmark_time(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self.counter += 1

        previous_dtype = x.dtype
        torch_result_dtype = x.dtype # Use torch_result_dtype to restore at the end

        # GPU Warm-up: Perform a few small operations
        if x.is_cuda:
            for _ in range(3): # Number of warm-up iterations
                warmup_tensor_a = torch.randn(10, 10, device=x.device, dtype=previous_dtype)
                warmup_tensor_b = torch.randn(10, 10, device=x.device, dtype=previous_dtype)
                _ = warmup_tensor_a @ warmup_tensor_b
            torch.cuda.synchronize(device=x.device) # Ensure warm-up ops are complete

        # Timing the whole forward method
        total_start_time = time.time()
        timings = {}

        if self.disable_adapters:
            if self.merged:
                unmerge_start = time.time()
                self.unmerge()
                timings["unmerge"] = time.time() - unmerge_start
            base_layer_start = time.time()
            result = self.base_layer(x, *args, **kwargs)
            timings["base_layer"] = time.time() - base_layer_start
        elif self.merged:
            base_layer_start = time.time()
            result = self.base_layer(x, *args, **kwargs)
            timings["base_layer_merged"] = time.time() - base_layer_start
        else:
            # Time initialization
            init_start = time.time()
            oft_rotation = torch.eye(self.in_features, device=x.device, dtype=previous_dtype) # Changed to previous_dtype
            oft_scale = torch.ones((int(self.out_features), 1), device=x.device, dtype=previous_dtype) # Changed to previous_dtype
            timings["initialization"] = time.time() - init_start

            adapter_loop_total_time = 0
            coft_projection_total_time = 0
            cayley_total_time = 0
            dropout_total_time = 0
            block_diag_total_time = 0 # Removed as _block_diagonal is commented out
            oft_matmul_total_time = 0 # Added for self.oft_matmul
            scaling_total_time = 0

            for active_adapter in self.active_adapters:
                adapter_loop_start = time.time()
                if active_adapter not in self.oft_r.keys():
                    continue

                # Param retrieval is usually very fast, not timed separately
                oft_r = self.oft_r[active_adapter]
                oft_block_size = self.oft_block_size[active_adapter]
                oft_s = self.oft_s[active_adapter]
                dropout = self.oft_dropout[active_adapter]
                rank = self.r[active_adapter] # rank needed if using _block_diagonal
                coft = self.coft[active_adapter]
                eps = self.eps[active_adapter]

                # Time COFT projection
                coft_start = time.time()
                if coft:
                    with torch.no_grad():
                        oft_r.copy_(self._project_batch(oft_r, eps=eps))
                coft_projection_total_time += time.time() - coft_start

                # Time Cayley transform
                cayley_start = time.time()
                orth_rotate = self._cayley_batch(oft_r, oft_block_size)
                cayley_total_time += time.time() - cayley_start

                # Time dropout
                dropout_start = time.time()
                orth_rotate = dropout(orth_rotate)
                dropout_total_time += time.time() - dropout_start

                # Time block diagonal (if used)
                oft_matmul_start = time.time()
                # breakpoint()
                # oft_mat = self._block_diagonal(orth_rotate, rank)
                # oft_rotation = oft_mat @ oft_rotation
                current_oft_rot_dtype = oft_rotation.dtype
                if orth_rotate.dtype != current_oft_rot_dtype:
                    orth_rotate = orth_rotate.to(current_oft_rot_dtype)
                oft_rotation = self.oft_matmul(orth_rotate, oft_rotation.unsqueeze(0)).squeeze(0)
                oft_rotation = oft_rotation.to(current_oft_rot_dtype)

                # Time oft_matmul
                # oft_matmul_start = time.time()
                # Ensure dtypes match for matmul if using custom kernel
                # oft_rotation might need casting depending on oft_matmul implementation
                # Assuming oft_matmul handles dtypes or orth_rotate/oft_rotation match
                oft_matmul_total_time += time.time() - oft_matmul_start

                # Time scaling
                scaling_start = time.time()
                # Ensure oft_s and oft_scale have compatible dtypes for multiplication
                oft_scale = oft_s.to(oft_scale.dtype) * oft_scale
                scaling_total_time += time.time() - scaling_start

                adapter_loop_total_time += time.time() - adapter_loop_start

            timings["adapter_loop_total"] = adapter_loop_total_time
            timings["coft_projection_total"] = coft_projection_total_time
            timings["cayley_transform_total"] = cayley_total_time
            timings["dropout_total"] = dropout_total_time
            timings["oft_matmul_total"] = oft_matmul_total_time # Added
            timings["scaling_total"] = scaling_total_time # Added


            # Time weight transformation
            weight_transform_start = time.time()
            # Cast x right before F.linear as per original logic
            # x = x.to(self.get_base_layer().weight.data.dtype) # Moved down

            orig_weight = self.get_base_layer().weight.data
            orig_weight_dtype = orig_weight.dtype # Store original weight dtype
            orig_weight = torch.transpose(orig_weight, 0, 1)
            # Ensure dtypes match for torch.mm
            oft_rotation_dtype = oft_rotation.dtype
            rotated_weight = torch.mm(oft_rotation, orig_weight.to(oft_rotation_dtype))
            rotated_weight = torch.transpose(rotated_weight, 0, 1)
            # Ensure dtypes match for scaling
            scaled_rotated_weight = rotated_weight * oft_scale.to(rotated_weight.dtype)
            timings["weight_transformation"] = time.time() - weight_transform_start

            # Time final linear operation (including dtype casting)
            linear_op_start = time.time()
            final_weight_dtype = scaled_rotated_weight.dtype
            x = self._cast_input_dtype(x, final_weight_dtype) # Cast x to final weight dtype
            bias = self.get_base_layer().bias
            if bias is not None:
                bias = self._cast_input_dtype(bias, final_weight_dtype) # Cast bias to final weight dtype

            result = F.linear(input=x, weight=scaled_rotated_weight, bias=bias)
            timings["final_linear_op_incl_cast"] = time.time() - linear_op_start

        total_end_time = time.time()
        timings["total_forward_time"] = total_end_time - total_start_time

        # Print timings
        if self.counter == 100:
            total_time = total_end_time - total_start_time
            print("--- Forward Backup Timings ---")
            for op, duration in timings.items():
                print(f"{op}: {duration:.6f}s, percentage: {duration/total_time:.6f}")
                print("----------------------------")

            breakpoint()

        result = result.to(torch_result_dtype) # Restore original dtype
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "oft." + rep


class Conv2d(nn.Module, OFTLayer):
    """OFT implemented in Conv2d layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 8,
        oft_block_size: int = 0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        module_dropout: float = 0.0,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        init_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        OFTLayer.__init__(self, base_layer)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name

        # Create adapter and set it active
        self.update_layer(adapter_name, r, oft_block_size, module_dropout, coft, eps, block_share, init_weights)

    def update_layer(self, adapter_name, r, oft_block_size, module_dropout, coft, eps, block_share, init_weights):
        """
        Update the conv2d layer with trainable OFT weights.
        """
        # Initialize the MultiplicativeDropoutLayer for module_dropout > 0.0.
        if module_dropout > 0.0:
            # oft_dropout_layer = MultiplicativeDropoutLayer(p=module_dropout)
            oft_dropout_layer = nn.Dropout(p=module_dropout)
        else:
            oft_dropout_layer = nn.Identity()
        self.oft_dropout.update(nn.ModuleDict({adapter_name: oft_dropout_layer}))

        # layer information from the base layer
        base_layer = self.get_base_layer()
        conv_filter_dim = self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0]

        if r == 0 and oft_block_size != 0:
            if conv_filter_dim % oft_block_size != 0 or oft_block_size > conv_filter_dim:
                old_oft_block_size = oft_block_size
                oft_block_size = self.adjust_oft_parameters(conv_filter_dim, oft_block_size)
                warnings.warn(
                    f"Invalid `oft_block_size` ({old_oft_block_size})! Adjusted `oft_block_size` to ({oft_block_size})."
                )
            r = int(conv_filter_dim // oft_block_size)
        elif r != 0 and oft_block_size == 0:
            if conv_filter_dim % r != 0 or r > conv_filter_dim:
                old_r = r
                r = self.adjust_oft_parameters(conv_filter_dim, r)
                warnings.warn(f"Invalid `r` ({old_r})! Adjusted `r` to ({r}).")
            oft_block_size = int(conv_filter_dim // r)
        else:
            raise ValueError(
                "Something went wrong, please report this error: https://github.com/huggingface/peft/issues"
            )

        self.coft[adapter_name] = coft
        self.block_share[adapter_name] = block_share
        self.eps[adapter_name] = eps * math.ceil(self.out_features / r) * math.ceil(self.out_features / r)

        # Create weights with provided shape
        if block_share:
            self.oft_r[adapter_name] = nn.Parameter(
                torch.empty(1, math.ceil(conv_filter_dim / r), math.ceil(conv_filter_dim / r))
            )
        else:
            self.oft_r[adapter_name] = nn.Parameter(
                torch.empty(r, math.ceil(conv_filter_dim / r), math.ceil(conv_filter_dim / r))
            )
        # self.oft_s[adapter_name] = nn.Parameter(torch.empty(int(self.out_features), 1))
        # self.oft_s[adapter_name] = nn.Parameter(torch.ones(1))

        # Initialize weights
        self.reset_oft_parameters(adapter_name, init_weights)

        # set oft r and block size
        self.r[adapter_name] = r
        self.oft_block_size[adapter_name] = oft_block_size

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

        if self.oft_r[adapter_name].device != base_layer.weight.device:
            breakpoint()

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.oft_r.keys():
                base_layer = self.get_base_layer()
                orig_dtype = base_layer.weight.dtype
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    oft_mat, oft_s = self.get_delta_weight(active_adapter)

                    orig_weights = orig_weights.view(
                        self.out_features, self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0]
                    )
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = torch.mm(oft_mat, orig_weights.to(oft_mat.dtype))
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = orig_weights * oft_s
                    orig_weights = orig_weights.view(
                        self.out_features, self.in_features, base_layer.kernel_size[0], base_layer.kernel_size[0]
                    )

                    base_layer.weight.data = orig_weights.contiguous().to(orig_dtype)
                else:
                    oft_mat, oft_s = self.get_delta_weight(active_adapter)

                    orig_weights = base_layer.weight.data.clone()
                    orig_weights = orig_weights.view(
                        self.out_features, self.in_features * base_layer.kernel_size[0] * base_layer.kernel_size[0]
                    )
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = torch.mm(oft_mat, orig_weights.to(oft_mat.dtype))
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                    orig_weights = orig_weights * oft_s
                    orig_weights = orig_weights.view(
                        self.out_features, self.in_features, base_layer.kernel_size[0], base_layer.kernel_size[0]
                    )

                    base_layer.weight.data = orig_weights.contiguous().to(orig_dtype)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        base_layer = self.get_base_layer()
        orig_dtype = base_layer.weight.dtype
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.oft_r.keys():
                oft_mat, oft_s = self.get_delta_weight(active_adapter)

                orig_weights = self.get_base_layer().weight.data.clone()
                orig_weights = orig_weights.view(
                    self.out_features,
                    self.in_features * self.get_base_layer().kernel_size[0] * self.get_base_layer().kernel_size[0],
                )
                orig_weights = torch.transpose(orig_weights, 0, 1)
                orig_weights = torch.mm(oft_mat.t(), orig_weights.to(oft_mat.dtype))
                orig_weights = torch.transpose(orig_weights, 0, 1)
                orig_weights = orig_weights * (1 / oft_s)
                orig_weights = orig_weights.view(
                    self.out_features,
                    self.in_features,
                    self.get_base_layer().kernel_size[0],
                    self.get_base_layer().kernel_size[0],
                )

                base_layer.weight.data = orig_weights.to(orig_dtype)

    def get_delta_weight(self, adapter_name) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        oft_r = self.oft_r[adapter_name]
        oft_s = self.oft_s[adapter_name]

        rank = self.r[adapter_name]
        coft = self.coft[adapter_name]
        eps = self.eps[adapter_name]

        if coft:
            with torch.no_grad():
                oft_r.copy_(self._project_batch(oft_r, eps=eps))

        orth_rotate = self._cayley_batch(oft_r)
        weight = self._block_diagonal(orth_rotate, rank)

        return weight, oft_s

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            oft_rotation = torch.eye(
                self.in_features * self.get_base_layer().kernel_size[0] * self.get_base_layer().kernel_size[0],
                device=x.device,
            )
            oft_scale = torch.ones((int(self.out_features), 1), device=x.device)

            for active_adapter in self.active_adapters:
                if active_adapter not in self.oft_r.keys():
                    continue
                oft_r = self.oft_r[active_adapter]
                # oft_s = self.oft_s[active_adapter]
                dropout = self.oft_dropout[active_adapter]

                rank = self.r[active_adapter]
                coft = self.coft[active_adapter]
                eps = self.eps[active_adapter]

                if coft:
                    with torch.no_grad():
                        oft_r.copy_(self._project_batch(oft_r, eps=eps))

                orth_rotate = self._cayley_batch(oft_r)
                orth_rotate = dropout(orth_rotate)
                oft_mat = self._block_diagonal(orth_rotate, rank)

                oft_rotation = oft_mat @ oft_rotation
                oft_scale = oft_s * oft_scale

            orig_weights = self.base_layer.weight.data
            orig_weights = orig_weights.view(
                self.out_features,
                self.in_features * self.get_base_layer().kernel_size[0] * self.get_base_layer().kernel_size[0],
            )
            orig_weights = torch.transpose(orig_weights, 0, 1)
            oft_rotation = oft_rotation.to(previous_dtype)
            orig_weights = orig_weights.to(previous_dtype)
            rotated_weight = torch.mm(oft_rotation, orig_weights)
            rotated_weight = torch.transpose(rotated_weight, 0, 1)

            scaled_rotated_weight = rotated_weight * oft_scale

            scaled_rotated_weight = scaled_rotated_weight.view(
                self.out_features,
                self.in_features,
                self.get_base_layer().kernel_size[0],
                self.get_base_layer().kernel_size[0],
            )
            x = self._cast_input_dtype(x, scaled_rotated_weight.dtype)
            bias = self._cast_input_dtype(self.get_base_layer().bias, scaled_rotated_weight.dtype)
            result = F.conv2d(
                input=x,
                weight=scaled_rotated_weight,
                bias=bias,
                padding=self.get_base_layer().padding[0],
                stride=self.get_base_layer().stride[0],
            )

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "oft." + rep


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    oft_config: OFTConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, nn.Conv1d):
        new_module = Conv1d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = oft_config.fan_in_fan_out = False
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = oft_config.fan_in_fan_out = True
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
