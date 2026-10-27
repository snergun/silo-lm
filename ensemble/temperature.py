# Copied verbatim from pos_lm_v2 (branch infinigram-sparse, commit 92c99d8) pos_lm/temperature.py:
# the piecewise temperature (PCT) used in the WikiText-103 ensembles. Only TemperatureScaler and MLP are used here.
import torch
from typing import Optional, Union, List
import torch.nn as nn
import torch.nn.functional as F
import math
# from line_profiler import profile

class MLP(nn.Module):
    def __init__(self, d_in, d_out, d_inner, n_layer, dropout):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_inner)
        self.layers =  nn.ModuleList([nn.Linear(d_inner,d_inner) for _ in range(n_layer)])
        self.out_proj = nn.Linear(d_inner, d_out)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        x = self.dropout(self.in_proj(x))
        for layer in self.layers:
            x = F.relu(self.dropout(layer(x)))
        x = self.out_proj(x)
        return x

def reverse_cumsum(x: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(x,-1)
    return x - cumsum + cumsum[..., -1:None]

def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))

class PiecewiseTemperatureFunction:
    @staticmethod
    def apply_simple_temperature(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        return logits * temperature
        
    @staticmethod
    def bin_by_rank(logits: torch.Tensor, # (B, V)
                    counts: torch.Tensor, # (P-1,) if summing up to V, (P,)
                    ) -> torch.Tensor:
        sorted_logits, idx = torch.sort(logits, dim=-1)
        total_count = counts.sum()
        if total_count < logits.size(-1):
            counts = torch.cat([
                (logits.size(-1) - total_count).unsqueeze(0),
                counts
                ], dim=-1)
        bin_idx = torch.arange(
            counts.size(-1), device=logits.device
            ).repeat_interleave(counts).unsqueeze(0).expand(logits.shape) # (B, V)
        bin_ids = logits.new_zeros(logits.shape, dtype=torch.int)
        bin_ids[idx] = bin_idx
        cut_ids = torch.cumsum(counts, dim=0)
        thresholds = sorted_logits[:,cut_ids[:-1]] # (B, P-1)
        return bin_ids, thresholds
    
    @staticmethod
    def relative_shifts(logits: torch.Tensor, # (B,V)
                        ratios: torch.Tensor, # (P,) or (B, P)
                        ) -> torch.Tensor: # (B, V)
        """Get bins by shifts relative to max logit. Ratios will be normalized to sum to 1.
         E.g., ratios = [0.5, 0.3, 0.2] means thresholds are min, max - (max-min)*0.5, max - (max-min)*0.2, max """
        if ratios.dim() == 1:
            ratios = ratios.unsqueeze(0).expand(logits.size(0), ratios.size(-1)) # (B, P)
        B, P = ratios.shape
        min_vals = logits.min(dim=-1).values.unsqueeze(-1).expand(B, P-1) # (B, P)
        max_vals = logits.max(dim=-1).values.unsqueeze(-1).expand(B, P-1) # (B, P)
        ratios = (ratios / ratios.sum(dim=-1))[:,1:] # Make sure ratios add up to one
        return torch.cumsum(ratios, dim=-1) * (max_vals-min_vals) + min_vals # (B, P-1)
    
    @staticmethod
    def absolute_shifts(logits: torch.Tensor, # (B,V)
                        shifts: torch.Tensor, # (P-1,) or (B, P-1),
                        k: int = 1,
                        ) -> torch.Tensor: # (B, V)
        """Get bins by relative shifts from topk logit.
         E.g., shifts = [10, 3, 1] means thresholds are topk-14, topk-4, topk-1"""
        if shifts.dim() == 1:
            shifts = shifts.unsqueeze(0).expand(logits.size(0), shifts.size(-1)) # (B, P-1)
        shifts = reverse_cumsum(shifts) # (B, P-1)
        return logits.topk(k, dim=-1).values[:, -1].unsqueeze(-1) - shifts # (B, P-1)

    @staticmethod  
    def get_constants_no_grad(temperatures: torch.Tensor,  # (B, P)
                    thresholds: torch.Tensor,    # (B, P-1)
                            ):
        temp_diff = - torch.diff(temperatures, dim=-1) # (a1-a2, a2-a3, ....) : (B, P-1)
        c = torch.cumsum(temp_diff * thresholds, dim=-1) # (B, P-1)
        return torch.nn.functional.pad(c, (1,0), value=0.0) # (B, P)
    
    @staticmethod
    def apply_temperature(
                    logits: torch.Tensor, # (B, V)
                    temperatures: torch.Tensor, # (P,) or (B,P)
                    thresholds: torch.Tensor, # (P-1,) or (B, P-1)
                    bin_ids: torch.Tensor, # (B, V)
                    use_no_grad_version: bool = False,
                    ) -> torch.Tensor: # (B, V)
        B = logits.size(0)
        P = temperatures.size(-1)
        if temperatures.dim() == 1:
            temperatures = temperatures.unsqueeze(0).expand(B, P) # (B,P)
        if thresholds.dim() == 1:
            thresholds = thresholds.unsqueeze(0).expand(B, P-1) # (B,P-1)
        require_grad = temperatures.requires_grad or thresholds.requires_grad or logits.requires_grad
        if require_grad and not use_no_grad_version:
            return PiecewiseTemperatureFunction.apply_temperature_with_grad(
                logits, temperatures, thresholds, bin_ids
            )
        else:
            return PiecewiseTemperatureFunction.apply_temperature_no_grad(
                logits, temperatures, thresholds, bin_ids
            )
        
    @staticmethod
    def apply_temperature_no_grad(
                logits: torch.Tensor, # (B, V)
                temperatures: torch.Tensor, # (B,P)
                thresholds: torch.Tensor, # (B, P-1)
                bin_ids: torch.Tensor, # (B, V)
                ) -> torch.Tensor: # (B, V)
        c = PiecewiseTemperatureFunction.get_constants_no_grad(temperatures, thresholds) # (B, P)
        return logits * temperatures.gather(1, bin_ids) + c.gather(1, bin_ids)

    @staticmethod  
    def get_constants_with_grad(temperatures: torch.Tensor,  # (B,P)
                                thresholds: torch.Tensor,    # (B, P-1)
                                ):
        differences = torch.diff(torch.nn.functional.pad(thresholds, (1, 0), value=0.0), dim=-1)
        return torch.cumsum(differences * temperatures[:, :-1], dim=-1)
    
    @staticmethod
    def apply_temperature_with_grad(
                logits: torch.Tensor, # (B, V)
                temperatures: torch.Tensor, # (B, P)
                thresholds: torch.Tensor, # (B, P-1)
                bin_ids: torch.Tensor, # (B, V)
                ) -> torch.Tensor: # (B, V)
        B  = logits.size(0)
        c = PiecewiseTemperatureFunction.get_constants_with_grad(temperatures, thresholds) # (B, P-1)
        thresholds = torch.cat([thresholds.new_zeros(B,1), thresholds], dim=-1) # (B, P-1) -> (B, P)
        c = torch.cat([c.new_zeros(B,1), c], dim=-1) # (B, P-1) -> (B,V,P)
        return temperatures.gather(-1, bin_ids) * (logits - thresholds.gather(-1,bin_ids)) + c.gather(-1,bin_ids)
    
class BaseTemp(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_features = []
        self.n_temp = 1
    def set_context(self, context: torch.Tensor):
        pass
    def clear_cache(self):
        pass
    def get_logs(self):
        return {}

    
class TemperatureScaler(BaseTemp):
    """Trainable temperature scaling module."""
    def __init__(self,
                mode: str = "absolute", # "rank", "absolute", "relative_shifts", "absolute_shifts", "none" for single piece
                apply_to_full_probs: bool = False,
                bin_cutoffs: Optional[List[int]] = None,
                n_temp: int = 1,
                pieces: Union[int, List[int]] = 1,
                normalize_first: bool = False,
                thresholds: Optional[List[torch.Tensor]] = None, # (pieces - 1,)
                temp: Optional[List[torch.Tensor]] = None, # (pieces,)
                counts: Optional[List[torch.Tensor]] = None, # (pieces - 1,) or (pieces,) if summing up to vocab size
                ratios: Optional[List[torch.Tensor]] = None, # (pieces,)
                shifts: Optional[List[torch.Tensor]] = None, # (pieces-1,)
                shifts_topk: int = 1,
                init_temp: str = "ones", # "ones" or "random", or "plif"
                init_thresholds: str = "linspace", # (pieces - 1,)
                threshold_min: float = -30.0,
                threshold_max: float = 30.0,
                temp_mask_threshold: Optional[float] = None,
                temp_mask: bool = False,
                context_dependent: bool = False,
                logit_dependent: bool = False,
                logit_features: List[str] = ["max", "max_10", "max_100",  "max_1000", "median", "min", "ent", "cluster_logits"],
                context_dim: int = 1024,
                n_layer: int = 2,
                d_inner: int = 256,
                dropout: float = 0.1,
                fixed_thresholds: bool = False,
                softplus_shifts: bool = False,
                softplus_temp: bool = False,
                linear_component: bool = False,
                degree: int = 1,
                divide_by_temp: bool = False,
                after_tail_probs: bool = False,
                ):

        super().__init__()
        self.fixed_linspace = init_thresholds == "linspace" and mode == "absolute" and fixed_thresholds
        self.mode = mode
        self.degree = degree
        self.divide_by_temp = divide_by_temp
        self.n_temp = n_temp
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.apply_to_full_probs = apply_to_full_probs # Used if we want to apply to full logits
        self.after_tail_probs = after_tail_probs
        self.bin_cutoffs = bin_cutoffs
        self.normalize_first = normalize_first
        self.temp_mask_threshold = temp_mask_threshold
        self.temp_mask = temp_mask
        self.use_no_grad_version = False
        self.context_dependent = context_dependent
        self.logit_dependent = logit_dependent
        self.logit_features = logit_features
        self.use_cluster_logits = "cluster_logits" in logit_features
        self._cached_temps = None
        self._cached_cluster_logits = None
        self.shifts_topk = shifts_topk
        self.softplus_shifts = softplus_shifts
        self.softplus_temp = softplus_temp
        self.sample_batch = [None] * n_temp
        if isinstance(pieces, int):
            pieces = [pieces] * n_temp
        elif len(pieces) != n_temp:
            raise ValueError(f"Expected len(pieces) == n_temp ({n_temp}), got {len(pieces)}.")
        self.pieces = pieces
        if counts is not None:
            self.register_buffer("counts", counts)  
        else:
            self.counts = None

        self.mlps = nn.ModuleList()
        self.temperature = nn.ParameterList()
        self.thresholds = nn.ParameterList()
        self.shifts = nn.ParameterList()
        self.ratios = nn.ParameterList()
        if self.degree > 1:
            self.coeffs = nn.ParameterList()

        # Initialize parameters and optional MLPs
        for j, p in enumerate(pieces):
            # Initial Temp Values
            if temp is not None:
                temp_init = temp[j]
            elif init_temp == "ones":
                temp_init = torch.ones(p, dtype=torch.float32)
            
            elif init_temp == "random":
                temp_init = torch.rand(p, dtype=torch.float32) *0.5 + 1.0

            elif init_temp == "plif":
                # Follow Ganea
                assert self.softplus_temp, "PLIF initialization only makes sense if using softplus temperature"
                temp_init =torch.randn(p, dtype=torch.float32)  * 1.0 + math.log(math.exp(1) - 1)

            else:
                raise ValueError(f"Unknown init_temp: {init_temp} and temp is None.")
            if self.softplus_temp and init_temp != "plif":
                temp_init = inv_softplus(temp_init)

            # Initial Threshold/Ratio/Shift Values (only if p > 1)
            if p > 1:
                if mode == "absolute":
                    if thresholds is not None:
                        init = thresholds[j]
                    elif init_thresholds == "linspace":
                        init = torch.linspace(threshold_min, threshold_max, p + 1)[1:-1]  # Evenly space between min and max, excluding endpoints
                    elif init_thresholds == "random":
                        init = (threshold_max - threshold_min) * torch.rand(p - 1) + threshold_min
                    else:
                        raise ValueError(f"Unknown init_thresholds: {init_thresholds}")
                elif mode == "absolute_shifts":
                    if shifts is not None:
                        init = shifts[j]
                    else:
                        init = torch.ones(p - 1) * (threshold_max - threshold_min) / p
                elif mode == "relative_shifts":
                    if ratios is not None:
                        init = ratios[j]
                    else:
                        init = torch.ones(p) / p
                else:
                    init = None
            else:
                init = None  # No thresholds for single piece

            if self.context_dependent or self.logit_dependent:
                d_in = len(self.logit_features) if self.logit_dependent else context_dim
                # cluster_logits feature adds 1 input for all clusters (head gets head prob, tails get their tail probs)
                # No additional input needed since it's already counted in len(self.logit_features)
                # For p==1, d_out is just 1 (only temperature, no thresholds)
                if p == 1:
                    d_out = 1
                else:
                    d_out = 2 * p if mode == "relative_shifts" else 2 * p - 1
                mlp = MLP(d_in, d_out, d_inner, n_layer, dropout)
                mlp.out_proj.weight.data.zero_()
                if init is not None:
                    mlp.out_proj.bias.data.copy_(torch.cat([temp_init, init], dim=0))
                else:
                    mlp.out_proj.bias.data.copy_(temp_init)
                self.mlps.append(mlp)
                self.temperature.append(None)
                self.thresholds.append(None)
                self.shifts.append(None)
                self.ratios.append(None)
            else:
                self.temperature.append(nn.Parameter(temp_init) if p > 0 else None)
                param = nn.Parameter(init) if init is not None else None
                if fixed_thresholds and param is not None:
                    param.requires_grad = False
                self.thresholds.append(param if mode == "absolute" else None)
                self.shifts.append(param if mode == "absolute_shifts" else None)
                self.ratios.append(param if mode == "relative_shifts" else None)
                # Initialize higher order coefficients if degree > 1
                if self.degree > 1:
                    self.coeffs.append(nn.Parameter(torch.zeros(p, self.degree - 1)))  # (P, degree-1)

        self._register_legacy_hook()

    def mask_temp_and_threshold(self, thresholds, temps):
        if self.temp_mask_threshold is None or not self.temp_mask:
            return thresholds, temps
        temp_diff = torch.diff(temps, dim=-1).abs()
        temp_mask = temp_diff > self.temp_mask_threshold
        if temp_mask.sum() == 0: #Return at least one threshold and two temps
            temp_mask[-1] = True  
        return thresholds[temp_mask], temps[nn.functional.pad(temp_mask, (1,0), value=True)]

    def forward(self,
                logits: torch.Tensor,
                i: int = 0,
                return_thresholds: bool = False,
                row_indices: Optional[torch.Tensor] = None,
                plotting: bool = False,):
        
        thresholds = None

        if self.normalize_first and not plotting:
            logits = torch.log_softmax(logits, dim=-1)
        if self.after_tail_probs and i > 0 and not plotting:
            assert getattr(self, "_cached_cluster_logits") is not None, "Cluster logits must be set for after_tail_probs=True"
            tail_probs = self._cached_cluster_logits[:, i] if row_indices is None else self._cached_cluster_logits[row_indices, i] # (B,)
            logits = logits + tail_probs.unsqueeze(-1)
        if self.context_dependent:
            assert self._cached_temps is not None and self._cached_thresholds is not None
            temp = (self._cached_temps[i] if row_indices is None else self._cached_temps[i][row_indices])
            mlp_thr = (self._cached_thresholds[i] if row_indices is None else self._cached_thresholds[i][row_indices])
        
        elif self.logit_dependent:
            logit_features = self.extract_logit_features(logits, i, row_indices) # (B, num_features)
            out = self.mlps[i](logit_features) # (B, d_out)
            if self.pieces[i] == 1:
                # For single piece, output is just temperature
                temp = out  # (B, 1)
                mlp_thr = None
            else:
                # For multiple pieces, split into temp and thresholds
                temp = out[:, : self.pieces[i]] # (B, pieces)
                mlp_thr = out[:, self.pieces[i]:] # (B, pieces - 1) or # (B, pieces)

        # Cache this batch for plotting
        if not plotting: 
            self.sample_batch[i] = logits.clone().detach()

        if self.temperature[i] is not None:
            temp = self.temperature[i]
            if self.softplus_temp:
                temp = F.softplus(temp)
        
        if self.pieces[i] == 0:
            # No temperature scaling
            return logits
        
        elif self.pieces[i] == 1:
            # For single-piece, apply simple temperature scaling
            if temp.dim() == 2:
                # Batch-dependent temperature from MLP: (B, 1)
                return logits * temp.squeeze(-1).unsqueeze(-1) if not self.divide_by_temp else logits / temp.squeeze(-1).unsqueeze(-1)
            else:
                # Fixed temperature parameter: (1,)
                if self.divide_by_temp:
                    if self.degree > 1: 
                        logits_poly = torch.pow(logits.unsqueeze(-1), torch.arange(1, self.degree, device=logits.device).float()).clamp(max = 1e3, min=-1e3) # (B, V, degree-1)
                        divide_temp = temp + (logits_poly * self.coeffs[i]).sum(dim=-1, keepdims=True)
                    else:
                        divide_temp = temp
                    out = (logits.unsqueeze(-1) / divide_temp).squeeze(-1)
                else:
                    out = logits * temp
                    if self.degree > 1:
                        logits_poly = torch.pow(logits.unsqueeze(-1), torch.arange(2, self.degree+1, device=logits.device).float()).clamp(max = 1e3, min=-1e3)  # (B, V, degree-1)
                        out += (logits_poly * self.coeffs[i]).sum(dim=-1)
                if return_thresholds:
                    return out, None
                return out
        else:
            # Get thresholds
            if self.mode == "absolute":
                thresholds = self.thresholds[i] if self.thresholds[i] is not None else mlp_thr
                if self.fixed_linspace:
                    start = thresholds[0]
                    step = thresholds[1] - thresholds[0]
                    max_id = len(thresholds)
                    # 2. Calculate bins mathematically instead of searching
                    # torch.ceil mimics the default behavior of torch.searchsorted(right=False)
                    bin_ids = torch.ceil((logits - start) / step).long()
                    
                    # 3. Clamp out-of-bounds values to match searchsorted limits
                    bin_ids = torch.clamp(bin_ids, min=0, max=max_id)
                else:
                    thresholds, _ = torch.sort(thresholds)
                    bin_ids = torch.searchsorted(thresholds, logits)
    
            elif self.mode == "absolute_shifts":
                shifts = self.shifts[i] if self.shifts[i] is not None else mlp_thr
                shifts = F.softplus(shifts) if self.softplus_shifts else shifts
                thresholds = PiecewiseTemperatureFunction.absolute_shifts(logits, shifts, self.shifts_topk)
                bin_ids = torch.searchsorted(thresholds, logits)
            elif self.mode == "relative_shifts":
                ratios = self.ratios[i] if self.ratios[i] is not None else mlp_thr
                thresholds = PiecewiseTemperatureFunction.relative_shifts(logits, ratios)
                bin_ids = torch.searchsorted(thresholds, logits)
            
            logits = PiecewiseTemperatureFunction.apply_temperature(
                logits, temp, thresholds, bin_ids,
                use_no_grad_version=self.use_no_grad_version
            )
            # if logits.isnan().any():
            #     raise ValueError("NaN encountered in temperature scaling.")
            return (logits, thresholds) if return_thresholds else logits
                
    def set_context(self,
                    context: torch.Tensor # (B,d)
                    ):
        """Compute and cache temperatures for this context batch."""
        if not self.context_dependent:
            return
        self._cached_temps = []
        self._cached_thresholds = []
        for i, p in enumerate(self.pieces):
            out = self.mlps[i](context) # (B, 2*p - 1) or (B, 2*p)
            temps = out[:,:p]
            if self.softplus_temp:
                temps = F.softplus(temps) # (B, p)
            thresholds = out[:,p:] # (B, p-1) or (B, p)
            self._cached_temps.append(temps)
            self._cached_thresholds.append(thresholds)
    
    def set_cluster_logits(self, cluster_logits: torch.Tensor):
        """Cache cluster logits (normalized) from head output for use in all cluster MLPs.

        Args:
            cluster_logits: (B, n_clusters + 1) normalized cluster logits where:
                - cluster_logits[:, 0] is the head cluster log probability
                - cluster_logits[:, 1:] are the tail cluster log probabilities
        """
        self._cached_cluster_logits = cluster_logits

    def clear_cache(self):
        """Clear cached temperatures and cluster logits."""
        self._cached_temps = None
        self._cached_thresholds = None
        self._cached_cluster_logits = None
    
    def extract_logit_features(self, logits: torch.Tensor, cluster_idx: int = 0, row_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract features from logits for logit-dependent temperature/threshold MLPs.

        Args:
            logits: (B, V) logits for this cluster
            cluster_idx: Index of the cluster (0 for head, 1+ for tail clusters)
            row_indices: Optional row indices if processing a subset of the batch
        """
        features = []
        sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)

        if "max" in self.logit_features:
            features.append(sorted_logits[:,0])

        topks = [int(n.split("_")[-1]) for n in self.logit_features if "max_" in n]
        if len(topks) > 0:
            features.append(sorted_logits[:, topks])

        if "median" in self.logit_features:
            features.append(sorted_logits[:, sorted_logits.size(-1)//2])

        if "min" in self.logit_features:
            features.append(sorted_logits[:,-1])

        if "ent" in self.logit_features:
            p = F.log_softmax(logits, dim=-1)
            ent = -(p * p.exp()).sum(dim=-1)
            features.append(ent)

        if "cluster_logits" in self.logit_features:
            # Add the appropriate cluster probability for this cluster
            if self._cached_cluster_logits is None:
                raise RuntimeError(f"cluster_logits feature requested but cluster logits not cached. "
                                 f"Make sure set_cluster_logits() is called before forward().")
            if cluster_idx == 0:
                # Head cluster: use the head cluster probability (index 0)
                cluster_logit = self._cached_cluster_logits[:, 0:1]  # (B, 1)
            else:
                # Tail cluster: use this tail cluster's probability (index cluster_idx)
                cluster_logit = self._cached_cluster_logits[:, cluster_idx:cluster_idx+1]  # (B, 1)

            if row_indices is not None:
                cluster_logit = cluster_logit[row_indices]
            features.append(cluster_logit)  # (B, 1)

        return torch.cat([x.unsqueeze(1) if x.dim()==1 else x for x in features], dim=-1)  # (B, num_features)


    @torch.no_grad()
    def get_logs(self):
        """Return dictionary of scalar and line plots for visualization (e.g., WandB)."""
        if self.context_dependent or self.logit_dependent:
            return {}

        import wandb
        out = {}

        for i in range(self.n_temp):
            p = self.pieces[i]
            xmax = getattr(self, "threshold_max", 30.0)
            xmin = getattr(self, "threshold_min", -30.0)
            if xmax < 0.0: # If max threshold is negative, we are probably applying to log probabilities, so set xmax to 0 for better visualization
                xmax = 0.0

            # Over ride with sample batch inputs if available and are wider than thresholds
            if self.sample_batch[i] is not None:
                batch_min = self.sample_batch[i].min().item()
                batch_max = self.sample_batch[i].max().item()
                xmin = min(xmin, batch_min - 5)
                xmax = max(xmax, batch_max + 5)
                
            # --- Generate function plot ---
            xs = torch.linspace(xmin, xmax, 500).to(self.temperature[i].device)
            y, thresholds_out = self.forward(xs.unsqueeze(0), i, return_thresholds=True, plotting=True) 
            y = y.squeeze()
            thresholds_out = thresholds_out.squeeze(0) if thresholds_out is not None and thresholds_out.dim() == 2 else thresholds_out

            # Shift y so that f(x_max) = x_max for better visualization
            # y = y - y[-1] + xs[-1]

            # --- Line plot ---
            xs_list = xs.squeeze().cpu().tolist()
            y_list = y.squeeze().cpu().tolist()
            out[f"group_{i}_plot"] = wandb.plot.line_series(
                xs=xs_list,
                ys=[y_list],
                keys=["f(x)"],
                title=f"Temperature {i} Plot",
                xname="x"
            )
            # --- Scatter Plot of Temperatures --- #
            if thresholds_out is not None:
                bin_ids = torch.searchsorted(thresholds_out, xs)
                if hasattr(self, "temperature") and self.temperature[i] is not None:
                    temp_y = self.temperature[i][bin_ids]
                    if self.softplus_temp:
                        temp_y = F.softplus(temp_y)
                out[f"group_{i}_temp_plot"] = wandb.plot.line_series(
                    xs=xs_list,
                    ys=[temp_y.squeeze().cpu().tolist()],
                    keys=["temperature"],
                    title=f"Temperature {i} Bar Plot",
                    xname="x"
                )
            
            # # Histogram of logits in intervals
            # if self.sample_batch[i] is not None:
            #     sample_logits = self.sample_batch[i]
            #     bin_ids = torch.searchsorted(thresholds_out, sample_logits)
            #     hist_counts = torch.bincount(bin_ids.flatten().cpu(), minlength=p+1)
            #     out[f"group_{i}_hist"] = wandb.plot_table(
            #     vega_spec_name="ucsd-alon/logit_distribution",
            #     data_table=wandb.Table(data=[[j, count.item()] for j, count in enumerate(hist_counts)], columns=["bin", "count"]),
            #     fields={
            #     "x": "bin",
            #     "y": "count",
            #     "title": f"Group {i} Logit Distribution"
            # }
            # )

            # Histogram of logits
            if self.sample_batch[i] is not None:
                n_samples = 10000
                flat_logits = self.sample_batch[i].view(-1)
                random_indices = torch.randperm(len(flat_logits))[:n_samples]
                sample_batch = flat_logits[random_indices]
                out[f"group_{i}_logit_hist"] = wandb.plot.histogram(
                    table= wandb.Table(data=sample_batch.unsqueeze(1).cpu().tolist(), columns=["logit"]),
                    value='logit',
                    title=f"Group {i} Logit Distribution"
                )
                out[f"group_{i}_logits_max"] = self.sample_batch[i].max().item()
                out[f"group_{i}_logits_min"] = self.sample_batch[i].min().item()
                out[f"group_{i}_logits_mean"] = self.sample_batch[i].mean().item()
                out[f"group_{i}_logits_std"] = self.sample_batch[i].std().item()
            
            # --- Log scalar values if not more than 50 pieces---
            if p > 50:
                continue
            
            if thresholds_out is not None:
                out.update({
                    f"group_{i}_threshold_{j}": t
                    for j, t in enumerate(thresholds_out.cpu().tolist())
                })

            for name in ["temperature", "coeffs", "shifts", "ratios"]:
                if name in ["shifts", "ratios"] and p in [0,1]:
                    continue  # No shifts or ratios for single piece
                param_list = getattr(self, name, None)
                if param_list is not None and len(param_list) > i and param_list[i] is not None:
                    values = param_list[i].detach()
                    if name == "temperature" and self.softplus_temp:
                        values = F.softplus(values)
                    values = values.cpu().tolist()
                    if param_list[i].dim() == 1: 
                        out.update({
                            f"group_{i}_{name}_{j}": v
                            for j, v in enumerate(values)
                        })
                        # data = [[i, val] for i, val in enumerate(values)]
                        # table = wandb.Table(data=data, columns=["index", name])
                        # wandb.log({f"group_{i}_{name}_bar_plot": wandb.plot.bar(table, name, "index", title=f"{name} {i} Bar Plot")})
                    else:
                        for j, val in enumerate(values):
                            out.update({
                                f"group_{i}_{name}_{j}_{k}": v
                                for k, v in enumerate(val)
                            })
                            # data = [[i, v] for i, v in enumerate(val)]
                            # table = wandb.Table(data=data, columns=["index", f"{name}_{j}"])
                            # wandb.log({f"group_{i}_{name}_{j}_bar_plot": wandb.plot.bar(table, f"{name}_{j}", "index", title=f"{name} {i} {j} Bar Plot")})
        return out
    
    def _register_legacy_hook(self):

        """Registers a hook to migrate old checkpoints on the fly."""
        def hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            # 1. Detection: Check if the un-indexed 'temperature' key exists
            # New code expects 'temperature.0', 'temperature.1', etc.
            key_temp = prefix + "temperature"
            
            if key_temp in state_dict:
                print(f"Detected legacy checkpoint at {prefix}. Migrating...")
                self.softplus_temp = False  # Old checkpoints used direct temperature values
                # 2. Extract and Transform Temperature
                # Old shape: (n_temp, pieces)
                # New shape: ParameterList where each item is (pieces,)
                old_temp_tensor = state_dict.pop(key_temp)
                
        
                # Split and map to ParameterList keys
                for i in range(old_temp_tensor.shape[0]):
                    state_dict[f"{prefix}temperature.{i}"] = old_temp_tensor[i]

                # 4. Extract and Split Thresholds/Shifts/Ratios
                # These do not typically require inv_softplus unless softplus_shifts was used
                for param_name in ["thresholds", "shifts", "ratios"]:
                    key_param = prefix + param_name
                    if key_param in state_dict:
                        old_tensor = state_dict.pop(key_param)
                        for i in range(old_tensor.shape[0]):
                            state_dict[f"{prefix}{param_name}.{i}"] = old_tensor[i]

        self._register_load_state_dict_pre_hook(hook)

class VectorScaler(BaseTemp):
    def __init__(self, vocab_size: Optional[int] = None, init_scale: float = 1.0, bias: bool = True, cutoffs: Optional[List[int]] = None):
        super().__init__()
        self.cutoffs = cutoffs
        n_bins = len(cutoffs) if cutoffs is not None else vocab_size
        self.scale = nn.Parameter(torch.ones(n_bins) * init_scale)
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_bins))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.cutoffs is not None:
            # Determine bin indices based on cutoffs
            bin_ids = torch.searchsorted(
                torch.tensor(self.cutoffs, device=logits.device),
                torch.arange(logits.size(-1), device=logits.device)
            )  # (V,)
            return logits * self.scale[bin_ids] + (self.bias[bin_ids] if hasattr(self, "bias") else 0.0)
        return logits * self.scale + (self.bias if hasattr(self, "bias") else 0.0)
    
    def get_logs(self):
            import wandb
            out = {}
            xs_list = list(range(len(self.scale)))
            scale_list = self.scale.detach().squeeze().cpu().tolist()
            bias_list = self.bias.detach().squeeze().cpu().tolist() if hasattr(self, "bias") else None
            out[f"scale"] = wandb.plot.line_series(
                xs=xs_list,
                ys=[scale_list],
                title=f"Scale Parameters",
                xname="bin index"
            )
            if bias_list is not None:
                out[f"bias"] = wandb.plot.line_series(
                    xs=xs_list,
                    ys=[bias_list],
                    title=f"Bias Parameters",
                    xname="bin index"
                )
            return out

class CachedProbsModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temp_scalers = nn.ModuleList([])
        for _ in range(len(config.lm_prob_paths)):
            temp_scaler = TemperatureScaler(
                    mode=config.mode,
                    n_temp=1,
                    pieces=config.lm_pieces,
                    normalize_first=config.normalize_first,
                    thresholds= torch.tensor(config.lm_thresholds) if config.lm_thresholds else None,
                    counts=torch.tensor(config.counts) if config.counts else None,
                    ratios=torch.tensor(config.ratios) if config.ratios else None,
                    shifts=torch.tensor(config.shifts) if config.shifts else None,
                    shifts_topk=config.shifts_topk,
                    softplus_shifts=config.softplus_shifts,
                    init_temp=config.init_temp,
                    init_thresholds=config.init_thresholds,
                    threshold_min=config.threshold_min,
                    threshold_max=config.threshold_max,
                    temp_mask_threshold=config.temp_mask_threshold,
                    context_dependent=config.context_dependent,
                    context_dim=config.context_dim,
                    logit_dependent=config.logit_dependent,
                    logit_features=config.logit_features,
                    n_layer=config.n_layer,
                    d_inner=config.d_inner,
                    dropout=config.dropout,
                    fixed_thresholds=config.fixed_thresholds,
                    degree=config.degree,
                    divide_by_temp=config.divide_by_temp,
                )
            self.temp_scalers.append(temp_scaler)

        if len(self.temp_scalers) > 1 and not config.separate_losses and config.optimize_model_weights:
            self.model_weights = nn.Parameter(torch.ones(len(self.temp_scalers)))

    def forward(self, V, targets=None, return_full_prob=False):
        if V.dim() == 2:
            V = V.unsqueeze(1)  # (B, M, V)
        new_logits = V.new_zeros(V.shape, dtype=torch.float32)
        for i in range(V.size(1)):
            new_logits[:, i, :] = self.temp_scalers[i](V[:, i, :])
        new_logits = torch.log_softmax(new_logits, dim=-1)
        target_ps = new_logits.gather(-1, targets.unsqueeze(1).unsqueeze(2).expand(-1,new_logits.size(1),-1)).squeeze(-1)
        if return_full_prob:
            return target_ps, new_logits # (B, M, V)
        return target_ps  # (B, M) or (B, M+1)
    
    def load_state_dict(self, state_dict, strict = True, assign = False):
        if 'model' in state_dict:
            state_dict = state_dict['model']
        return super().load_state_dict(state_dict, strict, assign)
    