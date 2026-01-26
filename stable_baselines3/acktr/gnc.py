import math
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class GNCOptimizer(optim.Optimizer): 
    """ 
    Generalized Normal Coordinate (GNC) Optimizer. 
    
    Fixed Version based on "Simplifying Momentum-based Positive-definite Submanifold Optimization 
    with Applications to Deep Learning" (Lin et al., ICML 2023). 

    Fixes applied:
    1. Added `stat_decay` for Exponential Moving Average (EMA) of curvature stats (Critical).
    2. Corrected Weight Decay application (Before preconditioning).
    3. Added cold-start guards in hooks.
    """ 

    def __init__( 
        self, 
        model: nn.Module, 
        lr: float = 0.01,              # stepsize_2
        momentum: float = 0.9,         # alpha_2
        
        # GNC Factor Hyperparams
        factor_lr: float = 0.1,        # stepsize_1 (Tip: usually 0.1 ~ 0.5 works better than 0.01)
        factor_momentum: float = 0.9,  # alpha_1
        damping: float = 0.01,         # lambda
        
        # Stats Hyperparams
        stat_decay: float = 0.95,      # [FIX] Added stat decay for EMA
        update_freq: int = 10,         # Training steps between Factor updates
        
        # General
        weight_decay: float = 0, 
        cold_start_steps: int = 100, 
        max_grad_norm: float | None = 0.5, 
    ): 
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay) 
        super().__init__(model.parameters(), defaults) 

        self.model = model
        
        self.factor_lr = factor_lr
        self.factor_momentum = factor_momentum
        self.damping = damping
        self.stat_decay = stat_decay   # [FIX] Store stat decay
        self.update_freq = update_freq
        self.cold_start_steps = cold_start_steps
        self.max_grad_norm = max_grad_norm

        self.steps = 0
        
        # Modules handling
        self.known_modules: dict[nn.Module, str] = {} 
        self.modules: list[nn.Module] = [] 
        self.param_to_module: dict[torch.Tensor, nn.Module] = {} 

        # Stats Buffers (EMA storage)
        self.activations: dict[nn.Module, torch.Tensor | None] = {} 
        self.gradients: dict[nn.Module, torch.Tensor | None] = {} 
        self.m_aa: dict[nn.Module, torch.Tensor | None] = {} 
        self.m_gg: dict[nn.Module, torch.Tensor | None] = {} 

        # GNC Specific Buffers: Factors K, C and their momenta
        self.factors: dict[nn.Module, dict[str, torch.Tensor]] = {} 
        self.factor_momenta: dict[nn.Module, dict[str, torch.Tensor]] = {} 

        self._prepare_model() 

    def _prepare_model(self) -> None: 
        for module in self.model.modules(): 
            classname = module.__class__.__name__ 
            if classname in ["Linear", "Conv2d"]: 
                self.modules.append(module) 
                self.known_modules[module] = classname
                self.param_to_module[module.weight] = module
                if module.bias is not None: 
                    self.param_to_module[module.bias] = module
                
                module.register_forward_pre_hook(self._save_input) 
                module.register_full_backward_hook(self._save_grad_output) 

        for module in self.modules: 
            self.m_aa[module] = None
            self.m_gg[module] = None
            self.activations[module] = None
            self.gradients[module] = None
            
            self.factors[module] = {} 
            self.factor_momenta[module] = {} 

    def _save_input(self, module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None: 
        # [FIX] Don't collect stats during cold start
        if self.steps < self.cold_start_steps:
            return

        if torch.is_grad_enabled() and self.steps % self.update_freq == 0: 
            a = inputs[0].detach() 
            classname = self.known_modules[module] 
            if classname == "Linear" and a.dim() != 2: return
            if classname == "Conv2d" and a.dim() != 4: return
            self.activations[module] = a

    def _save_grad_output( 
        self, module: nn.Module, _grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...] 
    ) -> None: 
        # [FIX] Don't collect stats during cold start
        if self.steps < self.cold_start_steps:
            return

        # Note: self.acc_stats flag logic removed for simplicity, relying on update_freq
        if self.steps % self.update_freq == 0: 
            if not grad_output or grad_output[0] is None: 
                return
            g = grad_output[0].detach() 
            classname = self.known_modules[module] 
            if classname == "Linear" and g.dim() != 2: return
            if classname == "Conv2d" and g.dim() != 4: return
            self.gradients[module] = g

    def _update_stats_and_factors(self) -> None: 
        """ 
        1. Compute raw covariances.
        2. Update Exponential Moving Averages (EMA). [Critical Fix]
        3. Update factors K and C using GNC Matrix-Free method. 
        """ 
        for module in self.modules: 
            if self.activations[module] is None or self.gradients[module] is None: 
                continue

            classname = self.known_modules[module] 
            a = self.activations[module] 
            g = self.gradients[module] 

            # --- 1. Compute Raw Batch Covariances ---
            if classname == "Linear": 
                B = a.size(0) 
                if module.bias is not None: 
                    a = torch.cat([a, a.new_ones(a.size(0), 1)], 1) 
                cov_a = a.t() @ a / B
                g_scaled = g * B
                cov_g = g_scaled.t() @ g_scaled / B

            elif classname == "Conv2d": 
                # Unfold input to patch matrix
                a_unfold = F.unfold(a, module.kernel_size, dilation=module.dilation, 
                                    padding=module.padding, stride=module.stride) 
                B = a_unfold.size(0) 
                T = a_unfold.size(2) 
                a_unfold = a_unfold.permute(0, 2, 1).contiguous().view(-1, a_unfold.size(1)) 
                a_unfold = a_unfold / T
                if module.bias is not None: 
                    a_unfold = torch.cat( 
                        [a_unfold, a_unfold.new_ones(a_unfold.size(0), 1) / T], 
                        dim=1, 
                    ) 
                cov_a = a_unfold.t() @ a_unfold / B

                # Reshape gradient
                g_2d = g.permute(0, 2, 3, 1).contiguous().view(-1, g.size(1)) 
                g_2d = g_2d * T * B
                cov_g = g_2d.t() @ g_2d / (B * T) 
            else:
                continue

            # --- 2. Update EMA (Exponential Moving Average) ---
            # [FIX] This is critical. Using raw cov_a/cov_g makes optimization unstable.
            if self.m_aa[module] is None:
                self.m_aa[module] = cov_a.clone()
            else:
                self.m_aa[module].mul_(self.stat_decay).add_(cov_a, alpha=1 - self.stat_decay)

            if self.m_gg[module] is None:
                self.m_gg[module] = cov_g.clone()
            else:
                self.m_gg[module].mul_(self.stat_decay).add_(cov_g, alpha=1 - self.stat_decay)
            
            # Clear buffers
            self.activations[module] = None
            self.gradients[module] = None

            # --- 3. GNC Factor Update using EMA Stats ---
            
            # Initialize Factors if needed
            if 'K' not in self.factors[module]: 
                p_dim = self.m_aa[module].size(0) 
                d_dim = self.m_gg[module].size(0) 
                device = self.m_aa[module].device
                dtype = self.m_aa[module].dtype
                
                self.factors[module]['K'] = torch.eye(p_dim, device=device, dtype=dtype) 
                self.factors[module]['C'] = torch.eye(d_dim, device=device, dtype=dtype) 
                
                self.factor_momenta[module]['mK'] = torch.zeros_like(self.factors[module]['K']) 
                self.factor_momenta[module]['mC'] = torch.zeros_like(self.factors[module]['C']) 

            K = self.factors[module]['K'] 
            C = self.factors[module]['C'] 
            mK = self.factor_momenta[module]['mK'] 
            mC = self.factor_momenta[module]['mC'] 
            
            # Use EMA stats for calculating H_K, H_C
            m_aa_ema = self.m_aa[module]
            m_gg_ema = self.m_gg[module]

            # Compute Projected Gradients
            H_K = K.t() @ m_aa_ema @ K
            H_C = C.t() @ m_gg_ema @ C
            
            d_in = float(K.size(0))
            d_out = float(C.size(0))
            
            kappa2 = self.damping * torch.trace(K.t() @ K) 
            c2 = self.damping * torch.trace(C.t() @ C) 
            trace_HK = torch.trace(H_K) 
            trace_HC = torch.trace(H_C) 
            
            eye_in = torch.eye(int(d_in), device=K.device, dtype=K.dtype)
            eye_out = torch.eye(int(d_out), device=C.device, dtype=C.dtype)

            # Factors Gradient Update (Paper Figure 5)
            # mK gradient
            term_K = (trace_HC * H_K + c2 * (K.t() @ K) - d_in * eye_in) 
            grad_mK = term_K / (2.0 * d_out) 
            
            # mC gradient
            term_C = (trace_HK * H_C + kappa2 * (C.t() @ C) - d_out * eye_out) 
            grad_mC = term_C / (2.0 * d_in) 
            
            # Update Momentum of Factors
            mK.mul_(self.factor_momentum).add_(grad_mK) 
            mC.mul_(self.factor_momentum).add_(grad_mC) 
            
            # Update Factors (Linear Truncation: K <- K(I - lr * mK))
            # [Note] Applying update using subtraction: K - K @ (mK * lr)
            update_K = K @ (mK * self.factor_lr) 
            K.add_(-update_K) 
            
            update_C = C @ (mC * self.factor_lr) 
            C.add_(-update_C) 

            # Save back state (optional if assignment is by ref, but good for clarity)
            self.factors[module]['K'] = K
            self.factors[module]['C'] = C

    def _get_momentum_buffer(self, param: torch.Tensor) -> torch.Tensor: 
        param_state = self.state[param] 
        if "momentum_buffer" not in param_state: 
            param_state["momentum_buffer"] = torch.zeros_like(param.data) 
        return param_state["momentum_buffer"] 

    def _apply_gnc_preconditioning(self, module: nn.Module, grad: torch.Tensor) -> torch.Tensor: 
        if 'K' not in self.factors[module]: 
            return grad 
            
        K = self.factors[module]['K'] 
        C = self.factors[module]['C'] 
        
        P_in = K @ K.t() 
        P_out = C @ C.t() 
        
        classname = self.known_modules[module] 
        if classname == "Linear": 
            g_mat = grad
        elif classname == "Conv2d": 
            g_mat = grad.view(grad.size(0), -1) 
        else: 
            return grad

        # Safe guard on dimensions
        if g_mat.size(1) != P_in.size(0) or g_mat.size(0) != P_out.size(0):
            return grad

        v_mat = P_out @ g_mat @ P_in
        
        if classname == "Conv2d": 
            v = v_mat.view_as(grad) 
        else: 
            v = v_mat
            
        return v

    def _sgd_step(self) -> None: 
        """Cold start standard SGD.""" 
        if self.max_grad_norm is not None: 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) 

        for group in self.param_groups: 
            weight_decay = group["weight_decay"]
            lr = group["lr"]
            momentum = group["momentum"]
            
            for param in group["params"]: 
                if param.grad is None: continue
                grad = param.grad.data

                # Standard SGD Weight Decay applied to Gradient
                if weight_decay != 0:
                    grad.add_(param.data, alpha=weight_decay)

                v = self._get_momentum_buffer(param) 
                v.mul_(momentum).add_(grad) 
                param.data.add_(v, alpha=-lr) 

    def _gnc_step(self) -> None: 
        """Perform GNC Optimization step.""" 

        if self.max_grad_norm is not None: 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) 
        
        # 1. Update Factors (K, C)
        if self.steps % self.update_freq == 0: 
            self._update_stats_and_factors() 

        # 2. Apply preconditioned gradient to weights
        for group in self.param_groups: 
            weight_decay = group["weight_decay"]
            lr = group["lr"] 
            momentum = group["momentum"] 

            for param in group["params"]: 
                if param.grad is None: continue
                grad = param.grad.data

                # [FIX] Apply Weight Decay BEFORE preconditioning
                # This ensures we are preconditioning the gradient of the regularized objective.
                if weight_decay != 0:
                    grad.add_(param.data, alpha=weight_decay)

                module = self.param_to_module.get(param, None) 
                
                # Handling Preconditioning logic
                preconditioned = False
                w_up = None
                b_up = None

                if module is not None: 
                    # Only process when we encounter the WEIGHT parameter
                    if getattr(module, "weight", None) is param: 
                        
                        # Gather full gradient (W + B)
                        if module.bias is not None: 
                            if module.bias.grad is not None: 
                                bg = module.bias.grad.data.clone()
                                # [FIX] Don't forget weight decay on bias grad if we are merging it
                                if weight_decay != 0:
                                    bg.add_(module.bias.data, alpha=weight_decay)
                            else: 
                                bg = torch.zeros_like(module.bias.data) 

                            if self.known_modules[module] == "Linear": 
                                grad_to_precondition = torch.cat([grad, bg.unsqueeze(1)], 1) 
                            elif self.known_modules[module] == "Conv2d": 
                                g_flat = grad.view(grad.size(0), -1) 
                                grad_to_precondition = torch.cat([g_flat, bg.unsqueeze(1)], 1) 
                        else:
                            # No bias
                            grad_to_precondition = grad

                        # Do Preconditioning
                        pre_grad = self._apply_gnc_preconditioning(module, grad_to_precondition) 
                        
                        # Split gradients back
                        if module.bias is not None:
                            if self.known_modules[module] == "Linear":
                                w_up = pre_grad[:, :-1]
                                b_up = pre_grad[:, -1]
                            else: # Conv2d
                                w_up = pre_grad[:, :-1].view_as(grad)
                                b_up = pre_grad[:, -1]
                        else:
                            # No Bias
                            if self.known_modules[module] == "Linear":
                                w_up = pre_grad
                            else:
                                w_up = pre_grad.view_as(grad)
                            b_up = None
                        
                        preconditioned = True

                    elif getattr(module, "bias", None) is param:
                        # Skip Bias parameter in the main loop, 
                        # because it is updated as a side-effect of the weight update above.
                        continue

                # --- 3. Update Parameters ---
                if preconditioned:
                    # Update Weight (param)
                    v = self._get_momentum_buffer(param) 
                    v.mul_(momentum).add_(w_up) 
                    param.data.add_(v, alpha=-lr) 
                    
                    # Update Bias (Side-effect)
                    if b_up is not None and module.bias is not None:
                        vb = self._get_momentum_buffer(module.bias)
                        vb.mul_(momentum).add_(b_up)
                        module.bias.data.add_(vb, alpha=-lr)
                else:
                    # Fallback for non-supported layers (BN, etc.) or just weights without module mapping
                    # Note: grad already has weight_decay added at the top of loop
                    w_up = grad
                    # Standard SGD update
                    v = self._get_momentum_buffer(param) 
                    v.mul_(momentum).add_(w_up) 
                    param.data.add_(v, alpha=-lr) 

    @torch.no_grad() 
    def step(self, closure: Any = None) -> torch.Tensor | None: 
        loss = None
        if closure is not None: 
            with torch.enable_grad(): 
                loss = closure() 

        if self.steps < self.cold_start_steps: 
            self._sgd_step() 
        else: 
            self._gnc_step() 

        self.steps += 1
        return loss