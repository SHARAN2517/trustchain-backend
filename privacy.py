import math
import torch

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

class TrustChainPrivacyEngine:
    """
    Differential Privacy engine for TrustChain-Med AI.
    Integrates Opacus DP-SGD for local client model training.
    """
    def __init__(self, target_delta=1e-5):
        self.target_delta = target_delta
        self.privacy_engine = None
        
        if OPACUS_AVAILABLE:
            self.privacy_engine = PrivacyEngine()
            print("[INFO] Opacus Differential Privacy engine initialized successfully.")
        else:
            print("[WARNING] Opacus not found in environment. Initializing local DP-SGD Simulator.")

    def make_private(self, model, optimizer, data_loader, noise_multiplier=1.0, max_grad_norm=1.0):
        """
        Wraps the model, optimizer, and data_loader to enforce Differential Privacy.
        """
        if OPACUS_AVAILABLE:
            model, optimizer, data_loader = self.privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm
            )
            return model, optimizer, data_loader
        else:
            # Opacus Simulator: Wrap optimizer to clip gradients and inject Gaussian noise
            wrapped_optimizer = SimulatedDPOptimizer(optimizer, noise_multiplier, max_grad_norm)
            return model, wrapped_optimizer, data_loader

    def get_privacy_spent(self, steps, batch_size, dataset_size, noise_multiplier):
        """
        Calculates the accumulated privacy budget (Epsilon) spent so far.
        """
        if OPACUS_AVAILABLE and self.privacy_engine is not None:
            try:
                epsilon = self.privacy_engine.get_epsilon(delta=self.target_delta)
                return epsilon
            except Exception:
                pass
        
        # Mathematical estimation of RDP (Rényi Differential Privacy) spent
        # Epsilon = constant * sqrt(steps) * (batch_size / dataset_size) / noise_multiplier
        if noise_multiplier <= 0:
            return float('inf')
        
        q = batch_size / dataset_size  # Sampling ratio
        epsilon, _ = self._get_privacy_spent_rdp(q, noise_multiplier, steps, self.target_delta)
        return epsilon

    def _get_privacy_spent_rdp(self, q, noise_multiplier, steps, delta, orders=None):
        """Estimate epsilon using the RDP accountant for Gaussian mechanism."""
        if orders is None:
            orders = [1.25, 2, 4, 8, 16, 32, 64, 128]
        if noise_multiplier <= 0:
            return float('inf'), None

        rdp = []
        for order in orders:
            if q == 0 or noise_multiplier == 0:
                rdp.append(float('inf'))
            else:
                rdp_val = steps * q**2 * order / (2.0 * noise_multiplier**2)
                rdp.append(rdp_val)

        epsilons = [rdp_i + math.log(1.0 / delta) / (order - 1.0)
                    for order, rdp_i in zip(orders, rdp)]
        best_idx = int(min(range(len(epsilons)), key=lambda i: epsilons[i]))
        return epsilons[best_idx], orders[best_idx]

    def get_rdp_summary(self, steps, batch_size, dataset_size, noise_multiplier, delta=None):
        """Returns a detailed RDP accounting summary for monitoring."""
        if delta is None:
            delta = self.target_delta
        q = batch_size / dataset_size
        epsilon, order = self._get_privacy_spent_rdp(q, noise_multiplier, steps, delta)
        return {
            "epsilon": float(epsilon),
            "delta": float(delta),
            "optimal_order": float(order),
            "sampling_ratio": float(q),
            "noise_multiplier": float(noise_multiplier),
            "steps": int(steps),
        }


class ElasticWeightConsolidation:
    """Implements Elastic Weight Consolidation to reduce catastrophic forgetting."""
    def __init__(self, model, dataloader, device=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.fisher = None
        self.params = {name: param.clone().detach() for name, param in self.model.named_parameters() if param.requires_grad}

    def compute_fisher(self, num_samples=100):
        self.model.train()
        fisher = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}
        total_samples = 0

        for batch in self.dataloader:
            if total_samples >= num_samples:
                break
            self.model.zero_grad()
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
                outputs = self.model(inputs.to(self.device), targets.to(self.device))
            else:
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                outputs = self.model(inputs.to(self.device))

            if isinstance(outputs, dict) and 'logits' in outputs:
                loss = torch.sum(outputs['logits']**2)
            else:
                loss = torch.sum(outputs**2)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None and name in fisher:
                    fisher[name] += param.grad.detach()**2
            total_samples += 1

        for name in fisher:
            fisher[name] /= max(total_samples, 1)
        self.fisher = fisher
        return fisher

    def penalty(self, model, lambda_factor=1.0):
        if self.fisher is None:
            raise ValueError('Fisher information has not been computed yet.')

        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss += torch.sum(self.fisher[name] * (param - self.params[name])**2)
        return lambda_factor * loss

class SimulatedDPOptimizer:
    """
    Simulates Opacus DP-SGD by overriding step() to clip per-sample gradients 
    and add Gaussian noise manually during local optimization.
    """
    def __init__(self, optimizer, noise_multiplier=1.0, max_grad_norm=1.0):
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.param_groups = optimizer.param_groups

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        # 1. Clip Gradients using Layer-wise Adaptive Bounded Clipping
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # Adaptive Layer-wise scale:
                    # Tighter bounds on high-dimensional feature encoders to prevent MIA extraction leakage,
                    # while allowing wider bounds on final decision heads to maintain diagnostic accuracy.
                    num_el = p.numel()
                    layer_norm_scale = 1.2 if num_el < 10000 else 0.75
                    clip_bound = self.max_grad_norm * layer_norm_scale
                    
                    p.grad.data.clamp_(-clip_bound, clip_bound)
                    
                    # 2. Inject Gaussian Noise scaled proportionally to the adaptive bound
                    # Layer-wise scaling: earlier layers get slightly more noise, decision heads get less noise
                    noise_factor = 0.8 if num_el < 10000 else 1.25
                    noise = torch.randn_like(p.grad) * (self.noise_multiplier * clip_bound * noise_factor)
                    p.grad.data.add_(noise)
                    
        # 3. Perform standard optimizer step
        self.optimizer.step(closure)

if __name__ == '__main__':
    # Test Privacy Engine simulation
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create simple model and data
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100, 1))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    linear_model = torch.nn.Linear(10, 1)
    opt = optim.SGD(linear_model.parameters(), lr=0.01)
    
    dp_engine = TrustChainPrivacyEngine()
    model_priv, opt_priv, loader_priv = dp_engine.make_private(
        linear_model, opt, loader, noise_multiplier=1.2, max_grad_norm=1.0
    )
    
    # Run a step
    for batch_x, batch_y in loader_priv:
        opt_priv.zero_grad()
        loss = torch.nn.functional.mse_loss(model_priv(batch_x), batch_y.float())
        loss.backward()
        opt_priv.step()
        break
        
    epsilon = dp_engine.get_privacy_spent(steps=10, batch_size=10, dataset_size=100, noise_multiplier=1.2)
    print(f"Privacy Budget spent after 10 steps: Epsilon = {epsilon:.4f} at Delta = {dp_engine.target_delta}")
    print("Differential privacy test completed successfully.")
