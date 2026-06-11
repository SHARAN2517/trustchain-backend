import copy
import numpy as np
import torch

try:
    import flwr as fl
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False

class FlowerMultimodalClient:
    """
    Flower Federated Learning Client for TrustChain-Med AI.
    Integrates secure decentralized parameters extraction and local training.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, privacy_engine=None, client_id="hospital_1"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.privacy_engine = privacy_engine
        self.client_id = client_id
        
        # State tracking
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config=None):
        """Extracts model weights as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Updates local model weights from global parameters."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Trains the model locally with global model weights."""
        self.set_parameters(parameters)
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Train for 1 epoch locally (simulated)
        for images, text_ids in self.train_loader:
            images, text_ids = images.to(self.device), text_ids.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(images, text_ids)
            # Create synthetic target for test labels (multi-label)
            targets = torch.zeros_like(outputs['logits'])
            targets[:, 0] = 1.0  # mock label positive
            
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs['logits'], targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            preds = (outputs['probabilities'] > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.numel()

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = correct / total
        
        # Return local parameters, size of local dataset, and logs
        return self.get_parameters(), len(self.train_loader.dataset), {
            "loss": epoch_loss, 
            "accuracy": epoch_acc, 
            "client_id": self.client_id
        }

    def evaluate(self, parameters, config):
        """Evaluates the global model locally."""
        self.set_parameters(parameters)
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, text_ids in self.val_loader:
                images, text_ids = images.to(self.device), text_ids.to(self.device)
                outputs = self.model(images, text_ids)
                targets = torch.zeros_like(outputs['logits'])
                targets[:, 0] = 1.0
                
                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs['logits'], targets)
                running_loss += loss.item() * images.size(0)
                preds = (outputs['probabilities'] > 0.5).float()
                correct += (preds == targets).sum().item()
                total += targets.numel()
                
        eval_loss = running_loss / len(self.val_loader.dataset)
        eval_acc = correct / total
        
        return eval_loss, len(self.val_loader.dataset), {"accuracy": eval_acc}


class SimulatedFederation:
    """
    Enhanced simulated multi-client federated training with:
      - Byzantine-robust aggregation (Krum, FLTrust, Bulyan, FedAvg)
      - Poisoning detection (composing verdicts → aggregator weights)
      - Per-round metrics tracking
    """
    def __init__(self, central_model, client_loaders, val_loader, lr=0.01,
                 aggregation_method="fedavg", byzantine_f=1):
        self.central_model = central_model
        self.client_loaders = client_loaders
        self.val_loader = val_loader
        self.lr = lr
        self.aggregation_method = aggregation_method
        self.byzantine_f = byzantine_f
        self.round_history = []

        # Import defense modules (optional — graceful if missing)
        try:
            from byzantine import aggregate as byz_aggregate
            self._aggregate = byz_aggregate
        except ImportError:
            self._aggregate = None

        try:
            from poisoning_detector import EnsemblePoisoningDetector
            self._detector = EnsemblePoisoningDetector()
        except ImportError:
            self._detector = None

    def _squared_distance(self, params_a, params_b):
        """Compute squared Euclidean distance between two sets of model parameters."""
        return sum(np.sum((a - b) ** 2) for a, b in zip(params_a, params_b))

    def krum_score(self, candidate_idx, client_parameters, f=1):
        """Assign a Krum score by summing the smallest pairwise distances (legacy helper)."""
        distances = []
        for idx, params in enumerate(client_parameters):
            if idx == candidate_idx:
                continue
            distances.append(self._squared_distance(client_parameters[candidate_idx], params))
        distances.sort()
        return sum(distances[: max(0, len(distances) - f - 2)])

    def aggregate_parameters(self, client_parameters, method=None, **kwargs):
        """
        Aggregates client parameters using the configured strategy.

        Pipeline: poisoning_detector → verdicts → byzantine aggregator.
        """
        method = method or self.aggregation_method

        # Step 1: Run poisoning detection (if available)
        verdicts = {}
        detection_report = {}
        if self._detector is not None:
            full_report = self._detector.detect(client_parameters)
            verdicts = {idx: info["verdict"] for idx, info in full_report.items()}
            detection_report = {idx: info["risk_score"] for idx, info in full_report.items()}
            rejected = [idx for idx, v in verdicts.items() if v == "MALICIOUS"]
            if rejected:
                print(f"  [DEFENSE] Poisoning detected: clients {rejected} marked MALICIOUS")

        # Step 2: Aggregate using byzantine-robust strategy
        if self._aggregate is not None:
            aggregated, meta = self._aggregate(
                method, client_parameters,
                f=self.byzantine_f, verdicts=verdicts, **kwargs,
            )
            meta["poisoning_verdicts"] = verdicts
            meta["risk_scores"] = detection_report
            return aggregated, meta
        else:
            # Fallback to basic FedAvg
            aggregated = []
            n = len(client_parameters)
            for layer_idx in range(len(client_parameters[0])):
                layers = [client_parameters[c][layer_idx] for c in range(n)]
                aggregated.append(np.mean(layers, axis=0))
            return aggregated, {"method": "fedavg_fallback", "poisoning_verdicts": verdicts}

    def run_round(self, round_idx, method=None):
        """Executes a single federated learning round with defense pipeline."""
        method = method or self.aggregation_method
        print(f"\n--- Federated Round {round_idx} (aggregation: {method}) ---")
        client_updates = []
        client_metrics = []

        global_params = [val.cpu().numpy() for _, val in self.central_model.state_dict().items()]

        for i, client_loader in enumerate(self.client_loaders):
            local_model = copy.deepcopy(self.central_model)
            local_optimizer = torch.optim.SGD(local_model.parameters(), lr=self.lr)

            client = FlowerMultimodalClient(
                model=local_model,
                train_loader=client_loader,
                val_loader=self.val_loader,
                optimizer=local_optimizer,
                client_id=f"hospital_{i+1}"
            )

            new_params, data_size, metrics = client.fit(global_params, config={})
            print(f"  Hospital {i+1}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")
            client_updates.append(new_params)
            client_metrics.append(metrics)

        # Byzantine-robust aggregation with poisoning detection
        aggregated_params, agg_meta = self.aggregate_parameters(
            client_updates, method=method,
        )

        # Update central model
        params_dict = zip(self.central_model.state_dict().keys(), aggregated_params)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.central_model.load_state_dict(state_dict, strict=True)

        # Evaluate
        eval_optimizer = torch.optim.SGD(self.central_model.parameters(), lr=self.lr)
        eval_client = FlowerMultimodalClient(
            self.central_model, self.val_loader, self.val_loader, eval_optimizer,
        )
        loss, size, eval_metrics = eval_client.evaluate(aggregated_params, config={})
        print(f"  Global Model: loss={loss:.4f}, acc={eval_metrics['accuracy']:.4f}")
        print(f"  Aggregation: {agg_meta.get('method', method)}")

        round_record = {
            "round": round_idx,
            "method": agg_meta.get("method", method),
            "loss": loss,
            "accuracy": eval_metrics["accuracy"],
            "client_metrics": client_metrics,
            "aggregation_meta": agg_meta,
        }
        self.round_history.append(round_record)
        return loss, eval_metrics["accuracy"], agg_meta


# Standalone integration entry point for Flower server setup
def start_flower_server(server_address="0.0.0.0:8080", rounds=3):
    if FLWR_AVAILABLE:
        print(f"[INFO] Initializing Centralized Flower Federated Server at {server_address}...")
        # Define strategy (FedAvg)
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2
        )
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy
        )
    else:
        print("[WARNING] flwr package not installed. Flower server cannot run in server-mode.")

if __name__ == '__main__':
    # Test local simulated federation loop
    from torch.utils.data import TensorDataset, DataLoader
    from model import TrustChainMedModel
    
    # 2 hospital clients
    h1_dataset = TensorDataset(torch.randn(10, 3, 224, 224), torch.randint(0, 30522, (10, 64)))
    h2_dataset = TensorDataset(torch.randn(10, 3, 224, 224), torch.randint(0, 30522, (10, 64)))
    val_dataset = TensorDataset(torch.randn(5, 3, 224, 224), torch.randint(0, 30522, (5, 64)))
    
    loaders = [DataLoader(h1_dataset, batch_size=2), DataLoader(h2_dataset, batch_size=2)]
    val_loader = DataLoader(val_dataset, batch_size=2)
    
    model = TrustChainMedModel()
    federation = SimulatedFederation(model, loaders, val_loader)
    
    for round_idx in range(1, 3):
        federation.run_round(round_idx)
    print("\nFederated Learning simulation test completed successfully.")
