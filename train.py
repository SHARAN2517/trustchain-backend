import argparse
import csv
import json
import os
from typing import List, Optional

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from model import TrustChainMedModel
from privacy import TrustChainPrivacyEngine

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


DEFAULT_LABELS = [
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "infiltration",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
]


class ClinicalManifestDataset(Dataset):
    """
    Loads real multimodal clinical training rows from a CSV manifest.

    Required columns:
    - image_path: path to a PNG/JPG image, absolute or relative to --image-root
    - note: clinical note/report text
    - optional age, sex, study_description metadata columns

    Labels can be supplied either as a JSON/list column named "labels" or as one
    binary column per class name passed through --labels.
    """

    def __init__(
        self,
        manifest_path: str,
        image_root: str,
        labels: List[str],
        tokenizer,
        max_length: int = 128,
    ):
        self.manifest_path = manifest_path
        self.image_root = image_root
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        with open(manifest_path, newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))

        if not self.rows:
            raise ValueError(f"{manifest_path} contains no training rows.")
        missing = {"image_path", "note"} - set(self.rows[0])
        if missing:
            raise ValueError(f"{manifest_path} is missing required columns: {sorted(missing)}")

    def __len__(self):
        return len(self.rows)

    def _image_path(self, row):
        raw_path = row["image_path"]
        return raw_path if os.path.isabs(raw_path) else os.path.join(self.image_root, raw_path)

    def _labels(self, row):
        if row.get("labels"):
            values = json.loads(row["labels"])
            if len(values) != len(self.labels):
                raise ValueError("labels JSON length does not match --labels.")
            return torch.tensor(values, dtype=torch.float32)
        return torch.tensor([float(row.get(label, 0.0)) for label in self.labels], dtype=torch.float32)

    def _metadata(self, row):
        return torch.tensor(
            [[
                normalize_age(row.get("age", "")),
                encode_sex(row.get("sex", "")),
                encode_study_description(row.get("study_description", "")),
            ]],
            dtype=torch.float32,
        ).squeeze(0)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(self._image_path(row)).convert("RGB")
        tokenized = self.tokenizer(
            row["note"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return self.transform(image), tokenized["input_ids"].squeeze(0), self._metadata(row), self._labels(row)


class SmokeTestDataset(Dataset):
    """Synthetic plumbing test only. Never use this to produce deployable weights."""

    def __init__(self, size: int, num_classes: int, max_length: int):
        self.size = size
        self.num_classes = num_classes
        self.max_length = max_length

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (
            torch.randn(3, 224, 224),
            torch.randint(0, 30522, (self.max_length,)),
            torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32),
            torch.randint(0, 2, (self.num_classes,)).float(),
        )


class TinySmokeModel(torch.nn.Module):
    """Fast model for smoke-testing the training loop only."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.image_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        self.text_embedding = torch.nn.Embedding(30522, 8)
        self.classifier = torch.nn.Linear(16, num_classes)

    def forward(self, images, clinical_text_ids, metadata_features=None):
        image_features = self.image_encoder(images)
        text_features = self.text_embedding(clinical_text_ids).mean(dim=1)
        if metadata_features is None:
            metadata_features = torch.zeros((images.shape[0], 3), device=images.device)
        metadata_features = metadata_features.to(images.device, dtype=image_features.dtype)
        metadata_features = torch.nn.functional.pad(metadata_features, (0, 5))
        text_features = text_features + metadata_features
        logits = self.classifier(torch.cat([image_features, text_features], dim=1))
        return {"logits": logits, "probabilities": torch.sigmoid(logits)}


def normalize_age(age_value) -> float:
    text = str(age_value or "").strip().upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return 0.0
    age = float(digits)
    if text.endswith("M"):
        age = age / 12.0
    elif text.endswith("W"):
        age = age / 52.0
    elif text.endswith("D"):
        age = age / 365.0
    return max(0.0, min(age / 100.0, 1.2))


def encode_sex(sex_value) -> float:
    sex = str(sex_value or "").strip().lower()
    if sex in {"m", "male"}:
        return 1.0
    if sex in {"f", "female"}:
        return -1.0
    return 0.0


def encode_study_description(description: str) -> float:
    if not description:
        return 0.0
    import hashlib
    digest = hashlib.sha256(description.lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def build_tokenizer():
    if AutoTokenizer is None:
        raise RuntimeError(
            "Training requires transformers. Install requirements.txt so "
            "AutoTokenizer can load emilyalsentzer/Bio_ClinicalBERT."
        )
    return AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_items = 0
    correct = 0
    total_labels = 0
    with torch.no_grad():
        for images, text_ids, metadata, labels in loader:
            images = images.to(device)
            text_ids = text_ids.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
            outputs = model(images, text_ids, metadata)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["logits"], labels)
            total_loss += loss.item() * images.size(0)
            total_items += images.size(0)
            correct += ((outputs["probabilities"] > 0.5).float() == labels).sum().item()
            total_labels += labels.numel()
    return total_loss / max(total_items, 1), correct / max(total_labels, 1)


def train(args):
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        raise ValueError("--labels must contain at least one class name.")

    if args.smoke_test:
        if args.save_model:
            raise ValueError("--smoke-test cannot be combined with --save-model.")
        tokenizer = None
        dataset = SmokeTestDataset(size=8, num_classes=len(labels), max_length=args.max_length)
        min_samples = 1
        print("[SMOKE TEST] Using synthetic tensors for code-path validation only.")
    else:
        tokenizer = build_tokenizer()
        dataset = ClinicalManifestDataset(args.manifest, args.image_root, labels, tokenizer, args.max_length)
        min_samples = args.min_samples

    if len(dataset) < min_samples:
        raise ValueError(
            f"Refusing to train on {len(dataset)} samples. Provide at least {min_samples} real samples "
            "or use --smoke-test for a non-deployable plumbing check."
        )

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        TinySmokeModel(num_classes=len(labels))
        if args.smoke_test
        else TrustChainMedModel(num_classes=len(labels), embed_dim=768)
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dp_engine: Optional[TrustChainPrivacyEngine] = None
    if args.dp:
        if len(dataset) < args.min_dp_samples:
            raise ValueError(
                f"DP training needs at least {args.min_dp_samples} samples; got {len(dataset)}. "
                "Train without --dp first, then enable DP on a sufficiently large dataset."
            )
        dp_engine = TrustChainPrivacyEngine()
        model, optimizer, train_loader = dp_engine.make_private(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, text_ids, metadata, batch_labels in train_loader:
            images = images.to(device)
            text_ids = text_ids.to(device)
            metadata = metadata.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, text_ids, metadata)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["logits"], batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        val_loss, val_acc = evaluate(model, val_loader, device)
        dp_msg = ""
        if dp_engine:
            epsilon = dp_engine.get_privacy_spent(
                steps=epoch * len(train_loader),
                batch_size=args.batch_size,
                dataset_size=len(dataset),
                noise_multiplier=args.noise_multiplier,
            )
            dp_msg = f" | DP epsilon={epsilon:.2f}, delta={dp_engine.target_delta}"
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={running_loss / max(seen, 1):.4f} "
            f"val_loss={val_loss:.4f} val_label_acc={val_acc:.4f}{dp_msg}"
        )

    if args.save_model:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        torch.save(model.state_dict(), args.output)
        print(f"Saved trained weights to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train TrustChain-Med on a real clinical manifest.")
    parser.add_argument("--manifest", default="data/train_manifest.csv")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--output", default=os.path.join("models", "trustchain_med_model.pth"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--min-dp-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--dp", action="store_true")
    parser.add_argument("--noise-multiplier", type=float, default=1.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
