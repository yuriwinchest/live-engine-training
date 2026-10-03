"""Laço de treino CTC: AdamW, aquecimento + cosseno, early stopping e retomada (research R5 da 002).

A métrica de parada é a do operador: acerto do número inteiro na validação, decodificando pela gramática
(sem rejeição). Perda de validação só desempata.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from live_lab.decode import CompiledTrie, decide
from live_lab.grammar import Grammar
from live_lab.ingest_real import running_in_cloud
from live_lab.manifest import Example, Split, read_manifest
from live_lab.metrics import Outcome, rates
from live_lab.model.network import LiveNet, NetConfig
from live_lab.train import checkpoint
from live_lab.train.data import Batch, DurationBatchSampler, ManifestDataset, collate
from live_lab.vocab import VOCAB_VERSION


@dataclass(frozen=True, slots=True)
class TrainConfig:
    manifest: Path
    out_dir: Path
    seed: int = 7
    epochs: int = 80
    patience: int = 8
    batch_seconds: float = 240.0
    lr: float = 1e-3
    weight_decay: float = 1e-3
    warmup_steps: int = 500
    max_seconds: float = 8.0
    val_limit: int = 1500
    workers: int = 2
    init_from: Path | None = None
    net: NetConfig = field(default_factory=NetConfig)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}


@dataclass(frozen=True, slots=True)
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_wrong_rate: float
    lr: float


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _schedule(warmup: int, total: int) -> Any:
    def factor(step: int) -> float:
        rise = min(1.0, (step + 1) / max(1, warmup))
        return rise * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, total))))

    return factor


def _check_manifest(examples: list[Example], vocab_version: str) -> None:
    if vocab_version != VOCAB_VERSION:
        raise ValueError(f"manifesto do vocabulário {vocab_version}, modelo usa {VOCAB_VERSION}")
    if running_in_cloud() and any(e.split is Split.TEST for e in examples):
        raise ValueError("manifesto com partição de teste (voz real) não pode treinar na nuvem (FR-006)")


def _loader(
    dataset: ManifestDataset | Subset[Any], durations: list[float], config: TrainConfig, shuffle: bool
) -> DataLoader[Any]:
    sampler = DurationBatchSampler(durations, config.batch_seconds, config.seed, shuffle)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=config.workers)


def _ctc_loss(
    model: LiveNet, batch: Batch, ctc: nn.CTCLoss, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = model(batch.audio.to(device), batch.samples.to(device))
    frames = model.output_frames(batch.samples).clamp(max=log_probs.shape[1])
    loss = ctc(
        log_probs.transpose(0, 1),
        batch.targets.to(device),
        frames.to(device),
        batch.target_lengths.to(device),
    )
    return loss, log_probs


@torch.no_grad()
def validate(model: LiveNet, loader: DataLoader[Any], examples: list[Example], trie: CompiledTrie,
             ctc: nn.CTCLoss, device: torch.device) -> tuple[float, float, float]:  # fmt: skip
    model.eval()
    losses: list[float] = []
    outcomes: list[Outcome] = []
    for batch in loader:
        loss, log_probs = _ctc_loss(model, batch, ctc, device)
        losses.append(float(loss))
        frames = model.output_frames(batch.samples)
        for row, index in enumerate(batch.indices.tolist()):
            decision = decide(log_probs[row, : int(frames[row])].cpu().numpy(), trie)
            outcomes.append(Outcome(examples[index].number, decision.best))
    result = rates(outcomes)
    return float(np.mean(losses)) if losses else math.inf, result.accuracy, result.wrong_rate


def train(config: TrainConfig) -> list[EpochResult]:
    header, examples = read_manifest(config.manifest)
    _check_manifest(examples, header.vocab_version)
    _seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = config.manifest.parent

    train_set = ManifestDataset(examples, root, Split.TRAIN, config.max_seconds)
    val_full = ManifestDataset(examples, root, Split.VAL, config.max_seconds)
    pick = np.random.default_rng(config.seed).permutation(len(val_full))[: config.val_limit].tolist()
    val_set = Subset(val_full, sorted(pick))
    train_loader = _loader(train_set, [e.duration_s for e in train_set.examples], config, shuffle=True)
    val_loader = _loader(
        val_set, [val_full.examples[i].duration_s for i in val_set.indices], config, shuffle=False
    )
    trie = CompiledTrie(Grammar.build().export_trie())

    model = LiveNet(config.net).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = config.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _schedule(config.warmup_steps, total_steps))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    out = config.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config.as_dict(), indent=2, default=str), encoding="utf-8")
    start, best_accuracy, best_loss, stale = 0, -1.0, math.inf, 0
    history: list[EpochResult] = []
    if (out / checkpoint.LAST).exists():
        state = checkpoint.load(out / checkpoint.LAST)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        checkpoint.restore_rng(state["rng"])
        start, best_accuracy, best_loss, stale = (
            state["epoch"] + 1,
            state["best_accuracy"],
            state["best_loss"],
            state["stale"],
        )
        history = [EpochResult(**item) for item in state["history"]]
        print(f"retomando da época {start}", file=sys.stderr)
    elif config.init_from is not None:
        model.load_state_dict(checkpoint.load(config.init_from)["model"])

    for epoch in range(start, config.epochs):
        if stale >= config.patience:
            break
        train_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            loss, _ = _ctc_loss(model, batch, ctc, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss))
        val_loss, accuracy, wrong = validate(model, val_loader, val_full.examples, trie, ctc, device)
        result = EpochResult(
            epoch, float(np.mean(losses)), val_loss, accuracy, wrong, scheduler.get_last_lr()[0]
        )
        history.append(result)
        improved = accuracy > best_accuracy + 1e-9 or (accuracy == best_accuracy and val_loss < best_loss)
        if improved:
            best_accuracy, best_loss, stale = accuracy, val_loss, 0
        else:
            stale += 1
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch, "best_accuracy": best_accuracy, "best_loss": best_loss, "stale": stale,
            "rng": checkpoint.rng_state(), "history": [asdict(h) for h in history],
            "net": config.net.as_dict(), "vocab_version": VOCAB_VERSION,
        }  # fmt: skip
        checkpoint.save(out / checkpoint.LAST, state)
        if improved:
            checkpoint.save(out / checkpoint.BEST, state)
        print(f"época {epoch}: perda {result.train_loss:.3f} | val perda {val_loss:.3f} "
              f"acerto {accuracy:.1%} errado {wrong:.1%}", file=sys.stderr)  # fmt: skip
    (out / "history.json").write_text(json.dumps([asdict(h) for h in history], indent=2), encoding="utf-8")
    return history
