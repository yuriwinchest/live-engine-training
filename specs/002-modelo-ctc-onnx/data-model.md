# Data Model — 002

## TrainConfig

`manifest`, `out_dir`, `seed`, `epochs` (80), `patience` (8), `batch_seconds` (orçamento de áudio por lote),
`lr` (1e-3), `weight_decay` (1e-3), `warmup_steps` (500), `channels` (144), `blocks` (5), `init_from` (pacote ou
checkpoint opcional, para ajuste fino). Gravado em `config.json` no `out_dir`.

## Checkpoint (`last.pt`, `best.pt`; privados)

`model`, `optimizer`, `scheduler`, `epoch`, `best_metric`, `epochs_without_gain`, `rng` (python, numpy, torch),
`config`, `vocab_version`.

## RejectionThresholds

`min_posterior`, `min_margin`, `min_adherence` (floats). Escolhidos por `calibrate.py`; vão para o pacote.

## Decision (saída do reconhecimento)

`number: int | None`, `posterior`, `margin`, `adherence`, `candidates` (top-3 com escore). `number = None` = rejeição.

## EvalReport

`model_sha256`, `provisional`, `totals` {accuracy, wrong_rate, reject_rate, n}, `by_snr_level`, `by_mode`,
`by_digits`, `errors` [{example_id, expected, got, posterior}], `thresholds`.

## ModelPackage (pasta privada)

Contrato em [contracts/model-package.md](contracts/model-package.md).

## Holdout

`reserved.txt` (uma chave `fonte/caminho-relativo` por linha), `playback/*.wav`, `LEIA-ME.md`.
Regra: reservado ⇔ `int(sha256("holdout-v1|chave")[:8], 16) / 0xFFFFFFFF < 0.10`.
