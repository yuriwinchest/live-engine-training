# Quickstart — 002

## 1. Ruído reservado e faixas para a gravação (local)

```powershell
cd training
uv run python -m live_lab noise holdout --registry data/fontes.toml --out data/playback --seed 1
```

Esperado: `data/playback/reserved.txt`, `data/playback/*.wav` (multidão e rua) e `LEIA-ME.md` com o roteiro da
gravação. Rodar a preparação de dados de novo e conferir que nenhum reservado aparece em `noise_source`.

## 2. Testes

```powershell
uv sync --extra train
uv run pytest
```

## 3. Treino (Colab)

`training/colab/treino.ipynb`: monta o Drive, roda `train` sobre o dataset sintético; se a sessão cair, rodar de
novo continua do `last.pt`.

## 4. Exportar e avaliar (local)

```powershell
uv run python -m live_lab export --checkpoint <best.pt> --manifest data/ds-v1/manifest.jsonl --out data/models/v1
uv run python -m live_lab evaluate --package data/models/v1 --manifest data/ds-v1/manifest.jsonl --split val
uv run python -m live_lab evaluate --package data/models/v1 --manifest data/ds-real/manifest.jsonl --split test --inscritos data/inscritos.txt
```

Conferir SC-001 (tamanho), SC-002 (queda INT8), SC-004/005 (metas) e SC-007 (ruído puro rejeitado).
