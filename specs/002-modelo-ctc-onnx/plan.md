# Implementation Plan: Modelo acústico CTC, treino, avaliação e exportação

**Branch**: `002-modelo-ctc-onnx` | **Date**: 2026-10-03 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/002-modelo-ctc-onnx/spec.md`

## Summary

Modelo convolucional 1-D separável em profundidade (família QuartzNet/TC-ResNet) com front-end log-Mel próprio,
implementado com operações que exportam para ONNX (convolução com pesos de DFT fixos). Treino CTC em PyTorch
(Colab com GPU para volume; CPU local para testes). Decodificação por *prefix beam search* restrita à trie da
feature 001, com três sinais de rejeição calibrados na validação. Avaliação executa o **ONNX exportado** via ONNX
Runtime — o mesmo arquivo que vai ao celular. Exportação FP32 → INT8 estático (QDQ), front-end mantido em FP32.
Antes do modelo, o **conjunto reservado de ruído** (US5) para a gravação real do PO.

## Technical Context

**Language/Version**: Python 3.12 (uv), pacote `live_lab` da feature 001.

**Primary Dependencies**: extra `train`: torch (CPU no Windows; CUDA pré-instalado no Colab), onnx, onnxscript,
onnxruntime, psutil. Verificado em 2026-10-03: torch 2.14.1+cpu, onnxruntime 1.30.0, onnx 1.23.1 e onnxscript
0.7.2 carregam nesta máquina apesar do Smart App Control.

**Storage**: checkpoints e pacotes de modelo em `training/data/` (local) ou no Drive (Colab); nunca no Git.

**Testing**: pytest; testes de modelo com dimensões reduzidas em CPU; decodificador e calibração testados com
probabilidades sintéticas, sem modelo.

**Target Platform**: treino no Colab (GPU); avaliação e medição no Windows local (CPU); destino final Android.

**Project Type**: biblioteca + CLI (`python -m live_lab train|evaluate|export|noise`).

**Performance Goals**: modelo INT8 < 5 MB (meta interna ≤ 1 MB); clipe de 3 s em < 100 ms com 1 thread no
notebook [estimativa: ~0,5 M parâmetros × 150 quadros ≈ 75 M multiplicações].

**Constraints**: conv-only (sem GRU/LSTM: quantização INT8 de recorrência é menos madura no ONNX Runtime
[memória]); entrada = áudio bruto 16 kHz; voz real só local.

**Scale/Scope**: 50–200 mil exemplos de treino; ~45 tokens; ≤ 8 s por aperto.

## Constitution Check

| Princípio | Como o plano cumpre | Status |
|---|---|---|
| I. Contrato estreito | Entrada áudio, saída inteiro ou rejeição; inscritos chegam como trie. | ✅ |
| II. Nunca inventar número | Beam search só percorre a trie; sequência vazia compete como candidata; rejeição calibrada priorizando menor taxa de erro (SC-005). | ✅ |
| III. Paridade | Log-Mel dentro do grafo; avaliação roda o ONNX exportado; limiar viaja no pacote. | ✅ |
| IV. Dados e privacidade | Treino em nuvem recusa voz real; avaliação real só local; checkpoints e modelos fora do Git; ruído reservado fora do treino. | ✅ |
| V. Medir | Relatório por nível de ruído, modo e algarismos; teste sintético marcado provisório; latência/memória medidas. | ✅ |
| VI. Engenharia | Módulos < 500 linhas; semente e versões registradas; dependências de treino num extra fixado. | ✅ |

**Re-check pós-design**: mantido.

## Project Structure

### Documentation (this feature)

```text
specs/002-modelo-ctc-onnx/
├── plan.md · research.md · data-model.md · quickstart.md · tasks.md
└── contracts/
    ├── model-package.md
    └── cli.md
```

### Source Code

```text
training/live_lab/
├── holdout.py          # US5: arquivos reservados + faixas de reprodução (multidão)
├── decode.py           # prefix beam search na trie + sinais de rejeição (numpy puro)
├── calibrate.py        # escolha dos limiares na validação
├── metrics.py          # acerto, número errado, rejeição, recortes
├── model/
│   ├── frontend.py     # log-Mel exportável (STFT por convolução)
│   ├── network.py      # encoder separável + cabeça CTC
│   └── specaugment.py  # máscaras de tempo/frequência (só treino)
├── train/
│   ├── data.py         # Dataset/DataLoader com lotes por duração
│   ├── loop.py         # época, CTC, AdamW, cosseno com aquecimento, early stopping
│   └── checkpoint.py   # salvar/retomar
├── export.py           # ONNX FP32 → INT8 QDQ, equivalência, pacote
├── evaluate.py         # avaliação via ONNX Runtime + relatório
├── bench.py            # tamanho, latência 1 thread, memória
└── cli_model.py        # subcomandos train/evaluate/export/noise
training/colab/treino.ipynb
training/tests/ (test_holdout, test_decode, test_calibrate, test_frontend, test_network, test_train_smoke, test_export)
```

**Structure Decision**: estende o pacote `live_lab`; modelo e treino em subpacotes porque dependem do extra
`train`, e o núcleo (gramática, dados, decodificador) segue sem torch.

## Complexity Tracking

Nenhuma violação a justificar.
