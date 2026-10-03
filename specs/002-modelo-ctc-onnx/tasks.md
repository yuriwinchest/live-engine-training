---
description: "Tarefas da feature 002 — modelo CTC, treino, avaliação e exportação"
---

# Tasks: Modelo acústico CTC, treino, avaliação e exportação

**Input**: `specs/002-modelo-ctc-onnx/` · **Tests**: pedidos (pytest) · caminhos relativos à raiz.

Ordem de execução: US5 primeiro (destrava a gravação real do PO), depois US2 → US1 → US4 → US3.

## Phase 1: Setup

- [X] T001 Adicionar extra `train` (torch, onnx, onnxscript, onnxruntime, psutil) em training/pyproject.toml e atualizar uv.lock
- [X] T002 [P] Registrar `musan-speech` (kind `distractor`, CC-BY-4.0) em training/sources.example.toml

## Phase 2: User Story 5 - Ruído reservado (P2, executada primeiro) 🎯

**Goal**: faixas de multidão/rua para a gravação do PO, com arquivos fora do treino.
**Independent Test**: preparação não usa nenhum reservado; reserva independe da semente.

- [X] T003 [P] [US5] Testes de reserva, exclusão na preparação e faixas (pico ≤ −1 dBFS, duração) em training/tests/test_holdout.py
- [X] T004 [US5] Implementar reserva determinística e faixas de reprodução em training/live_lab/holdout.py
- [X] T005 [US5] Excluir reservados dos pools de ruído em training/live_lab/prep.py
- [X] T006 [US5] Comando `noise holdout` em training/live_lab/cli_model.py e registro em training/live_lab/cli.py
- [X] T007 [US5] Gerar as faixas reais a partir do MUSAN em training/data/playback/ (fora do Git) e conferir

## Phase 3: User Story 2 - Reconhecer sem inventar (P1)

- [X] T008 [P] [US2] Testes do beam search restrito com probabilidades sintéticas (nunca sai da trie; vazio vence em ruído; inscritos) em training/tests/test_decode.py
- [X] T009 [US2] Implementar decodificador e sinais de rejeição em training/live_lab/decode.py
- [X] T010 [P] [US2] Testes da calibração (prioriza menor taxa de número errado) em training/tests/test_calibrate.py
- [X] T011 [US2] Implementar métricas e calibração em training/live_lab/metrics.py e training/live_lab/calibrate.py

## Phase 4: User Story 1 - Treinar (P1)

- [X] T012 [P] [US1] Testes do front-end (forma, paridade com STFT de referência) e da rede (forma de saída, parâmetros) em training/tests/test_frontend.py e training/tests/test_network.py
- [X] T013 [US1] Implementar front-end log-Mel exportável em training/live_lab/model/frontend.py
- [X] T014 [US1] Implementar rede separável e SpecAugment em training/live_lab/model/network.py e training/live_lab/model/specaugment.py
- [X] T015 [US1] Implementar dados, laço de treino e checkpoints em training/live_lab/train/data.py, loop.py, checkpoint.py
- [X] T016 [US1] Teste de fumaça: treino curto em CPU com dataset falso, interrupção e retomada, em training/tests/test_train_smoke.py
- [X] T017 [US1] Comando `train` em training/live_lab/cli_model.py e notebook training/colab/treino.ipynb

## Phase 5: User Story 4 - Exportar (P2)

- [X] T018 [US4] Implementar exportação FP32 → INT8, equivalência e pacote em training/live_lab/export.py
- [X] T019 [US4] Implementar medição (tamanho, latência 1 thread, memória) em training/live_lab/bench.py
- [X] T020 [US4] Testes de exportação com modelo pequeno (equivalência, entrada = áudio bruto) em training/tests/test_export.py
- [X] T021 [US4] Comando `export` em training/live_lab/cli_model.py

## Phase 6: User Story 3 - Avaliar (P1)

- [X] T022 [US3] Implementar avaliação via ONNX Runtime e relatório em training/live_lab/evaluate.py
- [X] T023 [US3] Comando `evaluate` (recusa voz real na nuvem) e teste ponta a ponta em training/tests/test_evaluate.py

## Phase 7: Polish

- [ ] T024 Exportar requirements, atualizar training/README.md e README.md
- [ ] T025 Rodar quickstart local (passos 1, 2 e 4 com modelo de fumaça) e registrar no HANDOFF.md

## Dependencies

US5 independente. US2 depende só da trie (001). US1 depende de US2 para a métrica de validação. US4 depende de US1.
US3 depende de US2 e US4.
