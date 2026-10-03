---
description: "Tarefas da feature 001 — vocabulário, gramática e preparação de dados"
---

# Tasks: Vocabulário, gramática de números e preparação de dados

**Input**: `specs/001-gramatica-dataprep/` (plan, spec, research, data-model, contracts, quickstart)

**Tests**: pedidos no plano (pytest) e exigidos pelos critérios SC-001 a SC-010.

**Organization**: por história de uso; caminhos relativos à raiz do repositório.

## Phase 1: Setup

- [X] T001 Criar projeto uv em training/pyproject.toml (Python 3.12, pacote live_lab, script `live-lab`, grupo dev com pytest; motores TTS fora do lock, instalados no Colab)
- [X] T002 Criar estrutura de pacotes training/live_lab/__init__.py, training/live_lab/synth/__init__.py e training/tests/conftest.py
- [X] T003 [P] Conferir que .gitignore cobre training/data/, áudio e manifestos (incluir *.opus, *.jsonl em training/data)

---

## Phase 2: Foundational

- [X] T004 Implementar vocabulário fechado e versionado em training/live_lab/vocab.py ("Índice 0 = `<blank>` (CTC); demais em ordem fixa (FR-001)")

**Checkpoint**: vocabulário pronto; todas as histórias dependem dele.

---

## Phase 3: User Story 1 - Converter fala em número, sem ambiguidade (P1) 🎯 MVP

**Goal**: inteiro ↔ falas válidas, com rejeição do que não é número.

**Independent Test**: varredura 0–9999: toda forma gerada volta ao inteiro; nenhuma fala serve a dois inteiros.

- [X] T005 [P] [US1] Testes das regras do português (cem/cento, mil e "e" opcional, meia, catorze, uma/duas, zeros à esquerda, casos de rejeição da spec) em training/tests/test_numbers_pt.py
- [X] T006 [P] [US1] Testes exaustivos da gramática (SC-001, SC-002) e do contrato da trie em training/tests/test_grammar.py
- [X] T007 [US1] Implementar gerador inteiro → falas em training/live_lab/numbers_pt.py (regras de research R5)
- [X] T008 [US1] Implementar Grammar (tabela com colisão abortando, parse, forms, export_trie) em training/live_lab/grammar.py
- [X] T009 [US1] Implementar comandos `grammar parse|forms|export` em training/live_lab/cli.py (saída 3 = REJEITADO)

**Checkpoint**: gramática completa testada sem áudio.

---

## Phase 4: User Story 2 - Restringir aos inscritos (P2)

**Goal**: com lista de inscritos, só números da lista são aceitos.

**Independent Test**: inscritos {7, 233, 1500}: 233 aceito, 234 rejeitado.

- [X] T010 [US2] Testes de restrição (SC-003, lista vazia ou fora de 0–9999 → erro) em training/tests/test_grammar.py
- [X] T011 [US2] Implementar Grammar.restrict e `--inscritos` no export ("Vazio ou fora da faixa → erro") em training/live_lab/grammar.py e training/live_lab/cli.py

---

## Phase 5: User Story 3 - Conjunto de dados rotulado e reprodutível (P1)

**Goal**: manifesto JSONL determinístico, separado por locutor, com fontes licenciadas.

**Independent Test**: duas execuções com a mesma semente → resultado idêntico; nenhum locutor em duas partições.

- [X] T012 [P] [US3] Testes da política de licenças (NC recusada, ausente recusada) em training/tests/test_sources.py
- [X] T013 [P] [US3] Testes de partição por locutor ("`test` só fonte real") em training/tests/test_split.py
- [X] T014 [P] [US3] Testes do planejador de síntese (balanceamento por modo e quantidade de algarismos, cobertura do vocabulário) em training/tests/test_planner.py
- [X] T015 [US3] Implementar registro de fontes e lista de permissão (research R3) em training/live_lab/sources.py e training/sources.example.toml
- [X] T016 [P] [US3] Implementar leitura/reamostragem/gravação WAV 16 kHz mono PCM16 em training/live_lab/audio_io.py
- [X] T017 [P] [US3] Implementar CleanClip, Example, cabeçalho e JSONL com UUID v5 em training/live_lab/manifest.py
- [X] T018 [P] [US3] Implementar partição por locutor com pseudônimo estável em training/live_lab/split.py
- [X] T019 [US3] Implementar protocolo TtsEngine/Voice e motor falso para testes em training/live_lab/synth/base.py
- [X] T020 [US3] Implementar planejador de trabalhos de síntese em training/live_lab/synth/planner.py
- [X] T021 [US3] Implementar renderização retomável → clean.jsonl em training/live_lab/synth/render.py
- [X] T022 [P] [US3] Implementar adaptador Kokoro (vozes pf_dora, pm_alex, pm_santa; 24 kHz → 16 kHz) em training/live_lab/synth/kokoro_engine.py — *escrito, não executado: exige espeak-ng/torch (Colab)*
- [X] T023 [P] [US3] Implementar adaptadores Chatterbox (referência CC0) e Parler-TTS em training/live_lab/synth/chatterbox_engine.py e training/live_lab/synth/parler_engine.py — *escritos, não executados (GPU/Colab)*
- [X] T024 [US3] Implementar ingestão da gravação real (recusa no Colab) em training/live_lab/ingest_real.py
- [X] T025 [US3] Implementar exemplos "sem número" (ruído puro e fala distratora sem palavras do vocabulário) em training/live_lab/negatives.py
- [X] T026 [US3] Implementar orquestração da preparação em training/live_lab/prep.py
- [X] T027 [US3] Implementar relatório JSON/Markdown e ATRIBUICOES.md em training/live_lab/report.py
- [X] T028 [US3] Implementar comandos `sources check`, `synth plan|render`, `ingest-real`, `prep` em training/live_lab/cli.py
- [X] T029 [US3] Teste ponta a ponta com motor falso e ruído gerado (SC-004, SC-005, SC-007) em training/tests/test_prep_e2e.py

---

## Phase 6: User Story 4 - Simular o ruído de largada (P2)

**Goal**: degradação controlada sem alterar rótulo e sem saturar.

**Independent Test**: SNR medida a até 1 dB da pedida (SC-008); pico ≤ 0,98.

- [X] T030 [P] [US4] Testes de SNR, anti-clipping, stretch e pitch em training/tests/test_augment.py
- [X] T031 [US4] Implementar mix_at_snr, stretch, pitch, microfone e folga PTT (research R6) em training/live_lab/augment.py
- [X] T032 [US4] Ligar a degradação ao prep (níveis leve/medio/extremo, `--copies`) em training/live_lab/prep.py

---

## Phase 7: Polish

- [X] T033 Exportar training/requirements.txt do uv.lock e criar notebook de síntese em training/colab/sintese.ipynb
- [X] T034 [P] Documentar o laboratório em training/README.md e atualizar README.md (seção de estado)
- [X] T035 Rodar quickstart (passos 1–3) e registrar resultado real no HANDOFF.md — *rodado também o passo 4 com motor falso*

## Dependencies

- Setup → Foundational → US1 → US2.
- US3 depende de US1 (rótulos vêm da gramática). US4 depende de US3 (prep) para a integração T032; T030–T031 são independentes.
- Polish depois de tudo.

## Parallel Examples

- US1: T005 e T006 juntos; depois T007 → T008 → T009.
- US3: T012, T013, T014, T016, T017, T018 em paralelo; T022 e T023 em paralelo.

## Implementation Strategy

1. MVP = US1 (gramática exaustivamente testada, sem áudio).
2. US2 (inscritos) — pequeno, fecha a garantia "nunca inventar".
3. US3 + US4 — dataset; validação final só quando o PO trouxer a gravação real.
