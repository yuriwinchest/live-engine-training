# Implementation Plan: Vocabulário, gramática de números e preparação de dados

**Branch**: `001-gramatica-dataprep` | **Date**: 2026-10-03 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-gramatica-dataprep/spec.md`

## Summary

Laboratório Python (`training/live_lab`) com três blocos independentes:

1. **Gramática**: gerador inteiro → todas as falas válidas (por extenso e dígito a dígito) e conversor
   fala → inteiro construído **a partir do próprio gerador** (tabela exata, com checagem de colisão na construção).
   Exporta uma árvore de prefixos (trie) em JSON, completa ou restrita aos inscritos, para a decodificação CTC.
2. **Síntese**: planeja falas balanceadas a partir da gramática e as renderiza com geradores TTS de licença
   comercial clara (Kokoro-82M primeiro; Chatterbox Multilingual e Parler-TTS Multilingual como adaptadores).
3. **Preparação**: degrada as falas limpas com ruído MUSAN em SNR controlado, *time stretch*, *pitch shift* e
   coloração de microfone; gera exemplos "sem número"; separa por locutor; grava manifesto JSONL e relatório.

O recorte de palavras do Common Voice fica fora desta versão (decisão do PO). A gravação real do PO entra só pela
ingestão local, na partição de teste.

## Technical Context

**Language/Version**: Python 3.12 (venv gerenciado por `uv`; ADR-005)

**Primary Dependencies**: numpy, scipy (filtros, reamostragem), soundfile (WAV), pytest. Phase vocoder próprio
(`dsp.py`): o librosa foi retirado porque o numba é bloqueado pelo Smart App Control desta máquina. Motores TTS
(`kokoro` + espeak-ng, `chatterbox-tts`, `parler-tts`) ficam fora do lock e são instalados no Colab.

**Storage**: arquivos locais fora do Git: `training/data/` (áudio WAV 16 kHz mono PCM16, manifestos JSONL,
relatórios JSON/Markdown).

**Testing**: pytest (gramática exaustiva 0–9999, determinismo, SNR, separação por locutor, política de licenças).

**Target Platform**: Windows 11 local (CPU, sem GPU NVIDIA) e Google Colab (Linux, GPU) para síntese em volume.

**Project Type**: biblioteca + CLI (`live-lab`).

**Performance Goals**: construção da gramática completa < 5 s em CPU; preparação ≥ 20 clipes/s por núcleo em
CPU local [estimativa: clipes de ~2 s, stretch por phase vocoder domina o custo]; a medir.

**Constraints**: offline após download das fontes; determinístico por semente no mesmo ambiente; nenhum arquivo
de código > 500 linhas; nada de voz real em ambiente de nuvem.

**Scale/Scope**: ~100 mil sequências válidas na gramática completa [estimativa: 10 000 inteiros × ~10 variantes];
dataset inicial de 50–200 mil exemplos de treino.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Princípio | Como o plano cumpre | Status |
|---|---|---|
| I. Contrato estreito | Gramática pura: tokens → inteiro/rejeição; nada de banco, tela ou regra de prova. | ✅ |
| II. Nunca inventar número | Conversor = tabela gerada; colisão aborta a construção; inscritos restringem por exportação. | ✅ |
| III. Paridade treino/aparelho | Áudio padronizado em 16 kHz mono; log-Mel fica no modelo (feature seguinte). Trie JSON exportável. | ✅ |
| IV. Dados e privacidade | Lista de licenças permitidas; NC recusada; `training/data/` fora do Git; ingestão real recusa rodar no Colab. | ✅ |
| V. Medir, não prometer | Partição `test` só aceita fonte real; relatório marca resultado sintético como provisório. | ✅ |
| VI. Engenharia | Módulos < 500 linhas, tipados; UUID v5 derivado da semente; `uv.lock` + `requirements.txt` exportado. | ✅ |

Sem violações; *Complexity Tracking* vazio.

**Re-check pós-design**: mantido. O contrato da trie ([contracts/grammar-export.md](contracts/grammar-export.md))
não carrega dado pessoal; o manifesto guarda só pseudônimo de locutor.

## Project Structure

### Documentation (this feature)

```text
specs/001-gramatica-dataprep/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── cli.md
│   └── grammar-export.md
└── tasks.md
```

### Source Code (repository root)

```text
training/
├── pyproject.toml            # projeto uv, extras por motor TTS
├── uv.lock
├── requirements.txt          # exportado do lock, para o Colab
├── sources.example.toml      # registro de fontes (modelo, sem caminhos locais)
├── colab/                    # notebook de síntese em GPU
├── live_lab/
│   ├── vocab.py              # vocabulário fechado e índices
│   ├── numbers_pt.py         # inteiro → falas (regras do português)
│   ├── grammar.py            # tabela, conversão, restrição, exportação trie
│   ├── sources.py            # registro e política de licenças
│   ├── audio_io.py           # leitura, reamostragem, gravação 16 kHz
│   ├── dsp.py                # phase vocoder, pitch shift, corte de silêncio
│   ├── augment.py            # SNR, stretch, pitch, microfone, anti-clipping
│   ├── manifest.py           # Example, JSONL, UUID determinístico
│   ├── split.py              # partição por locutor
│   ├── negatives.py          # exemplos "sem número"
│   ├── prep.py               # orquestra a preparação
│   ├── report.py             # relatório
│   ├── ingest_real.py        # gravação real do PO → teste (só local)
│   ├── cli.py
│   └── synth/
│       ├── base.py           # protocolo TtsEngine, Voice
│       ├── planner.py        # escolha balanceada de números e formas
│       ├── render.py         # renderização retomável
│       ├── kokoro_engine.py
│       ├── chatterbox_engine.py
│       └── parler_engine.py
└── tests/
    ├── test_numbers_pt.py
    ├── test_grammar.py
    ├── test_sources.py
    ├── test_augment.py
    ├── test_split.py
    ├── test_manifest.py
    ├── test_planner.py
    └── test_prep_e2e.py      # ponta a ponta com motor TTS falso e ruído sintético
```

**Structure Decision**: projeto Python único em `training/`, separado do futuro `android/`. Pacote `live_lab`
com um módulo por responsabilidade; motores TTS isolados atrás de um protocolo, instalados como extras para que
o núcleo (gramática e preparação) rode sem dependências pesadas.

## Complexity Tracking

Nenhuma violação a justificar.
