# Quickstart — validação da feature 001

## Pré-requisitos

- `uv` instalado; Python 3.12 obtido pelo próprio `uv`.
- Para síntese real: espeak-ng (Kokoro) — no Colab via `apt`; no Windows, instalador MSI (decisão do PO).
- MUSAN baixado do OpenSLR para um diretório fora do repositório.

## 1. Ambiente e testes (sem áudio externo)

```powershell
cd training
uv sync
uv run pytest
```

Esperado: todos os testes passam, incluindo a varredura exaustiva 0–9999 (SC-001, SC-002, SC-003),
determinismo (SC-004), separação por locutor (SC-005) e SNR (SC-008), com motor TTS falso e ruído gerado.

No Windows desta máquina, use `uv run python -m live_lab …` no lugar de `uv run live-lab …`: o Smart App
Control bloqueia o `live-lab.exe` gerado pelo instalador.

## 2. Gramática na mão

```powershell
uv run live-lab grammar parse "zero zero sete"        # 7
uv run live-lab grammar parse "trinta e"              # REJEITADO
uv run live-lab grammar forms 1200                    # inclui "mil e duzentos" e "um dois zero zero"
uv run live-lab grammar export --out ..\training\data\grammar.json
```

## 3. Fontes

Copiar `sources.example.toml` para `training/data/fontes.toml`, ajustar caminhos e rodar
`uv run live-lab sources check --registry data/fontes.toml`. Uma fonte com `license = "CC-BY-NC-4.0"`
deve ser recusada (saída 2).

## 4. Pipeline (Colab ou local com espeak-ng)

```powershell
uv run live-lab synth plan --out data/jobs.jsonl --count 3000 --seed 7 --engine kokoro
uv run live-lab synth render --jobs data/jobs.jsonl --out data/clean/kokoro --engine kokoro
uv run live-lab prep --registry data/fontes.toml --clean data/clean/kokoro/clean.jsonl --out data/ds-v1 --seed 7 --copies 4
```

Conferir em `data/ds-v1/report.md`: contagens por partição, nível de ruído e cobertura do vocabulário;
`provisional_test: true` enquanto não houver gravação real.

## 5. Gravação real do PO (só local)

`uv run live-lab ingest-real --labels rotulos.csv --audio <pasta> --out data/clean/real` e repetir o `prep`
incluindo esse `clean.jsonl`; a fonte real cai inteira em `test` e `provisional_test` passa a `false`.
