# live-lab — laboratório do L.I.V.E.

Gramática de números, síntese de voz e preparação de dados para o modelo acústico CTC.
Especificação e decisões: [`specs/001-gramatica-dataprep/`](../specs/001-gramatica-dataprep/).

## Ambiente

```powershell
uv sync          # Python 3.12 gerenciado pelo uv; dependências fixadas em uv.lock
uv run pytest    # inclui a varredura exaustiva de 0 a 9999
```

`requirements.txt` é exportado do lock para o Colab: `uv export --no-hashes --no-dev --no-emit-project -o requirements.txt`.

## Gramática

```powershell
uv run live-lab grammar parse "zero zero sete"      # 7
uv run live-lab grammar forms 1200                  # todas as falas válidas
uv run live-lab grammar export --out data/grammar.json [--inscritos inscritos.txt]
```

A linguagem é definida uma única vez em `live_lab/numbers_pt.py`; o conversor fala → número é a tabela
inversa (55 908 falas). Uma fala que servisse a dois números aborta a construção.

## Dados

1. Copie `sources.example.toml` para `data/fontes.toml` e ajuste os caminhos. Licença não comercial é recusada.
2. Síntese (Colab, [`colab/sintese.ipynb`](colab/sintese.ipynb)): `synth plan` → `synth render`.
3. Preparação: `live-lab prep --registry data/fontes.toml --clean <clean.jsonl…> --out data/ds-v1 --seed 7`.
4. Gravação real do PO, só local: `live-lab ingest-real` → vai inteira para a partição `test`.

Saída: `manifest.jsonl` (cabeçalho + um exemplo por linha, UUID), `audio/<partição>/`, `report.md`,
`report.json`, `ATRIBUICOES.md`, `run.json`. Tudo em `training/data/`, fora do Git.

## Restrições desta máquina

O Smart App Control do Windows bloqueia DLLs do `numba` e do `mypy`. Por isso o processamento de sinal é próprio
(`live_lab/dsp.py`, sem librosa) e a checagem de tipos (`uv run mypy live_lab`) roda no Colab ou no Linux.
