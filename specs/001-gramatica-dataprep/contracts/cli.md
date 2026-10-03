# Contrato — CLI `live-lab`

Saída humana em stderr; resultados em arquivos. Código de saída 0 = sucesso, 2 = entrada inválida
(licença recusada, inscritos inválidos), 1 = erro inesperado.

| Comando | Entrada | Saída |
|---|---|---|
| `live-lab grammar parse "duzentos e trinta e três"` | fala | inteiro em stdout ou `REJEITADO` (saída 3) |
| `live-lab grammar forms 1200` | inteiro | uma fala por linha |
| `live-lab grammar export --out g.json [--inscritos ids.txt]` | lista opcional (um inteiro por linha) | trie JSON ([grammar-export.md](grammar-export.md)) |
| `live-lab sources check --registry fontes.toml` | registro | lista de fontes aceitas/recusadas com motivo |
| `live-lab synth plan --out jobs.jsonl --count N --seed S --engine kokoro` | — | trabalhos de síntese (número, forma, voz) |
| `live-lab synth render --jobs jobs.jsonl --out DIR --engine kokoro` | trabalhos | WAV 16 kHz + `clean.jsonl`; retomável |
| `live-lab ingest-real --labels rotulos.csv --audio DIR --out DIR` | CSV `arquivo,numero` | `clean.jsonl` da fonte real (recusa rodar no Colab) |
| `live-lab prep --registry fontes.toml --clean a.jsonl [b.jsonl…] --out DIR --seed S --copies K` | clipes limpos + ruído | `manifest.jsonl`, `audio/`, `report.json`, `report.md`, `ATRIBUICOES.md` |
