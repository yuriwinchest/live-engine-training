# Data Model — 001

## Vocabulary

| Campo | Tipo | Regra |
|---|---|---|
| `version` | str | Muda quando a lista muda; gravado em todo manifesto e exportação. |
| `tokens` | tuple[str, ...] | Índice 0 = `<blank>` (CTC); demais em ordem fixa (FR-001). |

## Grammar

| Campo | Tipo | Regra |
|---|---|---|
| `table` | dict[tuple[str, ...], int] | Toda fala válida → inteiro; colisão aborta a construção. |
| `allowed` | frozenset[int] \| None | Inscritos; `None` = 0–9999. Vazio ou fora da faixa → erro. |

Operações: `parse(tokens) -> int | None`, `forms(n) -> list[tuple[str, ...]]`, `restrict(ids) -> Grammar`,
`export_trie() -> dict` (contrato em [contracts/grammar-export.md](contracts/grammar-export.md)).

## Source

| Campo | Tipo | Regra |
|---|---|---|
| `name` | str | Único no registro. |
| `kind` | `speech_synthetic` \| `speech_real` \| `noise` \| `distractor` | `speech_real` só entra na partição `test`. |
| `license` | str | Deve estar na lista de permissão (research R3). |
| `origin` | str | URL ou descrição da procedência. |
| `obtained_at` | date | ISO 8601. |
| `path` | str | Diretório local; nunca dentro do repositório versionado. |
| `attribution` | str \| None | Obrigatório para CC-BY. |

## CleanClip (saída da síntese ou da ingestão real)

| Campo | Tipo | Regra |
|---|---|---|
| `id` | UUID | v5 de (semente, motor, voz, número, forma). |
| `audio` | caminho relativo | WAV 16 kHz mono. |
| `tokens` | list[str] | Vazio para distrator. |
| `number` | int \| None | `None` = sem número. |
| `mode` | `extenso` \| `digitos` \| `nenhum` | |
| `source` | str | Nome da fonte. |
| `speaker` | str | Pseudônimo estável (`spk-` + hash curto); nunca o identificador original. |

## Example (linha do manifesto final)

Todos os campos de `CleanClip`, mais:

| Campo | Tipo | Regra |
|---|---|---|
| `split` | `train` \| `val` \| `test` | Por locutor; `test` só fonte real. |
| `noise_source` | str \| None | |
| `snr_level` | `limpo` \| `leve` \| `medio` \| `extremo` | |
| `snr_db` | float \| None | Valor sorteado e aplicado. |
| `stretch` | float | 1.0 = sem alteração. |
| `pitch_semitones` | float | |
| `duration_s` | float | |

## Manifest header (primeira linha do JSONL, `"type": "header"`)

`seed`, `vocab_version`, `created_at`, `lib_versions`, `sources` (nome + licença), `provisional_test`
(true enquanto a partição `test` não tiver fonte real).

## Transições

`CleanClip` (síntese/ingestão) → `Example` (preparação). Um `CleanClip` de treino gera N `Example`
(cópias degradadas); um `CleanClip` real gera exatamente um `Example` sem degradação.
