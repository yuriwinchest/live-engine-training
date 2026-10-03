# Research — 001 gramática e preparação de dados

Marcas de origem: **[verificado]** (conferido em 2026-10-03, com fonte), **[memória]**, **[estimativa]**.

## R1. Geradores de voz sintética (licença comercial clara)

**Decision**: Kokoro-82M primeiro; Chatterbox Multilingual e Parler-TTS Mini Multilingual v1.1 como adaptadores.

| Motor | Licença | pt-BR | Observação |
|---|---|---|---|
| Kokoro-82M | Apache-2.0 [verificado: github.com/hexgrad/kokoro] | 3 vozes: pf_dora, pm_alex, pm_santa | 24 kHz; exige espeak-ng no sistema (no Windows, instalador MSI que o PO precisa autorizar e instalar; no Colab, `apt`). Leve, roda em CPU. |
| Chatterbox Multilingual | MIT [verificado: resemble.ai] | Português | Clonagem de voz a partir de áudio de referência → muitas vozes se a referência vier do Common Voice (CC0). **Marca d'água PerTh embutida** em toda saída; inaudível, e o ruído da preparação a cobre [estimativa]. Pesado: GPU no Colab. |
| Parler-TTS Mini Multilingual v1.1 | Apache-2.0 [verificado: huggingface.co/parler-tts] | Português (sotaque pode tender ao europeu [memória]) | Voz controlada por descrição em texto → diversidade sem clonagem. GPU no Colab. |

**Excluídos** (licença não comercial ou ambígua): XTTS-v2/Coqui (CPML, não comercial) [memória]; MMS-TTS (CC-BY-NC)
[memória]; F5-TTS (pesos CC-BY-NC) [memória]; edge-tts (termos de serviço do Edge não autorizam o uso) [memória];
Piper pt_BR (licença por voz ambígua) [verificado: discussão rhasspy/piper #271]. APIs gratuitas temporárias
de nuvem ficam fora: os termos de uso da saída para treino de modelo não foram verificados.

**Rationale**: o núcleo não depende de nenhum motor; trocar ou somar motor é um adaptador novo.
A licença de cada motor é registrada como fonte e passa pela mesma política das demais (R3).

## R2. Ruído

**Decision**: MUSAN (CC BY 4.0, ~109 h; atribuição exigida; montado para permitir uso comercial)
[verificado: openslr.org/17]. Subconjuntos `noise` (ruído) e `music` (som ambiente); `speech` do MUSAN é
majoritariamente em inglês e serve como "fala alheia" nos exemplos sem número.

**Alternatives**: UrbanSound8K e ESC-50 (CC BY-NC) [memória] — recusados pela política (R3).

## R3. Política de licenças

**Decision**: lista de permissão explícita: `CC0-1.0`, `CC-BY-4.0`, `Apache-2.0`, `MIT`, `public-domain`,
`own-recording-consented`. Qualquer outra, ou ausente, é recusada antes do processamento (FR-007). Licenças
com atribuição geram arquivo `ATRIBUICOES.md` junto do dataset.

## R4. Gramática: tabela gerada, não analisador escrito à mão

**Decision**: um gerador inteiro → falas define a linguagem; o conversor fala → inteiro é a tabela inversa.
Uma colisão (mesma fala para dois inteiros) aborta a construção.

**Rationale**: a linguagem é finita (~100 mil sequências [estimativa]); gerar e inverter garante, por construção,
que conversor e gerador nunca divergem — a falha clássica de ter dois códigos para a mesma regra.
**Exportação**: árvore de prefixos (trie) com o inteiro nos nós terminais; é o formato que um *beam search*
restrito consome. Tamanho da trie completa será medido; no app, a trie restrita aos inscritos é pequena.

**Alternatives**: FST com aritmética nos arcos (menor, mas duas implementações da regra); OpenFST/pynini
(dependência pesada, compilação difícil no Windows) [memória].

## R5. Regras do português adotadas

- Por extenso: "cem" só 100; "cento e …" 101–199; "mil" e "um mil" → 1000; após "mil", o "e" é aceito com e sem
  (normativo e coloquial); "uma/duas" aceitos como 1/2 fora do multiplicador de milhar; "quatorze/catorze".
- Dígito a dígito: 1–4 algarismos, zeros à esquerda, "meia" = 6, "uma/duas" = 1/2.
- Mistura de modos e agrupamento em pares: fora desta versão.

## R6. Degradação

**Decision**:
- SNR por nível, sorteado uniformemente: leve 15–25 dB, médio 5–15 dB, extremo −5–5 dB [estimativa inicial,
  a calibrar com o áudio real do PO]. Potência medida sobre a fala ativa (amostras acima de −40 dBFS do pico).
- *Time stretch* 0,85–1,15× e *pitch shift* ±2 semitons por phase vocoder próprio (`dsp.py`); conferido: 440 Hz
  +2 semitons → 494,0 Hz (esperado 493,9) [verificado em 2026-10-03]. O librosa saiu: o numba é bloqueado pelo
  Smart App Control do Windows.
- Coloração de microfone: passa-altas 80–200 Hz e passa-baixas 3,4–7 kHz sorteados (alto-falante/microfone de
  celular) [estimativa].
- Folga de Push-to-Talk: 0,1–0,6 s de ruído antes e depois da fala.
- Anti-clipping: se o pico passar de 0,98, a mistura inteira é atenuada (preserva a SNR).

## R7. Determinismo

**Decision**: um `numpy.random.Generator` por exemplo, semeado por `hash(semente global, chave do exemplo)`;
UUID v5 do mesmo par. A ordem de processamento não muda o resultado. Igualdade byte a byte garantida no mesmo
ambiente; entre CPUs ou bibliotecas diferentes pode haver diferença de ponto flutuante [memória], registrada no
manifesto (versões das bibliotecas).

## R8. Formato de áudio

**Decision**: WAV PCM 16 bits, 16 kHz, mono. Padrão de modelos de fala pequenos [memória] e do microfone do
Android em modo de reconhecimento.
