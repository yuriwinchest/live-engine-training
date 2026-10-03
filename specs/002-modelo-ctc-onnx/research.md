# Research — 002

Marcas: **[verificado]** (conferido em 2026-10-03), **[memória]**, **[estimativa]**.

## R1. Ambiente local

torch 2.14.1+cpu, onnxruntime 1.30.0 (CPUExecutionProvider), onnx 1.23.1 e onnxscript 0.7.2 instalam e carregam no
Windows com Smart App Control ligado [verificado: venv de sondagem]. Consequência: testes de modelo, exportação,
avaliação da voz real e medição rodam localmente; só o treino em volume vai ao Colab.

## R2. Arquitetura

**Decision**: front-end log-Mel (janela 25 ms, salto 10 ms, 40 bandas, normalização por fala) → convolução de
entrada com passo 2 (quadros de 20 ms) → 5 blocos residuais de convolução 1-D separável (profundidade + ponto),
BatchNorm, ReLU, dropout → convolução 1×1 para 45 saídas. ~128–160 canais, ~0,3–0,6 M parâmetros [estimativa].

**Rationale**: separáveis com CTC são comprovadas em reconhecimento de fala pequeno (QuartzNet) e em detecção de
palavras (TC-ResNet) [memória]; só convoluções → quantização INT8 estável e execução rápida em ARM. Quadros de
20 ms dão ≥ 50 quadros por segundo, folga para as ≤ 9 palavras de um número.

**Alternatives**: GRU/LSTM (quantização menos madura); Conformer pequeno (atenção custa memória e latência no A11).

## R3. Front-end exportável

**Decision**: STFT como `Conv1d` com pesos fixos (partes real e imaginária da DFT × janela de Hann), potência,
produto pela matriz Mel e `log(x + 1e-6)`. Só operações básicas de ONNX (Conv, Mul, Add, MatMul, Log, ReduceMean).

**Rationale**: o operador `STFT` do ONNX (opset 17) tem suporte irregular em runtimes móveis [memória]; a
convolução é universal. Front-end fica fora da quantização (FP32), pois log e potência perdem precisão em 8 bits.

## R4. Decodificação e rejeição

**Decision**: *prefix beam search* CTC (feixe 8) onde cada hipótese é um nó da trie; só tokens com filho na trie
expandem. Candidatos finais = nós com valor + a **sequência vazia** (todos brancos). Sinais:
1. **posterior** do melhor candidato entre todos os candidatos finais (softmax dos escores);
2. **margem** em log entre o 1º e o 2º;
3. **aderência**: (escore restrito − escore do melhor caminho livre) / quadros — baixo quando o áudio não parece
   nenhuma fala da gramática.
Rejeita se o melhor for a sequência vazia ou se qualquer sinal ficar abaixo do limiar.

**Calibração**: busca em grade na validação; entre os limiares que cumprem acerto ≥ meta, escolhe o de **menor
taxa de número errado** (decisão do PO, opção A); empate → maior acerto.

## R5. Treino

AdamW (lr 1e-3, decaimento 1e-3), aquecimento linear 500 passos + cosseno, até 80 épocas, early stopping com
paciência 8 sobre o acerto do número inteiro na validação (decodificação restrita, sem rejeição). CTCLoss com
`blank=0` e `zero_infinity`; alvo vazio (sem número) é aceito pela perda CTC [memória]. SpecAugment leve (2 máscaras
de frequência ≤ 8 bandas, 2 de tempo ≤ 10% da fala). Lotes agrupados por duração. Checkpoint `last` a cada época e
`best` quando melhora; retomada restaura modelo, otimizador, agendador, época e estados aleatórios.

## R6. Exportação e quantização

`torch.onnx.export` (opset 17, eixo dinâmico de amostras) → `onnxruntime.quantization.quantize_static`, formato QDQ,
por canal, calibração com ~300 clipes de validação, nós do front-end excluídos [memória: API]. Equivalência:
FP32 ONNX × PyTorch (diferença máxima de log-probabilidade < 1e-3) e INT8 × FP32 (queda de acerto ≤ 1 p.p., SC-002).

## R7. Ruído reservado (US5)

Reserva **independente da semente**: um arquivo é reservado se `hash("holdout-v1", fonte/arquivo)` cair nos 10%
inferiores. Mudar a semente da preparação nunca devolve um reservado ao treino. A preparação ignora reservados.
Faixas de reprodução: **multidão** = 6–8 falas reservadas do MUSAN sobrepostas com ganhos e deslocamentos
aleatórios; **rua** = multidão + ruído/música reservados. Normalização em −20 dBFS RMS e pico ≤ −1 dBFS.
O MUSAN fala majoritariamente inglês; com 6+ vozes sobrepostas o idioma pesa pouco [estimativa], mas o ruído real
de prova continua necessário.
