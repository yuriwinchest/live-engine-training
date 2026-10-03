# Arquitetura do L.I.V.E. — decisões

Registro de decisões (ADR). Cada decisão diz o que foi escolhido, o que foi descartado e por quê. Mudar uma decisão
exige nova entrada aqui, não só código novo.

---

## ADR-001 — Modelo de sequência com CTC + gramática, não classificador de palavra única

**Data:** 2026-10-03 · **Decisão:** Yuri (PO), com análise de Claude Code (TONE)

**Contexto.** O primeiro esboço (pesquisa com o Gemini) propunha uma CNN de *Keyword Spotting* que classifica o
espectrograma em uma de ~40 palavras. Um classificador de rótulo único responde **uma** palavra por janela; o número
de peito falado é uma **sequência**: 233 = "duzentos · e · trinta · e · três" (~1,5–2 s). Para usar o classificador
seria preciso segmentar palavra por palavra — exatamente a parte que quebra com ruído.

**Decisão.** Modelo acústico pequeno que emite uma **sequência de tokens** com perda **CTC** (*Connectionist Temporal
Classification*: aprende o alinhamento áudio↔tokens sozinho, sem marcar onde cada palavra começa). Encoder
convolucional + recorrente ou mini-Conformer, a decidir por medição. A decodificação é restrita por uma **gramática
de números válidos** (beam search sobre um autômato finito), então a saída só pode ser um número bem formado.

**Vocabulário (≈ 45 tokens).** zero · um/uma · dois/duas · três … nove · **meia** (6 no ditado) · dez … dezenove
(com quatorze/catorze) · vinte … noventa · cem · cento · duzentos … novecentos · mil · e · `<branco>` do CTC.
Dois modos de fala, ambos aceitos pela gramática: **por extenso** ("duzentos e trinta e três") e **dígito a
dígito** ("dois três três").

**Descartado.**
- KWS de rótulo único: não lê sequência (acima).
- Uma classe por número (10 000 classes): dados impossíveis de reunir.
- STT genérico (Vosk, Whisper, Google): medido no LGDA, perde falas inteiras e parte números no ruído
  (relatório do LGDA, 2026-10-03).

## ADR-002 — Captura por *Push-to-Talk*

O operador segura um botão (tela, tecla de volume ou controle Bluetooth) enquanto fala. Começo e fim da fala deixam
de ser adivinhados por um VAD; conversa em volta fora do aperto não entra. O **horário da passagem é o do aperto**,
mais exato para cronometragem do que o horário em que o motor termina de decodificar.

## ADR-003 — Robustez vem dos dados; o app é o coletor

"O ruído é ignorado pela rede" só vale para o ruído que a rede viu no treino. Fontes, em ordem de valor:
1. **Clipes reais do LGDA:** cada aperto confirmado pelo operador é um exemplo rotulado com ruído real.
   Só com **consentimento registrado** (LGPD); áudio nunca entra no Git.
2. Gravações dirigidas (várias vozes lendo listas de números) e ruído gravado em provas.
3. Fala sintética variada (vozes TTS) para cobrir combinações raras.
4. *Augmentation*: mistura com ruído em SNR controlado (leve, médio, extremo), *time stretch*, *pitch shift*,
   resposta de microfone/ambiente.

Etapa 2 (planejada): **ajuste ao operador** — ele fala ~50 números uma vez e o modelo se adapta à voz dele.

## ADR-004 — Exportação ONNX, espectrograma dentro do modelo

O cálculo do log-Mel entra **no grafo exportado**: celular e treino calculam exatamente igual (elimina a classe de
bug "treinou com um espectrograma, roda com outro"). Formato padrão **ONNX** com ONNX Runtime no Android; TFLite só
se uma medição no aparelho justificar. Sem C++/JNI próprio enquanto não houver gargalo medido.

## ADR-005 — Ambiente do laboratório

Python **3.11 ou 3.12** em ambiente virtual próprio (o 3.14 da máquina pode não ter todas as ferramentas de
exportação). Dependências fixadas em `requirements.txt`.

---

## Critério de pronto (a medir, não a prometer)

| Métrica | Como medir |
|---|---|
| Acerto do número inteiro | % de clipes cujo número sai exato; nunca "8" no lugar de "248" |
| Rejeição | clipe sem número ou ambíguo → "não entendi", nunca um número inventado |
| Latência | do soltar o botão ao número na tela, medida no aparelho mais fraco de campo |
| Tamanho e memória | MB do modelo e pico de RAM no aparelho (o LGDA já foi encerrado por falta de memória) |

Os valores-alvo são definidos pelo PO com base no áudio real gravado em prova.
