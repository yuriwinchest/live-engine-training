# Feature Specification: Vocabulário, gramática de números e preparação de dados

**Feature Branch**: `001-gramatica-dataprep`

**Created**: 2026-10-03

**Status**: Draft

**Input**: User description: "Fase 1, Etapa 1 do L.I.V.E.: vocabulário, gramática textual de números e preparação
de dados (data_prep) para treinar o modelo acústico CTC. Vocabulário fechado de ~45 tokens em pt-BR; gramática
por extenso e dígito a dígito com zeros à esquerda, inversa e restringível aos inscritos; manifesto de exemplos a
partir de fontes com licença compatível com uso comercial; ruído sintético por SNR, time stretch e pitch shift;
exemplos sem número para treinar rejeição; separação por locutor; determinístico; local e Colab; áudio nunca no
Git. Risco temporário: sem áudio real de prova."

## Contexto

O L.I.V.E. transforma a fala do operador de cronometragem ("duzentos e trinta e três") no número de peito `233`.
Esta feature entrega a base sem a qual nenhum modelo pode ser treinado nem avaliado: **o que pode ser dito**
(vocabulário), **o que é um número válido** (gramática) e **o material de treino e teste** (dados preparados).
O modelo acústico, o treino e a exportação para o celular são features seguintes.

## User Scenarios & Testing *(mandatory)*

Os "usuários" desta feature são o engenheiro de treino (quem prepara dados e treina) e, indiretamente, o operador
de cronometragem, cuja segurança depende de o motor nunca inventar número.

### User Story 1 - Converter fala transcrita em número, sem ambiguidade (Priority: P1)

O engenheiro entrega uma sequência de palavras do vocabulário (ex.: "duzentos e trinta e três", "dois três três",
"zero zero sete") e recebe o inteiro correspondente, ou uma rejeição quando a sequência não forma um número válido.
O inverso também existe: dado um inteiro, obtêm-se todas as formas válidas de falá-lo, usadas para rotular dados.

**Why this priority**: é a garantia central do produto ("nunca inventar número") e o rótulo de todo exemplo de
treino. Funciona e é testável sem nenhum áudio.

**Independent Test**: percorrer todos os inteiros de 0 a 9999, gerar suas falas válidas, convertê-las de volta e
conferir que cada uma retorna exatamente o inteiro de origem; e verificar que sequências malformadas são rejeitadas.

**Acceptance Scenarios**:

1. **Given** a sequência "duzentos e trinta e três", **When** convertida, **Then** resulta em `233`.
2. **Given** a sequência "dois três três", **When** convertida, **Then** resulta em `233`.
3. **Given** a sequência "zero zero sete", **When** convertida, **Then** resulta em `7` (nunca no texto "007").
4. **Given** a sequência "meia meia", **When** convertida, **Then** resulta em `66`.
5. **Given** a sequência "trinta e" (incompleta) ou "vinte dez" (malformada), **When** convertida, **Then** é rejeitada.
6. **Given** uma sequência vazia, **When** convertida, **Then** é rejeitada.
7. **Given** o inteiro `1200`, **When** pedidas suas formas, **Then** incluem "mil e duzentos" e "um dois zero zero".

---

### User Story 2 - Restringir a gramática aos inscritos da prova (Priority: P2)

Quem integra o motor informa a lista de números inscritos na prova; a partir daí, uma fala só é aceita se resultar
em um número dessa lista. Qualquer outro número válido passa a ser rejeitado.

**Why this priority**: reduz drasticamente a chance de erro em campo (o motor escolhe entre centenas de números, não
entre dez mil) e é o mecanismo previsto para o app no Android.

**Independent Test**: com a lista {7, 233, 1500}, "duzentos e trinta e três" retorna `233` e "duzentos e trinta e
quatro" é rejeitado.

**Acceptance Scenarios**:

1. **Given** inscritos {7, 233, 1500}, **When** chega "mil e quinhentos", **Then** resulta em `1500`.
2. **Given** inscritos {7, 233, 1500}, **When** chega "duzentos e trinta e quatro", **Then** é rejeitada.
3. **Given** lista de inscritos vazia ou com valor fora de 0–9999, **When** aplicada, **Then** a configuração é
   recusada com motivo explícito, sem afetar a gramática completa.

---

### User Story 3 - Preparar um conjunto de dados rotulado e reprodutível (Priority: P1)

O engenheiro aponta as fontes de áudio e de ruído autorizadas e executa a preparação. Recebe um manifesto de
exemplos, cada um com o áudio, a sequência de palavras, o inteiro (ou "sem número"), a fonte, a licença, o locutor
pseudonimizado, a partição (treino, validação, teste) e os parâmetros de degradação aplicados.

**Why this priority**: sem dados rotulados não há treino; sem reprodutibilidade, nenhuma comparação entre modelos é
confiável.

**Independent Test**: executar a preparação duas vezes com a mesma semente e as mesmas fontes e obter manifestos e
áudios idênticos; conferir que nenhum locutor aparece em duas partições.

**Acceptance Scenarios**:

1. **Given** fontes autorizadas e uma semente, **When** a preparação roda duas vezes, **Then** os resultados são
   idênticos byte a byte.
2. **Given** uma fonte sem licença registrada ou com licença não comercial, **When** a preparação roda, **Then** a
   fonte é recusada e o motivo é informado antes de qualquer processamento.
3. **Given** a preparação concluída, **When** se inspecionam as partições, **Then** nenhum locutor aparece em mais
   de uma partição.
4. **Given** a preparação concluída, **When** se inspeciona o resultado, **Then** existem exemplos "sem número"
   (só ruído ou fala sem número) com alvo vazio, para ensinar a rejeição.
5. **Given** a preparação concluída, **When** se lê o relatório, **Then** ele mostra contagens por partição, por
   fonte, por modo de fala, por nível de ruído e a cobertura de cada palavra do vocabulário.

---

### User Story 4 - Simular o ruído de largada (Priority: P2)

Cada exemplo de fala pode ser degradado de forma controlada: mistura com ruído urbano em três níveis de relação
sinal-ruído (leve, médio, extremo), variação de velocidade e variação de tom, simulando microfone de celular em
evento de rua.

**Why this priority**: não há gravação real de prova; a robustez inicial depende dessa simulação (risco temporário
declarado).

**Independent Test**: degradar um exemplo com nível de ruído conhecido e medir que a relação sinal-ruído obtida
corresponde à pedida, dentro da tolerância.

**Acceptance Scenarios**:

1. **Given** um exemplo e o nível "médio", **When** degradado, **Then** a relação sinal-ruído medida fica dentro da
   faixa definida para "médio".
2. **Given** um exemplo e uma variação de velocidade, **When** degradado, **Then** o rótulo (palavras e inteiro)
   permanece o mesmo.
3. **Given** um exemplo, **When** degradado, **Then** o áudio resultante não satura (sem picos cortados).

---

### Edge Cases

- **Zeros à esquerda**: "zero", "zero zero", "zero zero zero zero" → `0`; "zero sete" → `7`. Mais de quatro
  algarismos falados → rejeição.
- **"mil" sozinho** → `1000`; "um mil" → aceito como `1000` (forma rara, mas inequívoca).
- **"cem" × "cento"**: "cem" só vale para 100 exatos; "cento e um" para 101–199; "cem e um" → rejeição.
- **"e" extra após "mil"**: "mil e duzentos e trinta" (coloquial) é aceito como `1230`, além da forma normativa
  "mil duzentos e trinta".
- **Feminino**: "uma" e "duas" aceitos como 1 e 2; "duzentas" não pertence ao vocabulário.
- **"meia" fora do modo dígito a dígito** ("trinta e meia") → rejeição.
- **"quatorze" e "catorze"**: ambos → `14`.
- **Mistura de modos** ("dois trinta e três") → rejeição; agrupamento em pares ("doze trinta e quatro") está fora
  desta versão (ver Assumptions).
- **Clipe muito curto, silencioso ou corrompido** na fonte → descartado e contado no relatório, nunca rotulado.
- **Transcrição da fonte com palavras fora do vocabulário** → exemplo não entra como fala de número.
- **Ruído com voz humana audível de outro locutor** → permitido apenas no nível "extremo" e marcado no manifesto.

## Requirements *(mandatory)*

### Functional Requirements

**Vocabulário e gramática**

- **FR-001**: O sistema MUST definir um vocabulário fechado e versionado com as palavras: zero, um, uma, dois, duas,
  três, quatro, cinco, seis, meia, sete, oito, nove, dez, onze, doze, treze, quatorze, catorze, quinze, dezesseis,
  dezessete, dezoito, dezenove, vinte, trinta, quarenta, cinquenta, sessenta, setenta, oitenta, noventa, cem, cento,
  duzentos, trezentos, quatrocentos, quinhentos, seiscentos, setecentos, oitocentos, novecentos, mil, e — mais o
  símbolo "branco" exigido pelo treino CTC.
- **FR-002**: O sistema MUST converter uma sequência de palavras em um inteiro de 0 a 9999 ou em rejeição, aceitando
  o modo por extenso e o modo dígito a dígito (1 a 4 algarismos, com zeros à esquerda e "meia" = 6).
- **FR-003**: O sistema MUST gerar, para qualquer inteiro de 0 a 9999, todas as sequências de palavras válidas que o
  representam.
- **FR-004**: Nenhuma sequência aceita pela gramática pode corresponder a mais de um inteiro.
- **FR-005**: O sistema MUST aceitar uma lista de inscritos (inteiros de 0 a 9999) e, com ela ativa, rejeitar
  qualquer número fora da lista.
- **FR-006**: A gramática MUST poder ser exportada em forma utilizável pela decodificação do modelo (feature
  seguinte) sem reescrever suas regras.

**Fontes e licenças**

- **FR-007**: Toda fonte de áudio ou ruído MUST ser registrada com nome, origem, licença e data de obtenção; fontes
  sem licença registrada ou com licença não comercial MUST ser recusadas.
- **FR-008**: As falas de números de **treino** MUST vir de voz sintética gerada a partir da gramática, apenas com
  geradores de licença clara para uso comercial, degradada com ruído público licenciado. O recorte e a emenda de
  palavras do Common Voice ficam como fonte opcional, desligada nesta versão. As falas de **validação final e
  teste** vêm de gravação real do PO na rua, no microfone do celular, com termo de consentimento próprio; ficam só
  na máquina local e nunca entram no treino. *(Decisão do PO, 2026-10-03: opção C.)*
- **FR-009**: O conjunto de teste MUST conter voz humana real; enquanto não contiver, todo resultado medido nele
  MUST ser marcado como provisório.

**Preparação e degradação**

- **FR-010**: O sistema MUST produzir um manifesto em que cada exemplo tem identificador UUID, referência ao áudio,
  sequência de palavras, inteiro ou "sem número", modo de fala, fonte, licença, locutor pseudonimizado, partição e
  parâmetros de degradação.
- **FR-011**: O sistema MUST misturar ruído em três níveis nomeados de relação sinal-ruído (leve, médio, extremo),
  aplicar variação de velocidade e de tom, e manter o rótulo inalterado.
- **FR-012**: O sistema MUST incluir exemplos "sem número" (ruído puro e fala sem número) com alvo vazio.
- **FR-013**: O sistema MUST separar treino, validação e teste por locutor, sem locutor compartilhado entre
  partições.
- **FR-014**: Com a mesma semente, as mesmas fontes e a mesma versão do vocabulário, o resultado MUST ser idêntico.
- **FR-015**: O sistema MUST padronizar o áudio de saída (um canal, taxa de amostragem única definida no plano) e
  descartar, contando no relatório, clipes inválidos.
- **FR-016**: O sistema MUST gerar um relatório com contagens por partição, fonte, modo de fala, nível de ruído,
  cobertura de cada palavra do vocabulário e exemplos descartados com motivo.
- **FR-017**: O sistema MUST rodar tanto na máquina local (sem GPU) quanto em ambiente de nuvem descartável, com
  as mesmas entradas e saídas.

**Privacidade e guarda**

- **FR-018**: Áudio, manifestos com identificador de locutor e relatórios com dados de fonte MUST ficar fora do
  repositório Git.
- **FR-019**: O ambiente de nuvem MUST receber apenas dados públicos ou sintéticos; gravação de voz consentida só é
  processada localmente.
- **FR-020**: Identificadores de locutor das fontes MUST ser substituídos por pseudônimos estáveis antes de entrar
  no manifesto.

### Key Entities

- **Vocabulário**: lista fechada e versionada de palavras e o símbolo branco; cada palavra tem índice fixo.
- **Gramática**: conjunto de regras que define quais sequências de palavras formam um inteiro 0–9999 e qual inteiro;
  pode ser restringida por uma lista de inscritos.
- **Fonte**: origem de áudio de fala ou de ruído, com licença, procedência e data de obtenção.
- **Locutor**: pessoa (ou voz sintética) por trás de uma fala, conhecida só por pseudônimo; define a partição.
- **Exemplo**: um clipe preparado, com rótulo (palavras + inteiro ou "sem número"), fonte, locutor, partição e
  parâmetros de degradação; identificado por UUID.
- **Manifesto**: a lista completa de exemplos de uma execução, acompanhada da semente e da versão do vocabulário.
- **Relatório de preparação**: resumo quantitativo da execução e dos descartes.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% dos 10 000 inteiros (0–9999) têm ao menos uma forma por extenso e uma dígito a dígito, e cada
  forma gerada volta exatamente ao inteiro de origem.
- **SC-002**: Zero sequências aceitas correspondem a mais de um inteiro (verificação exaustiva).
- **SC-003**: Com lista de inscritos ativa, 0% de números fora da lista são aceitos em teste exaustivo.
- **SC-004**: Duas execuções com a mesma semente produzem resultados 100% idênticos.
- **SC-005**: Zero locutores compartilhados entre treino, validação e teste.
- **SC-006**: Toda palavra do vocabulário aparece em ao menos 1% dos exemplos de treino com fala.
- **SC-007**: Entre 10% e 20% dos exemplos de treino são "sem número".
- **SC-008**: A relação sinal-ruído medida de cada exemplo degradado fica a até 1 dB do nível pedido.
- **SC-009**: Zero arquivos de áudio ou manifestos com locutor no histórico do Git após a feature.
- **SC-010**: 100% das fontes usadas têm licença registrada e compatível com uso comercial.

## Assumptions

- Esta feature não treina nem exporta modelo; entrega gramática e dados para a feature seguinte (modelo CTC).
- Faixa de números: 0–9999. Peitos impressos com zeros ("007") não mudam nada: a saída é sempre o inteiro.
- Modos de fala em campo: por extenso e dígito a dígito, conforme o PO. Agrupamento em pares ("doze trinta e
  quatro") fica fora desta versão e pode entrar depois sem quebrar as regras existentes.
- Níveis de ruído iniciais (ajustáveis no plano): leve ≈ 20 dB, médio ≈ 10 dB, extremo ≈ 0 dB de relação
  sinal-ruído; velocidade entre 0,85× e 1,15×; tom até ±2 semitons.
- Fontes candidatas com licença comercial: Mozilla Common Voice pt (CC0) e MUSAN (CC BY 4.0, exige atribuição).
  Vozes sintéticas só entram após verificação da licença de cada voz, pois várias são não comerciais.
- O download de conjuntos de dados que exigem aceite de termos é feito pelo PO; a preparação lê arquivos locais.
- Áudio real de prova ainda não existe (risco temporário); quando existir, entra como fine-tuning e no conjunto de
  teste, sem mudar esta especificação.
- A máquina local não tem GPU NVIDIA; a preparação é viável em CPU, e o treino (feature seguinte) usa nuvem.
