# Feature Specification: Modelo acústico CTC, treino, avaliação e exportação para o celular

**Feature Branch**: `002-modelo-ctc-onnx`

**Created**: 2026-10-03

**Status**: Draft

**Input**: User description: "Modelo acústico CTC leve, treino no Colab com dados sintéticos/públicos, decodificação
restrita pela gramática com rejeição, avaliação por número inteiro (inclusive na gravação real do PO, só local),
exportação ONNX INT8 < 5 MB para o Samsung A11, com o espectrograma dentro do modelo."

## Contexto

A feature 001 entrega a gramática e os dados preparados. Esta feature transforma esses dados no **motor em si**: um
modelo que ouve o aperto do botão e devolve o número de peito ou "não entendi", pequeno o bastante para o celular
mais fraco do campo. Integrar no app Android (SDK, botão, latência no aparelho) é a feature seguinte.

## User Scenarios & Testing *(mandatory)*

Atores: o **engenheiro de treino** (treina, compara e exporta modelos) e o **PO** (avalia com a própria gravação e
decide se o modelo vai para o app).

### User Story 1 - Treinar um modelo a partir do dataset preparado (Priority: P1)

O engenheiro aponta o manifesto da feature 001 e inicia o treino em ambiente de nuvem com GPU. O treino guarda
pontos de retomada; se a sessão cair, continua de onde parou. Ao final, o melhor modelo segundo a validação fica
salvo junto com o histórico do treino.

**Why this priority**: sem modelo treinado não há motor; tudo o mais depende disso.

**Independent Test**: treinar com um dataset pequeno por poucas épocas, interromper no meio, retomar, e verificar
que o treino continua da época seguinte e que a perda na validação cai em relação ao início.

**Acceptance Scenarios**:

1. **Given** um manifesto válido, **When** o treino roda, **Then** a cada época são registradas a perda e o acerto
   do número inteiro na validação.
2. **Given** um treino interrompido, **When** é reiniciado com a mesma pasta, **Then** retoma do último ponto salvo
   sem repetir épocas.
3. **Given** a validação sem melhora por um número configurado de épocas, **When** o treino segue, **Then** ele
   para e mantém o melhor modelo, não o último.
4. **Given** um manifesto com exemplos de voz real, **When** o treino é iniciado em nuvem, **Then** ele recusa
   começar.

---

### User Story 2 - Reconhecer um número com a garantia de nunca inventar (Priority: P1)

Dado um clipe de áudio, o motor devolve um inteiro válido da gramática (completa ou restrita aos inscritos) ou uma
rejeição. A rejeição acontece quando o melhor candidato tem confiança baixa ou quando dois candidatos ficam perto
demais para decidir com segurança.

**Why this priority**: é o comportamento que o operador sente em campo; um número errado é o pior resultado.

**Independent Test**: com um modelo treinado, decodificar a validação com gramática completa e com uma lista de
inscritos, e conferir que nenhuma saída fica fora da gramática ativa e que o limiar de rejeição pode ser ajustado.

**Acceptance Scenarios**:

1. **Given** um clipe de "duzentos e trinta e três" bem audível, **When** reconhecido, **Then** a saída é `233`.
2. **Given** inscritos {7, 233, 1500} e um clipe de "duzentos e trinta e quatro", **When** reconhecido, **Then** a
   saída é rejeição ou um número da lista, nunca `234`.
3. **Given** um clipe só de ruído, **When** reconhecido, **Then** a saída é rejeição.
4. **Given** o limiar de rejeição calibrado na validação, **When** aplicado, **Then** fica registrado junto do
   modelo exportado, para o app usar o mesmo valor.

---

### User Story 3 - Medir o modelo como o operador vai senti-lo (Priority: P1)

O engenheiro roda a avaliação e recebe um relatório com acerto do número inteiro, taxa de número errado, taxa de
rejeição, separados por nível de ruído, modo de fala (por extenso ou dígito a dígito) e quantidade de algarismos.
O PO roda a mesma avaliação **na própria máquina** sobre a gravação real, usando exatamente o modelo exportado que
iria para o celular.

**Why this priority**: o critério de pronto da constituição é medido, não prometido; e a medida que vale é a da
voz real.

**Independent Test**: avaliar um modelo no conjunto de validação e no conjunto de teste e conferir que o relatório
traz todas as quebras e marca como provisório o resultado sem voz real.

**Acceptance Scenarios**:

1. **Given** um modelo exportado e o manifesto, **When** a avaliação roda, **Then** o relatório traz as três taxas
   (acerto, número errado, rejeição) no total e por nível de ruído, modo de fala e quantidade de algarismos.
2. **Given** a partição de teste sem voz real, **When** avaliada, **Then** o relatório marca o resultado como
   provisório.
3. **Given** a gravação real do PO, **When** avaliada, **Then** a avaliação roda só localmente e usa o modelo
   exportado (o mesmo arquivo que iria ao celular), não o modelo de treino.
4. **Given** um erro (número diferente do falado), **When** listado no relatório, **Then** aparece com o número
   esperado, o devolvido e a confiança, para inspeção.

---

### User Story 4 - Exportar para o celular (Priority: P2)

O engenheiro exporta o melhor modelo em formato de celular, quantizado em 8 bits, com o cálculo do espectrograma
dentro do arquivo. A exportação confere que o modelo exportado reconhece igual ao modelo de treino e mede tamanho,
tempo de reconhecimento em processador de um núcleo e memória.

**Why this priority**: sem exportação o modelo não chega ao app; mas ela só faz sentido com um modelo bom (US1–US3).

**Independent Test**: exportar, rodar o exportado e o original sobre os mesmos clipes e comparar números
reconhecidos e taxas.

**Acceptance Scenarios**:

1. **Given** o melhor modelo, **When** exportado, **Then** o arquivo recebe áudio bruto (sem espectrograma calculado
   fora) e devolve as probabilidades por quadro.
2. **Given** o modelo exportado e quantizado, **When** comparado ao original na validação, **Then** a perda de acerto
   do número inteiro fica dentro do limite definido.
3. **Given** o modelo exportado, **When** medido, **Then** o relatório traz tamanho do arquivo, tempo por clipe com um
   núcleo e pico de memória.
4. **Given** o pacote exportado, **When** inspecionado, **Then** contém o modelo, a versão do vocabulário, o limiar de
   rejeição e o resumo das métricas, e nada disso vai para o repositório público.

---

### User Story 5 - Separar o ruído da gravação de teste (Priority: P2)

O PO vai gravar o teste real reproduzindo ruído em caixa de som. Os arquivos de ruído usados nessa reprodução saem de
um conjunto **reservado**, que nunca entra no treino; o sistema gera esse conjunto, incluindo faixas de "multidão"
(várias falas sobrepostas), e o exclui da preparação de dados.

**Why this priority**: se o mesmo ruído estiver no treino e no teste, o teste mede memória, não robustez.

**Independent Test**: gerar o conjunto reservado, rodar a preparação e conferir que nenhum arquivo reservado aparece
como ruído de treino ou validação.

**Acceptance Scenarios**:

1. **Given** as fontes de ruído registradas, **When** o conjunto reservado é gerado, **Then** ele lista os arquivos
   reservados e produz faixas de reprodução (incluindo multidão) com duração configurável.
2. **Given** o conjunto reservado, **When** a preparação de dados roda, **Then** nenhum arquivo reservado é usado.

---

### Edge Cases

- Clipe mais curto que a janela mínima do modelo ou totalmente silencioso → rejeição, sem erro de execução.
- Clipe muito longo (operador segurou o botão por muito tempo) → processado até um limite definido; acima disso,
  rejeição.
- Fala com número fora da lista de inscritos → rejeição, mesmo que o modelo "ouça" claramente o número.
- Fala com dois números ("duzentos e trinta e três, duzentos e trinta e quatro") → rejeição.
- Áudio em outra taxa de amostragem ou estéreo na avaliação local → convertido antes, como na preparação.
- Sessão de nuvem cai durante a gravação de um ponto de retomada → o ponto anterior continua válido.
- Manifesto de uma versão de vocabulário diferente da do modelo → recusa explícita.

## Requirements *(mandatory)*

### Functional Requirements

**Modelo e treino**

- **FR-001**: O modelo MUST emitir, a cada quadro de tempo, uma distribuição sobre os tokens do vocabulário da
  feature 001 mais o símbolo branco, treinado com alinhamento automático (CTC).
- **FR-002**: O modelo MUST ser leve o bastante para que a versão exportada e quantizada tenha **menos de 5 MB**.
- **FR-003**: O treino MUST ler o manifesto da feature 001, usar só as partições `train` e `val`, e recusar manifesto
  de outra versão de vocabulário.
- **FR-004**: O treino MUST usar otimização com decaimento de pesos, taxa de aprendizado com decaimento em cosseno e
  parada antecipada pela validação, mantendo o melhor modelo.
- **FR-005**: O treino MUST salvar pontos de retomada periódicos e retomar deles sem repetir épocas.
- **FR-006**: O treino em ambiente de nuvem MUST recusar manifesto que contenha voz real.
- **FR-007**: O treino MUST aceitar um modelo já treinado como ponto de partida (ajuste fino com áudio real futuro).
- **FR-008**: Execuções com a mesma semente e os mesmos dados MUST ser reproduzíveis dentro da tolerância do
  hardware, com semente, configuração e versões registradas.

**Reconhecimento**

- **FR-009**: A decodificação MUST ser restrita pela gramática exportada da feature 001 (completa ou só inscritos);
  nenhuma saída pode estar fora dela.
- **FR-010**: O reconhecimento MUST rejeitar quando a confiança do melhor candidato ficar abaixo do limiar ou quando
  a diferença para o segundo candidato for pequena demais; os dois limites são calibrados na validação.
- **FR-011**: O limiar calibrado MUST acompanhar o modelo exportado.

**Avaliação**

- **FR-012**: A avaliação MUST medir acerto do número inteiro, taxa de número errado e taxa de rejeição, no total e
  por nível de ruído, modo de fala e quantidade de algarismos, e listar os erros com esperado, devolvido e confiança.
- **FR-013**: A avaliação MUST poder rodar sobre o modelo exportado, na máquina local, sem dependências de GPU.
- **FR-014**: A avaliação da voz real MUST rodar só localmente; resultado sem voz real MUST ser marcado provisório.

**Exportação**

- **FR-015**: A exportação MUST incluir o cálculo do espectrograma dentro do modelo: entrada = áudio bruto 16 kHz.
- **FR-016**: A exportação MUST quantizar pesos e ativações em 8 bits com calibração em dados de validação.
- **FR-017**: A exportação MUST verificar a equivalência entre modelo exportado e original e medir tamanho, tempo por
  clipe com um núcleo e pico de memória.
- **FR-018**: Modelos, pesos e pontos de retomada MUST NOT entrar no repositório (ADR-006).

**Ruído reservado**

- **FR-019**: O sistema MUST gerar um conjunto reservado de ruído, com faixas de reprodução e de multidão, para a
  gravação de teste do PO, e a preparação de dados MUST excluí-lo.

### Key Entities

- **Configuração de treino**: semente, hiperparâmetros, caminho do manifesto, versão do vocabulário.
- **Ponto de retomada**: estado do modelo, do otimizador e do agendador, época, melhor métrica; local e privado.
- **Modelo exportado (pacote)**: arquivo do modelo, versão do vocabulário, limiar de rejeição, taxa de amostragem,
  resumo de métricas e de medições; privado.
- **Relatório de avaliação**: taxas totais e por recorte, lista de erros, marca de provisório, modelo avaliado.
- **Conjunto reservado de ruído**: lista de arquivos excluídos do treino e faixas de reprodução geradas.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Modelo exportado e quantizado com menos de 5 MB.
- **SC-002**: A quantização custa no máximo 1 ponto percentual de acerto do número inteiro na validação.
- **SC-003**: 0 saídas fora da gramática ativa em avaliação exaustiva da validação, com e sem lista de inscritos.
- **SC-004**: Na validação sintética (provisória): acerto do número inteiro ≥ 97% e número errado ≤ 0,3%.
- **SC-005**: Na gravação real do PO, com lista de inscritos: acerto ≥ 90%, **número errado ≤ 0,5%**, rejeição ≤ 10%.
  Na calibração, errar é pior que rejeitar: entre dois limiares que cumprem as metas, vence o de menor taxa de
  número errado. *(Decisão do PO, 2026-10-03, opção A: o viés é a segurança, nunca a adivinhação.)*
- **SC-006**: Treino retomado após interrupção perde no máximo uma época de trabalho.
- **SC-007**: Clipes de ruído puro da validação: 95% ou mais rejeitados.
- **SC-008**: Nenhum arquivo do conjunto reservado de ruído aparece em exemplos de treino ou validação.
- **SC-009**: Tempo por clipe de 3 s com um núcleo de processador de computador fica registrado como indicador; o
  valor no aparelho é medido na feature do SDK Android.

## Assumptions

- Reconhecimento **não contínuo**: o áudio inteiro do aperto é processado ao soltar o botão (Push-to-Talk), o que
  simplifica o modelo e a decodificação.
- Duração máxima de um aperto: 8 segundos; acima disso, rejeição.
- O treino roda na nuvem só com dados sintéticos e públicos; a máquina local não tem GPU e só executa o modelo
  exportado (avaliação e medição).
- A arquitetura exata (quantidade de blocos, canais, com ou sem camada recorrente) é escolhida no plano por medição,
  dentro do limite de tamanho.
- O conjunto reservado de ruído usa cerca de 10% dos arquivos de ruído e de fala do MUSAN; as faixas de multidão são
  misturas de várias falas reservadas.
- A integração no app (SDK Android, botão, latência e memória no Samsung A11) é a feature 003.
