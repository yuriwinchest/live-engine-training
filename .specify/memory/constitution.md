<!--
Sync Impact Report
- Versão: modelo vazio → 1.0.0 (primeira ratificação)
- Princípios definidos: I. Contrato estreito · II. Nunca inventar número · III. Paridade treino/aparelho ·
  IV. Dados e privacidade · V. Medir, não prometer · VI. Engenharia sustentável
- Seções adicionadas: Restrições técnicas; Fluxo de trabalho e portões de qualidade; Governança
- Seções removidas: nenhuma
- Templates: lidos em tempo de execução; nenhum alterado (escopo restrito à constituição)
- TODOs: nenhum
-->

# Constituição do L.I.V.E. (Largada Intent Voice Engine)

## Core Principles

### I. Contrato estreito

Entra áudio de uma fala capturada por Push-to-Talk; sai um inteiro de 0 a 9999 **ou** uma rejeição
explícita ("não entendi"). O motor MUST funcionar offline e MUST NOT conhecer banco de dados, tela,
inscrição ou regra de corrida. Qualquer conhecimento de prova chega de fora, como restrição de gramática.

**Por quê:** um contrato pequeno é testável isoladamente e troca de app sem reescrita.

### II. Nunca inventar número

A decodificação MUST ser restrita por uma gramática de números válidos sobre a saída CTC do modelo
acústico. A gramática aceita o número por extenso ("duzentos e trinta e três") e dígito a dígito
("dois três três"), incluindo zeros à esquerda ("zero zero sete"), sempre normalizados para o inteiro
(`7`). O app MAY restringir a gramática aos números inscritos na prova. Fala ambígua, incompleta ou
sem número MUST resultar em rejeição, nunca no número "mais provável" fora da gramática.
Classificador de palavra única (KWS de um rótulo por clipe) está descartado (ADR-001).

**Por quê:** na cronometragem, um número errado é pior que nenhum: o operador corrige um "não entendi",
mas não percebe um "8" no lugar de "248".

### III. Paridade treino/aparelho

O cálculo do log-Mel MUST estar dentro do grafo exportado; treino e celular executam o mesmo cálculo.
O formato padrão é ONNX com quantização INT8. TFLite só entra se uma medição no aparelho justificar.
O modelo MUST caber em menos de 5 MB e rodar no Samsung A11 (aparelho de referência mais fraco);
a arquitetura é leve (convoluções separáveis em profundidade, família MobileNet/TC-ResNet).

**Por quê:** elimina a classe de bug "treinou com um espectrograma, roda com outro" e respeita a
memória limitada que já encerrou o LGDA em campo.

### IV. Dados e privacidade (LGPD)

- Áudio de pessoas MUST NOT entrar no Git; voz própria só com consentimento registrado.
- Ambiente de nuvem (ex.: Google Colab) MUST receber apenas dados públicos ou sintéticos.
- Toda fonte de dados MUST ter licença registrada e compatível com o licenciamento comercial
  futuro do modelo; licenças não comerciais (NC) não entram no treino.
- Pesos, checkpoints, modelos exportados, datasets e listas de locutores MUST NOT ser publicados (ADR-006).

**Por quê:** voz é dado pessoal, e o modelo treinado é o ativo comercial do projeto.

### V. Medir, não prometer

Métricas (acerto do número inteiro, taxa de rejeição, latência, tamanho e pico de RAM) MUST ser
medidas em um conjunto de teste separado do treino, com separação por locutor. A avaliação final
MUST incluir voz humana real; resultado medido só com voz sintética é provisório e assim declarado.
Nenhum "teste feito" é registrado sem execução real; valores-alvo são definidos pelo PO.

**Por quê:** modelo que só viu a própria fonte de dados mede a si mesmo, não o campo.

### VI. Engenharia sustentável

- Nenhum arquivo de código com mais de 500 linhas; uma responsabilidade por módulo, testável isoladamente.
- Código Python tipado; dependências fixadas; Python 3.11 ou 3.12 em ambiente virtual próprio.
- Persistência ou log com identificador usa apenas UUID.
- Processamento de dados MUST ser determinístico dado uma semente registrada (reprodutível).

**Por quê:** treino que não se reproduz não se depura.

## Restrições técnicas

- Domínio: corrida de rua, números de peito 0–9999 em português do Brasil.
- Captura: Push-to-Talk; o horário da passagem é o do aperto, não o do fim da decodificação (ADR-002).
- Treino em GPU de nuvem é permitido nas condições do Princípio IV; a máquina local não tem GPU NVIDIA.
- Ausência de áudio real de prova é risco declarado; a chegada dele leva a fine-tuning, não a novo desenho.

## Fluxo de trabalho e portões de qualidade

- Operação TONE: fase declarada (A construir, B homologar, C auditar); a Fase C só com gatilho do PO.
- Spec Kit: constituição → especificação → esclarecimento → plano → tarefas → implementação.
- Toda alteração é registrada no `HANDOFF.md` (data, autor, pedido, arquivos, validação real, riscos).
- Build, testes automáticos e métricas não substituem a homologação do PO.

## Governance

Esta constituição prevalece sobre preferências de implementação. Mudar uma decisão de arquitetura exige
nova ADR em `docs/ARQUITETURA.md` aprovada pelo PO (Yuri) antes do código. Emendas seguem versionamento
semântico: MAJOR para remover ou redefinir princípio, MINOR para princípio ou seção nova, PATCH para
redação. Toda especificação e plano verifica conformidade com estes princípios; exceção só por escrito
do PO, com risco nomeado e prazo.

**Version**: 1.0.0 | **Ratified**: 2026-10-03 | **Last Amended**: 2026-10-03
