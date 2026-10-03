# L.I.V.E. — Largada Intent Voice Engine 🎙️🏃

> Motor de voz offline que transforma **"duzentos e trinta e três"** em **`233`** — no meio da rua, com música, apito e multidão em volta.

![status](https://img.shields.io/badge/status-em%20desenvolvimento-orange)
![offline](https://img.shields.io/badge/edge%20AI-100%25%20offline-2ea44f)
![python](https://img.shields.io/badge/lab-Python%203.11%2B%20%C2%B7%20PyTorch-3776ab)
![android](https://img.shields.io/badge/SDK-Kotlin%20%C2%B7%20ONNX%20Runtime-7f52ff)

O **L.I.V.E.** é o motor de reconhecimento de voz do **LGDA**, o app de cronometragem de corridas de rua da
plataforma **Largada Brasil**. O fiscal na linha de chegada fala o número de peito do atleta; o motor devolve o
número inteiro e o horário exato da passagem. Sem internet, em celulares Android de entrada.

---

## 🧭 O problema

Testamos os motores prontos no campo. O resultado (diagnóstico real de um Galaxy A56, 2026-10-02):

| Motor | O que aconteceu na rua |
|---|---|
| Reconhecedor do Google no Android | **8 de 11 falas** voltaram sem texto nenhum |
| Vosk (modelo pequeno PT) | número partido: "duzentos e trinta e **seis**" → `6` |

Motores genéricos foram feitos para **transcrever qualquer frase** e precisam adivinhar sozinhos **quando a fala
começa e termina**. Na rua, essa adivinhação quebra. O nosso problema é muito mais estreito: **só números de peito
de 0 a 9999**. Um problema estreito merece um motor feito para ele.

## 💡 A abordagem

```
 [ segura o botão ]                                                        [ solta ]
        │                                                                      │
  microfone 16 kHz ──► log-Mel (dentro do modelo) ──► rede acústica ──► CTC ──► gramática ──► 233
        │                                                                                     │
  horário = instante do aperto                                    "não entendi" se não for número válido
```

1. **Push-to-Talk.** O fiscal segura um botão (tela, tecla de volume ou controle Bluetooth) enquanto fala. Começo e
   fim da fala deixam de ser adivinhados, e conversa em volta não entra.
2. **Rede pequena que lê sequências.** "233" é falado como uma *sequência* — `duzentos · e · trinta · e · três`. A
   rede emite tokens de um vocabulário de ~45 palavras numéricas, treinada com **CTC** (aprende o alinhamento
   áudio↔palavras sem marcação manual).
3. **Gramática de números válidos.** A decodificação só aceita sequências que formam um número bem formado, por
   extenso ("duzentos e trinta e três") ou dígito a dígito ("dois três três", com o nosso "**meia**" = 6).
4. **Robustez vem dos dados.** Fala limpa misturada com ruído urbano em vários níveis de SNR, variações de
   velocidade e tom — e, em produção, clipes reais confirmados pelos fiscais (com consentimento).
5. **Edge AI.** Modelo exportado em ONNX, com o espectrograma *dentro* do grafo: o celular calcula exatamente como o
   treino.

As decisões e o que foi descartado estão em [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md).

---

## 🏗️ Estrutura

```text
live-engine-training/
├── training/        # Laboratório de ML (Python · PyTorch · torchaudio)
│   ├── scripts/     # Preparação de dados e augmentation (ruído, SNR, stretch, pitch)
│   ├── live/        # Pacote: vocabulário, gramática, modelo, treino, avaliação
│   └── data/        # Datasets locais (fora do Git)
├── android/         # SDK de inferência (Kotlin · ONNX Runtime)
├── docs/            # Arquitetura e decisões
└── .specify/        # Especificações (Spec-Driven Development com GitHub Spec Kit)
```

> As pastas são criadas conforme cada etapa é implementada — este repositório mostra o projeto sendo construído.

## 🗺️ Roteiro

- [x] Diagnóstico em campo dos motores prontos
- [x] Arquitetura: Push-to-Talk + sequência CTC + gramática
- [ ] Vocabulário e gramática de números (0–9999, por extenso e dígito a dígito)
- [ ] Laboratório de dados: injeção de ruído com SNR controlado, *time stretch*, *pitch shift*
- [ ] Modelo acústico compacto + treino + avaliação por número inteiro
- [ ] Exportação ONNX com espectrograma embutido
- [ ] SDK Android: `LiveEngine.start()` · `onNumber(Int, instante)`
- [ ] Integração no app LGDA e coleta de clipes reais (com consentimento)
- [ ] Ajuste à voz de cada fiscal

## 🛡️ Padrões de engenharia

1. **Regra dos 500** — nenhum arquivo de código passa de 500 linhas; responsabilidades quebradas em módulos.
2. **SOLID estrito** — o SDK não conhece banco, tela nem regra de corrida: entra áudio, sai número.
3. **UUID** — qualquer registro ou log do motor usa identificador universal, nunca sequencial.
4. **Medir antes de prometer** — acerto, latência, tamanho e memória são medidos no aparelho mais fraco de campo.
5. **Dados de voz são pessoais** — nenhum áudio de pessoa entra no repositório.

## ⚙️ Ambiente (laboratório)

```bash
cd training
python -m venv .venv          # Python 3.11 ou 3.12
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

---

## 👤 Autor

**Yuri Winchester** — desenvolvedor full stack (React · Python · PHP · Kotlin), criador da plataforma Largada Brasil.
Projeto desenvolvido com engenharia assistida por IA (Claude Code, Gemini, Codex) sob especificação e homologação
do autor.

## 📄 Licença

© 2026 Yuri Winchester. **Todos os direitos reservados.** O código está público para leitura e avaliação; uso,
cópia ou redistribuição exigem autorização do autor. Veja [`LICENSE`](LICENSE).
