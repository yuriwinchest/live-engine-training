# Contrato — pacote do modelo (consumido pelo SDK Android, feature 003)

Pasta privada (nunca no Git):

```text
live-model-<data>-<sha8>/
├── model.int8.onnx      # entrada "audio" float32 [1, amostras] 16 kHz; saída "log_probs" [1, quadros, 45]
├── model.fp32.onnx      # referência para equivalência
├── grammar.json         # trie completa (contrato da feature 001)
├── meta.json
└── report.md            # avaliação e medições
```

`meta.json`:

```json
{
  "format": "live-model",
  "format_version": 1,
  "vocab_version": "1",
  "tokens": ["<blank>", "zero", "..."],
  "sample_rate": 16000,
  "frame_ms": 20,
  "max_seconds": 8.0,
  "thresholds": {"min_posterior": 0.0, "min_margin": 0.0, "min_adherence": 0.0},
  "beam_width": 8,
  "sha256": {"model.int8.onnx": "…", "grammar.json": "…"},
  "metrics": {"val": {}, "test": {}, "provisional": true},
  "bench": {"size_bytes": 0, "latency_ms_3s_1thread": 0.0, "peak_rss_mb": 0.0}
}
```

- O app gera a trie restrita aos inscritos com a mesma regra da feature 001; o decodificador é o de `decode.py`
  portado, com os mesmos limiares.
- Saída `log_probs` já em log-softmax; o índice 0 é o branco.
