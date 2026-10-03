# Contrato — CLI (acréscimos da 002)

No Windows: `uv run python -m live_lab …`.

| Comando | Faz |
|---|---|
| `noise holdout --registry R --out DIR [--tracks 6 --minutes 3 --seed S]` | lista reservados, gera faixas de multidão/rua e `LEIA-ME.md` |
| `train --manifest M --out DIR --seed S [--epochs --init-from P]` | treina; retoma se `DIR/last.pt` existir; recusa voz real na nuvem |
| `export --checkpoint DIR/best.pt --manifest M --out PKG` | ONNX FP32 → INT8, equivalência, calibração de limiares, medição, pacote |
| `evaluate --package PKG --manifest M --split val\|test [--inscritos F]` | avalia o ONNX INT8; `test` com voz real só local |
