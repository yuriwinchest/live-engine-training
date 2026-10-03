# Contrato — exportação da gramática (trie JSON)

Consumidor: decodificação CTC restrita (feature seguinte, treino e Android).

```json
{
  "format": "live-grammar-trie",
  "format_version": 1,
  "vocab_version": "1",
  "tokens": ["<blank>", "zero", "um", "..."],
  "restricted": false,
  "numbers": 10000,
  "nodes": [
    {"next": {"5": 1, "12": 2}, "value": null},
    {"next": {}, "value": 0}
  ]
}
```

- `nodes[0]` é a raiz. `next` mapeia **índice do token** (como string, exigência do JSON) → índice do nó.
- `value` não nulo marca fim de fala válida com aquele inteiro. Um nó pode ter `value` e `next` ao mesmo tempo
  ("dois" = 2 e prefixo de "dois mil").
- `restricted: true` quando gerada a partir de uma lista de inscritos; `numbers` = quantidade de inteiros aceitos.
- Garantia: todo caminho da raiz até um nó com `value` é uma fala aceita por `Grammar.parse`, e vice-versa.
- O índice 0 (`<blank>`) nunca aparece em `next`.
