# Benchmark results (strict, real)

## Language-model quality on fixed evaluation text

Lower bits-per-byte is better. BPB is tokenizer-independent, so
models with different vocabularies are compared fairly.

| Model | bits/byte | CE (nats) | tokens |
|---|---|---|---|
| llama-1b | 0.608 | 1.917 | 117 |
| qwen-0.5b | 0.656 | 2.086 | 116 |
| teacher-11m | 1.220 | 3.814 | 118 |
| nova-base | 1.287 | 4.023 | 118 |
| nova-soup | 1.291 | 4.035 | 118 |

## Agentic ability (same task suite, graded by ground truth)

`correct` is an independent check, not the model's own opinion.
`false pass` counts answers the loop accepted but that were wrong.

| Model | correct | accuracy | false passes |
|---|---|---|---|
| qwen-0.5b | 11/11 | 100% | 0 |
| llama-1b | 2/11 | 18% | 0 |
| nova-agent | 1/11 | 9% | 3 |
| nova-base | 0/11 | 0% | 10 |

