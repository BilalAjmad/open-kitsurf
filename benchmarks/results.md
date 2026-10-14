# Benchmark results (strict, real)

## Language-model quality on fixed evaluation text

Lower bits-per-byte is better. BPB is tokenizer-independent, so
models with different vocabularies are compared fairly.

| Model | bits/byte | CE (nats) | tokens |
|---|---|---|---|
| llama3.2-1b | 0.608 | 1.917 | 117 |
| qwen2.5-0.5b | 0.656 | 2.086 | 116 |
| qwen3-0.6b | 0.732 | 2.325 | 116 |
| gemma3-1b | 0.742 | 2.339 | 117 |
| deepseek-r1-1.5b | 0.997 | 3.141 | 117 |
| teacher-11m | 1.220 | 3.814 | 118 |
| nova-base | 1.287 | 4.023 | 118 |
| nova-soup | 1.291 | 4.035 | 118 |
| chimera-base | 1.295 | 4.048 | 118 |
| ageis-soup | 1.359 | 4.247 | 118 |
| ageis-base | 1.371 | 4.285 | 118 |

## Agentic ability (same task suite, graded by ground truth)

`correct` is an independent check, not the model's own opinion.
`false pass` counts answers the loop accepted but that were wrong.

| Model | correct | accuracy | false passes |
|---|---|---|---|
| qwen2.5-0.5b | 11/11 | 100% | 0 |
| qwen3-0.6b | 11/11 | 100% | 0 |
| deepseek-r1-1.5b | 8/11 | 73% | 2 |
| gemma3-1b | 5/11 | 45% | 3 |
| llama3.2-1b | 2/11 | 18% | 0 |
| ageis-agent | 1/11 | 9% | 3 |
| chimera-agent | 1/11 | 9% | 1 |
| nova-agent | 1/11 | 9% | 3 |
| nova-base | 0/11 | 0% | 10 |

