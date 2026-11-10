# Technical notes

## Precision profiles

NV-Reason-CT is a BF16 model. Both profiles hold NVIDIA's BF16 weights exactly;
the published bundle stores them as FP32 only because it was converted that way.

| Profile | CT encoding and prompt | Generated tokens |
| :--- | :--- | :--- |
| `bfloat16_fast` (default) | BF16 matmuls with FP32 accumulation, product rounded to BF16 | FP32 arithmetic on BF16 weights (`bf16_gemv`); FP32 norms, recurrence, attention and caches |
| `bfloat16` | NVIDIA's BF16 casts and cache boundaries, reproduced by custom Metal kernels | Same source BF16 boundaries |

Version 0.6.0 removed the `float32` and `bfloat16_fp32` profiles. They read the
same BF16 weights; FP32 matmuls on the CT and prompt took about twice as long
and produced the same report on the measured CT.

## Measurements

Apple M5 Max (128 GB), MLX 0.32.4, LUNA16 chest CT
`1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260`, default
structured chest report (13,844 prompt tokens), greedy decoding.

| Measurement | `float32` (0.4.x) | `bfloat16_fast` |
| :--- | ---: | ---: |
| Prompt prefill, warm, 6 paired rounds | 11.89 s | 6.18 s |
| Full report, `elapsed_s` (load, encode, prefill, 208 tokens) | 17.9 s | 12.4 s |
| Peak MLX memory | 12.13 GB | 12.19 GB |

- The two profiles generated the same 208 tokens and the same `report.txt`.
- An abdominal report (`anatomy_region="abdomen"`, whole-body CT, 321 tokens)
  peaked at 12.19 GB, the same as the chest report: every scan is resampled to
  one 192³ grid, so the peak does not grow with the input.
- The source-exact `bfloat16` profile peaked at 12.46 GB; a three-token answer
  took 380 s, almost all of it in its emulated prefill.
- Generation runs at 21.3–21.6 ms per token at this context in both profiles,
  which share the decode path. Each token reads about 9.5 GB (7.2 GB of BF16
  decoder weights, 1.27 GB of LM head, 0.9 GB of FP32 KV cache), about
  446 GB/s, so decoding is bound by memory bandwidth.
