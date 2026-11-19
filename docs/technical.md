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

## Sessions

Every question about one CT repeats the whole prompt, and the tokens up to the end
of the CT (13,824 image tokens, after any tool block) are the same each time. A
`bfloat16_fast` session keeps the decoder state at a boundary inside those tokens
and decodes only what follows it.

| `prefill_reuse` | State kept at | Decoded again per question | Kept MLX memory |
| :--- | :--- | :--- | ---: |
| `exact` (default) | A multiple of 2,048 tokens, more than 2,560 before the end of the CT | 2,561–4,608 CT tokens and the question | 0.72 GB |
| `fast` | The last multiple of 64 tokens before the question | The question, 20–340 tokens | 0.96 GB |

The boundary decides whether two decoder calls repeat the arithmetic of one call
over the whole prompt. MLX 0.32.4 chooses its Metal matmul kernel from the row
count, the feed-forward runs in 2,048-row chunks, and the gated-delta recurrence
runs in 64-token chunks, 16 at a time, with a cumulative-sum partition that the
source derives from the whole sequence's chunk count (a continuation is told how
many tokens precede it). At the exact boundary every operation meets the kernel
and chunking it has in the one-call prefill. A short remainder takes the small-row
kernels, whose products round differently in rare elements.

Apple M5 Max, MLX 0.32.4:

- **Exact boundary.** On 15 prompts over two CTs (reports, yes/no questions,
  thinking, history, and tool blocks that start the CT 277–2,377 tokens into the
  prompt; 2,601–5,122 tokens decoded after the boundary), the logits were
  bit-identical to the one-call prefill, and so was the whole decoder cache on
  the 10 prompts where it was compared. On the reference CT above, a one-shot
  run and sessions in both modes produced the same 208 tokens, the report that
  is token-identical to NVIDIA's implementation.
- **Fast boundary.** Maximum logit difference 0.04–0.19 against the one-call
  prefill. Over 303 prompts on 22 CT series, 300 gave the tokens of the exact
  mode: all 72 yes/no and left/right choices and all 176 constrained tool calls,
  33 of 34 short answers written from a tool result, and 19 of 21 free-text
  reports. In the two reports the wording diverged after a near-tie token
  ("liver masses" against "liver foci"). The first call of a fast session is
  split the same way, so its report can differ from a one-shot run.
- Peak MLX memory was 11.5 GB for an exact session and 12.2 GB for a fast one,
  against 12.19 GB one-shot.
- Time per question in a loaded session after its first report, on the
  reference CT. Each variant ran in six fresh processes, interleaved with the
  others, and timed five calls per process after two untimed ones; the values
  are the median and range over the processes.

  | Question | 0.7.0 | `exact` | `fast` |
  | :--- | ---: | ---: | ---: |
  | `Yes`/`No` answer | 7.7 s (7.3–8.0) | 2.4 s (2.3–2.8) | 0.23 s (0.21–0.24) |
  | Tool call chosen among ten | 9.7 s (9.2–10.7) | 3.1 s (3.0–3.7) | 0.64 s (0.64–0.70) |

  The first question after the declared tools change rebuilds the kept state
  and took 9.0–11.2 s in both modes.
- Apple M1 Max (64 GB), MLX 0.32.4: the exact boundary was also bit-identical to
  the one-call prefill, in the logits and the whole decoder cache, on 18 prompts
  over two CTs (reports, yes/no questions, thinking, history, and tool blocks
  that start the CT 260–1,891 tokens into the prompt; 3,445–4,063 tokens decoded
  after the boundary). The reference CT gave the same 208-token report one-shot
  and in an exact session. Bit-identity has not been checked on other Apple GPU
  generations, which may dispatch matmuls differently.

## Constrained answers

`response_choices` and `tool_choice` restrict greedy decoding to a finite set of
strings with LLGuidance. Each string is a literal, so the model may spell it
with any tokens; only the checkpoint's special tokens, which include the tool
tags, are fixed token rules. This matters where the tokenizer merges across a
boundary: `<function=liver_lesions>` begins `=l`, `iver` but
`<function=lung_nodules>` begins `=`, `lung`. Restricting each choice to its
canonical tokens made `=l` commit to `liver_lesions`, and with those tool names
both lung-nodule questions went to the liver tool in 34 of 34 trials; as
literals, `=l` leaves both names open and the next token decides.

The model is consulted only where the permitted strings diverge. Text they all
share is emitted in its canonical tokens from the grammar, and decoded only when
a later decision needs its state; once one string remains, the rest of it and
the end of the turn are emitted without the model.

| Answer | Tokens emitted | Tokens decoded |
| :--- | ---: | ---: |
| `Yes` / `No` | 2 | 0 |
| Tool call without arguments, among 10 permitted calls | 19 | 4–5 |
| Tool call with one `enum` argument, among 10 permitted calls | 28–30 | 16–17 |

`choice_probability` is the product, over the tokens the model chose, of the
softmax across the legal tokens. Selecting the call with the greatest
unconstrained likelihood of its whole text was tested as an alternative and
rejected: it chose the expected call in 51 of 133 cases, because every extra
argument token costs likelihood.

## Answers against ground truth

Greedy decoding, thinking off, `bfloat16_fast`, exact session mode. Chest: 10
LUNA16 CTs, 6 with a consensus nodule of 10–25 mm and 4 with no consensus nodule
of 3 mm or more. Abdomen: 6 HCC-TACE-Seg patients with a segmented liver tumour
(4–1,608 ml), each as a contrast-enhanced and an unenhanced series (liver
43–59 HU). The counts are too small for accuracy estimates; they show where the
answers can be relied on.

| Question or output | With the finding | Without it |
| :--- | ---: | ---: |
| Chest report labels include `Lung nodule` or `Lung mass` | 6 of 6 | 2 of 4 without |
| "Is a lung nodule present?" (`Yes`/`No`) | 5 of 6 Yes | 3 of 4 No |
| "Which lung contains the largest nodule?" (`Left`/`Right`) | 5 of 6 correct | |
| "Was intravenous contrast administered?" (`Yes`/`No`) | 6 of 6 Yes | 3 of 6 No |
| Abdomen report technique line | 6 of 6 "IV contrast CT" | 1 of 6 "Non-contrast CT" |
| "Is a liver lesion present?" (`Yes`/`No`), enhanced series | 6 of 6 Yes | |
| "Is a liver lesion present?" (`Yes`/`No`), unenhanced series | 5 of 6 Yes | |

**Tool selection.** Three tools were declared (`detect_lung_nodules`,
`measure_liver_lesions`, and `segment_organ` with eight organs), giving ten
permitted calls, and eight questions were asked of each of the 22 series with
`tool_choice="required"`. Unconstrained, the checkpoint requested no tool in
any prompt tried, with thinking on or off.

| Question | Expected call selected |
| :--- | ---: |
| "What is the diameter of the largest lung nodule?", "How many lung nodules are there?" | 44 of 44 |
| "What is the volume of the left lung / liver / right kidney?" | 64 of 66 |
| "Is the heart enlarged? Measure it.", "Is the spleen enlarged? Measure it." | 31 of 44 |
| "How large is the largest liver lesion?" | 10 of 22 |

Chest series: 77 of 80. Abdominal series: 72 of 96; there the liver-lesion
question went to `detect_lung_nodules` in 10 of 12 series, and the heart and
spleen questions went to `segment_organ(liver)` in 9 of 24 and to
`detect_lung_nodules` in 3.

The order of the declarations decides more than the names do. The same eight
questions on the same 22 series, with the tools renamed and with the original
names reordered:

| Tools, in declared order | Chest | Abdomen | Lung-nodule questions | Liver-lesion question |
| :--- | ---: | ---: | ---: | ---: |
| `detect_lung_nodules`, `measure_liver_lesions`, `segment_organ` | 77 of 80 | 72 of 96 | 44 of 44 | 10 of 22 |
| `lung_nodules`, `liver_lesions`, `organ_volume` | 72 of 80 | 80 of 96 | 44 of 44 | 4 of 22 |
| `measure_liver_lesions`, `segment_organ`, `detect_lung_nodules` | 59 of 80 | 74 of 96 | 28 of 44 | 22 of 22 |

A question about a lesion tends to go to whichever lesion tool is declared
first: the liver-lesion question went to the nodule tool in 12 and 18 of 22
series when that tool came first, and the nodule questions went to the liver
tool in 16 of 44 when it came first. `choice_probability` does not flag these
calls: for the first declaration its median was 0.995 when the call was the
expected one and 0.90 when it was not, with 13 of the 27 unexpected calls above
0.9, and for the second the 24 unexpected calls had a median of 0.996.

Two changes to the request improved selection:

- **Name the measurement.** With the first declaration, "What is the volume of
  the heart?" and "What is the volume of the spleen?" selected the expected call
  in 44 of 44, against 31 of 44 for the "Is the … enlarged? Measure it." forms.
- **Declare one lesion tool.** With `measure_liver_lesions` and `segment_organ`
  declared for the abdominal series, the liver-lesion question selected the
  liver tool in 12 of 12. With `detect_lung_nodules` and `segment_organ`
  declared for the chest series, 89 of 90 calls over nine questions were the
  expected ones.

The indirect heart and spleen questions stayed unreliable on the abdominal
series under the two-tool declaration (7 of 24, most going to
`segment_organ(liver)`), and the spleen-volume question selected the liver in 3
of 12 there.

**Using a tool result.** With the measured value returned as the tool result,
the answer quoted the liver volume in 12 of 12 series (once rounded to the
millilitre) and the lesion diameter in 9 of 12. For the largest lung nodule it
quoted the diameter in 2 of 6 and otherwise answered in categories
("moderate-sized"); asking it to "report the measured diameter of each lung
nodule" raised that to 5 of 6. With an empty detection result it reported no
nodule or a diameter of zero in 4 of 4. With thinking enabled it declined to
answer from a tool result in the one conversation tried, so keep thinking off
in tool turns.
