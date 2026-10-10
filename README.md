# Task 4: Speech-to-Reasoning Pipeline (Whisper + Unsloth Dynamic 4-bit LLM)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/codewithUswaFatima/whisper-quantized-reasoning/blob/main/Whisper_Quantized_Reasoning.ipynb)

A pipeline that takes **spoken audio**, transcribes it with **OpenAI Whisper**, and passes the text to an **Unsloth dynamic 4-bit reasoning model (Qwen3-4B)**, which thinks step by step and returns an answer. Everything runs in one Google Colab notebook on a free **Tesla T4** GPU.

> Internship project for ArchTechnologies  
> Author: **[Your Full Name]** | Domain: **[Your Internship Domain]** | Email: **[your.email@example.com]**  
> GitHub: [@codewithUswaFatima](https://github.com/codewithUswaFatima)

---

## What this project does

1. **Prepares audio:** three spoken test questions are generated with gTTS. You can also use your own recording (mp3, wav, m4a, or a WhatsApp voice note).
2. **Stage 1, speech recognition:** `openai/whisper-small` (fp16) transcribes the audio. Audio is decoded and resampled to 16 kHz mono, and clips are processed in batches with 30-second chunking.
3. **Memory hand-off:** Whisper is deleted and the CUDA cache is cleared before the LLM is used.
4. **Stage 2, reasoning:** `unsloth/Qwen3-4B-unsloth-bnb-4bit` (dynamic 4-bit, thinking mode) receives the transcripts, reasons inside `<think>` tags and gives a final answer. Prompts are left-padded and generated in batches.
5. **Output:** the reasoning trace and the final answer are separated and printed for each audio file.

## Architecture

```
Audio (mp3 / wav / mp4)
      |
Resample to 16 kHz mono (librosa / ffmpeg)
      |
Whisper-small (fp16, batched, 30 s chunks) --> Transcript
      |   (Whisper deleted, GPU cache cleared)
      v
Chat template + system prompt --> left-padded batch
      |
Qwen3-4B, Unsloth dynamic 4-bit, thinking mode
      |
<think> reasoning </think> + final answer
```

## Tech stack

| Component | Choice |
|---|---|
| Speech recognition | `openai/whisper-small` (Hugging Face pipeline, fp16) |
| Reasoning LLM | `unsloth/Qwen3-4B-unsloth-bnb-4bit` |
| LLM engine | Unsloth `FastLanguageModel` |
| Audio handling | `librosa`, `ffmpeg` |
| Sample audio | `gTTS` |
| Hardware | Google Colab, Tesla T4 (14.56 GB VRAM) |

## How to run

1. Click the **Open in Colab** badge above (or open `Whisper_Quantized_Reasoning.ipynb` from this repository in Google Colab).
2. Select `Runtime > Change runtime type > T4 GPU`.
3. Run Section 1 to Section 7 from top to bottom. Unsloth is imported before transformers, which is required.
4. Section 8 lets you try **your own audio** (see below).

### Using your own voice

**Upload a recording.** Run the upload cell and choose your file. mp3, wav and m4a work directly.

**WhatsApp voice notes** are saved as `.mp4`, which `librosa` cannot read (it raises `LibsndfileError`). Convert them first:

```python
import glob, subprocess
src = glob.glob("/content/audio/WhatsApp*")[0]
dst = "/content/audio/my_voice.wav"
subprocess.run(["ffmpeg", "-y", "-i", src, "-vn", "-ar", "16000", "-ac", "1", dst,
                "-loglevel", "error"], check=True)

res = speech_to_reasoning([dst])[0]
print("TRANSCRIPT:", res["transcript"])
print("ANSWER:", res["answer"])
```

If an earlier cell failed while Whisper was loaded, delete it first (`del asr`) and run `torch.cuda.empty_cache()`.

For Urdu, change `"language": "english"` to `"urdu"` in the `transcribe()` function.

## Notebook structure

| Section | Content |
|---|---|
| 1 | Install dependencies |
| 2 | Imports, GPU check, `vram()` and `free_gpu()` helpers |
| 3 | Generate the sample audio queries |
| 4 | Stage 1: Whisper transcription |
| 4b | Free Whisper before loading the LLM |
| 5 | Stage 2: load the dynamic 4-bit Qwen3-4B |
| 6 | Batched reasoning function `reason()` |
| 7 | End-to-end demo |
| 7b | Wrapper `speech_to_reasoning(paths)` |
| 8 | Your own audio |
| 9 | Memory summary |

## Key settings

| Setting | Value | Reason |
|---|---|---|
| Whisper model | `whisper-small`, fp16 | Fast and only about 0.46 GB |
| `chunk_length_s` | 30 | Handles audio longer than 30 seconds |
| Whisper `batch_size` | 4 | Several clips in parallel |
| `tokenizer.padding_side` | `left` | Required for correct batched generation |
| Sampling | temperature 0.6, top_p 0.95, top_k 20 | Recommended for Qwen3 thinking mode |
| Prompt truncation | 2048 tokens | Bounds memory |
| `max_new_tokens` | 1024 default, 2048 for long reasoning | See "Token budget" below |

## Results

### Transcription and answers

| Input | Transcript | Result |
|---|---|---|
| q1_math (generated) | A train travels 120 km in 2 hours and then 180 km in 3 hours. What is its average speed for the whole journey? | 60 km/h, with working. Correct |
| q2_logic (generated) | All roses are flowers. And some flowers fade quickly. Can we conclude that some roses fade quickly? Explain your reasoning. | Conclusion does not follow. Correct |
| q3_qa (generated) | Why does 4-bit quantization let a large language model fit on a small GPU? | Complete with 2048 tokens (cut off with 1024) |
| My own voice (WhatsApp) | If I have 5 apples and give away 2, then buy 4 more. How many apples do I have? | 7 apples. Correct |

All four transcripts matched what was spoken.

### Token budget

In thinking mode the reasoning tokens count against `max_new_tokens`. With 1024 tokens, the open-ended q3 question used all tokens on thinking, so no final answer was produced and the output began with `<think>`. With **`max_new_tokens=2048`** all three generated queries returned complete, correct answers.

```python
outputs = reason(transcripts, batch_size=3, max_new_tokens=2048)
```

For short factual questions you can also call `reason(..., thinking=False)`.

### GPU memory

| Stage | Allocated | Peak |
|---|---|---|
| Baseline | 0.01 GB | 0.01 GB |
| After loading Whisper-small | 0.46 GB | 0.46 GB |
| After transcription (3 files) | 0.46 GB | 1.33 GB |
| After freeing Whisper | 0.01 GB | 1.33 GB (not reset) |
| After loading Qwen3-4B dynamic 4-bit | 3.35 GB | 3.37 GB |
| After reasoning (1024 tokens) | 3.85 GB | 3.94 GB |
| Final (after own-voice test) | 3.47 GB | 4.60 GB |
| Re-run with 2048 tokens | 5.52 GB | 5.62 GB |

The highest peak was **5.62 GB of 14.56 GB (38.6 percent of the T4)**. `nvidia-smi` showed 3,743 MiB used out of 15,360 MiB at the end of the main run.

## Memory-saving techniques used

- Dynamic 4-bit LLM (a 4B model in 3.35 GB instead of roughly 8 GB in fp16)
- fp16 Whisper-small (about 0.46 GB)
- Sequential loading: Whisper is deleted and the cache cleared before the LLM runs
- Batched transcription and batched, left-padded generation
- Tensors deleted and `free_gpu()` called after every generated batch
- Truncated prompts and bounded `max_new_tokens`
- `torch.inference_mode()`

## Known limitations

- **Token budget:** long thinking can exhaust `max_new_tokens`. Raise it, switch thinking off for simple questions, or ask for short answers in the system prompt.
- **Warm-up time:** the first transcription is slow (14.6 s for three files). Later single-file calls took about 0.7 to 0.8 s.
- **Higher VRAM with longer generation:** the 2048-token run peaked at 5.62 GB because the KV-cache grows with length.
- **Formatting:** answers can contain LaTeX-style markers such as `$5 - 2 = 3$`. This is formatting only.
- **Harmless warnings:** `torch_dtype` deprecated, `chunk_length_s` experimental, and both `max_new_tokens` and `max_length` set (the first wins).
- **Sampling:** results vary slightly between runs.

## Possible improvements

- Use `whisper-large-v3-turbo` for higher accuracy
- Support more languages, including Urdu
- Stream the reasoning output
- Add text-to-speech to read the answer back

## Repository contents

| File | Description |
|---|---|
| `Whisper_Quantized_Reasoning.ipynb` | The Google Colab notebook (code and outputs) |
| `README.md` | This file |

## Links

- Colab notebook: [Whisper_Quantized_Reasoning.ipynb](https://colab.research.google.com/github/codewithUswaFatima/whisper-quantized-reasoning/blob/main/Whisper_Quantized_Reasoning.ipynb)
- GitHub: https://github.com/codewithUswaFatima/whisper-quantized-reasoning
- LinkedIn post: **[paste link]**
