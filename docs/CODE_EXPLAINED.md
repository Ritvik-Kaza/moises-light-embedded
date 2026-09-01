# Code Explained

A module-by-module walkthrough: what each file does, why it is designed that
way, and which decisions exist specifically because the target is a Raspberry
Pi. Written so you can answer "why did you do it like that?" in a review.

---

## Contents

1. [Execution flow](#1-execution-flow)
2. [`src/config.py`](#2-srcconfigpy)
3. [`src/audio_io.py`](#3-srcaudio_iopy)
4. [`src/models/`](#4-srcmodels)
5. [`src/separator.py`](#5-srcseparatorpy)
6. [`src/metrics.py`](#6-srcmetricspy)
7. [`main.py`](#7-mainpy)
8. [`scripts/`](#8-scripts)
9. [Design decisions worth defending](#9-design-decisions-worth-defending)

---

## 1. Execution flow

What happens when you run `python main.py separate song.wav`:

```
main.py
  parse arguments
  build_config()          profile defaults, then CLI overrides
      |
      v
  Separator(cfg)
      configure_threads()      pin PyTorch CPU thread count
      build_backend(cfg)       construct MoisesLight or OpenUnmix
      record model params, size, load time, RAM delta
      |
      v
  Separator.separate_file(path)
      load_audio()             decode -> float32 [2, N] at 44.1 kHz
      compute chunk/hop/overlap sample counts
      open StemWriter          4 output files, held open
      start MemorySampler      background RSS sampling thread
      |
      for each chunk i:
          slice mixture[start : start+chunk], zero-pad if short
          backend.separate_chunk()   -> [1, 4, 2, chunk]
          apply fade-in (i > 0) and fade-out (not last)
          acc += chunk output
          write acc[:hop] to disk        <- these samples are now final
          shift acc left by hop, in place
      |
      flush remaining overlap tail
      close writer, stop sampler
      |
      v
  RunMetrics.report()       printed table
  RunMetrics.save_json()    machine-readable, for laptop-vs-Pi diffing
```

The important structural point: `Separator` never knows which model it is
running. It only calls `backend.separate_chunk()`. That is what allows the same
pipeline to benchmark Moises-Light and produce audible stems with Open-Unmix
without any conditional logic in the pipeline itself.

---

## 2. `src/config.py`

Every tunable lives here so the laptop run and the Pi run differ by a config
profile, never by code. If they differed by code, the comparison would be
meaningless.

**Constants.** `SAMPLE_RATE = 44100` and `CHANNELS = 2` are fixed by the model
architecture — Moises-Light's STFT settings (`n_fft=6144`, `hop_size=1024`)
assume 44.1 kHz stereo. `STEM_NAMES` fixes the stem order to match the model's
output dimension.

**`RunConfig`.** A dataclass holding one run's settings. The two fields that
matter most for embedded viability:

- `chunk_seconds` — how much audio goes through the model per forward pass.
  This is the primary memory knob. Peak RAM scales with it.
- `overlap_seconds` — crossfade length between chunks. Larger means smoother
  boundaries but more redundant computation: the fraction of wasted work is
  `overlap / chunk`, so 0.5 s of overlap on 6 s chunks costs about 8% extra.

`validate()` catches the errors that would otherwise produce silently wrong
output — notably `overlap >= chunk`, which yields a non-positive hop and either
crashes or loops forever.

**`PROFILES`.** Named hardware presets (`laptop`, `pi5`, `pi4`, `pi_2gb`)
differing only in chunk size and thread count. `--profile pi4` is shorthand for
a chunk/thread pair that fits that board.

---

## 3. `src/audio_io.py`

### `load_audio(path, target_sr)`

Decodes any soundfile-supported format to float32, transposes from soundfile's
`[samples, channels]` to the model's `[channels, samples]`, forces stereo, and
resamples if needed. Returns a contiguous array — `np.ascontiguousarray` matters
because `torch.from_numpy` on a non-contiguous array silently copies, which on
a Pi is a measurable cost per chunk.

Errors are wrapped in `AudioLoadError` with an actionable message. A failed MP3
decode tells you the exact ffmpeg command to fix it, rather than surfacing a
libsndfile error code.

### `_to_stereo(audio)`

Mono is duplicated to both channels. More than two channels are downmixed by
averaging even-indexed channels into left and odd into right. The model's first
layer expects exactly two channels, so this cannot be skipped.

### `_resample_linear(audio, sr_in, sr_out)`

A dependency-free linear-interpolation resampler.

**Why not scipy or librosa?** Both are heavy installs on a Pi (scipy compiles
for a long time on ARM; librosa pulls numba). Resampling is only a fallback for
oddly-sampled input, and anything you actually benchmark should already be
44.1 kHz. Linear interpolation is a lower-quality resampler than polyphase
filtering, and it is the honest tradeoff here: accept slightly worse quality on
a path that should never run during a measurement, in exchange for a much
lighter dependency tree on the target hardware.

### `StemWriter`

Opens one `SoundFile` per stem in write mode and appends blocks as they are
finalised.

**Why streaming rather than accumulate-then-write?** Four full-length stereo
float32 stems for a 4-minute song is roughly 340 MB of buffers. On a 2 GB Pi
already holding a ~750 MB model, that is the difference between running and
being killed. Writing incrementally keeps output buffers bounded to one chunk.

Output is clipped to `[-1, 1]` before writing. Without it, model output slightly
above unity wraps around in 16-bit PCM and produces a loud click — a wrap-around
artefact that is easy to mistake for a chunking bug.

It is a context manager, so files close even if inference raises.

---

## 4. `src/models/`

### `base.py` — `SeparatorBackend`

The interface. Four members:

- `stems` — stem names in the model's output order
- `n_params()` — for the size report
- `separate_chunk(mixture)` — `[1, 2, N]` in, `[1, 4, 2, N]` out
- `is_pretrained` — **the most important flag in the project**

`is_pretrained` propagates into the metrics report and triggers an explicit
warning. It exists so nobody can accidentally present noise as separated audio.
Random weights give valid *timing* and *memory* numbers and meaningless audio;
the code refuses to let that distinction get lost.

### `moises_light_backend.py`

Wraps the `moises-light` package.

**The weights situation.** The paper does not release code or weights, and the
unofficial package ships architecture only. Constructing this backend without
`--checkpoint` gives random initialisation. The module docstring says so and the
CLI prints a warning on every such run.

**Why random weights still produce valid performance numbers.** A forward pass
executes the same convolutions, the same STFT, the same attention over the same
tensor shapes regardless of what numbers are in the weights. Latency, FLOPs and
memory are functions of architecture and input shape. Only the *values* change.
So the resource characterisation — the actual embedded contribution — is sound.

**Checkpoint loading.** `_load_checkpoint` accepts a bare `state_dict` or a
training-framework wrapper, and strips the `model.` prefix that PyTorch
Lightning adds. `weights_only=True` is set on `torch.load`: it refuses to
unpickle arbitrary Python objects, which is the difference between loading a
tensor file and executing whatever a downloaded checkpoint contains. Missing
keys raise with a message pointing at the likely cause (preset mismatch) rather
than letting a partially-loaded model produce quiet garbage.

**Warning filter.** The package calls `torch.amp.autocast` with an unsupported
CPU dtype and emits a `UserWarning` on every forward pass. It is benign — it
falls back to fp32 — but unfiltered it would print once per chunk and bury the
progress output.

### `openunmix_backend.py`

Loads pretrained Open-Unmix. Exists solely so you have audible stems for a demo.

`niter=0` disables Wiener-filter refinement. The upstream default of 1 sounds
marginally better and costs significantly more compute — the wrong tradeoff on a
Pi, and unnecessary for a laptop demo.

Target order is read off `model.target_models` rather than assumed, so a variant
that reorders its outputs cannot silently swap your vocals and drums files.

Weight download failures are caught and re-raised with the actual cause, because
the default failure mode is a network traceback that does not mention that the
fix is "connect to the internet once".

### `__init__.py` — `build_backend(cfg)`

A registry with **lazy imports**. Each backend is imported inside its branch, so
a Pi with `requirements-pi.txt` installed — no `openunmix` — starts fine instead
of failing at import time on a dependency it will never use.

---

## 5. `src/separator.py`

The core of the project.

### Why chunking exists

A whole-song forward pass allocates STFT intermediates measured in gigabytes.
Fine on a workstation, fatal on a Pi. Chunking bounds peak memory by chunk
length, so a 30-second clip and a 10-minute track have the same memory profile.
Length costs only time.

This is verifiable and it is the claim to make in a review:

| Chunk | Peak RAM (60 s input) |
|---|---|
| 3 s | 729 MB |
| 6 s | 922 MB |

### Overlap-add, and why it must be exact

Independent forward passes do not produce output that is continuous across
chunk boundaries. Butting outputs together produces an audible click at every
boundary — every 6 seconds, for the whole track.

The fix is a crossfade. `_make_ramps(overlap)` builds:

```python
fade_in  = np.linspace(0.0, 1.0, overlap, endpoint=False)
fade_out = 1.0 - fade_in
```

`endpoint=False` is load-bearing. It gives `[0, 1/n, 2/n, ..., (n-1)/n]`, so
`fade_in[k] + fade_out[k] == 1.0` exactly for every k. In any overlap region the
two contributing chunks sum to **exactly unity gain** — no dip, no boost. With
`endpoint=True` the ramps would not be complements and every boundary would have
a small gain error.

Each chunk gets a fade-in unless it is the first, and a fade-out unless it is
the last. The edges of the track must not be faded, or the song starts and ends
with a ramp.

### The rolling accumulator

```python
acc = np.zeros((n_stems, channels, chunk_n), dtype=np.float32)
```

Per chunk:

1. `acc += out` — add the windowed chunk output
2. Write `acc[:, :, :hop_n]` to disk. Those samples can no longer be modified,
   because the next chunk starts at `start + hop_n`.
3. Shift the buffer left by `hop_n`, **in place**:
   ```python
   acc[:, :, :overlap_n] = acc[:, :, hop_n:]
   acc[:, :, overlap_n:] = 0.0
   ```

Step 3 was originally `np.concatenate`, which allocates a fresh array every
chunk. The in-place version makes the loop allocation-free, which matters on a
board where allocator churn shows up in both latency and peak RSS.

After the loop, the accumulator still holds the final `overlap` samples of the
last chunk; those are flushed explicitly.

### Length handling

The final chunk is zero-padded to full length so the model always sees its
expected input shape. `remaining` tracks how many real samples are left and
clips every write, so the output is exactly as long as the input — no trailing
padding in the stems.

### `separate_file()`

Loads audio, computes chunk geometry, opens the writer and memory sampler, runs
the loop, and populates `RunMetrics`. It separates two timings deliberately:

- `inference_seconds` — sum of `separate_chunk()` calls only
- `total_seconds` — the whole pipeline including decode, overlap-add and disk

The gap between them is your I/O overhead. On a Pi writing to an SD card that
gap is worth reporting, because it is a real cost that a GPU-server benchmark
would never surface.

### `benchmark()`

Runs the model on synthetic noise with no file I/O. Two reasons it exists:

1. It needs no input file and no trained weights, so it is the one measurement
   you can run identically on the laptop and the Pi from day one.
2. It isolates model compute from storage speed, so a slow SD card cannot be
   mistaken for a slow model.

The `warmup` iteration is discarded. The first forward pass includes lazy
allocator setup and kernel selection; including it would inflate the mean.

### `configure_threads()`

Sets `torch.set_num_threads()` explicitly. On a Pi, pinning to the core count
(4) is worth doing: the default heuristic sometimes oversubscribes, and thread
thrashing on small ARM cores costs more than the parallelism gains.

---

## 6. `src/metrics.py`

For an embedded-systems project this module is the deliverable. Separation is
published work; what is being evaluated is whether it fits the budget.

### `MemorySampler`

Samples RSS from a background thread every 50 ms.

**Why a thread instead of reading RSS at the end?** Python frees the peak
allocation before the run finishes. Reading once at the end reports the
*residual*, not the *peak* — often less than half the true figure. The peak is
what has to fit in physical RAM, so it is what must be measured.

RSS specifically, rather than `tracemalloc` or `torch` memory stats, because RSS
is what the OOM killer looks at.

### `system_info()`

Captures platform, CPU count, RAM, PyTorch version and thread count — enough to
make a laptop run and a Pi run comparable. It also reads
`/proc/device-tree/model`, which exists only on a Pi and yields the exact board
name. Its presence in the JSON is itself proof the numbers came from real
hardware.

### `RunMetrics`

Stores raw measurements; derives the rest as properties.

- **`rtf`** = `total_seconds / audio_seconds`. Below 1.0 is faster than real
  time. This is the single number that answers "is this deployable?"
- **`inference_rtf`** excludes I/O, isolating model cost.
- **`mean_chunk_ms` / `max_chunk_ms`** — mean is throughput; max is the
  worst-case latency a real-time system would have to absorb.

`report()` produces the printable table. It renders `is_pretrained` as
`NO - audio is not meaningful` rather than `False`, so the caveat is impossible
to miss on a screenshot.

`save_json()` writes the same data machine-readably, which is how you diff
laptop against Pi.

### `model_size_mb(n_params, bytes_per_param)`

fp32 by default; pass `1` for an int8 estimate. That second figure is the
argument for quantisation as future work: 9.8 MB → 2.4 MB for `paper_small`.

---

## 7. `main.py`

Three subcommands: `info`, `benchmark`, `separate`.

**`build_config()`** applies a hardware profile first, then CLI overrides. So
`--profile pi4 --chunk 2` means "pi4 defaults, but chunk 2 s". Overrides are
checked against `None`, not falsiness, so `--threads 0` (meaning auto) is
distinguishable from "not specified".

**Error handling.** `main()` catches exceptions and prints
`ERROR: <message>` instead of dumping a traceback. Tracebacks are for
developers; this is a tool that gets run on a Pi over SSH during a demo.
`KeyboardInterrupt` returns 130, the conventional shell code.

**`cmd_separate`** re-prints the untrained-weights warning after the report,
because the report is the part people screenshot.

---

## 8. `scripts/`

### `make_sample_audio.py`

Generates a synthetic four-part mixture — bass line, drum pattern, chord pad,
vibrato tone — plus optional ground-truth stems. It exists so the pipeline can
be tested without sourcing a copyrighted song, and so results are reproducible:
same seed, same file, every time.

It is a test fixture, not music. Separation quality on it means nothing, because
it is nothing like the training distribution of any real separator.

### `test_pipeline.py`

14 checks. The important ones are the five `test_overlap_add_is_exact` cases.

They substitute an `IdentityBackend` that returns the input unchanged as all
four stems. If chunking and crossfading are correct, every output stem must
equal the original mixture sample for sample. Measured maximum error:
**5.96e-08 — one float32 ULP.**

This catches the entire class of bugs that are otherwise nearly invisible: a
wrong ramp direction, an off-by-one in the hop, a mis-timed flush. All of them
produce output that looks and mostly sounds fine while being quietly wrong at
every boundary.

The cases cover a last chunk needing padding, zero overlap, a ragged tail, input
shorter than one chunk, and 50% overlap. Output files use `FLOAT` subtype rather
than `PCM_16`, because 16-bit quantisation would mask errors smaller than
1/32768 and the test would pass while broken.

---

## 9. Design decisions worth defending

Anticipated review questions and the honest answers.

**"Why is the model not trained?"**
The Moises-Light paper does not release code or weights, and the only public
implementation ships architecture only. Training it requires MUSDB18-HQ and
substantial GPU time. The project scope is resource characterisation for
embedded deployment, which does not require trained weights — latency and memory
depend on architecture and input shape, not weight values. Open-Unmix is
included as a pretrained reference for anything requiring audible output.

**"Why chunked inference instead of processing the whole song?"**
Peak memory would scale with song length and exceed the Pi's RAM. Chunking
bounds it by chunk length instead. Demonstrated: 3 s chunks use 729 MB,
6 s chunks 922 MB, on the same 60 s input.

**"How do you know the chunking does not corrupt the audio?"**
An identity-backend test reconstructs the input to one float32 ULP across five
chunk/overlap configurations. `python scripts/test_pipeline.py`.

**"Why measure peak RSS instead of just checking it ran?"**
"It ran" on a 4 GB Pi does not tell you whether it runs on a 2 GB Pi, or whether
it was one background process away from being OOM-killed. The peak is the
deployment constraint.

**"Why is the CPU-only PyTorch build recommended on the laptop?"**
The Pi has no GPU, so a CUDA-accelerated laptop baseline would not be comparable
to the target platform. Matching CPU execution on both sides makes the
comparison meaningful.

**"What is the biggest remaining risk?"**
No trained Moises-Light weights. Everything about deployment viability can be
answered without them; nothing about separation quality can.

**"What would you do next?"**
Three things, in order: int8 quantisation (9.8 MB → ~2.4 MB estimated, plus a
latency gain worth measuring); ONNX Runtime as an alternative to PyTorch on ARM;
and a real-time streaming path with a ring buffer, which the sub-1.0 RTF
suggests is reachable.
