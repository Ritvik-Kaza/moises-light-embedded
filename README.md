# Embedded AI-Based Music Source Separation using Moises-Light

Separates a music recording into four stems — **vocals, drums, bass, other** —
and measures whether the model fits inside a Raspberry Pi's compute and memory
budget.

**Team:** Ritvik Kaza (24BCE5351), Vishal Narayan V (24BCE5427)

---

## Read this first: the weights problem

Moises-Light is a real, published architecture (Hung, Pereira & Korzeniowski,
*Moises-Light: Resource-efficient Band-split U-Net for Music Source Separation*,
WASPAA 2025, [arXiv:2510.06785](https://arxiv.org/abs/2510.06785)).

**The paper does not release code or trained weights.** The `moises-light` PyPI
package is an unofficial reimplementation of the architecture that ships no
weights either — the wheel is 16 KB.

That has one consequence you must understand before running anything:

| What you measure | Needs trained weights? | Valid out of the box |
|---|---|---|
| Parameter count, model size | No | Yes |
| Latency, real-time factor | No | Yes |
| Peak RAM, CPU utilisation | No | Yes |
| Separation *quality* (SDR, how it sounds) | **Yes** | **No** |

Compute cost depends on the architecture and the input shape, not on the numeric
values in the weight tensors. So every performance number this project reports is
real even with random initialisation. The output *audio*, with random weights, is
noise.

This project therefore ships two backends:

- **`moises_light`** — the architecture under study. Used for all resource
  measurements. This is the embedded-systems contribution.
- **`openunmix`** — a published 4-stem separator with downloadable pretrained
  weights. Used to produce stems a listener can actually judge. It is a much
  larger model; do not report its numbers as Moises-Light numbers.

If you later get GPU access and train Moises-Light on MUSDB18-HQ, pass the
checkpoint with `--checkpoint` and both roles collapse into one backend.

---

## Quick start

```bash
# 1. install
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. make a test file (no music download needed)
python scripts/make_sample_audio.py --duration 15

# 3. verify the pipeline is correct
python scripts/test_pipeline.py

# 4. measure the hardware
python main.py benchmark --duration 30

# 5. separate a file
python main.py separate data/input/test_mixture.wav

# 6. audible stems from a real song (downloads weights on first run)
python main.py separate my_song.wav --backend openunmix
```

Full install instructions, including Raspberry Pi: **[docs/SETUP.md](docs/SETUP.md)**
How every module works: **[docs/CODE_EXPLAINED.md](docs/CODE_EXPLAINED.md)**

---

## Measured results

Taken on a single-core x86 container with 4 GB RAM, PyTorch 2.13, 1 thread.
This is deliberately a *pessimistic* proxy for a Raspberry Pi 4: the Pi has four
cores, each weaker than this one. Treat these as a conservative baseline, then
re-run the same commands on the Pi and compare.

| Preset | Params | Size (fp32) | 6 s chunk | Inference RTF | Peak RAM |
|---|---|---|---|---|---|
| `paper_small` | 2.56 M | 9.8 MB | 2482 ms | **0.41x** | 827 MB |
| `paper_large` | 5.45 M | 20.8 MB | 4997 ms | **0.83x** | 934 MB |

RTF below 1.0 means faster than real time — a 3-minute song processes in under 3
minutes, and real-time streaming is theoretically reachable.

Peak RAM is controlled by chunk length, not song length:

| Chunk | Peak RAM (60 s input) |
|---|---|
| 3 s | 729 MB |
| 6 s | 922 MB |

**This is the headline finding: Moises-Light is Pi-viable.** Under 1 GB peak RSS
and faster than real time on one weak core means it fits a 2 GB Pi 4 with room
to spare, and should be comfortably faster on a 4-core Pi 5.

---

## How it works

```
audio file
    |
    v
[ audio_io.load_audio ]      decode -> float32 stereo 44.1 kHz
    |
    v
[ separator.Separator ]      slice into overlapping chunks
    |                        \
    |                         for each chunk:
    |                             [ backend.separate_chunk ] -> 4 stems
    |                             linear crossfade at boundaries
    |                             overlap-add into rolling buffer
    |                             flush finalised samples to disk
    v
[ audio_io.StemWriter ]      vocals.wav drums.wav bass.wav other.wav
    |
    v
[ metrics.RunMetrics ]       latency, RTF, peak RSS, CPU% -> report + JSON
```

The chunking is the design decision that makes this an embedded project rather
than a desktop one. A whole-song forward pass allocates multi-gigabyte STFT
intermediates. Chunked overlap-add bounds peak memory by chunk length, so a
30-second clip and a 10-minute track have the same memory profile — length only
costs time.

Chunk boundaries would click if outputs were simply butted together, because the
model's output is not continuous across independent forward passes. Each chunk
gets a linear fade-in and fade-out over the overlap region; the two ramps are
complements, so overlapping chunks sum to exactly unity gain.

`scripts/test_pipeline.py` proves this: it swaps in an identity backend that
returns the input unchanged, and asserts the reconstructed output matches the
original **to 6e-08, one float32 ULP**, across five chunk/overlap configurations.

---

## Project layout

```
.
├── main.py                       CLI: separate / benchmark / info
├── requirements.txt              laptop dependencies
├── requirements-pi.txt           Raspberry Pi dependencies
│
├── src/
│   ├── config.py                 all tunables + hardware profiles
│   ├── audio_io.py               loading, resampling, streaming stem writer
│   ├── separator.py              chunked overlap-add engine
│   ├── metrics.py                latency / RAM / CPU measurement + report
│   └── models/
│       ├── base.py               backend interface
│       ├── moises_light_backend.py
│       └── openunmix_backend.py
│
├── scripts/
│   ├── make_sample_audio.py      synthetic test mixture generator
│   └── test_pipeline.py          correctness tests (14 checks)
│
├── docs/
│   ├── SETUP.md                  install, troubleshooting, Pi deployment
│   └── CODE_EXPLAINED.md         module-by-module walkthrough
│
├── data/input/                   put source audio here
├── data/output/                  stems + metrics JSON land here
└── models/                       trained checkpoints, if you ever have any
```

---

## CLI reference

```
python main.py info                    model + system information, no inference
python main.py benchmark [options]     synthetic-audio throughput measurement
python main.py separate INPUT [opts]   separate a file into stems
```

Common options:

| Flag | Default | Meaning |
|---|---|---|
| `--profile` | `laptop` | hardware profile: `laptop`, `pi5`, `pi4`, `pi_2gb` |
| `--backend` | `moises_light` | `moises_light` or `openunmix` |
| `--preset` | `paper_small` | `paper_small`, `paper_large`, `fullband_small`, ... |
| `--checkpoint` | — | path to trained Moises-Light weights |
| `--chunk` | `6.0` | chunk length in seconds — the main memory knob |
| `--overlap` | `0.5` | crossfade length in seconds |
| `--threads` | auto | PyTorch CPU threads; set to `4` on a Pi |
| `--stems` | all four | subset, e.g. `--stems vocals drums` |
| `-o, --output` | `data/output` | output directory |

Benchmark-only: `--duration`, `--warmup`, `--save`.

Every run writes `<name>_metrics.json` next to the stems, so laptop and Pi runs
can be diffed directly.

---

## Reproducing the laptop-vs-Pi comparison

Run the identical command on both machines and diff the JSON:

```bash
# laptop
python main.py benchmark --duration 30 --save results_laptop.json

# Raspberry Pi
python main.py benchmark --duration 30 --profile pi4 --save results_pi.json
```

That table — same model, same input, two platforms — is the experimental result
the project exists to produce.

---

## Known limitations

- **No trained Moises-Light weights exist publicly.** Separation quality cannot
  be evaluated for this architecture without training it first. See the top of
  this file.
- **No SDR evaluation.** Measuring separation quality needs MUSDB18-HQ and
  reference stems; not implemented, because there are no weights to evaluate.
- **The Open-Unmix backend is untested end to end in the authoring environment**
  because weight downloads were blocked there. The API usage is standard, but
  run it once on your laptop with internet before relying on it for a demo.
- **The fallback resampler is linear interpolation**, chosen to avoid a scipy
  dependency on the Pi. Feed it 44.1 kHz audio and this path never runs.
- **Live microphone input is not implemented.** The current pipeline is
  file-in / file-out. RTF < 1 indicates streaming is feasible, but a real-time
  path needs a ring buffer and an audio callback, which is future work.

---

## References

- Hung, Pereira & Korzeniowski. *Moises-Light: Resource-efficient Band-split
  U-Net for Music Source Separation.* WASPAA 2025.
  [arXiv:2510.06785](https://arxiv.org/abs/2510.06785)
- Unofficial implementation: <https://github.com/crlandsc/moises-light> (MIT)
- Stöter et al. *Open-Unmix — A Reference Implementation for Music Source
  Separation.* JOSS 2019. <https://github.com/sigsep/open-unmix-pytorch>
