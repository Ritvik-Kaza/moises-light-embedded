# Embedded AI-Based Music Source Separation using Moises-Light

Separates a music recording into four stems — **vocals, drums, bass, other** —
and measures whether the model fits inside a Raspberry Pi's compute and memory
budget.

**Team:** Ritvik Kaza , Vishal Narayan V

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
- **`remote`** — offloads query-driven separation (e.g. "give me the bass
  guitar") to a heavier model (AudioSep) running on another machine, since
  that class of model is not realistically Pi-viable on-device. The Pi still
  does all the chunking, overlap-add, and I/O locally; only the forward pass
  is remote. See **[Offload setup](#offload-setup-query-driven-separation)**
  below — and do not present this as an on-device result the way
  Moises-Light's numbers are.

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

---

## Full setup

### Laptop

| Requirement | Notes |
|---|---|
| Python 3.9 – 3.12 | `python --version`. 3.13 may lack some wheels. |
| pip | `python -m pip --version` |
| ffmpeg *(optional)* | only needed to convert MP3s |
| ~3 GB disk | mostly PyTorch |

```bash
cd moises-light-embedded
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# If you have no NVIDIA GPU, the CPU-only build is ~200 MB instead of
# ~2.5 GB, and CPU is what the Pi uses anyway — keeps the two comparable:
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Verify:

```bash
python main.py info                # expect "parameters: 2,558,768 (2.56 M)"
python scripts/test_pipeline.py    # expect all tests passing
```

If any overlap-add test fails, stop and fix it before trusting any output —
a windowing bug silently corrupts every stem.

### Raspberry Pi

You must be on a **64-bit** OS (`uname -m` → `aarch64`, not `armv7l` — there
are no official PyTorch wheels for 32-bit Raspberry Pi OS).

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libsndfile1 ffmpeg

cd ~/moises-light-embedded
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-pi.txt   # 10-20 min; mostly compiling PyTorch
```

On a **2 GB** board, model load transiently needs more memory than
steady-state inference — raise swap first to avoid an OOM kill during load:

```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup && sudo dphys-swapfile swapon
```

`requirements-pi.txt` deliberately omits `openunmix` (several times larger
than Moises-Light, only needed for the laptop demo).

Verify, and watch for thermal throttling on any run you intend to report
(`watch -n 2 vcgencmd measure_temp`; above ~80°C the CPU downclocks —
`vcgencmd get_throttled` after a run, `0x0` means no throttling happened):

```bash
python main.py info                                              # board field should show your Pi model
python scripts/test_pipeline.py
python main.py benchmark --duration 30 --profile pi4 --save results_pi.json
```

Transfer the project over (don't copy `.venv` — x86 binaries won't run on
ARM):

```bash
rsync -av --exclude='.venv' --exclude='data/output' --exclude='__pycache__' \
      ./ pi@raspberrypi.local:~/moises-light-embedded/
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the project root, not from inside `src/` or `scripts/` |
| `OSError: sndfile library not found` | `sudo apt install libsndfile1` (Linux/Pi) or `brew install libsndfile` (macOS) |
| `could not decode '...mp3'` | Convert first: `ffmpeg -i song.mp3 -ar 44100 -ac 2 song.wav` |
| Process `Killed`, no traceback | Out of memory — reduce `--chunk` (try 3 or 2), add swap on a 2 GB Pi |
| `could not load pretrained Open-Unmix weights` | Needs internet on first use; weights cache to `~/.cache/torch/hub/checkpoints/`, copyable to the Pi |
| `checkpoint is missing N keys` | `--preset` doesn't match the checkpoint's preset |
| Slow on Pi, RTF much worse than expected | Check 64-bit OS, `--threads 4`, thermal throttling, and whether it's swapping (`free -h` during a run) |

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
├── requirements-server.txt       offload server dependencies (see Offload setup)
│
├── src/
│   ├── config.py                 all tunables + hardware profiles
│   ├── audio_io.py               loading, resampling, streaming stem writer
│   ├── separator.py              chunked overlap-add engine
│   ├── metrics.py                latency / RAM / CPU measurement + report
│   └── models/
│       ├── base.py               backend interface
│       ├── moises_light_backend.py
│       ├── openunmix_backend.py
│       └── remote_backend.py     offload client for query-driven separation
│
├── scripts/
│   ├── make_sample_audio.py      synthetic test mixture generator
│   ├── test_pipeline.py          correctness tests
│   └── serve_audiosep.py         offload server (runs on the laptop, not the Pi)
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
| `--backend` | `moises_light` | `moises_light`, `openunmix`, or `remote` |
| `--preset` | `paper_small` | `paper_small`, `paper_large`, `fullband_small`, ... |
| `--checkpoint` | — | path to trained Moises-Light weights |
| `--query` | — | target description, required for `--backend remote` (e.g. `"bass guitar"`) |
| `--remote-url` | — | offload server address, required for `--backend remote` |
| `--remote-timeout` | `60` | seconds to wait for one chunk's response from `--backend remote` |
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

## Offload setup (query-driven separation)

`--backend remote` lets you ask for an arbitrary target by text description —
`"bass guitar"`, `"the crowd noise"`, anything — instead of picking from the
fixed vocals/drums/bass/other vocabulary. That needs a different, much
heavier model ([AudioSep](https://github.com/Audio-AGI/AudioSep)), not
realistically Pi-viable on-device. So this feature **offloads compute**: the
Pi (or your laptop, for testing) still does all the chunking, overlap-add and
file I/O locally — only each chunk's forward pass is sent over HTTP to
AudioSep running on a more capable machine.

**Be precise about what this proves.** This does not demonstrate AudioSep
running on embedded hardware — it demonstrates the Pi correctly offloading a
workload too heavy for it, over a network boundary. Keep it visually separate
from the Moises-Light on-device numbers; they are two different claims.

**Honesty note:** the client side (`src/models/remote_backend.py`) has
automated tests and passes them. AudioSep itself was **not** run end to end
while building this feature — no GPU/conda in that environment, and the
checkpoint is a multi-gigabyte download. `scripts/serve_audiosep.py` is
written directly against AudioSep's real, published API — but run it once
yourself and confirm real output before relying on it for a demo, the same
caution this project already applies to Open-Unmix.

### Server (on your laptop)

```bash
# 1. get AudioSep — a separate environment from this project's .venv,
#    it pins its own torch/transformers/lightning versions
git clone https://github.com/Audio-AGI/AudioSep.git ~/AudioSep
cd ~/AudioSep
conda env create -f environment.yml
conda activate AudioSep

# 2. download the checkpoint per AudioSep's own README
#    (currently audiosep_base_4M_steps.ckpt on their HuggingFace repo)
#    into ~/AudioSep/checkpoint/

# 3. install the one extra thing AudioSep doesn't ship: a web server
pip install -r /path/to/moises-light-embedded/requirements-server.txt

# 4. run it
python /path/to/moises-light-embedded/scripts/serve_audiosep.py \
    --audiosep-repo ~/AudioSep \
    --checkpoint ~/AudioSep/checkpoint/audiosep_base_4M_steps.ckpt \
    --device cpu \
    --host 0.0.0.0 --port 8765
```

Verify it's alive: `curl http://localhost:8765/health`.

**Security:** this server has no authentication. `--host 0.0.0.0` makes it
reachable on your LAN, which is what the Pi needs — but never expose this
port to the public internet.

### Client (on the Pi, or your laptop for a dry run)

Nothing extra to install — `requests` is already in `requirements.txt` /
`requirements-pi.txt`.

```bash
python main.py separate data/input/clip.wav \
    --backend remote \
    --query "bass guitar" \
    --remote-url http://192.168.1.50:8765
```

The output file is named `<basename>_bass_guitar.wav` — the stem name comes
from your query text (lowercased, non-alphanumerics collapsed to
underscores). The performance report gains a **Network time** line: the
round-trip time to the laptop, the number that actually matters here — Pi-
side RAM/CPU should stay low relative to Moises-Light or Open-Unmix runs,
which is the point being demonstrated.

**Troubleshooting:** `could not reach remote separation server` means the
server isn't running or the URL/port is wrong — check
`curl http://<laptop-ip>:8765/health` from the Pi first. A timeout means CPU
inference is slow — try `--device cuda` on the server (if available) or a
smaller `--chunk`.

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
- **The `remote` backend's client side (chunking, wire protocol, error
  handling) has automated tests and passes them; the AudioSep model itself
  does not** — it was never run end to end while building this, since that
  needs a GPU-class machine, a conda environment, and a multi-gigabyte
  checkpoint. Run `scripts/serve_audiosep.py` once yourself and confirm real
  separated audio comes back before relying on it for a demo. See
  [Offload setup](#offload-setup-query-driven-separation) above.
- **The fallback resampler is linear interpolation**, chosen to avoid a scipy
  dependency on the Pi. Feed it 44.1 kHz audio and this path never runs.
- **Live microphone input is not implemented.** The current pipeline is
  file-in / file-out. RTF < 1 indicates streaming is feasible, but a real-time
  path needs a ring buffer and an audio callback, which is future work.

---

## Test status

`python scripts/test_pipeline.py` — **26/26 passed**, most recently verified
by applying this project's commits to a clean checkout from scratch (compile
check + full suite, independent of the working tree they were built in).
Covers config validation, mono/stereo/resampling handling, exact overlap-add
reconstruction (max error 5.96e-08, one float32 ULP, across five chunk/
overlap configurations), stem-subset output, Moises-Light's real output shape
and parameter count, and the `remote` backend's wire protocol, error handling,
and end-to-end run through the actual chunking pipeline against a fake
server.

---

## References

- Hung, Pereira & Korzeniowski. *Moises-Light: Resource-efficient Band-split
  U-Net for Music Source Separation.* WASPAA 2025.
  [arXiv:2510.06785](https://arxiv.org/abs/2510.06785)
- Unofficial implementation: <https://github.com/crlandsc/moises-light> (MIT)
- Stöter et al. *Open-Unmix — A Reference Implementation for Music Source
  Separation.* JOSS 2019. <https://github.com/sigsep/open-unmix-pytorch>
- Liu et al. *Separate Anything You Describe.* 2023.
  <https://github.com/Audio-AGI/AudioSep> — the query-driven model behind
  the `remote` offload backend.
