# Setup Guide

Covers laptop install, Raspberry Pi install, verification, and troubleshooting.

---

## 1. Laptop setup

### 1.1 Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.9 – 3.12 | `python --version`. 3.13 may lack some wheels. |
| pip | `python -m pip --version` |
| ffmpeg *(optional)* | only needed to convert MP3s |
| ~3 GB disk | mostly PyTorch |

### 1.2 Create a virtual environment

Do not skip this. Installing PyTorch system-wide will eventually collide with
something else on your machine.

```bash
cd moises-light-embedded

python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows cmd
.venv\Scripts\activate.bat
```

Your prompt should now be prefixed with `(.venv)`.

### 1.3 Install PyTorch

If you have **no NVIDIA GPU**, install the CPU-only build first. It is ~200 MB
instead of ~2.5 GB, and CPU is what the Pi will use anyway, so this also keeps
the two environments comparable:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you *do* have an NVIDIA GPU and want it for future training, skip this step
and let the next one pull the default build.

### 1.4 Install everything else

```bash
pip install -r requirements.txt
```

What that installs and why:

| Package | Role |
|---|---|
| `torch` | model execution |
| `numpy` | array handling, overlap-add buffers |
| `soundfile` | reading/writing WAV/FLAC (wraps libsndfile) |
| `psutil` | RSS and CPU sampling for the metrics report |
| `moises-light` | the Moises-Light architecture (no weights) |
| `einops`, `rotary-embedding-torch` | required by moises-light |
| `openunmix` | pretrained reference backend for audible stems |

### 1.5 Verify

```bash
python main.py info
```

Expect system information, the configuration, and a model summary ending with
`parameters: 2,558,768 (2.56 M)`.

Then run the correctness tests:

```bash
python scripts/test_pipeline.py
```

Expect `14/14 passed`. If any overlap-add test fails, stop and fix it before
trusting any output — a windowing bug silently corrupts every stem.

### 1.6 First real run

```bash
python scripts/make_sample_audio.py --duration 15
python main.py separate data/input/test_mixture.wav
```

Four WAVs and a metrics JSON appear in `data/output/`.

Remember: with the default `moises_light` backend and no checkpoint, those WAVs
contain noise. The performance report is the real output. For audible stems:

```bash
python main.py separate data/input/test_mixture.wav --backend openunmix
```

The first Open-Unmix run downloads weights (~140 MB) and needs internet.

---

## 2. Raspberry Pi setup

### 2.1 You must be on 64-bit

There are no official PyTorch wheels for 32-bit Raspberry Pi OS. Check:

```bash
uname -m
```

- `aarch64` → good, continue
- `armv7l` → 32-bit. Reflash with **Raspberry Pi OS (64-bit)** using Raspberry
  Pi Imager. Nothing below will work until you do.

Confirm the board and RAM:

```bash
cat /proc/device-tree/model
free -h
```

### 2.2 System packages

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libsndfile1 ffmpeg
```

`libsndfile1` is not optional — `soundfile` is a thin wrapper around it and
will fail to import without it.

### 2.3 Increase swap (2 GB boards only)

Model load transiently needs more memory than steady-state inference. On a 2 GB
Pi 4, raising swap prevents the OOM killer from ending the process during load:

```bash
sudo dphys-swapfile swapoff
sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
free -h
```

Swap on an SD card is slow. It is insurance against a crash, not a performance
feature. If your run is actually paging during inference, reduce `--chunk`
instead.

### 2.4 Python environment

```bash
cd ~/moises-light-embedded
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-pi.txt
```

This takes 10–20 minutes on a Pi. It is mostly compiling and installing PyTorch.
Leave it alone; a `Killed` message means you ran out of RAM, in which case add
swap (2.3) and retry.

`requirements-pi.txt` deliberately omits `openunmix` — it is several times
larger than Moises-Light and only needed for the laptop demo. Add it manually if
you want to compare the two on-device.

### 2.5 Verify on the Pi

```bash
python main.py info
```

The `board` field should now show your Pi model, confirming the metrics module
is reading the hardware correctly.

```bash
python scripts/test_pipeline.py
python main.py benchmark --duration 30 --profile pi4 --save results_pi.json
```

### 2.6 Choosing a profile

| Profile | Chunk | Threads | Use for |
|---|---|---|---|
| `laptop` | 6.0 s | auto | development machine |
| `pi5` | 6.0 s | 4 | Pi 5 (4/8 GB) |
| `pi4` | 4.0 s | 4 | Pi 4 (4/8 GB) |
| `pi_2gb` | 3.0 s | 4 | Pi 4 (2 GB) |

If a run is killed or thrashes swap, halve `--chunk`. Peak RAM scales with chunk
length; audio length costs only time.

### 2.7 Thermal throttling

Sustained inference will heat the SoC. A throttled Pi silently reports worse
latency, which will corrupt your results if you do not notice. Watch it:

```bash
watch -n 2 vcgencmd measure_temp
```

Above ~80 °C the CPU downclocks. Use a heatsink and fan for any run you intend
to report. Check afterwards whether throttling occurred:

```bash
vcgencmd get_throttled     # 0x0 means no throttling happened
```

Note this alongside your numbers. "RTF 0.9x, no throttling" is a result;
"RTF 0.9x" with an unnoticed throttle event is a measurement error.

---

## 3. Transferring the project to the Pi

Over the network, from your laptop:

```bash
rsync -av --exclude='.venv' --exclude='data/output' \
      --exclude='__pycache__' \
      ./ pi@raspberrypi.local:~/moises-light-embedded/
```

Or via git:

```bash
# on the Pi
git clone <your-repo-url> ~/moises-light-embedded
```

Do not copy `.venv` — it contains x86 binaries that cannot run on ARM.

To pull results back for the comparison table:

```bash
scp pi@raspberrypi.local:~/moises-light-embedded/results_pi.json .
```

---

## 4. Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
Run commands from the project root, not from inside `src/` or `scripts/`.

**`OSError: sndfile library not found`**
`sudo apt install libsndfile1` (Linux/Pi). On macOS: `brew install libsndfile`.

**`could not decode '...mp3'`**
Your libsndfile predates MP3 support. Convert first:
```bash
ffmpeg -i song.mp3 -ar 44100 -ac 2 data/input/song.wav
```

**Process is `Killed` with no traceback**
Out of memory. Reduce `--chunk` (try 3 or 2), close other applications, and on a
2 GB Pi add swap per §2.3.

**`could not load pretrained Open-Unmix weights`**
First use needs internet. If you are behind a proxy or offline, run it once on a
connected machine; weights cache to `~/.cache/torch/hub/checkpoints/`, which you
can copy to the Pi.

**`checkpoint is missing N keys`**
The `--preset` does not match the preset the checkpoint was trained with. Try
`--preset paper_large` or whichever the checkpoint used.

**Very slow on the Pi, RTF much worse than expected**
Check in order: 64-bit OS (`uname -m`), `--threads 4` actually set, thermal
throttling (§2.7), and whether the process is swapping (`free -h` during a run).

**`UserWarning: In CPU autocast, but the target dtype is not supported`**
Benign — the model falls back to fp32. It is filtered in
`moises_light_backend.py`; if you see it, an import ordering change has bypassed
the filter. It does not affect results.

---

## 5. Offline operation

Once installed, the `moises_light` backend needs no network at all. To run
Open-Unmix offline, warm the cache on a connected machine:

```bash
python -c "import openunmix; openunmix.umxl(device='cpu', pretrained=True)"
```

then copy `~/.cache/torch/hub/checkpoints/` to the same path on the Pi.
