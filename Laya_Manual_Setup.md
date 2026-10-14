# Laya: Manual Setup for the Demo

Oct 13, 2026 · @Lev Selector

## Overview

These steps get the Laya demo running on a machine where the repo is already cloned, with no AI help. Plan on about 15 minutes and 2 GB of disk: roughly 1 GB for Python packages and 0.8 GB for the English model. Every command was tested on a Mac on 2026-10-13 with laya 0.4.6, torch 2.14.1 and transformers 5.19.0.

You need:

- Python 3.10 or newer (`python3 --version`, or `py --version` on Windows)
- The cloned repo; all commands below run from its root folder
- Network access to `pypi.org` for packages and `huggingface.co` for the model. If `huggingface.co` is blocked at work, section 2 shows how to bring the model in from another computer.

Nothing is installed system-wide. Packages go into a `.venv` folder inside the repo and the model into a `models` folder, both already in `.gitignore`. Deleting those two folders undoes everything.

## 1. Create the .venv and install Laya

One install command pulls in everything Laya needs: torch, transformers, safetensors, huggingface\_hub and numpy. Use the block for your system.

**macOS or Linux**

```bash
cd path/to/laya
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

**Windows (PowerShell)**

```powershell
cd path\to\laya
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation and call `.venv\Scripts\python` directly.

**Linux without an NVIDIA GPU:** plain `pip install` fetches the CUDA build of torch, which is several GB. Install the CPU build first, then Laya:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

Check the install. It should print the laya version (0.4.6 or later) and the torch version:

```bash
python -c "import laya, torch; print(laya.__version__, torch.__version__)"
```

The rest of this guide assumes the `.venv` is activated, so `python`, `hf` and `laya` refer to the copies inside it. In a new terminal, activate it again first.

## 2. Download the model

Download the English checkpoint (about 0.8 GB) into `models/laya` inside the repo. The examples look there first, so after this step they never touch the network. The Hugging Face repo `convaiinnovations/laya` also holds the multilingual and typed-decisions checkpoints as subfolders; the two `--exclude` flags skip them.

```bash
hf download convaiinnovations/laya --exclude "multilingual/*" --exclude "typed-decisions/*" --local-dir models/laya
```

Each `--exclude` takes one pattern, so repeat the flag. Listing both patterns after one flag fails with `File not found`.

After the download, `models/laya` should contain `model.safetensors` (about 800 MB), `rl_agent_config.json`, `encoder/config.json` and `tokenizer/tokenizer.json`.

**If you later want all three checkpoints** (about 2.3 GB, needed for the multilingual examples), drop the excludes and add two links so the examples find them:

```bash
hf download convaiinnovations/laya --local-dir models/laya
ln -s laya/multilingual models/laya-multilingual
ln -s laya/typed-decisions models/laya-typed-decisions
```

On Windows, copy the two subfolders to `models\laya-multilingual` and `models\laya-typed-decisions` instead of linking them.

**If huggingface.co is blocked at work:** run the same `hf download` command on another computer (install it there with `pip install huggingface_hub`). Copy the resulting `models` folder into the repo root on the work machine. Then tell the Hugging Face library to stay offline:

- macOS or Linux: `export HF_HUB_OFFLINE=1`
- Windows PowerShell: `$env:HF_HUB_OFFLINE = "1"`

With that set, examples 01, 02 and 10 ran with no network access in testing.

## 3. Run the demo

Run these from the repo root with the `.venv` activated. The first call loads the weights in a few seconds; each answer after that takes milliseconds. On a Mac the examples use the Apple GPU automatically, elsewhere CUDA if present, otherwise the CPU.

1. One yes/no question about an email:

   ```bash
   python examples/02_hello_world_decision.py
   ```

   Expected: `refund_requested   noul    0.843 (true)`
2. Four questions of all three types, answered in one model call:

   ```bash
   python examples/10_mixed_primitives_one_pass.py
   ```

   Expected:

   ```text
   department         choice  billing (p=0.965)
   urgency            score   1.44 / 2
   churn_risk         noul    0.825 (true)
   refund_requested   noul    0.843 (true)
   ```
3. Your own text. Save this as `my_demo.py` in the repo root, change the email and questions, and run `python my_demo.py`:

   ```python
   import os
   os.environ.setdefault("USE_TF", "0")
   import laya

   agent = laya.load("models/laya")
   result = agent.predict(
       {"body": "The app crashes every time I open settings."},
       {"department": {
           "type": "choice",
           "instructions": "Which department should handle this?",
           "criteria": {"billing": "invoices, payments, refunds",
                        "technical": "bugs, outages, errors",
                        "other": "everything else"}}},
   )
   print(result["answers"]["department"])
   ```

Expected for step 3: `'choice': 'technical'` with probability about 0.96. You may also see a `RuntimeWarning` about calibration temperatures; it is harmless for the demo.

The `examples` folder has 41 numbered scripts, ordered from simplest to most advanced. Examples that use the multilingual model need the full download from section 2.

The `laya` command-line tool (`laya "My payment failed twice" --preset triage`) does not read the `models` folder. It downloads its own copy of the model into the Hugging Face cache on first use, so it needs network access to huggingface.co.

## 4. Optional: fine-tune on your own data

The same install includes `laya-train`, which fine-tunes a checkpoint on a labelled CSV or JSONL file on your own machine. For a CSV with a text column and a label column:

```bash
laya-train --data tickets.csv --text-column body --label-column department \
  --instructions "Which department should handle this?" --dry-run
```

`--dry-run` only checks the file and prints how many rows it can use. To train, replace it with an output folder and, ideally, a held-out file for a before/after score:

```bash
laya-train --data tickets.csv --text-column body --label-column department \
  --instructions "Which department should handle this?" \
  --base models/laya --eval holdout.csv --out my-checkpoint \
  --micro-batch 1 --grad-accum 16
```

Then load the result with `laya.load("my-checkpoint")`. Use hundreds of labelled rows or more. With very small data, lower `--grad-accum` or raise `--epochs`, otherwise the model gets only a handful of updates. `--micro-batch 1` keeps memory use low on a 16 GB laptop. On Windows, write each command on one line, since `\` does not continue lines in PowerShell.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `pip` times out or cannot connect | Corporate proxy. Set `HTTPS_PROXY=http://proxy.host:port` (ask IT for the address) before `pip` and `hf`, or pass `--proxy` to pip. |
| `SSL: CERTIFICATE_VERIFY_FAILED` | The proxy re-signs HTTPS traffic. Get the company root certificate from IT and point both tools at it: `pip config set global.cert C:\path\corp.pem`, plus `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` set to the same file. |
| `hf download` fails with `File not found` | Two patterns after one `--exclude`. Repeat the flag for each pattern. |
| `hf: command not found` | The `.venv` is not activated. Activate it, or run `.venv/bin/hf` (`.venv\Scripts\hf` on Windows). |
| Model loading hangs | A TensorFlow install elsewhere on the path can deadlock loading. Set `USE_TF=0` before running; the examples already do this. |
| Python 3.9 or older | Laya needs 3.10 or newer. Install a newer Python and recreate the `.venv`. |
| `Cannot find an appropriate cached snapshot` | `HF_HUB_OFFLINE=1` is set but the file being loaded is not in `models`. Check section 2, or unset the variable if you have network access. |

**To remove everything**, delete the two folders the setup created: `rm -rf .venv models` on macOS or Linux, `Remove-Item -Recurse -Force .venv, models` in PowerShell. If you ran the `laya` command-line tool, also delete `~/.cache/huggingface/hub/models--convaiinnovations--laya`.
