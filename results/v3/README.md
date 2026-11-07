# v3 final summaries

Fetched 2026-11-08 05:30–05:32 JST from `/mnt/home/unno/lab/rubiks_project/dreamcoder-rubik/results/v3` (shared NFS), via banshee. These are small experiment JSON outputs only, not checkpoints or raw datasets. Interrupted `.killed.json` files are excluded.

24/24 summaries contain 20 history entries, final arguments and E2E metrics. The 15 relaunched runs are all 12 `full_*` runs plus the three `stages_nonn_nolib_*` runs; the nine other staged runs finished earlier. Logs end in `E2E SOLVED`; no experiment runner was observed on banshee, bellatrix or herzog at the check. CPU/GPU utilization was inspected before any other server work; no new training was launched.

Rebuild and validate from the repository root:

```sh
python3 experiments/export_web.py
node experiments/verify_viewer.cjs
node --check docs/app.js
```

Exporter verifies 2,023 training/E2E solutions using Python goals. The viewer-model check independently replays all 716 E2E solutions through the JavaScript cube model, including orientation changes at saved stage boundaries. Actual browser rendering remains unverified: browser navigation to the local preview was blocked by tool policy during this job.
