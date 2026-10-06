# A0 poster

`abide_poster_A0.pptx` is an A0 portrait (841 x 1189 mm) PowerPoint poster summarising both
pipelines for readers without a machine-learning background. Open it in PowerPoint to edit text
or the native charts.

To rebuild it after new results (it reads `results/pitt10_check/*.csv`; the pipeline A numbers are
the README results table):

```bash
cd poster && npm install && node build_poster.js .. abide_poster_A0.pptx
```

`slice_*.png` are three real model inputs (one time point of `Pitt_0050003`, middle cut of each
plane), exported from `data/slices/Pitt_0050003.npz` and upscaled 8x without smoothing.
