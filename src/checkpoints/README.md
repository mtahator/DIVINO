# Trained operator weights

`pino.pt` is created by the training script (run from `src/`):

```bash
python train.py --N 128 --steps 9000 --out checkpoints/pino.pt
```

This is the configuration used for the results in the paper (Fourier neural operator with 4 spectral
layers, width 48, 24 retained modes, 10.6 M parameters, trained on a one/two-cavity mixture of
geometries, sources, and convective coefficients). The notebooks and the inverse and benchmark scripts
load the weights from this file. Large weight files are tracked with Git LFS (see `.gitattributes`).
