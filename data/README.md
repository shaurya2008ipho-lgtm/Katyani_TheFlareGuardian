# Data placement

Place downloaded observations locally under `data/raw/` (this folder is git-ignored). Keep original files unmodified.

- `data/raw/solexs/`: SoLEXS Level-1 light curves
- `data/raw/hel1os/`: HEL1OS Level-1 light curves
- `data/processed/`: generated aligned curves; git-ignored
- `data/demo/combined_demo.csv`: synthetic sample used only by the dashboard demo

Before processing all files, inspect one representative file from each payload. A Type-II spectrum or event-list file may not itself be a light curve; choose the Level-1 light-curve product or explicitly derive a light curve from spectral/event data using the instrument documentation.
