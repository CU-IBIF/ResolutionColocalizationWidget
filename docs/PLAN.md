# ColocWidget: design and model

A companion to NoiseWidget and MicroscopeWidget that answers one question with
numbers: *for this specimen and this microscope, how much of a colocalization
coefficient is the molecules, how much is the resolution, and how much is the noise?*

---

## 1. Decisions

| # | Decision | Consequence |
|---|---|---|
| 1 | **Same stack and conventions as NoiseWidget** | numpy model, PySide6 UI, QPainter plots (no matplotlib), one-file PyInstaller build, offscreen Qt tests, the same dark panes and status colours, rows that carry their equation, the same CI and release workflows. |
| 2 | **The widget is a `QWidget`** | MicroscopeWidget can host it in a tab or dock; `ColocWindow` only adds the toolbar and status bar. |
| 3 | **Ground truth is generated** | Scenes place individual labels at known 3D positions, so every coefficient has a molecular answer to compare with. |
| 4 | **Three stages, one analysis** | The same coefficients on the molecules, the noise-free image and the recorded image. The noise-free image reuses the recorded frame's thresholds, so the two differ only in noise. |
| 5 | **Scenes declare their parameters** | The lever panel and `--set` discover them; adding a scene needs no UI code. |
| 6 | **Lessons are tests** | Every preset's claim is an assertion in `tests/test_imaging.py`. |

## 2. The imaging model

```
labels (x, y, z) --deposit--> planes D_k on a 512 x 512 grid (20 nm flat scenes; 50 nm x 0.4 um cells)
                 --FFT------> D_k(f)                                           cached on the scene
sum over planes:  S(f) = sum_k D_k(f) G(f; z_k - z_f) w(z_k - z_f)             cached on the axial optics
                  G = exp(-2 pi^2 sigma^2 f^2),  sigma = |dz| tan(theta) / 2   geometric defocus
                  w = 1 (widefield)  or  1 / (1 + (2 dz / FWHM_z)^2)           pinhole rejection
lateral:          I(f) = S(f) OTF(f; FWHM)   circular-pupil OTF, cutoff 1.029 / FWHM
registration:     channel 2 x exp(-2 pi i f.shift), focus + shift_z
pixels:           exact box integral over pixels of any size
photons:          each channel scaled so its brightest 0.1% of pixels reach the peak; + beta ch1 into ch2
noise:            Poisson(signal + background) + Gaussian read noise; repeat frames for the spread
```

* **Lateral FWHM** by modality: widefield `0.514 lambda / NA`; confocal at 1 AU
  `x 0.9`; pixel reassignment `/ 1.7`; STED as set (capped at the confocal spot).
* **Optical section** (1 AU): `sqrt((0.88 lambda / (n - sqrt(n^2 - NA^2)))^2 + (sqrt2 n PH / NA)^2)`
  (Cole et al. 2011).
* Flat scenes are one plane. Cells use planes 0.4 µm apart from the coverslip to the
  top of the cell.

## 3. The coefficients

Computed on background-subtracted (clipped at 0) pixels inside the region of
interest. Formulas and references are in `physics/metrics.py`:

* **Pearson's r**, **Spearman's rho**, **Li's ICQ**, **van Steensel's CCF** (r vs
  x-shift of channel 2), **nMDP** map.
* **Manders' M1, M2**, thresholded as in Coloc 2:
  `M1 = sum a[a > T1 and b > T2] / sum a[a > T1]`. **MOC**, **k1**, **k2**.
* **Thresholds**: Costes (orthogonal regression, bisection on r of the pixels not above
  both thresholds; falls back to Otsu when the slope is not positive), Otsu, a fixed
  fraction of the 99.9th percentile, or none.
* **Costes randomization**: channel 1 scrambled in PSF-sized blocks, 40 trials.
* **Spot-based**: local maxima (after a binomial smooth) above the thresholds, matched
  within the larger FWHM, against the Poisson chance level.

## 4. The truth

* **Molecular M1, M2**: fraction of one channel's labels with a label of the other
  within distance d (default 25 nm). This is exact, using a cell list of side d.
* **Molecular r**: Pearson's r of the two label counts in d-sized voxels, computed
  sparsely so the empty voxels enter only through their number.
* **Compartment overlap** (cells): share of each channel in compartments the other
  occupies. Two diffuse proteins sharing the nucleus are rarely within 25 nm of each
  other, so molecular and compartmental truths answer different questions. The
  widget reports both.

## 5. Model cell

A dome-shaped cell on the coverslip, `(x/a)^2 + (y/b)^2 + (z/h)^2 <= 1, z >= 0`, with an
ellipsoidal nucleus raised 0.5 µm off the glass. Labels are sampled uniformly:

* in the nucleus volume, with a per-channel fraction in 30 nuclear bodies;
* in the cytosol volume (cell minus nucleus), with a per-channel fraction in 250
  puncta;
* on the membrane, per unit area across the apical dome and the basal disc.

A body or punctum is a Gaussian cloud of labels whose set diameter is about 4σ.
Nuclear bodies are kept a radius inside the nucleus. Each compartment has two pools
of body positions: channel 1 uses pool A, and each of channel 2's punctate labels
uses pool A with probability "puncta shared" and pool B otherwise. Body positions
come from a child generator and are drawn only when some channel has puncta, so a
cell without puncta is the same specimen as before.

Regions of interest are drawn at the focal plane: the cell's cross-section, a rim
`max(0.4 µm, 1.5 FWHM)` wide for the plasma membrane, the nucleus's cross-section,
and the remainder for the cytosol. Each compartment's apparent share is its fraction
of the cell's recorded signal at that plane. The true share is the 3D fraction of
labels. The two are not expected to match even with perfect optics, because one
plane samples part of a 3D cell. What the widget compares is how far they diverge
between modalities.

## 6. What is not modelled

* Vectorial high-NA PSF, aberrations, refractive-index mismatch, and the axial PSF
  beyond the geometric-defocus and Lorentzian-section approximations.
* Wrap-around: the FFT is periodic, so haze from planes far out of focus can wrap at
  the field edge. The default cell sits ~3 µm from the edge, which keeps this small;
  the longest cells (24 µm) come within 1 µm.
* Fluorophore photophysics, bleaching and blinking. Brightness is set by peak photons.
* Spectral detail: bleed-through is one coefficient, with no cross-excitation.
* Linkage error, label size and labelling efficiency beyond the scene parameters.
* Vesicles are imaged in one plane. Their 3D shape enters the truth, not the defocus.

## 7. Layout

```
colocwidget/
  physics/          headless, numpy only
    optics.py         modalities, OTF by FWHM, defocus, optical section, registration, pixels
    scenes.py         specimens with known label positions, their parameters, the model cell
    metrics.py        every coefficient, thresholds, significance, objects, and the truth
    imaging.py        levers -> images -> three stages of coefficients; the sweeps; caching
    report.py         every derived number, with its equation
    presets.py        starting points that each make one point, by case
  ui/               PySide6
    widget.py         ColocWidget (embeddable), ColocWindow, graphs, coefficient and compartment tabs
    controls.py       the lever panel; scene parameters built from the scene
    panes.py          two-colour merges, the nMDP map, the cytofluorogram
    plots.py          QPainter plots with hover read-out (shared with NoiseWidget)
```
