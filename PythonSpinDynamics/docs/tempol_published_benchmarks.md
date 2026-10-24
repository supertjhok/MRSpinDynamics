# Published TEMPOL/water benchmarks

Run from `PythonSpinDynamics` with the normal numerical/plot extras:

```bash
python validation/validate_tempol_water.py --output .tmp/tempol_report.json
python examples/plot_tempol_published_benchmarks.py \
  --output .tmp/tempol_comparison.png --report .tmp/tempol_report.json
python -m unittest tests.test_tempol_published_benchmarks
```

The example writes two PNGs: the requested path and a sibling `_anchors.png`.
Runs are offline, deterministic, and perform no parameter fitting. The frozen
data file `tempol_water_published.json` contains source locations, units,
extraction coordinates, experimental conditions and reduction choices.
`tempol_water_benchmark_report.json` records the current descriptive results.
Its data digest is SHA256 of canonical sorted JSON, independent of line endings.
No publisher PDF or source figure is redistributed.

## Evidence and results

| Comparison | Evidence | Result | Interpretation |
| --- | --- | --- | --- |
| 9.21238 T DNP frequency profile | 31 extracted experimental marker clusters, Sezer Fig. 5 | RMSE 0.48285, MAE 0.40420, largest absolute residual 1.05857 in epsilon | Partial empirical agreement; systematic shape differences remain |
| Saturation frequency profile | 19 manually sampled points on Sezer Fig. 6c | RMSE 0.01751, largest absolute residual 0.04476 | Published theory comparison, not measured saturation |
| Room-temperature EPR saturation product | Published fitted parameter, compared with another study's relaxation parameters | Products differ by 4.32286 times | Cross-study compatibility diagnostic; no statistical rejection without uncertainties |
| 0.34/3.4 T saturation and enhancement anchors | DNP-derived saturation and approximate enhancement maxima | Factor closures -98.59/-24.79 versus roughly -100/-20 | Consistency check; saturation and enhancement are dependent observations |

The descriptive errors weight extracted markers equally. They are not reduced
chi-square statistics or estimates of independent experimental sample size.
The fixed reduction envelopes, RMSE <=0.8 in enhancement and <=0.08 in
saturation, are engineering regression bounds. Passing them does **not** mean
agreement within experimental uncertainty. Point extraction half-widths are
about 1 MHz and 0.05 in enhancement; experimental error bars were unavailable.
Residuals exceed extraction precision. No measurement covariance is inferred.

## Primary source and reduction

[Sezer et al., PCCP 2009, DOI 10.1039/B906719C](https://doi.org/10.1039/B906719C)
is the primary comparison. The Fig. 5 data are 14N TEMPOL in water, 3.15 mM,
at approximately 313 K. Table 5 and Eqs. 38/41 supply the field, microwave
amplitude, hyperfine tensor, relaxation times, exchange time, leakage and
coupling factors. The production model uses the central-line T2 on all lines,
isotropic hyperfine spacing and no independently inferred nitrogen rate. It
omits the source calculation's full relaxation matrix, differential linewidths,
coherence transfer and shifts. These are stated reductions, not fitted terms.
Source parameters themselves include estimates and phenomenological choices.
This benchmark does not independently validate microscopic motion or FFHS
absolute rates: coupling and leakage are supplied from the publication.

The source convention is `epsilon=(M_on-M_off)/M_off=E-1`; production CW output
includes the thermal baseline. The benchmark subtracts one explicitly. The
source horizontal axis is microwave minus resonance frequency, opposite to
the API detuning. Microwave B1 is treated as the co-rotating amplitude; gauss
becomes tesla by multiplication by 1e-4. Exchange conversion uses `k=1/tau_H`
and `K=k/C` under the correlation-erasure convention. A 0.8-1.0 G B1 band
shows sensitivity only, not a confidence interval.

## Extraction audit

For Fig. 5, the original embedded JPEG is 375x304 pixels. Red marker outlines
are thresholded with `R>G+35` and `R>B+25`; enclosed interiors are identified
by hole filling and 4-connected components. Fragments with centers within four
pixels are merged, and centers are area weighted. This produces 31 clusters;
open/compressed squares are omitted. Nearby fragments can share a frequency,
so these clusters are not asserted to be distinct instrument acquisitions.
The frozen centers were checked visually against the square markers. Linear
axis calibration uses two tick anchors on each axis. Fig. 6c instead uses
19 frozen manual pixel samples on the magenta total-saturation curve.

To audit the exact source PDF, obtain the author-hosted URL in the JSON and run:

```bash
python validation/validate_tempol_water.py --source-pdf /path/to/PCCP_11_6638_6653.pdf
```

This optional audit requires PyMuPDF, Pillow and SciPy. It verifies the PDF and
embedded-image SHA256 digests, then re-extracts Fig. 5 marker centers and checks
them against the frozen transcription. The normal benchmark needs no PDF or
image-extraction dependency. The source PDF's zero-based pages 10/11 correspond
to printed pages 6646/6647. The JSON preserves every native pixel coordinate,
calibration anchor and extraction precision assumption.

## Additional diagnostics and limits

[Prandolini et al., JACS 2009, DOI 10.1021/ja901496g](https://doi.org/10.1021/ja901496g)
reports a TEMPOL/water homogeneous EPR saturation product of 3.5e-15 s² and
a linewidth-derived T2 of 28 ns. The plotted EPR response is reconstructed
from that fit parameter, not digitized raw EPR data. It is compared with
Sezer's room-temperature central-line product, 850 ns times 17.8 ns. Isotope,
sample and method differences preclude treating the discrepancy as a measured
failure at identical conditions or selecting one set as universal calibration.

[Höfer et al., JACS 2008, DOI 10.1021/ja0783207](https://doi.org/10.1021/ja0783207)
provides lower-field saturation estimates inferred from DNP and NMRD. Their
reported plus/minus values are plotted without interpreting them as standard
deviations. The NMRD identity reconstructs leakage for a factor-closure check.
Approximate enhancement maxima span 5-10 mM, whereas the NMRD reference is
10 mM. Full source PDF access was unavailable; numeric text and the repository
abstract were transcribed with locations recorded. The baseline convention of
these rough maxima is not independently verified; an offset of one remains
unresolved. No independently supplied B1/T1e/T2e supports a population-model
saturation prediction for these anchors.

Independent TEMPOL/water ELDOR saturation data with matched electron relaxation,
microwave calibration and irradiation temperature remain needed for a stronger
experimental saturation claim. TEMPONE, toluene, frozen samples and mixed
nitrogen-isotope datasets have not been substituted for TEMPOL/water controls.
