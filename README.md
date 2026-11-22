# TomatoMAP benchmark audit

Code, split lists and archived numerical results for a study of annotation
convention integrity and sampling-aware evaluation on the public TomatoMAP
four-camera RGB tomato imaging benchmark.

The repository is self-contained for the numerical claims: the AP values,
paired comparisons, cluster-resampling outputs, training-scope audit and
upstream annotation verification can be inspected and re-derived without
downloading a single photograph.

## What this repository contains

* `scripts/` — export, label repair, split construction, training, evaluation,
  paired comparison, cluster resampling, power scan and verification tools.
* `tables/` — archived result tables (T0, T1, T2, T7, T8, T13, T14, T15, T19,
  T20) exactly as used in the study.
* `work/tm_yolo/` — class list, image index, split membership lists, the two
  paired image lists, per-run manifests, training histories and the archived
  prediction statistics (`.json` and `.npz`) that the resampling scripts read.
* `USAGE.txt` — command-by-command notes, expected outputs and interpretation
  limits. Read it before rerunning anything.
* `release_metadata.json` — authors, funding, upstream DOIs, recorded
  dependency versions, initial-weight fingerprints and the list of archived
  inputs that each script expects.

## What is not included

Source photographs, exported label trees, detector checkpoints and pretrained
weights are not redistributed here. Nothing in this repository grants new
photographic rights; consult the upstream record for image licensing.

## Upstream data

* Data descriptor: <https://doi.org/10.1038/s41597-026-06926-9>
* Dataset: <https://doi.org/10.5447/ipk/2025/14> (CC BY 4.0)

Extract the upstream release so that `samples.json` and its image paths stay
intact, then point the package at it either by placing it in `data/TomatoMAP`
or by setting `TOMATOMAP_DATA_ROOT` in your shell. The upstream input is only
read; all generated material is written under `work/tm_yolo`.

## Environment

Python 3.10.9 was the recorded interpreter.

```
python -m venv .venv
python -m pip install -r requirements.txt
```

The recorded GPU build used `torch 2.6.0+cu124` and `torchvision 0.21.0+cu124`;
install those two from <https://download.pytorch.org/whl/cu124> before
`requirements.txt` if you need the same CUDA build. The pins record what was
observed, they are not recommendations for newer releases.

## Checking the archived results without images

```
python scripts/tm_archive_check.py
python scripts/sampling_audit.py
```

The first command verifies the five paired prediction archives, image order,
identical paired ground truth, plant counts, planned epochs and the archived
T7 values. The second reports split assignment separately from the camera
scope that actually entered training. Cluster resampling and power analyses
that reuse the archived arrays are documented in `USAGE.txt`.

## Rebuilding labels, splits and runs

```
python scripts/tm_export.py
python scripts/tm_prepare.py
python scripts/tm_train.py --protocol random_img --model yolo11s --imgsz 800 --epochs 60 --batch 16 --workers 12 --tag b1
python scripts/tm_eval.py --run work/tm_yolo/runs/random_img/random_img__yolo11s__img800__e60__b16__b1 --group camera --weights last.pt
```

`tm_export.py` rebuilds labels, index and splits and copies images locally;
`tm_prepare.py` only regenerates runtime configuration from the archived split
CSVs. Put `yolo11s.pt` and `yolo11n.pt` in `work/weights` before training and
compare the SHA256 values recorded in `release_metadata.json` if you need
identical initialisation. Repeated runs on different hardware are not expected
to be bit-identical.

Two further steps reproduce the convention and coverage analyses:

```
python scripts/verify_annotations.py --upstream data/upstream-labels --export work/tm_yolo/labels
python scripts/tm_export_broken.py
python scripts/tm_xcheck.py --run <run-directory>
python scripts/tm_xcheck_table.py
```

The first compares exported centre-format boxes with the annotation files of
the original release, matching is order independent and class constrained at a
tolerance of 2e-6, and the archived T20 table covers all 13,908 file
comparisons. `tm_export_broken.py` regenerates the raw-convention label set
that the cross-evaluation matrix needs. Run `tm_xcheck.py` for each trained
run, then collect the cells with `tm_xcheck_table.py`.

## Reading the comparisons

Camera `cam2_horizontal90` is the only camera whose images enter training and
validation; the other three cameras are evaluation domains only. Membership
overlap between views of the same acquisition set is reported as a descriptive
property of the partition, not as demonstrated leakage. Every reported
difference is a comparison between evaluation protocols, not a causal
decomposition of the channels that produce it. Both paired subsets contain 31
plants, so inference is by whole-plant cluster, and rare-class support should
be checked before a group average is compared across domains.

## Related repository

`tomatomap-stage-view-evaluation` holds the companion support-aware study of
viewpoint transfer and localisation across developmental stages. The two
packages share the same upstream source and the same split conventions; the
reverse-stage split built here is archived there.

## Rights and citation

Code copyright is retained by Lifen Tu and Qi Peng. No software licence has
been selected for this archive; dataset, dependency and model licences are
separate. Cite the upstream descriptor and dataset when using these results.

## Authors and funding

Lifen Tu (<https://orcid.org/0000-0002-5774-8138>) and Qi Peng
(<https://orcid.org/0000-0002-5693-0480>), School of Physics and Electronic
Information Engineering, Hubei Engineering University, Xiaogan, China.

Supported by the Natural Science Foundation of Hubei Province (2026AFB666) and
the Hubei Provincial Natural Science Foundation Xiaogan Innovation and
Development Joint Fund Key Project (2026AFC0666).
