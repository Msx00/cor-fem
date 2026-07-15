# COR-FEM NIfTI image registration on Windows

The entry point is `run_fem_image_registration.py`. It reads the original MR
and US NIfTI files, extracts `label0` surfaces in physical LPS millimetres,
runs the existing COR-FEM surface registration, and uses the resulting
tetrahedral deformation to pull-resample MR image and labels onto the US grid.

## Single case

```powershell
python .\run_fem_image_registration.py `
  --data-root ".\test" `
  --case case_0000 `
  --output-root ".\outputs" `
  --outside-mode identity `
  --label-outside-mode zero
```

## All cases

```powershell
python .\run_fem_image_registration.py `
  --data-root ".\test" `
  --all-cases `
  --output-root ".\outputs" `
  --continue-on-error
```

Use `--dry-run` to inspect required files and MR/US geometry without running
surface extraction, COR-FEM, or resampling.

## Warped outputs

Each case contains a `warped` directory with explicit mapping names and short
aliases:

```text
warped/
  warped_mr_image.nii.gz
  image.nii.gz
  warped_label0.nii.gz ... warped_label5.nii.gz
  warped_label0.stl    ... warped_label5.stl
  label0.nii.gz        ... label5.nii.gz
  label0.stl           ... label5.stl
  warped_label_map.nii.gz (when MR label_map exists)
  fixed_us_image.nii.gz
```

All warped NIfTI files copy size, spacing, origin, and direction from
`us/image.nii.gz`. Warped STL files are extracted from the already warped label
NIfTI files, so the surface and volume outputs share the same US physical
space.

MR intensities use linear interpolation. Binary labels use nearest-neighbour
interpolation and remain uint8 with values 0/1. Labels and `label_map` use
`--label-outside-mode zero` by default: only positions reached through the FEM
inverse map can be foreground. This prevents an undeformed MR label from being
copied into the US grid outside the tetrahedral domain.

## Deformation scope

The default `--outside-mode identity` applies FEM mapping inside the prostate
tetrahedral mesh. Outside the gland, MR is sampled at the same physical point.
Thus the FEM deformation is strictly defined inside the prostate; smooth
deformation of surrounding tissue requires a larger tetrahedral domain or a
displacement extrapolation model.

`source_surface_after_registration.stl` is the direct FEM deformation of the
smoothed source surface used for registration. `warped_label0.stl` is instead
extracted from a 0.8-mm nearest-neighbour warped NIfTI label. They are in the
same LPS physical space but will not have identical triangle-level detail:
surface smoothing and voxelization remove or alter sub-voxel features.

## Evaluate warped labels

`evaluate_warped_labels.py` compares each warped output label with the matching
US reference label. It reports TRE, Dice, binary-label MI, HD95, CD/ASSD, and
directed ASD in millimetres where applicable.

```powershell
python .\evaluate_warped_labels.py `
  --outputs-root ".\outputs" `
  --test-root ".\test" `
  --all-cases
```

Results are written to `outputs/label_evaluation/warped_label_metrics.csv` and
`warped_label_summary.csv`. The script requires matching NIfTI geometry and
records a mismatch instead of silently resampling labels.
