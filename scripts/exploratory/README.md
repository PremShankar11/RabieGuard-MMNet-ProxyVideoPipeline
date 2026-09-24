# Exploratory and Dataset Inspection Scripts

This directory contains standalone, ad-hoc inspection and exploratory scripts used during initial dataset discovery, metadata verification, and environment probing.

## Contents
- `inspect_352.py`, `count_species_352.py`: Verification of species counts across the Animal Kingdom 352-clip subset.
- `inspect_ak_annotations.py`, `inspect_ak_videos.py`: Structure and integrity validation of Animal Kingdom annotations and raw video files.
- `inspect_bioacoustic_refs.py`: Investigation of multi-modal bioacoustic references in canine datasets.
- `inspect_dog_pose.py`: Schema and keypoint validation for Ultralytics Dog-Pose dataset.
- `inspect_excel_metadata.py`, `check_sheets.py`, `list_all_sheets.py`: Excel workbook structure audits for dataset annotations.
- `inspect_multi_actions.py`: Analysis of concurrent and sequential multi-action labels in video segments.
- `inspect_notebook.py`: Utility to inspect cell outputs in exploratory Jupyter notebooks.
- `inspect_social_play_metadata.py`, `search_social_play.py`: Querying social play datasets for canine behavioral markers.
- `inspect_zenodo_api.py`: Zenodo API queries for public canine open-science archives.
- `calc_video_stats.py`, `compare_dog_actions.py`: Statistical breakdown of video lengths, resolutions, and action frequencies.
- `test_m1_m2.py`: Diagnostic script testing M1 vs M2 temporal sampling strategies.
- `scaffold.py`: Initial workspace folder structure scaffolding tool.
