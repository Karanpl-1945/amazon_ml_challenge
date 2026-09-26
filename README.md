# Amazon ML Challenge 2026 — Business Entity Resolution

The code lives in `6a7c763e089e4_aic_2026/student_resource/`. The data is **not** in this repo.

## Quick start
1. Clone this repo.
2. Copy the challenge `dataset/` folder (with `train/` and `test/` TSVs) into
   `6a7c763e089e4_aic_2026/student_resource/dataset/`.
3. Follow [`code/business_entity_resolution/README.md`](6a7c763e089e4_aic_2026/student_resource/code/business_entity_resolution/README.md):
   ```bash
   cd 6a7c763e089e4_aic_2026/student_resource/code/business_entity_resolution
   python -m pip install -r requirements.txt
   python src/run_pipeline.py --workers 8
   ```
4. Results: `student_resource/output/matching_results.tsv` (+ `candidate_pairs.tsv`);
   full log with validation scores: `student_resource/work/pipeline_log.txt`.
