# Matrix Experiment Plan (GenData + Opportunity)

## Scope

- Goal: fill mentor matrix rows with reproducible experiments.
- RocML means predefined rule/threshold events.
- ML generation in pipeline remains inference-only.
- Training, inference execution, and evaluation are done manually.
- In this iteration, Accuracy and AUC are reported as N/A; primary metrics are Recall and F1.

## Target Matrix

| Generiranje | Testiranje | Natančnost | Priklic | F1   | AUC |
| ----------- | ---------- | ---------- | ------- | ---- | --- |
| RocML       | RocML      | N/A        | fill    | fill | N/A |
| RocML       | SVN        | N/A        | fill    | fill | N/A |
| SVN(Op)     | SVN        | N/A        | fill    | fill | N/A |

## Global Rules

1. Freeze one Opportunity train/test split and use it for all matrix rows.
2. Never use Opportunity test labels for training or pseudo-labeling.
3. Keep run manifests for every run: row_id, split_id, train_dataset_id, infer_dataset_id, model_id, threshold, seed, commit.
4. For pipeline ML mode, keep leakage_guard enabled and ensure:
   - model_meta.trained_on_dataset_id != inference_meta.inference_dataset_id

## Phase 1: Prepare Shared Split ✅

1. Create one Opportunity split (time holdout) and keep it fixed. (s1-drill-run)
2. Define split_id (example: oppo_t80_v1).
3. Reuse the same test files in all rows.

Suggested script:

- evaluation/split-data.py

## Phase 2: Prepare Synthetic RocML Training Data ✅

1. Generate synthetic signals/events with predefined event generation.
2. Save dataset_id (example: synth_rocml_seed42_v1).

Suggested script:

- GenData/pipeline/run_pipeline.py with event_gen=predefined

## Phase 3: Row RocML -> SVN ✅

1. Train SVM model on synthetic RocML train data.
2. Run SVM inference on frozen Opportunity test signals.
3. Evaluate against frozen Opportunity test events.
4. Record Recall/F1.

Suggested scripts:

- baselines/svm-baseline/train.py
- baselines/svm-baseline/infer.py
- evaluation/evaluation.py

Note:

- This phase is the mentor row `Generiranje: RocML, Testiranje: SVN`.

## Phase 4: Row RocML -> RocML

1. Apply RocML rule-based detector on frozen Opportunity test signals.
2. Evaluate against frozen Opportunity test events.
3. Record Recall/F1.

Suggested scripts:

- GenData/generator/signal_generation_tools.py (rule definitions / criteria)
- evaluation/evaluation.py

Note:

- This row requires a rule-based RocML tester; SVM scripts do not cover this row.

## Phase 5: Row SVN(Op) -> SVN

1. Train SVM on Opportunity train split.
2. Run SVM inference on Opportunity test split.
3. Evaluate against Opportunity test events.
4. Record Recall/F1.

Suggested scripts:

- baselines/svm-baseline/train.py
- baselines/svm-baseline/infer.py
- evaluation/evaluation.py

## Phase 6: Consolidate Final Table

1. Fill Recall/F1 for all three rows.
2. Keep Natančnost and AUC as N/A in this cycle.
3. Add a short methods note that current evaluation is temporal-overlap based.

## Artifacts Per Matrix Row

- trained model artifact (model.pkl and related files)
- detected events artifact
- evaluation report artifact
- run manifest (json or md)

## Recommended Folder Layout

- evaluation/results/RocML_to_RocML/
- evaluation/results/RocML_to_SVN/
- evaluation/results/SVNOp_to_SVN/

Each folder should contain:

- detected events output
- evaluation output
- run_manifest.json (or run_manifest.md)

## Minimal Completion Checklist

- [ ] Same split_id across all rows
- [ ] No test-label leakage
- [ ] Recall/F1 computed for all rows
- [ ] Natančnost/AUC marked N/A
- [ ] All run manifests stored
