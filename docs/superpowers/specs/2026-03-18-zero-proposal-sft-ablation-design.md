# Zero-Proposal SFT Ablation — Design Spec

**Date:** 2026-03-18
**Status:** Approved
**Research question:** Can the SFT-trained Qwen3VL tool-use agent still produce valid HOI results when no YOLOE object detection proposals are provided?

---

## Overview

Create zero-proposal variants of the 4 SFT evaluation scripts (SWIG action referring, SWIG grounding, HICO action referring, HICO grounding) to enable ablation study comparing performance with vs. without proposals.

---

## Files to Create

### Python eval scripts

| New file | Source |
|---|---|
| `eval_swig_action_referring_sft_noprop_qwen3vl.py` | `eval_swig_action_referring_sft_qwen3vl.py` |
| `eval_swig_ground_sft_noprop_qwen3vl.py` | `eval_swig_ground_sft_qwen3vl.py` |
| `eval_hico_action_referring_sft_noprop_qwen3vl.py` | `eval_hico_action_referring_sft_qwen3vl.py` |
| `eval_hico_ground_sft_noprop_qwen3vl.py` | `eval_hico_ground_sft_qwen3vl.py` |

### Bash wrappers

| New file | Source |
|---|---|
| `run_swig_action_sft_noprop_eval.sh` | `run_swig_action_sft_eval.sh` |
| `run_swig_ground_sft_noprop_eval.sh` | `run_swig_ground_sft_eval.sh` |
| `run_hico_action_sft_noprop_eval.sh` | `run_hico_action_sft_eval.sh` |
| `run_hico_ground_sft_noprop_eval.sh` | `run_hico_ground_sft_eval.sh` |

---

## Design Decisions

### Option A: Keep SYSTEM_PROMPT identical

The SYSTEM_PROMPT is kept **exactly as-is** from the original scripts. Rationale:

- The ablation changes exactly **one variable**: presence of proposals in the user message.
- The model was SFT-trained with this exact system prompt — changing it introduces a second variable that confounds the ablation signal.
- Any performance delta is attributable solely to the absence of proposals.

### Remove proposals block entirely (not empty array)

The `<proposals>` / `<candidate_objects>` block is **completely removed** from the user prompt. Sending `[]` is not a clean zero-proposal condition — it still tells the model "here are your proposals: none." Removing the block entirely gives the cleanest signal.

---

## Changes per Script Type

### Action referring scripts (swig + hico)

**`build_referring_prompt()`** — remove proposals parameter and the proposals intro + `<proposals>` block. The real source structure is:

```
"You will be identifying and describing the action...
Here are the candidate object proposals detected in the image:\n\n
<proposals>\n{proposals_text}\n</proposals>\n\n
The person you need to analyze is located at: ..."
```

After change — remove the two lines above ("Here are the candidate object proposals..." and the `<proposals>` block). Result:

```python
def build_referring_prompt(person_bbox_1000, object_bbox_1000):
    return (
        "You will be identifying and describing the action that a person "
        "is performing with a specific object in an image.\n\n"
        f"The person you need to analyze is located at: **{person_bbox_json}**\n\n"
        f"The object they are interacting with is located at: **{object_bbox_json}**\n\n"
        "Your task is to describe the action..."  # rest unchanged
    )
```

In `eval_model()`:
- Skip `load_proposals()` call entirely
- Call `build_referring_prompt(person_bbox_1000, object_bbox_1000)` — no proposals arg
- Remove `missing_proposals` counter and related print output
- Remove `"missing_proposal"` field from the partial checkpoint record dict (the `partial_rec` dict written to `.json.partial.jsonl`)
- Remove `rec.get("missing_proposal")` from the resume loading block

**Dead functions to remove:** `load_proposals()` and `format_proposals()` — no longer called.

**Argparse:** Remove `--proposals-dir` argument.

---

### Grounding scripts (swig + hico)

**`build_grounding_prompt()`** — remove proposals parameter, the entire proposals description sentence, the `<candidate_objects>` block, and the proposals-dependent bullet from the second `parts.append` block.

The full opening sentence to remove from `parts[0]`:
```
"You will be given a JSON array of candidate object proposals detected in an image, "
"each with a bounding box (``bbox_2d`` in 1000x1000 normalized coordinates), label, "
"and confidence score. Your goal is to identify specific objects..."
```
Replace with clean intro:
```
"Your goal is to identify specific objects and their spatial relationships based on the task description provided."
```

The connector line `"Here is the task you need to complete:\n\n"` is **preserved**.

The second `parts.append` block contains this bullet that must be **removed**:
```
"- **Objects visible in the image but missing from proposals** — the "
"detector may not have found every relevant person or object. If you "
"see additional interacting pairs in the image, estimate their "
"bounding boxes from visual inspection and include them.\n\n"
```

After changes:

```python
def build_grounding_prompt(action, object_category, is_person_person=False):
    parts = [
        "You will be performing a visual grounding task. "
        "Your goal is to identify specific objects and their spatial "
        "relationships based on the task description provided.\n\n"
        "Here is the task you need to complete:\n\n"
        # task description unchanged ...
    ]
    # second parts.append block: remove the "missing from proposals" bullet,
    # keep all other instructions (spatial proximity, semantic relationships,
    # object labels, formatting guidelines, answer format)
```

In `eval_model()`:
- Skip `load_proposals()` call entirely
- Call `build_grounding_prompt(action, object_category, is_person_person)` — no proposals arg
- Remove `missing_proposals` counter and related print output
- Remove `"missing_proposal"` field from the partial checkpoint record dict
- Remove `rec["missing_proposal"]` from the resume loading block (currently a hard dict access that would `KeyError` on resume if field is absent)

**Dead functions to remove:** `load_proposals()` and `format_proposals()` — no longer called.

**Argparse:** Remove `--proposals-dir` argument.

---

## Changes to Bash Scripts

| Element | Original | Zero-Proposal |
|---|---|---|
| Output dir | `results-sft/swig_action_sft` | `results-sft/swig_action_sft_noprop` |
| Python script called | `eval_swig_action_referring_sft_qwen3vl.py` | `eval_swig_action_referring_sft_noprop_qwen3vl.py` |
| `PROPOSALS_DIR` variable | present | removed |
| `--proposals-dir` flag | passed to Python | removed |
| Proposals dir warning/check block | present | removed |
| `echo "Proposals: $PROPOSALS_DIR"` in header | present | removed |
| Header echo title | `"... (SFT Qwen3VL)"` | `"... (SFT Qwen3VL - No Proposals)"` |
| `PRED_FILE` / `LOG_FILE` prefix | `swig_action_sft_results_` | unchanged — output dir `_noprop` suffix is sufficient disambiguation |

All other logic unchanged: GPU handling, vLLM health check, auto-start server, resume support, `IMAGE_ID`, `MAX_IMAGES`, `VERBOSE`, `WANDB`, `MAX_TURNS`.

---

## Output Structure

```
results-sft/
├── swig_action_sft_noprop/
├── swig_ground_sft_noprop/
├── hico_action_sft_noprop/
└── hico_ground_sft_noprop/
```

---

## What Is NOT Changed

- `SYSTEM_PROMPT` — identical to original
- Multi-turn tool-call loop (`run_sft_agent_loop`)
- All metrics computation (METEOR/CIDEr/BLEU/ROUGE-L for action; COCO AR for grounding)
- Resume support, partial checkpoints
- Visualization logic
- W&B logging
- All other argparse arguments
