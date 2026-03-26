# Dataset Format

Baseline annotation file: `data/raw/annotations.jsonl`

Example record:

```json
{"sample_id":"s1","rgb_path":"data/raw/img1.png","depth_path":"data/raw/depth1.npy","instruction":"pick the bottle","action_history":["observe"],"action_sequence":["approach_target","open_gripper","align_gripper","close_gripper","lift"]}
```

Required fields:

- `sample_id`
- `instruction`
- `rgb_path`
- `action_sequence`

Optional fields:

- `depth_path`
- `action_history`

Storage policy:

- Raw: `data/raw`
- Interim: `data/interim`
- Processed: `data/processed`
- Splits: `data/splits`
