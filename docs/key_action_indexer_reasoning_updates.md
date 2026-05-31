# Key Action Indexer Reasoning Updates

## T-06/T-07 SOP And History Scoring

- `process_reasoner` evaluates serializable SOP condition objects in `branch_condition`, `entry_conditions`, and `completion_conditions`.
- Supported condition keys include `all_actions`, `any_actions`, `not_actions`, legacy `when_any_action_observed`/`unless_action_observed`, `required_material`, `min_confidence`, and `max_elapsed_sec`.
- If `metadata/history_model.json` exists, process output includes top-level and step-level `history_prior` and `history_deviation`.
- Missing-step inference records `history_basis` when action or transition priors support the inferred completion.
