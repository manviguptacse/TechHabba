# Vaksha Benchmark Dataset

## Purpose
This dataset is used to compare how the model responds to semantically equivalent prompts written in English, Kannada script, and Kannada transliteration (Kanglish).

## Dataset size
- 12 scenarios
- 3 language formats per scenario
- 36 prompt records total

## Scenario categories
- 6 benign / defensive scenarios
- 4 harmful-intent scenarios
- 2 ambiguous / educational scenarios

## Files
- `benchmark.jsonl`: one JSON object per line, containing one prompt variant.

## Fields
- `scenario_id`: ID shared by the three language versions of one scenario.
- `category`: topic of the scenario.
- `expected_label`: expected intent label (`benign`, `harmful`, or `ambiguous`).
- `language`: language identifier (`en`, `kn`, or `kn-Latn`).
- `variant`: representation (`original`, `native_script`, or `transliteration`).
- `prompt`: the actual prompt text.
- `expected_response_language`: language expected in the response.
- `translation_reviewed`: set to `true` only after a Kannada-fluent reviewer checks meaning preservation.
- `human_reviewed`: set to `true` only after a reviewer checks the record and label.
- `safe_for_demo`: whether the prompt is suitable for a public live demonstration.

## Important review note
The Kannada and Kanglish text in this starter dataset is a draft. A Kannada-fluent person must review spelling, naturalness, and whether the meaning is equivalent across all three versions. Keep review flags as `false` until that review is completed.

## How to read the JSONL file
Each line is a separate JSON record. Do not add commas between lines. Keep each record on one line.

## Suggested validation checklist
1. Confirm all three versions express the same intent.
2. Confirm the expected label matches the intent.
3. Check Kannada spelling and natural phrasing.
4. Check that transliteration is understandable and consistent.
5. Mark review fields `true` only after human validation.
6. Keep harmful-intent examples non-operational and do not use them to request real-world instructions.

## Limitations
This is a small exploratory benchmark, not a statistically conclusive evaluation. Report actual counts and denominators, and do not claim perfect safety or perfect translation.
