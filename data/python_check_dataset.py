
import json

file_path = "data/benchmark.jsonl"
records = []

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            records.append(json.loads(line))

print("DATASET VALIDATION")
print("-------------------")
print("Total prompts:", len(records))

scenario_ids = set(r["scenario_id"] for r in records)
print("Total scenarios:", len(scenario_ids))

print("\nLanguage formats:")
for variant in ["original", "native_script", "transliteration"]:
    count = sum(1 for r in records if r["variant"] == variant)
    print(variant, ":", count)

print("\nValidation completed!")
