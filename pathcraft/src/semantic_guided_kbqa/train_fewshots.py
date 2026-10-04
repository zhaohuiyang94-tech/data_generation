"""Few-shot data from source-verified train splits, not anonymous test derivatives."""
import json
from .prompt_principles import question_requirements
from pathlib import Path

RESOURCE_DIR = Path(__file__).with_name("resources")
DECOMPOSITION_DATA = json.loads((RESOURCE_DIR / "decomposition_train_examples.json").read_text())
OPERATOR_DATA = json.loads((RESOURCE_DIR / "temporal_operator_train_examples.json").read_text())


def decomposition_examples(action):
    blocks = ["Examples are verified train questions and path outputs. Bad paths are controlled corruptions of those same train paths; no factual answers are supplied."]
    for item in DECOMPOSITION_DATA["examples"]:
        q, good = item["question"], item["decomposition"]
        # Critics use genuine positive train examples. Negative corruption
        # templates were empirically copied into unrelated valid test paths.
        cases = ([(c["decomposition"], c["issues"]) for c in item["corruptions"]]
                 if action == "rewrite" else [(good, [])] + [(c["decomposition"], c["issues"]) for c in item["corruptions"] if c["mutation"] in {"invert_relation_direction", "drop_terminal_hop", "replace_requested_attribute"}])
        for paths, issues in cases:
            if action == "rewrite" and not issues:
                continue
            payload = {"question": q, "decomposition": paths, "question_requirements": question_requirements(q)}
            if action == "rewrite":
                payload.update(issues=issues, review_reason="Restore the lost meaning of this train path.")
                response = {"decomposition": good}
            else:
                if action == "confirm":
                    payload.update(review_issues=issues or ["First review claims the path is wrong."],
                                   review_reason="Check the actual current path before agreeing.")
                response = {"is_reasonable": not bool(issues), "issues": issues,
                            "reason": "The whole current path covers the requested meaning." if not issues else issues[0]}
            blocks.extend(["Example request: " + json.dumps(payload, ensure_ascii=False),
                           "Example response: " + json.dumps(response, ensure_ascii=False)])
    return "\n".join(blocks)


def operator_examples():
    blocks = ["Operator examples from official CWQ train, excluded from both benchmark test question sets:"]
    for item in OPERATOR_DATA["examples"]:
        blocks.extend(["Example input: " + json.dumps(item["input"], ensure_ascii=False),
                       "Example output: " + json.dumps(item["output"], ensure_ascii=False)])
    return "\n".join(blocks)
