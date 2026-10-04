"""User-authorized WebQSP test demonstrations with recorded KG verification."""
from hashlib import sha256
import json
from pathlib import Path

EXAMPLE_PATH = Path(__file__).with_name("resources") / "glm_webqsp_error_examples.json"


def glm_webqsp_examples(action):
    if action not in {"review", "rewrite"}:
        raise ValueError("WebQSP GLM examples support review and rewrite only")
    data = json.loads(EXAMPLE_PATH.read_text())
    blocks = [
        "Temporal/numeric/first repairs from WebQSP and CWQ. Every original is nonperfect and its authored repair passed the full downstream pipeline. Apply only matching defects; preserve otherwise."
    ]
    for example in data["examples"]:
        verification = example["verification"]
        path_hash = sha256(json.dumps(example["corrected_decomposition"], ensure_ascii=False).encode()).hexdigest()
        graph_hash = sha256(json.dumps(example["verification_graph"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if (verification.get("exact_ids") is not True
                or verification.get("full_pipeline_verified") is not True
                or type(verification.get("baseline_f1")) not in {int, float}
                or not 0 <= verification["baseline_f1"] < 1
                or verification.get("candidate_f1") != 1
                or type(verification.get("candidate_f1")) not in {int, float}
                or verification.get("corrected_decomposition_sha256") != path_hash
                or verification.get("verification_graph_sha256") != graph_hash):
            raise ValueError(f"Unverified/changed GLM example: {example['source']['index']}; run scripts/verify_glm_webqsp_examples.py")
        issues = [example["prompt_issue"]] if example.get("prompt_issue") else example["issues"]
        if action == "rewrite" and not issues:
            continue
        if not issues and example["original_decomposition"] != example["corrected_decomposition"]:
            raise ValueError("An accepted preservation example must retain the original paths")
        payload = {"question": example["question"], "decomposition": example["original_decomposition"]}
        if action == "review":
            output = {"is_reasonable": not bool(issues), "issues": issues,
                      "reason": "Required target/condition differs." if issues else "The editions path already retrieves the requested books."}
        else:
            payload.update(issues=issues)
            output = {"decomposition": example["corrected_decomposition"]}
        # Never supply oracle IDs, Gold labels, verification queries or scores
        # as model inputs, even for the examples.
        blocks.extend(["WebQSP example request: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                       "WebQSP example response: " + json.dumps(output, ensure_ascii=False, separators=(",", ":"))])
    return "\n".join(blocks)
