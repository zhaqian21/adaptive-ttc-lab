"""Serve the released Qwen3-8B dPRM adapter via the general TTC scorer API."""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from serve_prm import STEP_MARKER, serve


REFERENCE_REVISION = "65b3e8e17a91a8a85807cb85035bddfd256db861"
ASSETS = {
    "base": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "adapter": ("dongboklee/dPRM-qwen", "7fd33134e5b8ffd42da28532e20aca710d718859"),
}


class DPRMQwen:
    def __init__(self, models, device, max_context):
        self.device = device
        self.max_context = max_context
        assets = json.loads((models / "downloads/dprm-ready.json").read_text(encoding="utf-8"))
        directories = {}
        self.manifest = {"reference_revision": REFERENCE_REVISION}
        for kind, (repo, revision) in ASSETS.items():
            expected = f"dPRM-qwen/{kind}"
            matches = [entry for entry in assets if entry["subdirectory"] == expected]
            if len(matches) != 1 or (matches[0]["repo"], matches[0]["revision"]) != (repo, revision):
                raise ValueError(f"Unexpected dPRM {kind} revision")
            directories[kind] = models / expected
            self.manifest[kind] = {"repo": repo, "revision": revision}

        self.tokenizer = AutoTokenizer.from_pretrained(directories["adapter"], local_files_only=True)
        self.labels = [self.tokenizer.encode(label, add_special_tokens=False)[-1] for label in ("-", "+")]
        self.marker = self.tokenizer.encode(STEP_MARKER, add_special_tokens=False)[-1]
        if self.labels != [12, 10] or self.marker != 22701:
            raise ValueError("Unexpected dPRM scoring tokens")
        base = AutoModelForCausalLM.from_pretrained(
            directories["base"], local_files_only=True, torch_dtype=torch.bfloat16,
            device_map={"": device}, low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(
            base, directories["adapter"], local_files_only=True, autocast_adapter_dtype=False,
        ).eval()

    def encode(self, question, steps):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(steps, list) or not steps or not all(isinstance(s, str) and s.strip() for s in steps):
            raise ValueError("steps must be a non-empty list of text")
        question = question.strip().replace(STEP_MARKER, "")
        tagged = [step.strip().replace(STEP_MARKER, "") + STEP_MARKER for step in steps]
        input_ids = self.tokenizer.encode(question + " \n\n" + "".join(tagged), return_tensors="pt")
        if input_ids.shape[1] > self.max_context:
            raise ValueError(f"PRM input exceeds {self.max_context} tokens")
        positions = (input_ids[0] == self.marker).nonzero().flatten()
        if positions.numel() != len(steps):
            raise ValueError(f"Expected {len(steps)} dPRM step markers, got {positions.numel()}")
        return input_ids.to(self.device), positions.to(self.device)

    @torch.inference_mode()
    def score(self, question, steps):
        input_ids, positions = self.encode(question, steps)
        # Causal logits at the step markers implement the upstream two-label scorer.
        logits = self.model(input_ids=input_ids, logits_to_keep=positions, use_cache=False).logits[0]
        scores = logits[:, self.labels].float().softmax(-1)[:, 1]
        return {"step_scores": scores.cpu().tolist(), "input_tokens": input_ids.shape[1]}

    @torch.inference_mode()
    def verify(self):
        question = "Question: What is 2 + 3?\nOptions:\n(A) 5\n(B) 6"
        steps = ["Add 2 and 3 to obtain 5.", "The answer is (A) 5."]
        input_ids, positions = self.encode(question, steps)
        logits = self.model(input_ids=input_ids, use_cache=False).logits[0]
        reference = logits[positions][:, self.labels].float().softmax(-1)[:, 1]
        result = torch.tensor(self.score(question, steps)["step_scores"], device=self.device)
        torch.testing.assert_close(result, reference, atol=0.003, rtol=0)
        print(f"dPRM marker logits verified against full logits. Scores {result.tolist()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--port", default=18423, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-context", default=4096, type=int)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    scorer = DPRMQwen(args.models, args.device, args.max_context)
    if args.verify:
        scorer.verify()
    serve(scorer, args.port)


if __name__ == "__main__":
    main()
