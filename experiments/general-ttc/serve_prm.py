"""Serve the released VersaPRM checkpoint and its Llama-PRM800K base."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare import ASSETS


STEP_MARKER = " \n\n\n\n"
STEP_TOKEN = 23535
LABEL_TOKENS = [12, 10]  # negative and positive in the author's LlamaPRM scorer


class VersaPRM:
    def __init__(self, directory, device, max_context):
        self.device = device
        self.max_context = max_context
        manifest = json.loads((directory / "assets.json").read_text(encoding="utf-8"))
        if manifest != {name: ASSETS[name] for name in ("prm_base", "prm_adapter")}:
            raise ValueError("Unexpected model asset revisions")
        self.manifest = manifest
        adapter = directory / "prm_adapter"
        base = AutoModelForCausalLM.from_pretrained(
            directory / "prm_base", local_files_only=True, torch_dtype=torch.bfloat16,
            device_map={"": device}, low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(
            base, adapter, local_files_only=True, autocast_adapter_dtype=False).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        if self.tokenizer.encode(STEP_MARKER, add_special_tokens=False) != [STEP_TOKEN]:
            raise ValueError("VersaPRM step marker does not match the pinned tokenizer")
        # The adapter targets decoder projections. Evaluate the same two LM-head logits
        # as the reference without allocating a vocabulary-sized tensor for every token.
        self.decoder = self.model.get_base_model().model
        self.classifier = self.model.get_base_model().get_output_embeddings().weight[LABEL_TOKENS].detach()

    @torch.inference_mode()
    def score(self, question, steps):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(steps, list) or not steps or not all(isinstance(s, str) and s.strip() for s in steps):
            raise ValueError("steps must be a non-empty list of text")
        question = "\n".join(line.strip() for line in question.splitlines() if line.strip())
        steps = [" ".join(step.split()) for step in steps]
        prompt = question + " \n\n" + "".join(step + STEP_MARKER for step in steps)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if input_ids.shape[1] > self.max_context:
            raise ValueError(f"PRM input exceeds {self.max_context} tokens")
        positions = (input_ids[0] == STEP_TOKEN).nonzero().flatten()
        if positions.numel() != len(steps):
            raise ValueError(f"Expected {len(steps)} step markers, got {positions.numel()}")
        hidden = self.decoder(input_ids=input_ids.to(self.device), use_cache=False).last_hidden_state
        step_hidden = hidden[0, positions.to(self.device)]
        logits = torch.nn.functional.linear(step_hidden, self.classifier)
        return {
            "step_scores": logits.float().softmax(-1)[:, 1].cpu().tolist(),
            "input_tokens": input_ids.shape[1],
        }


def serve(prm, port):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"status": "ready", "assets": prm.manifest, "dtype": "bfloat16",
                                 "max_context": prm.max_context})
            else:
                self.reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if self.path != "/score":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(size))
                result = prm.score(body["question"], body["steps"])
            except (KeyError, TypeError, ValueError) as error:
                self.reply(400, {"error": str(error)})
                return
            self.reply(200, result)

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"VersaPRM ready on 127.0.0.1:{port}", flush=True)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18423)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-context", type=int, default=4096)
    args = parser.parse_args()
    serve(VersaPRM(args.models, args.device, args.max_context), args.port)


if __name__ == "__main__":
    main()