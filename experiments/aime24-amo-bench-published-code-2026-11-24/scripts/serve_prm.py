"""Serve the pinned upstream PRM scorer on the assigned local GPU."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json

from validate_models import ASSETS, PRMSelector, asset_path, check_placement, require_h20


MODEL_ID = "Qwen2.5-Math-PRM-7B"
DEFAULT_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."
ADDRESS = ("127.0.0.1", 18422)


def serve():
    require_h20()
    scorer = PRMSelector(str(asset_path(ASSETS[2])), device="cuda:0")
    check_placement(scorer.model)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"status": "ready", "model": MODEL_ID, "device": "cuda:0"})
            elif self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
            else:
                self.reply(404, {"error": "Unknown route"})

        def do_POST(self):
            if self.path != "/score":
                self.reply(404, {"error": "Unknown route"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 262144:
                    raise ValueError("Expected a JSON request of up to 256 KiB")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                problem = data.get("problem")
                steps = data.get("reasoning_steps")
                system = data.get("system_prompt", DEFAULT_PROMPT)
                if not isinstance(problem, str) or not problem.strip():
                    raise ValueError("problem must be a non-empty string")
                if not isinstance(steps, list) or not steps or not all(isinstance(step, str) for step in steps):
                    raise ValueError("reasoning_steps must be a non-empty list of strings")
                if not isinstance(system, str):
                    raise ValueError("system_prompt must be a string")
            except (ValueError, UnicodeError) as exc:
                self.reply(400, {"error": str(exc)})
                return

            result = scorer.score_reasoning(problem, steps, system)
            if not result["step_rewards"]:
                self.reply(422, {"error": "The scorer returned no step rewards"})
                return
            self.reply(200, {"model": MODEL_ID, **result})

    with HTTPServer(ADDRESS, Handler) as server:
        print(f"{MODEL_ID} ready on {ADDRESS[0]}:{ADDRESS[1]}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    serve()
