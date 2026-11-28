from pathlib import Path
from serve_dprm import DPRMQwen
import torch

torch.set_num_threads(4)
scorer = DPRMQwen(Path("models-next"), "cpu", 4096)
scorer.verify()
print(scorer.score(
    "Question: What is the next term in 3, 5, 7?\nOptions:\n(A) 9\n(B) 10",
    ["The gap between successive terms is two.", "The answer is (A) 9."],
), flush=True)
