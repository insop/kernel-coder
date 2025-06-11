import sys
import types
import importlib.util
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")

# Provide dummy modules for optional dependencies
_dummy_modules = {
    "wandb": {},
    "datasets": {"Dataset": object},
    "deepspeed": {"DeepSpeedEngine": object},
    "transformers": {"AutoTokenizer": object, "PreTrainedModel": object},
    "vllm": {"LLM": object, "SamplingParams": object},
}
for name, attrs in _dummy_modules.items():
    if name not in sys.modules:
        module = types.ModuleType(name)
        for attr, val in attrs.items():
            setattr(module, attr, val)
        sys.modules[name] = module

# Dynamically load utils.py since the folder name contains a dash
utils_path = Path(__file__).resolve().parents[1] / "kernel-coder" / "utils.py"
spec = importlib.util.spec_from_file_location("kernel_utils", utils_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

prepare_model_inputs = utils.prepare_model_inputs

def test_prepare_model_inputs_basic():
    query_token_ids = [[1, 2], [3]]
    response_token_ids = [[4, 5, 6], [7, 8]]
    advantages = [[0.1, 0.2, 0.3], [0.4, 0.5]]
    device = torch.device("cpu")

    outputs = prepare_model_inputs(query_token_ids, response_token_ids, advantages, device)

    for key in ["input_ids", "attention_mask", "labels", "advantages", "labels_mask"]:
        assert key in outputs
        assert outputs[key].shape == (2, 5)

    # First sequence has no padding
    assert torch.equal(outputs["input_ids"][0], torch.tensor([1, 2, 4, 5, 6]))
    assert torch.equal(outputs["attention_mask"][0], torch.tensor([1, 1, 1, 1, 1]))
    assert torch.equal(outputs["labels"][0], torch.tensor([-100, -100, 4, 5, 6]))
    assert torch.allclose(outputs["advantages"][0], torch.tensor([0.0, 0.0, 0.1, 0.2, 0.3]))
    assert torch.equal(outputs["labels_mask"][0], torch.tensor([0, 0, 1, 1, 1]))

    # Second sequence padded to length 5
    assert torch.equal(outputs["input_ids"][1], torch.tensor([3, 7, 8, 0, 0]))
    assert torch.equal(outputs["attention_mask"][1], torch.tensor([1, 1, 1, 0, 0]))
    assert torch.equal(outputs["labels"][1], torch.tensor([-100, 7, 8, -100, -100]))
    assert torch.allclose(outputs["advantages"][1], torch.tensor([0.0, 0.4, 0.5, 0.0, 0.0]))
    assert torch.equal(outputs["labels_mask"][1], torch.tensor([0, 1, 1, 0, 0]))
