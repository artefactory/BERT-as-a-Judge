import os
import importlib
import json
import pkgutil

from datasets import (
    DatasetDict,
    concatenate_datasets,
    get_dataset_config_names as _get_dataset_config_names,
    load_dataset as _load_dataset,
    load_from_disk,
)
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def get_dataset_config_names(path):
    path = resolve_dataset_path(path)
    return _get_dataset_config_names(path)
    
    
def load_dataset(path, name=None, split=None, filter_fn=None, process_fn=None):
    path = resolve_dataset_path(path)
    names = [name] if not isinstance(name, list) else name
    splits = [split] if not isinstance(split, list) else split
    dataset = []

    try:
        for name in names:
            for split in splits:
                lfd_path = path + f"/{name}" * bool(name) + f"/{split}" * bool(split)
                dataset.append(load_from_disk(lfd_path))
        
    except:
        for name in names:
            for split in splits:
                ld_kwargs = {
                    "path": path, 
                    "split": split,
                    **({"name": name} if name is not None else {})
                }
                dataset.append(_load_dataset(**ld_kwargs))

    if isinstance(dataset[0], DatasetDict):
        dataset = concatenate_datasets(
            [ds_dict[split] for ds_dict in dataset for split in ds_dict]
        )
    else:
        dataset = concatenate_datasets(dataset)
    
    if filter_fn is not None:
        dataset = dataset.filter(
            filter_fn, 
            keep_in_memory=True,
            load_from_cache_file=False,
        )

    if process_fn is not None:
        dataset = dataset.map(
            process_fn,
            keep_in_memory=True,
            load_from_cache_file=False,
        )

    return dataset


def load_vllm_generator(
    path,
    trust_remote_code=False,
    dtype="bfloat16",
    tensor_parallel_size=1,
):
    try:
        LLM = importlib.import_module("vllm").LLM
    except Exception as exc:
        raise ImportError(
            "vLLM is required for `load_vllm_generator`. Install it with `pip install vllm`."
        ) from exc

    path = resolve_model_path(path)
    return LLM(
        path, 
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
    )


def load_hf_generator(
    path,
    trust_remote_code=False,
    dtype="bfloat16",
    device_map="auto",
):
    path = resolve_model_path(path)
    dtype = resolve_torch_dtype(dtype)
    return AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map=device_map, 
    )


def load_hf_encoder(
    path,
    trust_remote_code=False,
    dtype="bfloat16",
    device_map="auto",
):
    path = resolve_model_path(path)
    dtype = resolve_torch_dtype(dtype)
    return AutoModelForSequenceClassification.from_pretrained(
        path,
        num_labels=2,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map=device_map, 
    )


def load_hf_tokenizer(
    path,
    trust_remote_code=False,
):
    path = resolve_model_path(path)
    return AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=trust_remote_code,
    )


def resolve_dataset_path(path):
    if "LOCAL_DATASETS_DIR" in os.environ:
        path = os.path.join(
            os.environ["LOCAL_DATASETS_DIR"], 
            path.split("/")[-1],
        )
    return path


def resolve_model_path(path):
    if "LOCAL_MODELS_DIR" in os.environ:
        path = os.path.join(
            os.environ["LOCAL_MODELS_DIR"], 
            path.split("/")[-1],
        )
    return path


def resolve_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        return "auto"
    if isinstance(dtype, str) and hasattr(torch, dtype):
        return getattr(torch, dtype)
    raise ValueError(f"Unsupported dtype: {dtype}")


def parse_tasks(task_values):
    tasks = []
    for value in task_values:
        for task_name in value.split(","):
            task_name = task_name.strip()
            if task_name:
                tasks.append(task_name)
    return tasks


def discover_task_functions(package_name="bert_judge.tasks"):
    discovered = {}
    package = importlib.import_module(package_name)
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue

        module = importlib.import_module(f"{package_name}.{module_info.name}")
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(module, attr_name)
            if callable(attr_value) and getattr(attr_value, "__module__", None) == module.__name__:
                discovered[attr_name] = attr_value

    return discovered


def load_json_list(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list in {path}")

    return data


def get_model_name(model_path):
    return str(model_path).rstrip("/").split("/")[-1].replace("-", "_")
