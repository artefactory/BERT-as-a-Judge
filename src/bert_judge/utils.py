import os
import importlib

from datasets import (
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

    if not isinstance(name, list):
        name = [name]

    dataset = []

    try:
        for _name in name:
            ld_kwargs = {
                "path": path, 
                "split": split, 
                **({"name": _name} if _name is not None else {})
            }
            dataset.append(_load_dataset(**ld_kwargs))
    except:
        for _name in name:
            lfd_path = path + f"/{_name}" * bool(_name) + f"/{split}" * bool(split)
            dataset.append(load_from_disk(lfd_path))

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
