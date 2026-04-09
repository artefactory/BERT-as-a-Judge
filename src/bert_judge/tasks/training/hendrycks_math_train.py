import re

from ...utils import get_dataset_config_names, load_dataset


def hendrycks_math_train():
    def filter_fn(ex):
        return len(re.findall(r"\\boxed\{([^}]*)\}", ex["solution"])) > 0

    def process_fn(ex):
        prompt = ex["problem"]
        answer = re.findall(r"\\boxed\{([^}]*)\}", ex["solution"])[0]
        return {"prompt": prompt.strip(), "answer": answer.strip()}

    config_names = get_dataset_config_names("EleutherAI/hendrycks_math")
    return load_dataset(
        "EleutherAI/hendrycks_math",
        name=config_names,
        split="train",
        filter_fn=filter_fn,
        process_fn=process_fn,
    )


def hendrycks_math_train_soft():
    dataset = hendrycks_math_train()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nConclude your response with \"Final answer: X\", where X is the computed solution."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def hendrycks_math_train_strict():
    dataset = hendrycks_math_train()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the computed solution."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )
