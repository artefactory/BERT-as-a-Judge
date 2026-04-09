import re

from ..utils import load_dataset


def aime24():
    def process_fn(ex):
        question = ex["problem"]
        reference = re.findall(r"\\boxed\{([^}]*)\}", ex["solution"])[0]
        return {"question": question.strip(), "reference": reference.strip()}
    return load_dataset("math-ai/aime24", split="test", process_fn=process_fn)


def aime24_soft():
    dataset = aime24()
    return dataset.map(
        lambda ex: {"question": ex["question"] + "\n\nConclude your response with \"Final answer: X\", where X is the computed solution."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def aime24_strict():
    dataset = aime24()
    return dataset.map(
        lambda ex: {"question": ex["question"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the computed solution."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )
