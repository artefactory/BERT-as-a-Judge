from ..utils import load_dataset


def aime25():
    def process_fn(ex):
        question = ex["problem"]
        reference = ex["answer"]
        return {"question": question.strip(), "reference": reference.strip()}
    return load_dataset("math-ai/aime25", split="test", process_fn=process_fn)


def aime25_soft():
    dataset = aime25()
    return dataset.map(
        lambda ex: {"question": ex["question"] + "\n\nConclude your response with \"Final answer: X\", where X is the computed solution."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def aime25_strict():
    dataset = aime25()
    return dataset.map(
        lambda ex: {"question": ex["question"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the computed solution."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )
