from ..utils import load_dataset


def gsm8k_test():
    def process_fn(ex):
        question = ex["question"]
        reference = ex["answer"].split("#### ")[-1]
        return {"question": question.strip(), "reference": reference.strip()}
    return load_dataset("openai/gsm8k", name="main", split="test", process_fn=process_fn)


def gsm8k_test_soft():
    dataset = gsm8k_test()
    return dataset.map(
        lambda ex: {"question": ex["question"] + "\n\nConclude your response with \"Final answer: X\", where X is the computed solution."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def gsm8k_test_strict():
    dataset = gsm8k_test()
    return dataset.map(
        lambda ex: {"question": ex["question"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the computed solution."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )
