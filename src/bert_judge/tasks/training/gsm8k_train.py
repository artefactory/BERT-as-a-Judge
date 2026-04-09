from ...utils import load_dataset


def gsm8k_train():
    def process_fn(ex):
        prompt = ex["question"]
        answer = ex["answer"].split("#### ")[-1]
        return {"prompt": prompt.strip(), "answer": answer.strip()}
    return load_dataset("openai/gsm8k", name="main", split="train", process_fn=process_fn)


def gsm8k_train_soft():
    dataset = gsm8k_train()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nConclude your response with \"Final answer: X\", where X is the computed solution."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def gsm8k_train_strict():
    dataset = gsm8k_train()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the computed solution."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )
