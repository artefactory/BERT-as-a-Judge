from ...utils import load_dataset


def asdiv():
    def process_fn(ex):
        prompt = ex["body"] + " " + ex["question"]
        answer = ex["answer"]
        answer = answer[:answer.find(" (")]
        return {"prompt": prompt.strip(), "answer": answer.strip()}
    return load_dataset("EleutherAI/asdiv", split="validation", process_fn=process_fn)


def asdiv_soft():
    dataset = asdiv()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nConclude your response with \"Final answer: X\", where X is the computed solution."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def asdiv_strict():
    dataset = asdiv()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the computed solution."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )
