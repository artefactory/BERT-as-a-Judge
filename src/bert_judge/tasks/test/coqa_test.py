from ...utils import load_dataset


def coqa_test():
    def process_fn(ex):
        prompt = (
            "Answer the question based on the provided context.\n\n" +
            "Context:\n" + ex["story"] + "\n\n" +
            "Question: " + ex["questions"]["input_text"][0]
        )
        answer = ex["answers"]["input_text"][0]
        return {"prompt": prompt.strip(), "answer": answer.strip()}
    return load_dataset("EleutherAI/coqa", split="validation", process_fn=process_fn)


def coqa_test_soft():
    dataset = coqa_test()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nConclude your response with \"Final answer: X\", where X is the exact span from the context that answers the question."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def coqa_test_strict():
    dataset = coqa_test()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the exact span from the context that answers the question."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )
