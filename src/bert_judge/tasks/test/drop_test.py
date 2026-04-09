from ...utils import load_dataset


def drop_test():
    def filter_fn(ex):
        return len(set(ex["answers_spans"]["spans"])) == 1

    def process_fn(ex):
        prompt = (
            "Answer the question based on the provided context.\n\n" +
            "Context:\n" + ex["passage"] + "\n\n" +
            "Question: " + ex["question"]
        )
        answer = ex["answers_spans"]["spans"][0]
        return {"prompt": prompt.strip(), "answer": answer.strip()}

    return load_dataset("ucinlp/drop", split="validation", filter_fn=filter_fn, process_fn=process_fn)


def drop_test_soft():
    dataset = drop_test()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nConclude your response with \"Final answer: X\", where X is the exact span from the context that answers the question."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def drop_test_strict():
    dataset = drop_test()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the exact span from the context that answers the question."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )
