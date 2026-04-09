from ..utils import load_dataset


def hotpot_qa_train():
    def process_fn(ex):
        context = "\n".join([
            "".join(sentences) for title, sentences in
            zip(ex["context"]["title"], ex["context"]["sentences"])
            if title in ex["supporting_facts"]["title"]
        ])
        question = (
            "Answer the question based on the provided context.\n\n" +
            "Context:\n" + context + "\n\n" +
            "Question: " + ex["question"]
        )
        reference = ex["answer"]
        return {"question": question.strip(), "reference": reference.strip()}
    return load_dataset("hotpotqa/hotpot_qa", name="distractor", split="train", process_fn=process_fn)


def hotpot_qa_train_soft():
    dataset = hotpot_qa_train()
    return dataset.map(
        lambda ex: {"question": ex["question"] + "\n\nConclude your response with \"Final answer: X\", where X is the exact span from the context that answers the question."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def hotpot_qa_train_strict():
    dataset = hotpot_qa_train()
    return dataset.map(
        lambda ex: {"question": ex["question"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the exact span from the context that answers the question."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )
