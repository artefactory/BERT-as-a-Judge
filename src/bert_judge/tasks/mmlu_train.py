from string import ascii_uppercase as ALPHABET

from ..utils import load_dataset


def mmlu_train():
    def process_fn(ex):
        question = (
            "Answer the following multiple-choice question.\n\n" +
            "Question: " + ex["question"] + "\n\n" +
            "Choices:\n" + "\n".join([f"{letter}) {choice}" for letter, choice in zip(ALPHABET, ex["choices"])])
        )
        reference = ALPHABET[ex["answer"]] + ") " + ex["choices"][ex["answer"]]
        return {"question": question.strip(), "reference": reference.strip()}
    return load_dataset("cais/mmlu", name="all", split="auxiliary_train", process_fn=process_fn)


def mmlu_train_soft():
    dataset = mmlu_train()
    return dataset.map(
        lambda ex: {"question": ex["question"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the letter of the correct choice."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def mmlu_train_strict():
    dataset = mmlu_train()
    return dataset.map(
        lambda ex: {"question": ex["question"] + "\n\nConclude your response with \"Final answer: X\", where X is the letter of the correct choice."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )
