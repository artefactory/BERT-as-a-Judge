from string import ascii_uppercase as ALPHABET

from ..utils import load_dataset


def mmlu_pro():
    def process_fn(ex):
        question = (
            "Answer the following multiple-choice question.\n\n" +
            "Question: " + ex["question"] + "\n\n" +
            "Choices:\n" + "\n".join([f"{letter}) {choice}" for letter, choice in zip(ALPHABET, ex["options"])])
        )
        reference = ex["answer"] + ") " + ex["options"][ALPHABET.index(ex["answer"])]
        return {"question": question.strip(), "reference": reference.strip()}
    return load_dataset("TIGER-Lab/MMLU-Pro", split="test", process_fn=process_fn)


def mmlu_pro_soft():
    dataset = mmlu_pro()
    return dataset.map(
        lambda ex: {"question": ex["question"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the letter of the correct choice."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def mmlu_pro_strict():
    dataset = mmlu_pro()
    return dataset.map(
        lambda ex: {"question": ex["question"] + "\n\nConclude your response with \"Final answer: X\", where X is the letter of the correct choice."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )
