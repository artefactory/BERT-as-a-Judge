from string import ascii_uppercase as ALPHABET

from ..utils import load_dataset


def arc_challenge_test():
    def process_fn(ex):
        question = (
            "Answer the following multiple-choice question.\n\n" +
            "Question: " + ex["question"] + "\n\n" +
            "Choices:\n" + "\n".join([f"{letter}) {choice}" for letter, choice in zip(ALPHABET, ex["choices"]["text"])])
        )
        answer_key = ALPHABET[int(ex["answerKey"]) - 1] if ex["answerKey"] not in ALPHABET else ex["answerKey"]
        reference = answer_key + ") " + ex["choices"]["text"][ALPHABET.index(answer_key)]
        return {"question": question.strip(), "reference": reference.strip()}
    return load_dataset("allenai/ai2_arc", name="ARC-Challenge", split="test", process_fn=process_fn)


def arc_challenge_test_soft():
    dataset = arc_challenge_test()
    return dataset.map(
        lambda ex: {"question": ex["question"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the letter of the correct choice."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def arc_challenge_test_strict():
    dataset = arc_challenge_test()
    return dataset.map(
        lambda ex: {"question": ex["question"] + "\n\nConclude your response with \"Final answer: X\", where X is the letter of the correct choice."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )
