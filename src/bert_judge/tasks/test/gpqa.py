import random
from string import ascii_uppercase as ALPHABET

from ...utils import load_dataset


def gpqa():
    random.seed(0)

    def process_fn(ex):
        choices = [ex["Correct Answer"]] + [ex[f"Incorrect Answer {i+1}"] for i in range(3)]
        random.shuffle(choices)
        prompt = (
            "Answer the following multiple-choice question.\n\n" +
            "Question: " + ex["Question"] + "\n\n" +
            "Choices:\n" + "\n".join([f"{letter}) {choice}" for letter, choice in zip(ALPHABET, choices)])
        )
        answer = ALPHABET[choices.index(ex["Correct Answer"])] + ") " + ex["Correct Answer"]
        return {"prompt": prompt.strip(), "answer": answer.strip()}

    return load_dataset("Idavidrein/gpqa", name="gpqa_main", split="train", process_fn=process_fn)


def gpqa_soft():
    dataset = gpqa()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the letter of the correct choice."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def gpqa_strict():
    dataset = gpqa()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nConclude your response with \"Final answer: X\", where X is the letter of the correct choice."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def gpqa_loglik():
    def process_fn(ex):
        prompt = ex["prompt"]
        question = prompt[prompt.find("Question:")+10:prompt.find("\n\nChoices:")]
        choices_str = prompt[prompt.find("\nA)"):]
        choices = []
        for i in range(len(ALPHABET)):
            if i + 1 < len(ALPHABET) and f"\n{ALPHABET[i+1]}) " in choices_str:
                choices.append(
                    choices_str[choices_str.find(f"\n{ALPHABET[i]}) ")+4:choices_str.find(f"\n{ALPHABET[i+1]}) ")]
                )
            else:
                choices.append(
                    choices_str[choices_str.find(f"\n{ALPHABET[i]}) ")+4:]
                )
                break
        sequences = "<|seq_sep|>".join([question + "<|qa_sep|>" + choice for choice in choices])
        return {"prompt": sequences, "answer": ex["answer"][0]}
    dataset = gpqa()
    return dataset.map(
        process_fn,
        keep_in_memory=True,
        load_from_cache_file=False,
    )
