from string import ascii_uppercase as ALPHABET

from ...utils import load_dataset


def mmlu_train():
    def process_fn(ex):
        prompt = (
            "Answer the following multiple-choice question.\n\n" +
            "Question: " + ex["question"] + "\n\n" +
            "Choices:\n" + "\n".join([f"{letter}) {choice}" for letter, choice in zip(ALPHABET, ex["choices"])])
        )
        answer = ALPHABET[ex["answer"]] + ") " + ex["choices"][ex["answer"]]
        return {"prompt": prompt.strip(), "answer": answer.strip()}
    return load_dataset("cais/mmlu", name="all", split="auxiliary_train", process_fn=process_fn)


def mmlu_train_soft():
    dataset = mmlu_train()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the letter of the correct choice."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def mmlu_train_strict():
    dataset = mmlu_train()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nConclude your response with \"Final answer: X\", where X is the letter of the correct choice."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def mmlu_train_loglik():
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
    dataset = mmlu_train()
    return dataset.map(
        process_fn,
        keep_in_memory=True,
        load_from_cache_file=False,
    )
