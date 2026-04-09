from string import ascii_uppercase as ALPHABET

from ...utils import load_dataset


def arc_challenge_test():
    def process_fn(ex):
        prompt = (
            "Answer the following multiple-choice question.\n\n" +
            "Question: " + ex["question"] + "\n\n" +
            "Choices:\n" + "\n".join([f"{letter}) {choice}" for letter, choice in zip(ALPHABET, ex["choices"]["text"])])
        )
        answer_key = ALPHABET[int(ex["answerKey"]) - 1] if ex["answerKey"] not in ALPHABET else ex["answerKey"]
        answer = answer_key + ") " + ex["choices"]["text"][ALPHABET.index(answer_key)]
        return {"prompt": prompt.strip(), "answer": answer.strip()}
    return load_dataset("allenai/ai2_arc", name="ARC-Challenge", split="test", process_fn=process_fn)


def arc_challenge_test_soft():
    dataset = arc_challenge_test()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the letter of the correct choice."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def arc_challenge_test_strict():
    dataset = arc_challenge_test()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nConclude your response with \"Final answer: X\", where X is the letter of the correct choice."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def arc_challenge_test_loglik():
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
    dataset = arc_challenge_test()
    return dataset.map(
        process_fn,
        keep_in_memory=True,
        load_from_cache_file=False,
    )
