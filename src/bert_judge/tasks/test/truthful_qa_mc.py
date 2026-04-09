import random
from string import ascii_uppercase as ALPHABET

from ...utils import load_dataset


def truthful_qa_mc():
    random.seed(0)

    def process_fn(ex):
        choices_labels = list(zip(ex["mc1_targets"]["choices"], ex["mc1_targets"]["labels"]))
        random.shuffle(choices_labels)
        prompt = (
            "Answer the following multiple-choice question.\n\n" +
            "Question: " + ex["question"] + "\n\n" +
            "Choices:\n" + "\n".join([f"{letter}) {choice}" for letter, (choice, _) in zip(ALPHABET, choices_labels)])
        )
        answer = [f"{letter}) {choice}" for letter, (choice, label) in zip(ALPHABET, choices_labels) if label == 1][0]
        return {"prompt": prompt.strip(), "answer": answer.strip()}

    return load_dataset("truthfulqa/truthful_qa", name="multiple_choice", split="validation", process_fn=process_fn)


def truthful_qa_mc_soft():
    dataset = truthful_qa_mc()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the letter of the correct choice."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def truthful_qa_mc_strict():
    dataset = truthful_qa_mc()
    return dataset.map(
        lambda ex: {"prompt": ex["prompt"] + "\n\nConclude your response with \"Final answer: X\", where X is the letter of the correct choice."},
        keep_in_memory=True,
        load_from_cache_file=False,
    )


def truthful_qa_mc_loglik():
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
    dataset = truthful_qa_mc()
    return dataset.map(
        process_fn,
        keep_in_memory=True,
        load_from_cache_file=False,
    )
