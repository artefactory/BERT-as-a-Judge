import random
from string import ascii_uppercase as ALPHABET

from ..utils import load_dataset


def gpqa():
	random.seed(0)

	def process_fn(ex):
		choices = [ex["Correct Answer"]] + [ex[f"Incorrect Answer {i+1}"] for i in range(3)]
		random.shuffle(choices)
		question = (
			"Answer the following multiple-choice question.\n\n" +
			"Question: " + ex["Question"] + "\n\n" +
			"Choices:\n" + "\n".join([f"{letter}) {choice}" for letter, choice in zip(ALPHABET, choices)])
		)
		reference = ALPHABET[choices.index(ex["Correct Answer"])] + ") " + ex["Correct Answer"]
		return {"question": question.strip(), "reference": reference.strip()}

	return load_dataset("Idavidrein/gpqa", name="gpqa_main", split="train", process_fn=process_fn)


def gpqa_soft():
	dataset = gpqa()
	return dataset.map(
		lambda ex: {"question": ex["question"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the letter of the correct choice."},
		keep_in_memory=True,
		load_from_cache_file=False,
	)


def gpqa_strict():
	dataset = gpqa()
	return dataset.map(
		lambda ex: {"question": ex["question"] + "\n\nConclude your response with \"Final answer: X\", where X is the letter of the correct choice."},
		keep_in_memory=True,
		load_from_cache_file=False,
	)
