import random
from string import ascii_uppercase as ALPHABET

from ..utils import load_dataset


def truthful_qa_mc():
	random.seed(0)

	def process_fn(ex):
		choices_labels = list(zip(ex["mc1_targets"]["choices"], ex["mc1_targets"]["labels"]))
		random.shuffle(choices_labels)
		question = (
			"Answer the following multiple-choice question.\n\n" +
			"Question: " + ex["question"] + "\n\n" +
			"Choices:\n" + "\n".join([f"{letter}) {choice}" for letter, (choice, _) in zip(ALPHABET, choices_labels)])
		)
		reference = [f"{letter}) {choice}" for letter, (choice, label) in zip(ALPHABET, choices_labels) if label == 1][0]
		return {"question": question.strip(), "reference": reference.strip()}

	return load_dataset("truthfulqa/truthful_qa", name="multiple_choice", split="validation", process_fn=process_fn).select_columns(["question", "reference"])


def truthful_qa_mc_soft():
	dataset = truthful_qa_mc()
	return dataset.map(
		lambda ex: {"question": ex["question"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the letter of the correct choice."},
		keep_in_memory=True,
		load_from_cache_file=False,
	)


def truthful_qa_mc_strict():
	dataset = truthful_qa_mc()
	return dataset.map(
		lambda ex: {"question": ex["question"] + "\n\nConclude your response with \"Final answer: X\", where X is the letter of the correct choice."},
		keep_in_memory=True,
		load_from_cache_file=False,
	)
