from ..utils import load_dataset


def asdiv():
	def process_fn(ex):
		question = ex["body"] + " " + ex["question"]
		reference = ex["answer"]
		reference = reference[:reference.find(" (")]
		return {"question": question.strip(), "reference": reference.strip()}
	return load_dataset("EleutherAI/asdiv", split="validation", process_fn=process_fn).select_columns(["question", "reference"])


def asdiv_soft():
	dataset = asdiv()
	return dataset.map(
		lambda ex: {"question": ex["question"] + "\n\nConclude your response with \"Final answer: X\", where X is the computed solution."},
		keep_in_memory=True,
		load_from_cache_file=False,
	)


def asdiv_strict():
	dataset = asdiv()
	return dataset.map(
		lambda ex: {"question": ex["question"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the computed solution."},
		keep_in_memory=True,
		load_from_cache_file=False,
	)
