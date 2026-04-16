from ..utils import load_dataset


def squad_v2_test():
	def process_fn(ex):
		question = (
			"Answer the question based on the provided context.\n\n" +
			"Context:\n" + ex["context"] + "\n\n" +
			"Question: " + ex["question"]
		)
		reference = ex["answers"]["text"][0] if len(ex["answers"]["text"]) > 0 else "Unanswerable"
		return {"question": question.strip(), "reference": reference.strip()}
	return load_dataset("rajpurkar/squad_v2", split="validation", process_fn=process_fn).select_columns(["question", "reference"])


def squad_v2_test_soft():
	dataset = squad_v2_test()
	return dataset.map(
		lambda ex: {"question": ex["question"] + "\n\nConclude your response with \"Final answer: X\", where X is the exact span from the context that answers the question."},
		keep_in_memory=True,
		load_from_cache_file=False,
	)


def squad_v2_test_strict():
	dataset = squad_v2_test()
	return dataset.map(
		lambda ex: {"question": ex["question"] + "\n\nRespond only with the exact format \"Final answer: X\", where X is the exact span from the context that answers the question."},
		keep_in_memory=True,
		load_from_cache_file=False,
	)
