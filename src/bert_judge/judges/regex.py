import re
from string import ascii_uppercase


class RegexJudge:
	"""Judge candidates using regex extraction and deterministic metrics."""

	def __init__(
		self,
		pattern: str = "Final answer:\\s*(.+)",
		metric: str = "EM",
	) -> None:
		"""Initialize extraction pattern and scoring metric."""
		self.pattern = pattern
		self.metric = metric
		self._rouge_scorer = None
		self._math_parse = None
		self._math_verify = None

		if metric == "EM":
			self._compute_scores = self._compute_em_scores

		elif metric == "ROUGE":
			try:
				from rouge_score import rouge_scorer
			except Exception as exc:
				raise ImportError(
					"Rouge-score is required for using ROUGE metric. Install it with `pip install rouge-score`."
				) from exc

			self._rouge_scorer = rouge_scorer
			self._compute_scores = self._compute_rouge_scores

		elif metric == "Math-Verify":
			try:
				from math_verify import parse, verify
			except Exception as exc:
				raise ImportError(
					"Math-verify is required for using Math-Verify metric. Install it with `pip install math-verify`."
				) from exc

			self._math_parse = parse
			self._math_verify = verify
			self._compute_scores = self._compute_math_verify_scores

		else:
			raise ValueError(
				f"Unsupported metric: {metric}. Expected one of ['EM', 'ROUGE', 'Math-Verify']."
			)

	def predict(
		self,
		candidates: list[str],
		references: list[str],
	) -> list[float | int]:
		"""Score candidate answers against references."""
		references = self._process_references(references)
		extractions = self._extract_answers(candidates)
		scores = self._compute_scores(extractions, references)
		return scores

	def _extract_answers(self, candidates: list[str]) -> list[str | None]:
		"""Extract answers from model outputs with configured regex."""
		extractions: list[str | None] = []
		for candidate in candidates:
			match = re.findall(self.pattern, candidate)
			extractions.append(match[0] if match else None)

		return extractions

	def _compute_em_scores(self, extractions: list[str | None], references: list[str]) -> list[int]:
		"""Compute exact-match binary scores."""
		return [
			(extraction == reference) * 1
			for extraction, reference in zip(extractions, references, strict=False)
		]

	def _compute_rouge_scores(
		self, extractions: list[str | None], references: list[str]
	) -> list[float]:
		"""Compute ROUGE-L F1 scores."""
		metric = self._rouge_scorer.RougeScorer(["rougeL"])
		scores: list[float] = []
		for extraction, reference in zip(extractions, references, strict=False):
			if extraction is None or reference is None:
				scores.append(0.0)
				continue

			scores.append(metric.score(reference, extraction)["rougeL"].fmeasure)

		return scores

	def _compute_math_verify_scores(
		self, extractions: list[str | None], references: list[str]
	) -> list[int]:
		"""Compute symbolic math verification scores."""
		scores: list[int] = []
		for extraction, reference in zip(extractions, references, strict=False):
			if extraction is None or reference is None:
				scores.append(0)
				continue

			scores.append(self._math_verify(self._math_parse(reference), self._math_parse(extraction)) * 1)

		return scores

	def _process_references(self, references: list[str]) -> list[str]:
		"""Normalize multiple-choice references to option labels when present."""
		return [
			reference[0] if reference and reference.split(")")[0] in ascii_uppercase else reference
			for reference in references
		]
