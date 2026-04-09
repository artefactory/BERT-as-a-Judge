import re
from string import ascii_uppercase

from math_verify import parse, verify
from rouge_score import rouge_scorer


class RegexJudge:
    def __init__(
        self,
        pattern="Final answer:\\s*(.+)",
        metric="EM",
    ):
        self.pattern = pattern
        self.metric = metric

        if metric == "EM":
            self._compute_scores = self._compute_em_scores
        elif metric == "ROUGE":
            self._compute_scores = self._compute_rouge_scores
        elif metric == "Math-Verify":
            self._compute_scores = self._compute_math_verify_scores
        else:
            raise ValueError(
                f"Unsupported metric: {metric}. Expected one of ['EM', 'ROUGE', 'Math-Verify']."
            )

    def predict(
        self,
        candidates,
        references,
    ):
        references = self._process_references(references)
        extractions = self._extract_answers(candidates)
        return self._compute_scores(extractions, references)

    def _extract_answers(self, candidates):
        extractions = []
        for candidate in candidates:
            match = re.findall(self.pattern, candidate)
            extractions.append(match[0] if match else None)
        
        return extractions

    def _compute_em_scores(self, extractions, references):
        return [
            (extraction == reference) * 1
            for extraction, reference in zip(extractions, references)
        ]

    def _compute_rouge_scores(self, extractions, references):
        metric = rouge_scorer.RougeScorer(["rougeL"])
        scores = []
        for extraction, reference in zip(extractions, references):
            if extraction is None or reference is None:
                scores.append(0.0)
                continue

            scores.append(metric.score(reference, extraction)["rougeL"].fmeasure)

        return scores

    def _compute_math_verify_scores(self, extractions, references):
        scores = []
        for extraction, reference in zip(extractions, references):
            if extraction is None or reference is None:
                scores.append(0)
                continue

            scores.append(verify(parse(reference), parse(extraction)) * 1)

        return scores

    def _process_references(self, references):
        return [
            reference[0] if reference and reference.split(")")[0] 
            in ascii_uppercase else reference
            for reference in references
        ]
