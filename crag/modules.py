import dspy
from crag.signatures import RetrievalEvaluator, KnowledgeRefiner, QueryRewriter, ResponseGenerator


class Evaluator(dspy.Module):
    """Score document relevance to a query."""

    def __init__(self):
        super().__init__()
        self.evaluate = dspy.Predict(RetrievalEvaluator)

    def forward(self, query: str, document: str) -> float:
        result = self.evaluate(query=query, document=document)
        try:
            score = float(result.relevance_score)
        except (ValueError, TypeError):
            score = 0.0
        return max(0.0, min(1.0, score))


class Refiner(dspy.Module):
    """Extract key points from a document."""

    def __init__(self):
        super().__init__()
        self.refine = dspy.Predict(KnowledgeRefiner)

    def forward(self, document: str) -> str:
        result = self.refine(document=document)
        return result.key_points


class Rewriter(dspy.Module):
    """Rewrite a query for web search."""

    def __init__(self):
        super().__init__()
        self.rewrite = dspy.Predict(QueryRewriter)

    def forward(self, query: str) -> str:
        result = self.rewrite(query=query)
        return result.rewritten_query


class Generator(dspy.Module):
    """Generate a cited answer from knowledge."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(ResponseGenerator)

    def forward(self, query: str, references: str) -> str:
        result = self.generate(query=query, references=references)
        return result.answer
