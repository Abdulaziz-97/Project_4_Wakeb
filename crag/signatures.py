import dspy


class RetrievalEvaluator(dspy.Signature):
    """Evaluate how relevant a document is to a query. Return a score between 0.0 and 1.0."""
    query: str = dspy.InputField(desc="The user query")
    document: str = dspy.InputField(desc="The retrieved document text")
    relevance_score: float = dspy.OutputField(desc="Relevance score from 0.0 to 1.0")


class KnowledgeRefiner(dspy.Signature):
    """Extract key information from a document as concise bullet points."""
    document: str = dspy.InputField(desc="The document to extract key points from")
    key_points: str = dspy.OutputField(desc="Key information as bullet points")


class QueryRewriter(dspy.Signature):
    """Rewrite a query to be more suitable for web search."""
    query: str = dspy.InputField(desc="The original user query")
    rewritten_query: str = dspy.OutputField(desc="The rewritten search query")


class ResponseGenerator(dspy.Signature):
    """Answer the query using ONLY the provided references.
    For every claim, include the reference number in brackets like [1], [2].
    If no reference supports a claim, do not make that claim."""
    query: str = dspy.InputField(desc="The user query")
    references: str = dspy.InputField(desc="Numbered references: [1] text... [2] text...")
    answer: str = dspy.OutputField(desc="Answer with [N] citation markers")


# ── RAGAS Evaluation Signatures ─────────────────────────────────────────────

class FaithfulnessEvaluator(dspy.Signature):
    """Score how faithful the answer is to the context (0.0 = hallucinated, 1.0 = fully supported).
    Check every claim in the answer against the context. Return a float between 0.0 and 1.0."""
    question: str = dspy.InputField(desc="The user question")
    answer: str = dspy.InputField(desc="The generated answer to evaluate")
    context: str = dspy.InputField(desc="Retrieved context used to generate the answer")
    faithfulness_score: float = dspy.OutputField(
        desc="Float 0.0-1.0: fraction of answer claims supported by context"
    )


class AnswerRelevancyEvaluator(dspy.Signature):
    """Score how relevant the answer is to the question (0.0 = irrelevant, 1.0 = perfectly relevant).
    Penalise answers that are incomplete, contain redundant info, or miss the point."""
    question: str = dspy.InputField(desc="The user question")
    answer: str = dspy.InputField(desc="The generated answer to evaluate")
    relevancy_score: float = dspy.OutputField(
        desc="Float 0.0-1.0: how well the answer addresses the question"
    )


class ContextPrecisionEvaluator(dspy.Signature):
    """Score how precise the retrieved context is for answering the question (0.0 = noise, 1.0 = perfectly targeted).
    Consider whether the context contains exactly what is needed to answer the question."""
    question: str = dspy.InputField(desc="The user question")
    context: str = dspy.InputField(desc="Retrieved context passages")
    precision_score: float = dspy.OutputField(
        desc="Float 0.0-1.0: how much of the context is relevant to the question"
    )
