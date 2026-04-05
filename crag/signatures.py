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
