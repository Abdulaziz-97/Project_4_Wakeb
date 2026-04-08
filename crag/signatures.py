import dspy


class RetrievalEvaluator(dspy.Signature):
    """Score how relevant a retrieved document is to the user query (0.0 to 1.0)."""
    query: str = dspy.InputField(desc="The user query")
    document: str = dspy.InputField(desc="The retrieved document text")
    relevance_score: float = dspy.OutputField(desc="Relevance score from 0.0 to 1.0")


class KnowledgeRefiner(dspy.Signature):
    """Extract the key factual points from a weather document."""
    document: str = dspy.InputField(desc="The document to extract key points from")
    key_points: str = dspy.OutputField(desc="Key information as bullet points")


class QueryRewriter(dspy.Signature):
    """Rewrite the query for better web search results."""
    query: str = dspy.InputField(desc="The original user query")
    rewritten_query: str = dspy.OutputField(desc="The rewritten search query")


class ResponseGenerator(dspy.Signature):
    """Generate a weather answer from references, citing sources with [N] markers."""
    query: str = dspy.InputField(desc="The user query")
    references: str = dspy.InputField(desc="Numbered references: [1] text... [2] text...")
    answer: str = dspy.OutputField(desc="Answer with [N] citation markers")
