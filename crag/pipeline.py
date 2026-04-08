import json
import re
from datetime import date, timedelta
from tavily import TavilyClient
from config.settings import TAVILY_API_KEY, HIGH_RELEVANCE, LOW_RELEVANCE
from vectorstore.chroma_store import ChromaStore
from crag.modules import Evaluator, Refiner, Rewriter, Generator
from crag.metrics import RAGASEvaluator, MetricsLogger


class CRAGPipeline:
\
\
\
\
\
\
\
\
\
\
\
       

    def __init__(self, store: ChromaStore = None, evaluate_metrics: bool = True):
        self.store = store or ChromaStore()
        self.evaluator = Evaluator()
        self.refiner = Refiner()
        self.rewriter = Rewriter()
        self.generator = Generator()
        self.tavily = TavilyClient(api_key=TAVILY_API_KEY)
        self.evaluate_metrics = evaluate_metrics
        if evaluate_metrics:
            self.ragas_evaluator = RAGASEvaluator()
            self.metrics_logger = MetricsLogger()

    def run(self, query: str) -> dict:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
                          
        retrieved = self.store.query(query)
        if not retrieved:
            return self._web_search_path(query)

                          
        scores = []
        for doc in retrieved:
            score = self.evaluator(query=query, document=doc["text"])
            scores.append(score)

        if not scores:
            return self._web_search_path(query)

        max_score = max(scores)
        best_idx = scores.index(max_score)
        best_doc = retrieved[best_idx]

                       
        if max_score > HIGH_RELEVANCE:
            return self._correct_path(query, best_doc, scores)
        elif max_score < LOW_RELEVANCE:
            return self._incorrect_path(query, scores)
        else:
            return self._ambiguous_path(query, best_doc, scores)

    def _correct_path(self, query: str, best_doc: dict, scores: list) -> dict:
                                                     
        metadata = best_doc.get("metadata", {})
        is_ingested = "ingested_at" in metadata

        if metadata.get("doc_type") == "forecast":
            answer, contexts, date_range = self._aggregate_city_forecast(query, metadata)
            sources = self._extract_forecast_sources(metadata, date_range)
        elif is_ingested:
            answer = best_doc["text"]
            contexts = [answer]
            sources = self._extract_sources(metadata)
        else:
            ref_str = f"[1] {best_doc['text']}"
            answer = self.generator(query=query, references=ref_str)
            contexts = [best_doc["text"]]
            sources = self._extract_sources(metadata)

        result = {
            "answer": answer,
            "sources": sources,
            "action": "correct",
            "scores": scores,
            "is_ingested": is_ingested,
            "ingested_query": metadata.get("query", ""),
            "ingested_at": metadata.get("ingested_at", ""),
        }

        if self.evaluate_metrics:
            ragas_metrics = self.ragas_evaluator.evaluate(
                query=query,
                answer=answer,
                contexts=contexts,
            )
            result["ragas_metrics"] = ragas_metrics
            self.metrics_logger.log_metrics(query, ragas_metrics)

        return result

    _TODAY_RE = re.compile(r"\b(today|tonight|right now|current(ly)?)\b", re.I)
    _TOMORROW_RE = re.compile(r"\b(tomorrow)\b", re.I)
    _WEEK_RE = re.compile(r"\b(this week|next \d+ days|5.day|7.day|weekly|week)\b", re.I)

    def _aggregate_city_forecast(self, query: str, seed_metadata: dict) -> tuple[str, list[str], tuple]:
\
\
           
        city = seed_metadata.get("city", "")
        results = self.store.get_where({
            "$and": [
                {"doc_type": {"$eq": "forecast"}},
                {"city": {"$eq": city}},
            ]
        })
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        fallback = seed_metadata.get("text", "")
        if not docs:
            return fallback, [fallback], (None, None)

        paired = sorted(zip(metas, docs), key=lambda p: p[0].get("forecast_date", ""))

        today_str = date.today().isoformat()
        tomorrow_str = (date.today() + timedelta(days=1)).isoformat()

        if self._TODAY_RE.search(query):
            paired = [(m, d) for m, d in paired if m.get("forecast_date") == today_str]
        elif self._TOMORROW_RE.search(query):
            paired = [(m, d) for m, d in paired if m.get("forecast_date") == tomorrow_str]

        if not paired:
            return fallback, [fallback], (None, None)

        dates = [m.get("forecast_date", "") for m, _ in paired]
        date_range = (min(d for d in dates if d), max(d for d in dates if d))
        sorted_docs = [d for _, d in paired]
        return "\n\n".join(sorted_docs), sorted_docs, date_range

    def _incorrect_path(self, query: str, scores: list) -> dict:
                                             
        return self._web_search_path(query, scores=scores, action="incorrect")

    def _ambiguous_path(self, query: str, best_doc: dict, scores: list) -> dict:
                                                                     
        refined_local = self.refiner(document=best_doc["text"])
        web_contents, web_sources = self._do_web_search(query)

        local_sources = self._extract_sources(best_doc["metadata"])
        all_sources = local_sources + web_sources

                                   
        refs = [f"[1] {refined_local}"]
        for i, content in enumerate(web_contents, 2):
            refs.append(f"[{i}] {content}")
        ref_str = "\n\n".join(refs)

        answer = self.generator(query=query, references=ref_str)
        
        result = {
            "answer": answer,
            "sources": all_sources,
            "action": "ambiguous",
            "scores": scores,
            "is_ingested": False
        }
        
                                      
        if self.evaluate_metrics:
            all_contexts = [refined_local] + web_contents
            ragas_metrics = self.ragas_evaluator.evaluate(
                query=query,
                answer=answer,
                contexts=all_contexts,
            )
            result["ragas_metrics"] = ragas_metrics
            self.metrics_logger.log_metrics(query, ragas_metrics)
        
        return result

    def _web_search_path(self, query: str, scores: list = None, action: str = "web_only") -> dict:
                                                     
        web_contents, web_sources = self._do_web_search(query)

                                                    
        refs = []
        for i, content in enumerate(web_contents, 1):
            refs.append(f"[{i}] {content}")
        ref_str = "\n\n".join(refs) if refs else "No references found."

        answer = self.generator(query=query, references=ref_str)
        
        result = {
            "answer": answer,
            "sources": web_sources,
            "action": action,
            "scores": scores or [],
            "is_ingested": False
        }
        
                                      
        if self.evaluate_metrics:
            ragas_metrics = self.ragas_evaluator.evaluate(
                query=query,
                answer=answer,
                contexts=web_contents,
            )
            result["ragas_metrics"] = ragas_metrics
            self.metrics_logger.log_metrics(query, ragas_metrics)
        
        return result

    def _do_web_search(self, query: str) -> tuple[list[str], list[str]]:
                                                                                                               
        rewritten = self.rewriter(query=query)
        try:
            results = self.tavily.search(query=rewritten, max_results=3)
            contents = [r.get("content", "") for r in results.get("results", [])]
            sources = []
            for r in results.get("results", []):
                url = r.get("url", "")
                title = r.get("title", "")
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc.lstrip("www.")
                except Exception:
                    domain = url
                label = title if title else domain
                sources.append(f"{label} | {url}" if url else label)
        except Exception as e:
            contents = [f"Web search failed: {e}"]
            sources = []

        return contents, sources

    @staticmethod
    def _extract_forecast_sources(metadata: dict, date_range: tuple) -> list[str]:
                                                               
        city = metadata.get("city", "Unknown city")
        country = metadata.get("country", "")
        lat = metadata.get("latitude", "")
        lon = metadata.get("longitude", "")
        ingested_at = metadata.get("ingested_at", "")
        location = f"{city}, {country}".strip(", ")

        d_start, d_end = date_range
        if d_start and d_end and d_start != d_end:
            coverage = f"{d_start} → {d_end}"
        elif d_start:
            coverage = d_start
        else:
            coverage = metadata.get("forecast_date", "")

        coords = f"{lat}°N, {lon}°E" if lat and lon else ""

        parts = [
            "Open-Meteo Forecast API (open-meteo.com)",
            f"Location: {location}" + (f" ({coords})" if coords else ""),
            f"Forecast coverage: {coverage}",
            f"Data: temperature, precipitation, wind, humidity, UV index, sunrise/sunset",
        ]
        if ingested_at:
            retrieved = ingested_at.replace("T", " ").rstrip("Z")
            parts.append(f"Retrieved: {retrieved} UTC")
        parts.append("URL: https://open-meteo.com")
        return parts

    @staticmethod
    def _extract_sources(metadata: dict) -> list[str]:
\
\
\
\
\
\
           
        raw_sources = metadata.get("sources")
        if raw_sources and isinstance(raw_sources, str):
            try:
                parsed = json.loads(raw_sources)
                if isinstance(parsed, list) and parsed:
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass

        source = metadata.get("source", "unknown")
        source_type = metadata.get("source_type", "")

        if source == "open-meteo" or source_type == "forecast":
            city = metadata.get("city", "")
            country = metadata.get("country", "")
            forecast_date = metadata.get("forecast_date", "")
            ingested_at = metadata.get("ingested_at", "")
            lat = metadata.get("latitude", "")
            lon = metadata.get("longitude", "")
            location = f"{city}, {country}".strip(", ")
            coords = f"{lat}°N, {lon}°E" if lat and lon else ""
            parts = [
                "Open-Meteo Forecast API (open-meteo.com)",
                f"Location: {location}" + (f" ({coords})" if coords else ""),
            ]
            if forecast_date:
                parts.append(f"Forecast date: {forecast_date}")
            if ingested_at:
                parts.append(f"Retrieved: {ingested_at.replace('T', ' ').rstrip('Z')} UTC")
            parts.append("URL: https://open-meteo.com")
            return parts

        page = metadata.get("page", "")
        section = metadata.get("section", "")
        label = f"[{source_type}] {source}" if source_type else source
        parts = [label]
        if page:
            parts.append(f"Page {page}")
        if section:
            parts.append(f"Section: {section}")
        return parts

    def get_metrics_summary(self) -> dict:
                                                                 
        if not self.evaluate_metrics:
            return {"error": "Metrics evaluation not enabled"}
        return self.metrics_logger.get_summary()