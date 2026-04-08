import re
import time
from datetime import datetime, date, timedelta, timezone

import chromadb
import pytest

from config.settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, SAUDI_CITIES, FORECAST_TTL_DAYS

                                                                             
                                                                  
                                                                             

@pytest.fixture(scope="module")
def live_col():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    col = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return col


@pytest.fixture(scope="module")
def forecast_docs(live_col):
    results = live_col.get(
        where={"doc_type": {"$eq": "forecast"}},
        include=["metadatas", "documents"],
    )
    total = len(results.get("ids", []))
    if total == 0:
        pytest.skip("No forecast data in ChromaDB — run ingest_forecast.py first.")
    return results


@pytest.fixture(scope="module")
def fresh_docs(live_col):
    now_ts = int(time.time())
    results = live_col.get(
        where={
            "$and": [
                {"doc_type":      {"$eq": "forecast"}},
                {"expires_at_ts": {"$gt": now_ts}},
            ]
        },
        include=["metadatas", "documents"],
    )
    if not results.get("ids"):
        pytest.skip("All forecast chunks are expired — re-run ingest_forecast.py.")
    return results


                                                                             
                                         
                                                                             

class TestIngestPresence:

    def test_exactly_28_cities_indexed(self, forecast_docs):
        cities = {m["city"] for m in forecast_docs["metadatas"]}
        assert len(cities) == 28, (
            f"Expected 28 cities, got {len(cities)}: {sorted(cities)}"
        )

    def test_all_expected_cities_present(self, forecast_docs):
        cities = {m["city"] for m in forecast_docs["metadatas"]}
        missing = set(SAUDI_CITIES.keys()) - cities
        assert not missing, f"Missing cities: {sorted(missing)}"

    def test_no_unexpected_cities(self, forecast_docs):
        cities = {m["city"] for m in forecast_docs["metadatas"]}
        extra = cities - set(SAUDI_CITIES.keys())
        assert not extra, f"Unexpected cities in store: {sorted(extra)}"

    def test_exactly_196_total_chunks(self, forecast_docs):
        total = len(forecast_docs["ids"])
        assert total == 196, (
            f"Expected 196 chunks (28 cities × 7 days), got {total}"
        )

    def test_exactly_7_chunks_per_city(self, forecast_docs):
        from collections import Counter
        city_counts = Counter(m["city"] for m in forecast_docs["metadatas"])
        wrong = {city: n for city, n in city_counts.items() if n != 7}
        assert not wrong, f"Cities with wrong chunk count (expected 7): {wrong}"

    def test_all_chunks_are_fresh(self, forecast_docs):
        now_ts = int(time.time())
        expired = [
            m["city"] for m in forecast_docs["metadatas"]
            if int(m.get("expires_at_ts", 0)) <= now_ts
        ]
        assert not expired, (
            f"{len(expired)} chunks already expired. "
            f"Re-run ingest_forecast.py. Sample cities: {expired[:5]}"
        )

    def test_no_stale_expired_forecast_chunks_linger(self, live_col):
        now_ts = int(time.time())
        expired = live_col.get(
            where={
                "$and": [
                    {"doc_type":      {"$eq": "forecast"}},
                    {"expires_at_ts": {"$lte": now_ts}},
                ]
            },
            include=[],
        )
        assert len(expired.get("ids", [])) == 0, (
            f"{len(expired['ids'])} expired forecast docs still in store. "
            "Run ingest_forecast.py to clear them."
        )

    def test_doc_type_is_forecast_on_all_chunks(self, forecast_docs):
        bad = [
            i for i, m in enumerate(forecast_docs["metadatas"])
            if m.get("doc_type") != "forecast"
        ]
        assert not bad, f"Chunks with wrong doc_type at indices: {bad}"


                                                                             
                                                        
                                                                             

class TestIngestMetadata:

    REQUIRED_KEYS = {
        "doc_type", "city", "country", "forecast_date", "source",
        "ingested_at", "ingested_week", "ingested_at_ts", "expires_at_ts",
        "latitude", "longitude",
    }

    def test_all_required_metadata_keys_present(self, forecast_docs):
        for i, m in enumerate(forecast_docs["metadatas"]):
            missing = self.REQUIRED_KEYS - set(m.keys())
            assert not missing, (
                f"Chunk {i} ({m.get('city','?')} {m.get('forecast_date','?')}) "
                f"is missing metadata keys: {missing}"
            )

    def test_country_is_saudi_arabia_on_all_chunks(self, forecast_docs):
        bad = [
            f"{m['city']} {m['forecast_date']}"
            for m in forecast_docs["metadatas"]
            if m.get("country") != "Saudi Arabia"
        ]
        assert not bad, f"Wrong country on chunks: {bad}"

    def test_source_is_open_meteo_on_all_chunks(self, forecast_docs):
        bad = [
            m for m in forecast_docs["metadatas"]
            if m.get("source") != "open-meteo"
        ]
        assert not bad, f"{len(bad)} chunks with unexpected source."

    def test_forecast_date_format_yyyy_mm_dd(self, forecast_docs):
        bad = [
            m["forecast_date"] for m in forecast_docs["metadatas"]
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(m.get("forecast_date", "")))
        ]
        assert not bad, f"Malformed forecast_date values: {bad[:10]}"

    def test_ingested_at_is_iso_timestamp(self, forecast_docs):
        bad = [
            m["ingested_at"] for m in forecast_docs["metadatas"]
            if "T" not in str(m.get("ingested_at", ""))
        ]
        assert not bad, f"Malformed ingested_at values: {bad[:5]}"

    def test_ingested_week_format(self, forecast_docs):
        bad = [
            m["ingested_week"] for m in forecast_docs["metadatas"]
            if not re.match(r"^\d{4}-W\d{2}$", str(m.get("ingested_week", "")))
        ]
        assert not bad, f"Malformed ingested_week values: {bad[:5]}"

    def test_expires_at_ts_is_numeric(self, forecast_docs):
        for m in forecast_docs["metadatas"]:
            assert isinstance(m.get("expires_at_ts"), (int, float)), (
                f"expires_at_ts not numeric: {m.get('expires_at_ts')!r}"
            )

    def test_ingested_at_ts_is_numeric(self, forecast_docs):
        for m in forecast_docs["metadatas"]:
            assert isinstance(m.get("ingested_at_ts"), (int, float)), (
                f"ingested_at_ts not numeric: {m.get('ingested_at_ts')!r}"
            )

    def test_coordinates_match_settings(self, forecast_docs):
        for m in forecast_docs["metadatas"]:
            city = m["city"]
            expected = SAUDI_CITIES[city]
            assert abs(m["latitude"]  - expected["lat"]) < 0.001, \
                f"{city}: lat mismatch {m['latitude']} vs {expected['lat']}"
            assert abs(m["longitude"] - expected["lon"]) < 0.001, \
                f"{city}: lon mismatch {m['longitude']} vs {expected['lon']}"

    def test_ttl_math_expires_equals_ingested_plus_7_days(self, forecast_docs):
        expected_delta = FORECAST_TTL_DAYS * 86400
        bad = []
        for m in forecast_docs["metadatas"]:
            delta = int(m["expires_at_ts"]) - int(m["ingested_at_ts"])
            if abs(delta - expected_delta) > 5:
                bad.append(
                    f"{m['city']} {m['forecast_date']}: "
                    f"delta={delta}s expected={expected_delta}s"
                )
        assert not bad, f"TTL math wrong on {len(bad)} chunks:\n" + "\n".join(bad[:5])

    def test_all_chunks_share_same_ingested_week(self, forecast_docs):
        weeks = {m["ingested_week"] for m in forecast_docs["metadatas"]}
        assert len(weeks) == 1, (
            f"Chunks from multiple ingestion weeks detected: {weeks}. "
            "Run ingest_forecast.py --force to clean up."
        )

    def test_ingested_within_last_7_days(self, forecast_docs):
        now_ts = int(time.time())
        seven_days_ago = now_ts - FORECAST_TTL_DAYS * 86400
        stale = [
            m for m in forecast_docs["metadatas"]
            if int(m.get("ingested_at_ts", 0)) < seven_days_ago
        ]
        assert not stale, (
            f"{len(stale)} chunks were ingested more than {FORECAST_TTL_DAYS} days ago."
        )


                                                                             
                                                           
                                                                             

class TestIngestTextQuality:

    def test_text_is_not_empty(self, forecast_docs):
        empty = [i for i, d in enumerate(forecast_docs["documents"]) if not d or not d.strip()]
        assert not empty, f"Empty text at chunk indices: {empty}"

    def test_city_name_appears_in_text_lowercase(self, forecast_docs):
        bad = []
        for m, doc in zip(forecast_docs["metadatas"], forecast_docs["documents"]):
            city_lower = m["city"].lower()
            if city_lower not in doc:
                bad.append(f"{m['city']} {m['forecast_date']}: city not in text")
        assert not bad, f"{len(bad)} chunks missing lowercased city name:\n" + "\n".join(bad[:5])

    def test_saudi_arabia_appears_lowercase_in_text(self, forecast_docs):
        bad = [
            f"{m['city']} {m['forecast_date']}"
            for m, doc in zip(forecast_docs["metadatas"], forecast_docs["documents"])
            if "saudi arabia" not in doc
        ]
        assert not bad, f"{len(bad)} chunks missing 'saudi arabia' in text: {bad[:5]}"

    def test_no_uppercase_field_labels_in_text(self, forecast_docs):
        banned = ["Condition:", "Temperature:", "Precipitation:",
                  "Wind:", "Humidity:", "UV Index", "Sunrise:", "Sunset:"]
        violations = []
        for m, doc in zip(forecast_docs["metadatas"], forecast_docs["documents"]):
            found = [b for b in banned if b in doc]
            if found:
                violations.append(f"{m['city']} {m['forecast_date']}: {found}")
        assert not violations, (
            f"{len(violations)} chunks have uppercase labels (not normalized):\n"
            + "\n".join(violations[:5])
        )

    def test_lowercase_field_labels_present(self, forecast_docs):
        required_labels = ["condition:", "temperature:", "precipitation:",
                           "wind:", "humidity:", "uv index", "sunrise:", "sunset:"]
        for m, doc in zip(forecast_docs["metadatas"], forecast_docs["documents"]):
            for label in required_labels:
                assert label in doc, (
                    f"{m['city']} {m['forecast_date']}: missing '{label}' in text"
                )

    def test_forecast_date_appears_in_text(self, forecast_docs):
        bad = [
            f"{m['city']} {m['forecast_date']}"
            for m, doc in zip(forecast_docs["metadatas"], forecast_docs["documents"])
            if m["forecast_date"] not in doc
        ]
        assert not bad, f"{len(bad)} chunks missing their date in text: {bad[:5]}"

    def test_whitespace_collapsed_no_double_spaces(self, forecast_docs):
        bad = []
        for m, doc in zip(forecast_docs["metadatas"], forecast_docs["documents"]):
            if "  " in doc or "\n" in doc:
                bad.append(f"{m['city']} {m['forecast_date']}")
        assert not bad, (
            f"{len(bad)} chunks still have un-collapsed whitespace: {bad[:5]}"
        )


                                                                             
                                               
                                                                             

class TestIngestDateCoverage:

    def test_forecast_dates_start_from_today_or_tomorrow(self, forecast_docs):
        today = date.today()
        all_dates = sorted({
            date.fromisoformat(m["forecast_date"])
            for m in forecast_docs["metadatas"]
        })
        earliest = all_dates[0]
        assert (earliest - today).days <= 1, (
            f"Earliest forecast date {earliest} is more than 1 day away from today {today}"
        )

    def test_forecast_covers_7_distinct_days(self, forecast_docs):
        all_dates = {m["forecast_date"] for m in forecast_docs["metadatas"]}
        assert len(all_dates) == 7, (
            f"Expected 7 distinct forecast dates, got {len(all_dates)}: {sorted(all_dates)}"
        )

    def test_no_duplicate_date_per_city(self, forecast_docs):
        from collections import defaultdict, Counter
        city_dates = defaultdict(list)
        for m in forecast_docs["metadatas"]:
            city_dates[m["city"]].append(m["forecast_date"])
        bad = {
            city: [d for d, n in Counter(dates).items() if n > 1]
            for city, dates in city_dates.items()
            if len(set(dates)) != len(dates)
        }
        assert not bad, f"Duplicate dates found per city: {bad}"

    def test_per_city_dates_are_consecutive(self, forecast_docs):
        from collections import defaultdict
        city_dates = defaultdict(list)
        for m in forecast_docs["metadatas"]:
            city_dates[m["city"]].append(m["forecast_date"])

        bad = []
        for city, dates in city_dates.items():
            sorted_dates = sorted(dates)
            for i in range(1, len(sorted_dates)):
                prev = date.fromisoformat(sorted_dates[i - 1])
                curr = date.fromisoformat(sorted_dates[i])
                if (curr - prev).days != 1:
                    bad.append(f"{city}: gap between {sorted_dates[i-1]} and {sorted_dates[i]}")
        assert not bad, f"Non-consecutive forecast dates:\n" + "\n".join(bad[:10])

    def test_all_7_forecast_dates_within_next_8_days(self, forecast_docs):
        today = date.today()
        bad = [
            m["forecast_date"] for m in forecast_docs["metadatas"]
            if (date.fromisoformat(m["forecast_date"]) - today).days > 8
        ]
        assert not bad, f"Forecast dates too far in future: {sorted(set(bad))}"


                                                                             
                                                                
                                                                   
                                                                             

@pytest.mark.slow
class TestIngestRAGRetrieval:

    @pytest.fixture(scope="class")
    def store(self):
        from vectorstore.chroma_store import ChromaStore
        return ChromaStore()

    @pytest.mark.parametrize("city", [
        "Riyadh", "Jeddah", "Mecca", "Medina", "Dammam", "Tabuk", "Abha",
    ])
    def test_query_returns_correct_city(self, store, city):
        query = f"weather forecast for {city.lower()}, saudi arabia"
        results = store.query(query, k=5)
        assert results, f"No results returned for query: {query!r}"
        returned_cities = [r["metadata"].get("city", "") for r in results]
        assert city in returned_cities, (
            f"Query for {city!r} did not return a {city} chunk. "
            f"Top cities returned: {returned_cities}"
        )

    def test_query_temperature_riyadh(self, store):
        results = store.query("temperature in riyadh this week", k=3)
        assert results
        cities = [r["metadata"].get("city") for r in results]
        assert "Riyadh" in cities

    def test_query_returns_fresh_docs_only(self, store):
        now_ts = int(time.time())
        results = store.query("saudi arabia weather this week", k=10)
        expired = [
            r["metadata"].get("city")
            for r in results
            if int(r["metadata"].get("expires_at_ts", 0)) <= now_ts
        ]
        assert not expired, f"Query returned expired chunks for cities: {expired}"

    def test_query_doc_type_forecast_in_results(self, store):
        results = store.query("forecast rain wind humidity", k=10)
        non_forecast = [
            r for r in results
            if r["metadata"].get("doc_type") != "forecast"
        ]
                                                                                   
        forecast_results = [r for r in results if r["metadata"].get("doc_type") == "forecast"]
        assert forecast_results, "No forecast chunks returned for weather query."

    def test_all_28_cities_retrievable(self, store):
        failed = []
        for city in SAUDI_CITIES:
            results = store.query(f"weather forecast {city.lower()}", k=5)
            returned = [r["metadata"].get("city") for r in results]
            if city not in returned:
                failed.append(city)
        assert not failed, (
            f"{len(failed)} cities not retrievable by direct name query: {failed}"
        )
