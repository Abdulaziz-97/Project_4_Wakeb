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
   

import time
import logging
from datetime import datetime, timezone

from config.settings import SAUDI_CITIES, FORECAST_TTL_DAYS
from loaders.forecast_loader import fetch_city_forecast
from vectorstore.chroma_store import ChromaStore

logger = logging.getLogger("weather_agent")


class ForecastManager:
                                                                            

    def __init__(self, ttl_days: int = None):
        self.ttl_days = ttl_days or FORECAST_TTL_DAYS
        self._store = ChromaStore()

                                                                        
                
                                                                        

    def check_and_refresh(self, force: bool = False) -> bool:
\
\
\
\
\
\
\
\
           
        if not force and self._has_fresh_forecasts():
            fresh_until = self._fresh_until_iso()
            logger.info(
                f"[ForecastManager] Forecast data is fresh until {fresh_until}. "
                "Skipping re-ingestion."
            )
            return False

        if force:
            logger.info("[ForecastManager] Force-refresh requested.")
        else:
            logger.info("[ForecastManager] No fresh forecast data found. Starting weekly ingest.")

        expired_count = self._delete_expired_forecasts()
        if expired_count:
            logger.info(f"[ForecastManager] Deleted {expired_count} expired forecast chunks.")

        return self._ingest_all_cities()

    def status(self) -> dict:
                                                                          
        now_ts = int(time.time())
        try:
            all_fc = self._store.get_where({"doc_type": {"$eq": "forecast"}})
        except Exception:
            all_fc = {"ids": [], "metadatas": []}

        total = len(all_fc.get("ids", []))
        metas = all_fc.get("metadatas", []) or []

        fresh = sum(
            1 for m in metas
            if isinstance(m, dict) and int(m.get("expires_at_ts", 0)) > now_ts
        )
        cities = sorted({m["city"] for m in metas if isinstance(m, dict) and "city" in m})

        expires_values = [
            int(m["expires_at_ts"])
            for m in metas
            if isinstance(m, dict) and "expires_at_ts" in m
        ]
        next_expiry_iso = (
            datetime.fromtimestamp(min(expires_values), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if expires_values else "N/A"
        )

        return {
            "total_forecast_chunks": total,
            "fresh_chunks": fresh,
            "expired_chunks": total - fresh,
            "cities_indexed": cities,
            "next_expiry": next_expiry_iso,
        }

                                                                        
                      
                                                                        

    def _has_fresh_forecasts(self) -> bool:
                                                                            
        now_ts = int(time.time())
        try:
            results = self._store.get_where(
                {
                    "$and": [
                        {"doc_type":      {"$eq": "forecast"}},
                        {"expires_at_ts": {"$gt": now_ts}},
                    ]
                },
                limit=1,
            )
            return bool(results.get("ids"))
        except Exception as exc:
                                                                           
            logger.debug(f"[ForecastManager] Freshness check exception (treating as stale): {exc}")
            return False

    def _fresh_until_iso(self) -> str:
                                                                              
        now_ts = int(time.time())
        try:
            results = self._store.get_where(
                {
                    "$and": [
                        {"doc_type":      {"$eq": "forecast"}},
                        {"expires_at_ts": {"$gt": now_ts}},
                    ]
                },
            )
            metas = results.get("metadatas", []) or []
            exp_ts = [
                int(m["expires_at_ts"])
                for m in metas
                if isinstance(m, dict) and "expires_at_ts" in m
            ]
            if exp_ts:
                return datetime.fromtimestamp(
                    min(exp_ts), tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
        return "unknown"

    def _delete_expired_forecasts(self) -> int:
                                                                                     
        now_ts = int(time.time())
        try:
            deleted = self._store.delete_where(
                {
                    "$and": [
                        {"doc_type":      {"$eq": "forecast"}},
                        {"expires_at_ts": {"$lte": now_ts}},
                    ]
                }
            )
            return deleted
        except Exception as exc:
            logger.warning(f"[ForecastManager] Could not delete expired forecasts: {exc}")
                                                                                
            try:
                return self._store.delete_where({"doc_type": {"$eq": "forecast"}})
            except Exception:
                return 0

    def _ingest_all_cities(self) -> bool:
                                                                                            
        ingested_at_ts = int(time.time())
        total_chunks = 0
        failed = []

        for city_name, coords in SAUDI_CITIES.items():
            chunks = fetch_city_forecast(
                city_name=city_name,
                lat=coords["lat"],
                lon=coords["lon"],
                forecast_days=self.ttl_days,
                ingested_at_ts=ingested_at_ts,
                ttl_days=self.ttl_days,
            )
            if not chunks:
                failed.append(city_name)
                continue
            try:
                self._store.add_documents(chunks)
                total_chunks += len(chunks)
            except Exception as exc:
                logger.error(f"[ForecastManager] Failed to index {city_name}: {exc}")
                failed.append(city_name)

        expires_dt = datetime.fromtimestamp(
            ingested_at_ts + self.ttl_days * 86400, tz=timezone.utc
        ).strftime("%Y-%m-%d")

        if failed:
            logger.warning(
                f"[ForecastManager] Ingestion finished with {len(failed)} failures: {failed}"
            )
        logger.info(
            f"[ForecastManager] Indexed {total_chunks} forecast chunks for "
            f"{len(SAUDI_CITIES) - len(failed)}/{len(SAUDI_CITIES)} cities. "
            f"TTL expires {expires_dt}."
        )

        return total_chunks > 0


                                                                             
                                                                              
                                                                             

_manager: ForecastManager | None = None


def get_manager() -> ForecastManager:
    global _manager
    if _manager is None:
        _manager = ForecastManager()
    return _manager


def check_and_refresh(force: bool = False) -> bool:
                                                                        
    return get_manager().check_and_refresh(force=force)
