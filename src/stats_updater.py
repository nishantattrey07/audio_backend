# src/stats_updater.py
import logging
import json
import redis
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor

# ✨ NEW: Import the centralized settings object
from config import settings

# --- Configuration ---
STATS_CACHE_KEY = "api:system_stats_cache"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def aggregate_hourly_stats(redis_conn, key_prefix: str, days_to_scan=7):
    """Scans for hourly keys and aggregates them into daily totals."""
    daily_totals = {}
    total = 0
    
    # Scan keys for the last N days
    for i in range(days_to_scan):
        date = datetime.utcnow() - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        
        # Scan for all 24 hours of that day
        keys_to_fetch = [f"{key_prefix}:{date_str}:{h:02d}" for h in range(24)]
        values = redis_conn.mget(keys_to_fetch)
        
        day_total = sum(int(v) for v in values if v is not None)
        
        if day_total > 0:
            daily_totals[date_str] = day_total
            total += day_total
            
    return {"total": total, "by_day": daily_totals}

def get_trending_songs(redis_conn, limit=10):
    """Gets the top trending songs from the sorted set."""
    # ZREVRANGE gets the top N items from a sorted set. WITHSCORES includes their counts.
    trending_data = redis_conn.zrevrange("stats:trending:songs", 0, limit - 1, withscores=True)
    # We need to look up the song names from their IDs
    track_metadata = redis_conn.hgetall("track_metadata")

    trending_list = []
    for track_id, score in trending_data:
        trending_list.append({
            "track_id": int(track_id),
            "track_name": track_metadata.get(track_id, "Unknown Track"),
            "match_count": int(score)
        })
    return trending_list

def get_failure_reasons(redis_conn):
    """Gets the counts for different failure reasons."""
    keys = redis_conn.keys("stats:fail:reason:*")
    if not keys:
        return {}
    values = redis_conn.mget(keys)
    # Create a clean dictionary from the keys and values
    reasons = {key.split(":")[-1]: int(val) for key, val in zip(keys, values) if val}
    return reasons

def main():
    """Main function to run the statistics aggregation and caching process."""
    logger.info("🚀 Starting statistics aggregation worker...")
    redis_conn = None
    
    try:
        # ✨ REFACTORED: Use settings object for Redis connection
        redis_conn = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD or None,
            decode_responses=True
        )
        redis_conn.ping()

        # 1. Aggregate usage, success, and failure stats
        usage_stats = aggregate_hourly_stats(redis_conn, "stats:usage:match")
        success_stats = aggregate_hourly_stats(redis_conn, "stats:success:match")
        
        # 2. Calculate success rate
        success_rate = (success_stats["total"] / usage_stats["total"] * 100) if usage_stats["total"] > 0 else 0
        
        # 3. Get top 10 trending songs
        trending_songs = get_trending_songs(redis_conn)
        
        # 4. Get failure reason breakdown
        failure_reasons = get_failure_reasons(redis_conn)

        # 5. Assemble the final JSON object
        final_stats = {
            "last_updated_utc": datetime.utcnow().isoformat(),
            "overall": {
                "total_requests": usage_stats["total"],
                "successful_matches": success_stats["total"],
                "success_rate_percent": round(success_rate, 2),
            },
            "usage_by_day": usage_stats["by_day"],
            "trending_top_10": trending_songs,
            "failure_breakdown": failure_reasons
        }
        
        # 6. Serialize and cache the result in Redis for 10 minutes
        stats_json = json.dumps(final_stats, indent=2)
        redis_conn.set(STATS_CACHE_KEY, stats_json, ex=600)
        
        logger.info(f"✅ Successfully cached aggregated stats to Redis key '{STATS_CACHE_KEY}'.")
        print("--- Aggregated Statistics ---")
        print(stats_json)
        print("-----------------------------")

    except redis.RedisError as e:
        logger.error(f"🔴 A Redis error occurred: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"🔴 An unexpected error occurred: {e}", exc_info=True)
    
    finally:
        if redis_conn:
            redis_conn.close()
        logger.info("👋 Statistics aggregation worker finished.")

if __name__ == "__main__":
    main()