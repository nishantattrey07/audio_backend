# src/main.py
import logging
import os
import shutil
import uuid
import json  # Ensure json is imported at the top
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field
from .audio_fingerprinter import AudioFingerprinter

# ✨ NEW: Import the centralized settings object
from .config import settings

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Objects ---
app_globals = {}

# --- Pydantic Response Models (Final Polished Version) ---
class Artist(BaseModel):
    """Individual artist information with their role in the song."""
    artist_name: str = Field(..., description="Name of the artist")
    role: str = Field(..., description="Artist's role: primary, featured, or producer")

class SongMetadata(BaseModel):
    """Public song metadata - cleaned up for external API consumption."""
    song_id: int = Field(..., description="Unique song identifier")
    song_title: str = Field(..., description="Title of the song")
    duration_seconds: Optional[int] = Field(None, description="Song duration in seconds")
    youtube_id: str = Field(..., description="YouTube video ID for playback")
    song_url: Optional[str] = Field(None, description="Alternative streaming URL")
    album_title: Optional[str] = Field(None, description="Album name")
    album_main_artist: Optional[str] = Field(None, description="Primary album artist")
    artists: List[Artist] = Field(default_factory=list, description="List of all artists and their roles")
    # 🔧 REMOVED: `created_at` and `updated_at` are internal fields and no longer part of the public API response
    # This makes the response cleaner and more focused on what users actually need

class MatchResponse(BaseModel):
    """Complete API response for successful audio matches - optimized for end users."""
    match_found: bool = Field(True, description="Whether a match was found")
    score: float = Field(..., description="Confidence score (number of matching fingerprint hashes)")
    offset_seconds: float = Field(..., description="Time offset into the original track")
    offset_formatted: Optional[str] = Field(None, description="Human-readable time offset (mm:ss)")
    processing_time: float = Field(..., description="Total processing time in seconds")
    # ✨ NEW: Ready-to-use YouTube URL with automatic timestamp positioning
    youtube_playback_url: Optional[str] = Field(None, description="Direct YouTube playback URL with timestamp - click to play at the exact detected moment")
    metadata: SongMetadata = Field(..., description="Rich song metadata")

class ErrorResponse(BaseModel):
    """Standardized error response format."""
    match_found: bool = Field(False, description="Always false for error responses")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Audio Fingerprinting Backend API",
    description="A scalable API for matching audio samples with rich metadata and ready-to-use playback URLs.",
    version="2.2.0", # Version bump for response polishing
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- Application Lifecycle Events ---
@app.on_event("startup")
def startup_event():
    """Initialize all services with enhanced error handling."""
    logging.info("🚀 Application starting up...")
    
    # Initialize Redis connection for fingerprinter
    fingerprinter = AudioFingerprinter()
    # ✨ REFACTORED: Use settings object for Redis connection
    connected_redis = fingerprinter.init_redis_connection(
        host=settings.REDIS_HOST, 
        port=settings.REDIS_PORT, 
        password=settings.REDIS_PASSWORD or None
    )
    if not connected_redis:
        logging.error("🔴 FATAL: Could not connect to Redis. Fingerprinting service will be unavailable.")
    else:
        logging.info("✅ Fingerprinter successfully connected to Redis.")
    app_globals["fingerprinter"] = fingerprinter

    # Create PostgreSQL connection pool
    try:
        # ✨ REFACTORED: Use the DSN from the settings object for the pool
        pg_pool = psycopg2.pool.SimpleConnectionPool(1, 10, dsn=settings.postgres_dsn)
        app_globals["pg_pool"] = pg_pool
        logging.info("✅ PostgreSQL connection pool created successfully.")
        logging.info("   🔧 Pool configured: 1-10 connections for optimal performance")
    except psycopg2.Error as e:
        logging.error(f"🔴 FATAL: Failed to create PostgreSQL connection pool: {e}")
        app_globals["pg_pool"] = None

@app.on_event("shutdown")
def shutdown_event():
    """Gracefully shutdown all database connections."""
    logging.info("👋 Application shutting down...")
    if "fingerprinter" in app_globals and app_globals["fingerprinter"].redis_client:
        app_globals["fingerprinter"].redis_client.close()
        logging.info("🔌 Redis connection closed.")
    if "pg_pool" in app_globals and app_globals["pg_pool"]:
        app_globals["pg_pool"].closeall()
        logging.info("🔌 PostgreSQL connection pool closed.")

# --- API Endpoints ---
@app.get("/")
def read_root():
    """Health check endpoint with service status."""
    fingerprinter = app_globals.get("fingerprinter")
    pg_pool = app_globals.get("pg_pool")
    
    redis_status = fingerprinter and fingerprinter.redis_client
    postgres_status = pg_pool is not None
    
    return {
        "status": "online" if (redis_status and postgres_status) else "degraded",
        "message": "Audio Fingerprinting API v2.2.0 - Final Polished Version",
        "services": {
            "redis": "connected" if redis_status else "disconnected", 
            "postgresql": "pool_ready" if postgres_status else "unavailable"
        },
        "features": {
            "response_format": "polished_and_user_friendly",
            "youtube_urls": "automatic_timestamp_generation",
            "api_documentation": "available_at_/docs",
            "internal_fields": "cleaned_from_public_response"
        }
    }

@app.post("/api/v1/match", 
          response_model=MatchResponse,
          responses={
              200: {"description": "Successful match with rich metadata and playback URL"},
              404: {"description": "No match found", "model": ErrorResponse},
              503: {"description": "Service unavailable", "model": ErrorResponse}
          },
          summary="Match an audio sample and get polished, rich metadata")
async def match_audio_sample(
    audio_file: UploadFile = File(..., description="Audio file to analyze (MP3, WAV, FLAC, etc.)"),
    x_client_id: Optional[str] = Header(None, description="Optional unique ID for anonymous user tracking.")
):
    """
    🎵 **POLISHED AUDIO MATCHING WITH READY-TO-USE PLAYBACK URLS**
    
    This endpoint performs high-performance audio fingerprint matching and returns
    a clean, user-friendly response with automatic YouTube URL generation.
    
    **Key Features of v2.2.0:**
    - 🎯 **Instant Playback**: Automatic YouTube URLs with exact timestamps  
    - 🧹 **Clean Response**: Removed internal database timestamps
    - ⚡ **High Performance**: Connection pooling + Redis fingerprinting
    - 📱 **Mobile-Ready**: Optimized JSON structure for apps
    - 🔗 **Direct Integration**: URLs ready for iframe, links, or API calls
    
    **Process Flow:**
    1. 📤 Upload audio sample (any common format)
    2. ⚡ Generate audio fingerprints locally  
    3. 🔍 Search Redis database for matches (sub-second)
    4. 🎯 Fetch rich metadata from PostgreSQL (pooled connections)
    5. 🔗 Generate timestamped YouTube URL automatically
    6. ✨ Return polished JSON response
    
    **Example Response:**
    ```json
    {
        "match_found": true,
        "score": 1653.0,
        "offset_seconds": 23.96,
        "offset_formatted": "0:23",
        "processing_time": 2.13,
        "youtube_playback_url": "https://www.youtube.com/watch?v=tWuijqCvHp4&t=23s",
        "metadata": {
            "song_id": 177,
            "song_title": "Brodha V Forever [Music Video]",
            "duration_seconds": 239,
            "youtube_id": "tWuijqCvHp4",
            "album_title": null,
            "album_main_artist": null,
            "artists": [{"artist_name": "KR$NA", "role": "featured"}]
        }
    }
    ```
    
    **Perfect for:**
    - 🎵 Music discovery apps
    - 📱 Social media "now playing" features  
    - 🔍 Copyright detection systems
    - 🎧 Playlist generation tools
    - 📺 Video content identification
    """
    fingerprinter = app_globals.get("fingerprinter")
    pg_pool = app_globals.get("pg_pool")

    if not (fingerprinter and fingerprinter.redis_client and pg_pool):
        raise HTTPException(status_code=503, detail="Service Unavailable: A database connection is not ready.")

    temp_file_path = os.path.join("temp_samples", f"{uuid.uuid4()}-{audio_file.filename}")
    os.makedirs("temp_samples", exist_ok=True)
    
    pg_conn = None
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        result = fingerprinter.match_sample_from_redis(temp_file_path, verbose=False)
        
        # --- START: Comprehensive Analytics Block (Task 3.2) ---
        try:
            now = datetime.utcnow()
            hourly_key_suffix = now.strftime("%Y-%m-%d:%H")
            
            pipe = fingerprinter.redis_client.pipeline(transaction=False)
            
            # 1. Basic Usage Counter
            pipe.incr(f"stats:usage:match:{hourly_key_suffix}")
            
            # 2. Anonymous User Tracking
            if x_client_id:
                pipe.incr(f"stats:user:{x_client_id}:requests")

            if result and result.get("match_found"):
                # 3a. Success Analytics
                track_id = result.get("track_id")
                pipe.incr(f"stats:success:match:{hourly_key_suffix}")
                # 4a. Trending Songs Analytics
                pipe.zincrby("stats:trending:songs", 1, track_id)
            else:
                # 3b. Failure Analytics
                pipe.incr(f"stats:fail:match:{hourly_key_suffix}")
                # 4b. Failure Reason Analytics
                pipe.incr("stats:fail:reason:no_match")
            
            pipe.execute()
            logging.info(f"📊 Comprehensive analytics recorded for client '{x_client_id}'")
        except Exception as e:
            logging.error(f"🔴 Failed to record comprehensive analytics: {e}")
        # --- END: Comprehensive Analytics Block ---

        if not (result and result.get("match_found")):
            raise HTTPException(status_code=404, detail="No match found in the database.")

        track_id = result.get("track_id")
        pg_conn = pg_pool.getconn()
        cursor = pg_conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM song_details WHERE song_id = %s", (track_id,))
        metadata = cursor.fetchone()
        cursor.close()

        if not metadata:
            # Log this specific failure reason
            try:
                fingerprinter.redis_client.incr("stats:fail:reason:inconsistent_data")
            except Exception as e:
                logging.error(f"🔴 Failed to record inconsistency analytic: {e}")
            raise HTTPException(status_code=404, detail=f"Data inconsistency: track {track_id} not found in metadata.")

        youtube_playback_url = None
        if metadata.get("youtube_id"):
            offset = result.get("offset_seconds", 0)
            timestamp_seconds = max(0, int(offset))
            youtube_playback_url = f"https://www.youtube.com/watch?v={metadata['youtube_id']}&t={timestamp_seconds}s"

        return MatchResponse(
            match_found=True,
            score=result.get("score"),
            offset_seconds=result.get("offset_seconds"),
            offset_formatted=result.get("offset_formatted"),
            processing_time=result.get("processing_time"),
            youtube_playback_url=youtube_playback_url,
            metadata=SongMetadata(**metadata)
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"🔴 An unexpected error occurred during matching: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    
    finally:
        if pg_conn:
            pg_pool.putconn(pg_conn)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        await audio_file.close()

@app.get("/api/v1/stats", 
         summary="Get cached system and usage statistics")
def get_cached_system_stats():
    """
    Retrieves and returns the pre-aggregated system statistics from the Redis cache.
    
    This endpoint is designed to be extremely fast as it only performs a single
    read from the Redis cache. The heavy work of calculating these stats
    is handled by a separate background worker (`stats_updater.py`).
    """
    fingerprinter = app_globals.get("fingerprinter")
    if not (fingerprinter and fingerprinter.redis_client):
        raise HTTPException(status_code=503, detail="Service Unavailable: Redis connection is not ready.")
    
    try:
        # The key where our stats worker stores the aggregated data
        STATS_CACHE_KEY = "api:system_stats_cache"
        
        cached_stats = fingerprinter.redis_client.get(STATS_CACHE_KEY)
        
        if cached_stats:
            # If we found the cached data, parse the JSON string and return it
            return json.loads(cached_stats)
        else:
            # If the key doesn't exist, it means the worker hasn't run yet.
            return {
                "status": "pending_aggregation",
                "message": "Statistics are being compiled. Please check back in a few minutes."
            }
    except Exception as e:
        logging.error(f"🔴 Could not retrieve cached stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while fetching statistics.")