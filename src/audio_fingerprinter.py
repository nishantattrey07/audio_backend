import numpy as np
import librosa
import os
import pickle
import redis
import time
import uuid  # Added for temporary Redis key generation
from collections import defaultdict, Counter
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure, binary_erosion
import matplotlib.pyplot as plt
from typing import Optional  # Added for type hints

class AudioFingerprinter:
    """
    Implementation of a Shazam-like audio fingerprinting system based on
    Avery Wang's paper "An Industrial-Strength Audio Search Algorithm"
    Enhanced with Redis support for high-performance database operations
    """
    
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, 
                 peak_neighborhood_size=20, amplitude_min=0.5,
                 target_dt=100, target_df=50, fan_out=15,
                 min_matches_for_hit=5, min_histogram_peak=3,
                 # Redis Configuration Parameters
                 redis_batch_size=5000, redis_query_batch_size=100,
                 redis_socket_timeout=30, redis_connection_timeout=10,
                 redis_max_retries=3, redis_retry_delay=1):
        """
        Initialize the fingerprinter with parameters similar to the original implementation
        
        Parameters:
        -----------
        sample_rate : int
            The sample rate to use for audio processing
        n_fft : int
            FFT window size for spectrogram
        hop_length : int
            Hop length for STFT
        peak_neighborhood_size : int
            Size of neighborhood for peak detection
        amplitude_min : float
            Minimum amplitude for peak detection (as multiple of median)
        target_dt : int
            Maximum time difference for target zone
        target_df : int
            Maximum frequency difference for target zone
        fan_out : int
            Maximum number of points to pair with each anchor point
        min_matches_for_hit : int
            Minimum number of matching hashes for a valid match
        min_histogram_peak : int
            Minimum count in histogram peak for a valid match
        redis_batch_size : int
            Number of hashes to process in each Redis batch (default: 5000)
        redis_query_batch_size : int
            Number of queries to batch together for matching (default: 100)
        redis_socket_timeout : int
            Redis socket timeout in seconds (default: 30)
        redis_connection_timeout : int
            Redis connection timeout in seconds (default: 10)
        redis_max_retries : int
            Maximum number of retry attempts for failed Redis operations (default: 3)
        redis_retry_delay : int
            Base delay between retry attempts in seconds (default: 1)
        """
        # Audio processing parameters
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.peak_neighborhood_size = peak_neighborhood_size
        self.amplitude_min = amplitude_min
        self.target_dt = target_dt
        self.target_df = target_df
        self.fan_out = fan_out
        self.min_matches_for_hit = min_matches_for_hit
        self.min_histogram_peak = min_histogram_peak
        
        # Redis configuration parameters (user-configurable)
        self.redis_batch_size = redis_batch_size
        self.redis_query_batch_size = redis_query_batch_size
        self.redis_socket_timeout = redis_socket_timeout
        self.redis_connection_timeout = redis_connection_timeout
        self.redis_max_retries = redis_max_retries
        self.redis_retry_delay = redis_retry_delay
        
        # Database storage (similar to original code)
        # Structure: { hash_value: [(track_id, time_offset_in_frames), ...], ... }
        self.database_index = defaultdict(list)
        # Structure: { track_id: track_name, ... }
        self.track_metadata = {}
        
        # Redis connection (will be initialized when needed)
        self.redis_client = None
        
        # Load Lua matcher script
        try:
            possible_paths = [
            'matcher.lua',                    # Current directory
            'src/matcher.lua',               # src subdirectory
            os.path.join(os.path.dirname(__file__), 'matcher.lua')  # Same directory as this file
            ]
            
            lua_script_loaded = False
            for path in possible_paths:
                try:
                    with open(path, 'r') as f:
                        self.lua_matcher_script = f.read()
                    print(f"✅ Lua matcher script loaded from {path}")
                    lua_script_loaded = True
                    break
                except FileNotFoundError:
                    continue
                
            if not lua_script_loaded:
                raise FileNotFoundError("matcher.lua not found in any expected location")
        except FileNotFoundError:
            print("⚠️  Warning: matcher.lua not found. Redis matching will use fallback method.")
            print("   Searched locations: matcher.lua, src/matcher.lua, and src/ directory")
            self.lua_matcher_script = None
        except Exception as e:
            print(f"⚠️  Warning: Error loading matcher.lua: {e}")
            self.lua_matcher_script = None
            
        
        
        # SHA hash for the loaded script (will be set in init_redis_connection)
        self.lua_matcher_sha = None
        
        # Create directories if they don't exist
        self._create_directories()
    
    def update_redis_config(self, batch_size=None, query_batch_size=None, 
                           socket_timeout=None, connection_timeout=None,
                           max_retries=None, retry_delay=None):
        """
        Update Redis configuration parameters at runtime
        
        Parameters:
        -----------
        batch_size : int, optional
            Number of hashes to process in each Redis batch
        query_batch_size : int, optional
            Number of queries to batch together for matching
        socket_timeout : int, optional
            Redis socket timeout in seconds
        connection_timeout : int, optional
            Redis connection timeout in seconds
        max_retries : int, optional
            Maximum number of retry attempts
        retry_delay : int, optional
            Base delay between retry attempts in seconds
        """
        if batch_size is not None:
            self.redis_batch_size = batch_size
            print(f"🔧 Updated Redis batch size: {batch_size}")
        if query_batch_size is not None:
            self.redis_query_batch_size = query_batch_size
            print(f"🔧 Updated Redis query batch size: {query_batch_size}")
        if socket_timeout is not None:
            self.redis_socket_timeout = socket_timeout
            print(f"🔧 Updated Redis socket timeout: {socket_timeout}s")
        if connection_timeout is not None:
            self.redis_connection_timeout = connection_timeout
            print(f"🔧 Updated Redis connection timeout: {connection_timeout}s")
        if max_retries is not None:
            self.redis_max_retries = max_retries
            print(f"🔧 Updated Redis max retries: {max_retries}")
        if retry_delay is not None:
            self.redis_retry_delay = retry_delay
            print(f"🔧 Updated Redis retry delay: {retry_delay}s")
    
    def get_redis_config(self):
        """Get current Redis configuration"""
        config = {
            'batch_size': self.redis_batch_size,
            'query_batch_size': self.redis_query_batch_size,
            'socket_timeout': self.redis_socket_timeout,
            'connection_timeout': self.redis_connection_timeout,
            'max_retries': self.redis_max_retries,
            'retry_delay': self.redis_retry_delay
        }
        return config
    
    def print_redis_config(self):
        """Print current Redis configuration"""
        print("⚙️  Current Redis Configuration:")
        print(f"   Batch size: {self.redis_batch_size}")
        print(f"   Query batch size: {self.redis_query_batch_size}")
        print(f"   Socket timeout: {self.redis_socket_timeout}s")
        print(f"   Connection timeout: {self.redis_connection_timeout}s")
        print(f"   Max retries: {self.redis_max_retries}")
        print(f"   Retry delay: {self.redis_retry_delay}s")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = ['music', 'samples', 'database']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
    
    def init_redis_connection(self, host: str, port: int, password: Optional[str] = None):
        """
        Initialize Redis connection with robust timeout settings.
        Configuration is now passed in from the application's central settings.
        
        Parameters:
        -----------
        host : str
            Redis server host
        port : int
            Redis server port
        password : str, optional
            Redis password if required
            
        Returns:
        --------
        success : bool
            Whether connection was successful
        """
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                password=password,
                decode_responses=True,  # Automatically decode byte responses to strings
                socket_connect_timeout=self.redis_connection_timeout,
                socket_timeout=self.redis_socket_timeout,
                socket_keepalive=True,
                socket_keepalive_options={},
                retry_on_timeout=True,
                health_check_interval=30,
                max_connections=20  # Connection pooling
            )
            # Test the connection
            self.redis_client.ping()
            print(f"✅ Connected to Redis at {host}:{port}")
            print(f" Socket timeout: {self.redis_socket_timeout}s, Connection timeout: {self.redis_connection_timeout}s")
            print(f" Batch size: {self.redis_batch_size}, Max retries: {self.redis_max_retries}")
            # Load Lua script into Redis if available
            if self.lua_matcher_script:
                try:
                    self.lua_matcher_sha = self.redis_client.script_load(self.lua_matcher_script)
                    print(f"✅ Lua matcher script loaded (SHA: {self.lua_matcher_sha[:8]}...)")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to load Lua script into Redis: {e}")
                    self.lua_matcher_sha = None
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Redis: {e}")
            self.redis_client = None
            return False
    
    def _test_redis_connection(self):
        """Test Redis connection and reconnect if needed"""
        try:
            self.redis_client.ping()
            return True
        except:
            print("    🔧 Redis connection lost, attempting to reconnect...")
            # Try to reconnect with same settings
            return self.init_redis_connection()
    
    def _serialize_hash(self, hash_tuple):
        """
        Serialize hash tuple to string format for Redis key
        
        Parameters:
        -----------
        hash_tuple : tuple
            Hash tuple (f1, f2, dt)
            
        Returns:
        --------
        hash_key : str
            Serialized hash key as "f1:f2:dt"
        """
        f1, f2, dt = hash_tuple
        return f"{f1}:{f2}:{dt}"
    
    def _deserialize_hash(self, hash_string):
        """
        Deserialize hash string back to tuple
        
        Parameters:
        -----------
        hash_string : str
            Serialized hash key as "f1:f2:dt"
            
        Returns:
        --------
        hash_tuple : tuple
            Hash tuple (f1, f2, dt)
        """
        parts = hash_string.split(':')
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    
    def _serialize_track_offset(self, track_id, time_offset):
        """
        Serialize track_id and time_offset to string format for Redis value
        
        Parameters:
        -----------
        track_id : int
            Track identifier
        time_offset : int
            Time offset in frames
            
        Returns:
        --------
        value : str
            Serialized value as "track_id:time_offset"
        """
        return f"{track_id}:{time_offset}"
    
    def _deserialize_track_offset(self, value_string):
        """
        Deserialize track_offset string back to tuple
        
        Parameters:
        -----------
        value_string : str
            Serialized value as "track_id:time_offset"
            
        Returns:
        --------
        track_id : int
        time_offset : int
        """
        parts = value_string.split(':')
        return int(parts[0]), int(parts[1])
    
    def _load_audio(self, file_path):
        """
        Load audio file with robust error handling for different formats
        
        Parameters:
        -----------
        file_path : str
            Path to the audio file
            
        Returns:
        --------
        y : numpy.ndarray
            Audio time series
        sr : int
            Sample rate
        """
        try:
            # Use librosa which handles various audio formats
            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return y, sr
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None, None
    
    def _compute_spectrogram(self, y):
        """
        Compute the spectrogram of the audio signal
        
        Parameters:
        -----------
        y : numpy.ndarray
            Audio time series
            
        Returns:
        --------
        spectrogram : numpy.ndarray
            Magnitude spectrogram
        """
        if y is None:
            return None
            
        # Compute STFT
        stft_result = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Compute magnitude spectrogram
        spectrogram = np.abs(stft_result)
        
        return spectrogram
    
    def _find_peaks(self, spectrogram):
        """
        Find local peaks in the spectrogram using maximum filter
        
        Parameters:
        -----------
        spectrogram : numpy.ndarray
            Magnitude spectrogram
            
        Returns:
        --------
        peaks : list
            List of (freq_idx, time_idx) tuples representing peaks
        """
        if spectrogram is None:
            return []
            
        # Apply maximum filter to find local maxima
        max_filtered = maximum_filter(spectrogram, size=self.peak_neighborhood_size, mode='constant')
        
        # Find points equal to the max filter result (these are local peaks)
        local_maxima = (spectrogram == max_filtered)
        
        # Apply amplitude threshold based on median (similar to original code)
        magnitude_threshold = np.median(spectrogram) * self.amplitude_min
        peaks_mask = local_maxima & (spectrogram > magnitude_threshold)
        
        # Get coordinates of peaks
        freq_indices, time_indices = np.where(peaks_mask)
        
        # Package peaks as (time_index, freq_index) for consistency with original code
        peaks = list(zip(time_indices, freq_indices))
        
        # Sort peaks by time for more efficient target zone search
        peaks.sort()
        
        return peaks
    
    def _generate_hashes(self, peaks, track_id=None):
        """
        Generate hashes from peak points using the target zone approach
        
        Parameters:
        -----------
        peaks : list
            List of (time_idx, freq_idx) tuples
        track_id : any, optional
            ID of the track for database storage
            
        Returns:
        --------
        hashes : list
            List of (hash_value, time_offset) or (hash_value, (track_id, time_offset)) tuples
        """
        if not peaks:
            return []
            
        hashes = []
        
        # For each anchor point
        for i, anchor_peak in enumerate(peaks):
            anchor_time, anchor_freq = anchor_peak
            
            # Define target zone boundaries
            min_target_time = anchor_time + 1  # Target must be after anchor
            max_target_time = anchor_time + self.target_dt
            min_target_freq = max(0, anchor_freq - self.target_df)
            max_target_freq = anchor_freq + self.target_df
            
            # Count points paired with this anchor
            points_paired = 0
            
            # Iterate through subsequent peaks to find those in the target zone
            for j in range(i + 1, len(peaks)):
                target_peak = peaks[j]
                target_time, target_freq = target_peak
                
                # Check time constraint
                if target_time > max_target_time:
                    break  # No more valid targets for this anchor
                
                # Check frequency constraint (optional)
                if not (min_target_freq <= target_freq <= max_target_freq):
                    continue
                
                # Create the hash
                # Hash components: freq_anchor, freq_target, time_delta
                time_delta = target_time - anchor_time
                
                # Use a tuple as the hash key directly
                hash_key = (anchor_freq, target_freq, time_delta)
                
                # Store with track_id if provided
                if track_id is not None:
                    hashes.append((hash_key, (track_id, anchor_time)))
                else:
                    hashes.append((hash_key, anchor_time))  # For sample matching
                
                # Limit the number of points paired with each anchor
                points_paired += 1
                if points_paired >= self.fan_out:
                    break
        
        return hashes
    
    def add_track(self, audio_path, track_id=None, track_name=None):
        """
        Process an audio file and add its fingerprints to the database
        
        Parameters:
        -----------
        audio_path : str
            Path to the audio file
        track_id : any, optional
            ID for the track (will be generated if not provided)
        track_name : str, optional
            Name of the track (defaults to filename if not provided)
            
        Returns:
        --------
        track_id : any
            ID of the added track
        """
        # Generate track_id if not provided
        if track_id is None:
            track_id = len(self.track_metadata) + 1
            
        # Use filename as track_name if not provided
        if track_name is None:
            track_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        print(f"Processing {track_name}...")
        
        # 1. Load Audio
        y, sr = self._load_audio(audio_path)
        if y is None:
            print(f"  Failed to load {track_name}.")
            return None
        
        # 2. Compute Spectrogram
        spectrogram = self._compute_spectrogram(y)
        if spectrogram is None:
            print(f"  Failed to compute spectrogram for {track_name}.")
            return None
        
        # 3. Find Peaks
        peaks = self._find_peaks(spectrogram)
        if not peaks:
            print(f"  No peaks found for {track_name}.")
            return None
        
        # 4. Generate Hashes
        track_hashes = self._generate_hashes(peaks, track_id)
        if not track_hashes:
            print(f"  No hashes generated for {track_name}.")
            return None
        
        # 5. Add to Database Index
        hash_count = 0
        for hash_key, value in track_hashes:
            self.database_index[hash_key].append(value)
            hash_count += 1
        
        # Store track metadata
        self.track_metadata[track_id] = track_name
        print(f"  Added {track_name} (ID: {track_id}) with {len(peaks)} peaks and {hash_count} hashes.")
        
        return track_id
    
    def add_track_to_redis(self, audio_path, track_id=None, track_name=None):
        """
        Process an audio file and add its fingerprints to Redis database with robust batching
        
        Parameters:
        -----------
        audio_path : str
            Path to the audio file
        track_id : any, optional
            ID for the track (will be generated if not provided)
        track_name : str, optional
            Name of the track (defaults to filename if not provided)
            
        Returns:
        --------
        track_id : any
            ID of the added track
        """
        if self.redis_client is None:
            print("❌ Redis connection not initialized. Call init_redis_connection() first.")
            return None
        
        # Generate track_id if not provided
        if track_id is None:
            # Get next available track ID from Redis or start from 1
            try:
                existing_ids = self.redis_client.hkeys("track_metadata")
                if existing_ids:
                    track_id = max([int(id_str) for id_str in existing_ids]) + 1
                else:
                    track_id = 1
            except:
                track_id = 1
            
        # Use filename as track_name if not provided
        if track_name is None:
            track_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        print(f"Processing {track_name} for Redis...")
        
        # 1. Load Audio
        y, sr = self._load_audio(audio_path)
        if y is None:
            print(f"  Failed to load {track_name}.")
            return None
        
        # 2. Compute Spectrogram
        spectrogram = self._compute_spectrogram(y)
        if spectrogram is None:
            print(f"  Failed to compute spectrogram for {track_name}.")
            return None
        
        # 3. Find Peaks
        peaks = self._find_peaks(spectrogram)
        if not peaks:
            print(f"  No peaks found for {track_name}.")
            return None
        
        # 4. Generate Hashes
        track_hashes = self._generate_hashes(peaks, track_id)
        if not track_hashes:
            print(f"  No hashes generated for {track_name}.")
            return None
        
        # 5. Add to Redis using robust batching with non-transactional pipelines
        try:
            total_hashes = len(track_hashes)
            batch_size = self.redis_batch_size
            num_batches = (total_hashes + batch_size - 1) // batch_size
            
            print(f"  Adding {total_hashes} hashes in {num_batches} batches (batch size: {batch_size})...")
            
            # Process each batch independently with fresh, fast pipelines
            for i in range(num_batches):
                # 1. Get the current small batch
                start_index = i * batch_size
                end_index = min(start_index + batch_size, total_hashes)
                current_batch = track_hashes[start_index:end_index]
                
                # 2. Create a NEW, FAST pipeline for this batch
                # Using transaction=False for maximum speed during bulk loading
                # This sends commands without waiting for EXEC, much faster than transactional pipelines
                pipe = self.redis_client.pipeline(transaction=False)
                
                # 3. Populate the small pipeline with only the current batch
                for hash_key, (track_id_from_hash, time_offset) in current_batch:
                    redis_key = self._serialize_hash(hash_key)
                    redis_value = self._serialize_track_offset(track_id_from_hash, time_offset)
                    pipe.rpush(redis_key, redis_value)
                
                # 4. Execute just this one batch immediately
                pipe.execute()
                
                # 5. Show progress for this batch
                progress = (i + 1) / num_batches * 100
                print(f"    Batch {i + 1}/{num_batches} ({progress:.1f}%) completed - {len(current_batch)} hashes processed")
            
            # 6. Save metadata ONLY if all batches were successful
            self.redis_client.hset("track_metadata", track_id, track_name)
            print(f"  ✅ Successfully added {track_name} (ID: {track_id}) to Redis with {len(peaks)} peaks and {total_hashes} hashes.")
            
            # Also update local metadata for consistency
            self.track_metadata[track_id] = track_name
            
            return track_id
            
        except (redis.TimeoutError, redis.ConnectionError) as e:
            print(f"  ❌ A network error occurred while ingesting {track_name}: {e}")
            print(f"     The song may be partially indexed. It is recommended to re-ingest this track later.")
            return None
        except Exception as e:
            print(f"  ❌ An unexpected error occurred while adding {track_name} to Redis: {e}")
            return None
    
    def fingerprint_directory_to_redis(self, directory_path='music', recursive=True, show_progress=True):
        """
        Process all audio files in a directory and add fingerprints to Redis database
        
        Parameters:
        -----------
        directory_path : str
            Path to the directory containing audio files
        recursive : bool
            If True, scan subdirectories recursively (default: True)
        show_progress : bool
            If True, show detailed progress information (default: True)
        """
        if self.redis_client is None:
            print("❌ Redis connection not initialized. Call init_redis_connection() first.")
            return
        
        # Print current configuration
        if show_progress:
            self.print_redis_config()
            print()
        
        # Check if directory exists
        if not os.path.exists(directory_path):
            print(f"Directory does not exist: {directory_path}")
            return
        
        # Define supported audio extensions (expanded list)
        audio_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma', '.mp4', '.m4p')
        
        # Collect all audio files
        audio_files = []
        
        if recursive:
            # Use os.walk() for recursive directory traversal
            print(f"Scanning directory tree: {directory_path}")
            
            for root, directories, files in os.walk(directory_path):
                # Calculate relative path from the base directory
                relative_path = os.path.relpath(root, directory_path)
                if relative_path == '.':
                    relative_path = ''
                
                # Find audio files in current directory
                current_audio_files = [
                    f for f in files 
                    if f.lower().endswith(audio_extensions)
                ]
                
                # Add files with their full path and relative directory info
                for file_name in current_audio_files:
                    full_path = os.path.join(root, file_name)
                    audio_files.append({
                        'file_path': full_path,
                        'file_name': file_name,
                        'relative_dir': relative_path,
                        'directory': root
                    })
                
                if show_progress and current_audio_files:
                    dir_display = relative_path if relative_path else '(root)'
                    print(f"  Found {len(current_audio_files)} audio files in: {dir_display}")
        else:
            # Non-recursive: only scan the specified directory
            print(f"Scanning directory: {directory_path}")
            
            try:
                files = os.listdir(directory_path)
                current_audio_files = [
                    f for f in files 
                    if f.lower().endswith(audio_extensions)
                ]
                
                # Add files with their full path
                for file_name in current_audio_files:
                    full_path = os.path.join(directory_path, file_name)
                    audio_files.append({
                        'file_path': full_path,
                        'file_name': file_name,
                        'relative_dir': '',
                        'directory': directory_path
                    })
                    
            except PermissionError:
                print(f"Permission denied accessing directory: {directory_path}")
                return
        
        # Check if any audio files were found
        if not audio_files:
            print(f"No audio files found in {directory_path}")
            if recursive:
                print("  (searched recursively through all subdirectories)")
            return
        
        print(f"\nFound {len(audio_files)} audio files total. Starting Redis fingerprinting process...")
        
        # Get the next available track ID from Redis
        try:
            existing_ids = self.redis_client.hkeys("track_metadata")
            if existing_ids:
                next_track_id = max([int(id_str) for id_str in existing_ids]) + 1
            else:
                next_track_id = 1
        except:
            next_track_id = 1
        
        # Process each audio file
        successful_tracks = 0
        failed_tracks = 0
        start_time = time.time()
        
        for i, file_info in enumerate(audio_files):
            file_path = file_info['file_path']
            file_name = file_info['file_name']
            relative_dir = file_info['relative_dir']
            
            # Create a meaningful track name that includes directory structure
            base_name = os.path.splitext(file_name)[0]
            if relative_dir:
                # Include subdirectory in track name to avoid conflicts
                track_name = f"{relative_dir}/{base_name}".replace('\\', '/')
            else:
                track_name = base_name
            
            # Show progress
            if show_progress:
                progress_percent = (i + 1) / len(audio_files) * 100
                elapsed_time = time.time() - start_time
                avg_time_per_track = elapsed_time / (i + 1) if i > 0 else 0
                estimated_remaining = avg_time_per_track * (len(audio_files) - i - 1)
                
                print(f"\n[{i+1:3d}/{len(audio_files)}] ({progress_percent:5.1f}%) Processing: {track_name}")
                if i > 0:
                    print(f"    ⏱️  Avg time/track: {avg_time_per_track:.1f}s, Est. remaining: {estimated_remaining/60:.1f}min")
            
            # Process the track
            try:
                result_track_id = self.add_track_to_redis(
                    audio_path=file_path,
                    track_id=next_track_id,
                    track_name=track_name
                )
                
                if result_track_id is not None:
                    successful_tracks += 1
                    next_track_id += 1
                else:
                    failed_tracks += 1
                    if show_progress:
                        print(f"    ❌ Failed to process: {track_name}")
                    
            except Exception as e:
                failed_tracks += 1
                if show_progress:
                    print(f"    ❌ Error processing {track_name}: {str(e)}")
        
        # Print final summary
        total_time = time.time() - start_time
        print(f"\n" + "="*60)
        print(f"REDIS FINGERPRINTING COMPLETE")
        print(f"="*60)
        print(f"Total files found: {len(audio_files)}")
        print(f"Successfully processed: {successful_tracks}")
        print(f"Failed to process: {failed_tracks}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average time per track: {total_time/len(audio_files):.1f} seconds")
        
        # Get total tracks in Redis
        try:
            total_tracks = self.redis_client.hlen("track_metadata")
            print(f"Redis database now contains: {total_tracks} tracks")
        except:
            print("Could not get track count from Redis")
        
        # Show directory structure summary if recursive
        if recursive and show_progress:
            directory_summary = {}
            for file_info in audio_files:
                rel_dir = file_info['relative_dir'] if file_info['relative_dir'] else '(root)'
                directory_summary[rel_dir] = directory_summary.get(rel_dir, 0) + 1
            
            print(f"\nDirectory breakdown:")
            for dir_name, count in sorted(directory_summary.items()):
                print(f"  {dir_name}: {count} files")
    
    def match_sample_from_redis(self, sample_path, verbose=True):
        """
        Match a sample audio file against the Redis database using server-side Lua script execution
        
        This is the ultimate high-performance version that executes the entire matching process
        on the Redis server using a Lua script, eliminating all network latency bottlenecks.
        
        Parameters:
        -----------
        sample_path : str
            Path to the audio sample
        verbose : bool
            Whether to print detailed information
            
        Returns:
        --------
        result : dict
            Dictionary containing match information
        """
        if self.redis_client is None:
            print("❌ Redis connection not initialized. Call init_redis_connection() first.")
            return {"match_found": False, "error": "Redis not connected"}
        
        if self.lua_matcher_sha is None:
            print("❌ Lua matcher script not available. Ensure matcher.lua is loaded.")
            return {"match_found": False, "error": "Lua script not loaded"}
        
        start_time = time.time()
        
        if verbose:
            print(f"🎵 Matching sample: {os.path.basename(sample_path)} against Redis database...")
            print(f"🚀 Using server-side Lua script execution (ultimate performance mode)")
        
        # Phase 1: Local Processing - Generate Sample Hashes
        # 1. Load Sample
        y, sr = self._load_audio(sample_path)
        if y is None:
            print("  Failed to load sample.")
            return {"match_found": False, "error": "Failed to load sample"}
        
        # 2. Compute Spectrogram
        sample_spectrogram = self._compute_spectrogram(y)
        if sample_spectrogram is None:
            print("  Failed to compute spectrogram for sample.")
            return {"match_found": False, "error": "Failed to compute spectrogram"}
        
        # 3. Find Peaks
        sample_peaks = self._find_peaks(sample_spectrogram)
        if not sample_peaks:
            print("  No peaks found in sample.")
            return {"match_found": False, "error": "No peaks found"}
        
        # 4. Generate Hashes
        sample_hashes = self._generate_hashes(sample_peaks)
        if not sample_hashes:
            print("  No hashes generated for sample.")
            return {"match_found": False, "error": "No hashes generated"}
            
        if verbose:
            print(f"  📊 Sample generated {len(sample_peaks)} peaks and {len(sample_hashes)} hashes.")
        
        # Phase 2: Prepare Arguments for Lua Script
        # 5. Generate unique histogram key
        histogram_key = f"match_scores:{uuid.uuid4()}"
        
        # 6. Flatten sample_hashes into argument list for Lua script
        args = []
        for sample_hash, sample_anchor_time in sample_hashes:
            redis_key = self._serialize_hash(sample_hash)
            args.append(redis_key)
            args.append(str(sample_anchor_time))
        
        if verbose:
            print(f"  🔧 Prepared {len(args)//2} hash arguments for Lua script...")
        
        # Phase 3: Execute Server-Side Lua Script
        try:
            if verbose:
                print(f"  🚀 Executing server-side Lua script...")
            
            # Execute the Lua script on Redis server
            matches_found = self.redis_client.evalsha(self.lua_matcher_sha, 1, histogram_key, *args)
            
            if verbose:
                print(f"  ⚡ Server-side processing complete: {matches_found} potential matches found")
            
        except Exception as e:
            print(f"  ❌ Error executing Lua script: {e}")
            return {"match_found": False, "error": f"Lua script execution error: {e}"}
        
        # Phase 4: Fetch Final Scores and Find Winner
        try:
            # 7. Retrieve the pre-computed histogram from Redis
            final_scores = self.redis_client.hgetall(histogram_key)
            
            # 8. Clean up the temporary key
            self.redis_client.delete(histogram_key)
            
            if not final_scores:
                if verbose:
                    print(f"  📊 No aggregated scores found")
                return {"match_found": False, "error": "No aggregated scores found"}
            
            if verbose:
                print(f"  📈 Processing {len(final_scores)} aggregated alignment scores...")
            
            # 9. Find the best match from the aggregated scores
            best_match_track_id = None
            best_match_score = 0
            best_match_offset_frames = 0
            
            for field_key, count_str in final_scores.items():
                try:
                    count = int(count_str)  # Redis returns strings
                    
                    # Check if this is better than our current best and meets minimum requirements
                    if count > best_match_score and count >= self.min_histogram_peak:
                        # Parse the field_key (e.g., "101:2101") to extract track_id and offset
                        track_id_str, offset_str = field_key.split(':', 1)
                        track_id = int(track_id_str)
                        offset_frames = int(offset_str)
                        
                        # Update our best match
                        best_match_score = count
                        best_match_track_id = track_id
                        best_match_offset_frames = offset_frames
                        
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Could not parse final score '{field_key}:{count_str}': {e}")
            
        except Exception as e:
            if verbose:
                print(f"  ❌ Error processing final scores: {e}")
            return {"match_found": False, "error": f"Error processing final scores: {e}"}
        
        # Get track metadata from Redis
        try:
            all_track_metadata = self.redis_client.hgetall("track_metadata")
            # Convert string keys to integers
            self.track_metadata = {int(k): v for k, v in all_track_metadata.items()}
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load track metadata from Redis: {e}")
        
        # Phase 5: Return Result
        process_time = time.time() - start_time
        
        if best_match_track_id is not None:
            track_name = self.track_metadata.get(best_match_track_id, "Unknown Track")
            
            # Convert offset from frames to seconds
            offset_seconds = best_match_offset_frames * self.hop_length / self.sample_rate
            
            if verbose:
                print(f"\n🎉 --- Match Found! ---")
                print(f"  🎵 Track: {track_name} (ID: {best_match_track_id})")
                print(f"  📊 Score: {best_match_score} (matching hashes with consistent offset)")
                # Format time as minutes:seconds
                min_offset = int(abs(offset_seconds) // 60)
                sec_offset = int(abs(offset_seconds) % 60)
                time_str = f"{min_offset}:{sec_offset:02d}"
                print(f"  ⏰ Estimated Offset: {offset_seconds:.2f} seconds ({time_str}) into the track.")
                print(f"  🚀 Processing time: {process_time:.3f} seconds (Lua script optimized)")
            
            # Format offset as mm:ss
            min_offset = int(abs(offset_seconds) // 60)
            sec_offset = int(abs(offset_seconds) % 60)
            time_str = f"{min_offset}:{sec_offset:02d}"
            
            return {
                "match_found": True,
                "track_id": best_match_track_id,
                "track_name": track_name,
                "score": best_match_score,
                "offset_frames": best_match_offset_frames,
                "offset_seconds": offset_seconds,
                "offset_formatted": time_str,
                "processing_time": process_time
            }
        else:
            if verbose:
                print(f"\n❌ --- No confident match found. ---")
                print(f"  🚀 Processing time: {process_time:.3f} seconds (Lua script optimized)")
                print(f"  💡 Best score was {best_match_score}, but minimum required is {self.min_histogram_peak}")
                
            return {
                "match_found": False,
                "processing_time": process_time,
                "error": "No confident match"
            }
    
    def save_metadata(self, filename='database/track_metadata.pkl'):
        """
        Save track metadata to a pickle file (for backup or Redis fallback)
        
        Parameters:
        -----------
        filename : str
            Path to save the metadata file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.track_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Track metadata saved to {filename}")
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def load_metadata(self, filename='database/track_metadata.pkl'):
        """
        Load track metadata from a pickle file
        
        Parameters:
        -----------
        filename : str
            Path to the metadata file
            
        Returns:
        --------
        success : bool
            Whether the metadata was loaded successfully
        """
        try:
            with open(filename, 'rb') as f:
                self.track_metadata = pickle.load(f)
            
            print(f"Loaded metadata for {len(self.track_metadata)} tracks.")
            return True
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return False
    
    # Keep all the existing methods unchanged...
    def fingerprint_directory(self, directory_path='music', recursive=True, show_progress=True):
        """
        Process all audio files in a directory and its subdirectories, adding fingerprints to the database
        
        Parameters:
        -----------
        directory_path : str
            Path to the directory containing audio files
        recursive : bool
            If True, scan subdirectories recursively (default: True)
        show_progress : bool
            If True, show detailed progress information (default: True)
        """
        # Check if directory exists
        if not os.path.exists(directory_path):
            print(f"Directory does not exist: {directory_path}")
            return
        
        # Define supported audio extensions (expanded list)
        audio_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma', '.mp4', '.m4p')
        
        # Collect all audio files
        audio_files = []
        
        if recursive:
            # Use os.walk() for recursive directory traversal
            print(f"Scanning directory tree: {directory_path}")
            
            for root, directories, files in os.walk(directory_path):
                # Calculate relative path from the base directory
                relative_path = os.path.relpath(root, directory_path)
                if relative_path == '.':
                    relative_path = ''
                
                # Find audio files in current directory
                current_audio_files = [
                    f for f in files 
                    if f.lower().endswith(audio_extensions)
                ]
                
                # Add files with their full path and relative directory info
                for file_name in current_audio_files:
                    full_path = os.path.join(root, file_name)
                    audio_files.append({
                        'file_path': full_path,
                        'file_name': file_name,
                        'relative_dir': relative_path,
                        'directory': root
                    })
                
                if show_progress and current_audio_files:
                    dir_display = relative_path if relative_path else '(root)'
                    print(f"  Found {len(current_audio_files)} audio files in: {dir_display}")
        else:
            # Non-recursive: only scan the specified directory
            print(f"Scanning directory: {directory_path}")
            
            try:
                files = os.listdir(directory_path)
                current_audio_files = [
                    f for f in files 
                    if f.lower().endswith(audio_extensions)
                ]
                
                # Add files with their full path
                for file_name in current_audio_files:
                    full_path = os.path.join(directory_path, file_name)
                    audio_files.append({
                        'file_path': full_path,
                        'file_name': file_name,
                        'relative_dir': '',
                        'directory': directory_path
                    })
                    
            except PermissionError:
                print(f"Permission denied accessing directory: {directory_path}")
                return
        
        # Check if any audio files were found
        if not audio_files:
            print(f"No audio files found in {directory_path}")
            if recursive:
                print("  (searched recursively through all subdirectories)")
            return
        
        print(f"\nFound {len(audio_files)} audio files total. Starting fingerprinting process...")
        
        # Get the next available track ID
        next_track_id = max(self.track_metadata.keys()) + 1 if self.track_metadata else 1
        
        # Process each audio file
        successful_tracks = 0
        failed_tracks = 0
        
        for i, file_info in enumerate(audio_files):
            file_path = file_info['file_path']
            file_name = file_info['file_name']
            relative_dir = file_info['relative_dir']
            
            # Create a meaningful track name that includes directory structure
            base_name = os.path.splitext(file_name)[0]
            if relative_dir:
                # Include subdirectory in track name to avoid conflicts
                track_name = f"{relative_dir}/{base_name}".replace('\\', '/')
            else:
                track_name = base_name
            
            # Show progress
            if show_progress:
                progress_percent = (i + 1) / len(audio_files) * 100
                print(f"\n[{i+1:3d}/{len(audio_files)}] ({progress_percent:5.1f}%) Processing: {track_name}")
            
            # Process the track
            try:
                result_track_id = self.add_track(
                    audio_path=file_path,
                    track_id=next_track_id,
                    track_name=track_name
                )
                
                if result_track_id is not None:
                    successful_tracks += 1
                    next_track_id += 1
                else:
                    failed_tracks += 1
                    if show_progress:
                        print(f"    ❌ Failed to process: {track_name}")
                    
            except Exception as e:
                failed_tracks += 1
                if show_progress:
                    print(f"    ❌ Error processing {track_name}: {str(e)}")
        
        # Print final summary
        print(f"\n" + "="*60)
        print(f"FINGERPRINTING COMPLETE")
        print(f"="*60)
        print(f"Total files found: {len(audio_files)}")
        print(f"Successfully processed: {successful_tracks}")
        print(f"Failed to process: {failed_tracks}")
        print(f"Database now contains: {len(self.track_metadata)} tracks")
        
        # Show directory structure summary if recursive
        if recursive and show_progress:
            directory_summary = {}
            for file_info in audio_files:
                rel_dir = file_info['relative_dir'] if file_info['relative_dir'] else '(root)'
                directory_summary[rel_dir] = directory_summary.get(rel_dir, 0) + 1
            
            print(f"\nDirectory breakdown:")
            for dir_name, count in sorted(directory_summary.items()):
                print(f"  {dir_name}: {count} files")

    def get_directory_stats(self, directory_path='music'):
        """
        Get statistics about audio files in a directory structure without processing them
        
        Parameters:
        -----------
        directory_path : str
            Path to the directory to analyze
            
        Returns:
        --------
        stats : dict
            Dictionary containing directory statistics
        """
        if not os.path.exists(directory_path):
            print(f"Directory does not exist: {directory_path}")
            return None
        
        audio_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma', '.mp4', '.m4p')
        
        stats = {
            'total_files': 0,
            'directories': {},
            'file_types': {},
            'total_size_mb': 0
        }
        
        print(f"Analyzing directory structure: {directory_path}")
        
        for root, directories, files in os.walk(directory_path):
            relative_path = os.path.relpath(root, directory_path)
            if relative_path == '.':
                relative_path = '(root)'
            
            # Find audio files in current directory
            audio_files = [f for f in files if f.lower().endswith(audio_extensions)]
            
            if audio_files:
                stats['directories'][relative_path] = len(audio_files)
                stats['total_files'] += len(audio_files)
                
                # Count file types
                for file_name in audio_files:
                    ext = os.path.splitext(file_name)[1].lower()
                    stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
                    
                    # Calculate file size
                    try:
                        file_path = os.path.join(root, file_name)
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                        stats['total_size_mb'] += size_mb
                    except OSError:
                        pass  # Skip files we can't access
        
        # Print statistics
        print(f"\n📊 Directory Statistics:")
        print(f"   Total audio files: {stats['total_files']}")
        print(f"   Total size: {stats['total_size_mb']:.1f} MB")
        print(f"   Directories with audio: {len(stats['directories'])}")
        
        print(f"\n📁 Files by directory:")
        for dir_name, count in sorted(stats['directories'].items()):
            print(f"   {dir_name}: {count} files")
        
        print(f"\n🎵 Files by type:")
        for file_type, count in sorted(stats['file_types'].items()):
            print(f"   {file_type}: {count} files")
        
        return stats
    
    def match_sample(self, sample_path, verbose=True, plot=False):
        """
        Match a sample audio file against the database
        
        Parameters:
        -----------
        sample_path : str
            Path to the audio sample
        verbose : bool
            Whether to print detailed information
        plot : bool
            Whether to plot the match histogram
            
        Returns:
        --------
        result : dict
            Dictionary containing match information
        """
        start_time = time.time()
        
        if verbose:
            print(f"Matching sample: {os.path.basename(sample_path)}...")
        
        # 1. Load Sample
        y, sr = self._load_audio(sample_path)
        if y is None:
            print("  Failed to load sample.")
            return {"match_found": False, "error": "Failed to load sample"}
        
        # 2. Compute Spectrogram
        sample_spectrogram = self._compute_spectrogram(y)
        if sample_spectrogram is None:
            print("  Failed to compute spectrogram for sample.")
            return {"match_found": False, "error": "Failed to compute spectrogram"}
        
        # 3. Find Peaks
        sample_peaks = self._find_peaks(sample_spectrogram)
        if not sample_peaks:
            print("  No peaks found in sample.")
            return {"match_found": False, "error": "No peaks found"}
        
        # 4. Generate Hashes
        sample_hashes = self._generate_hashes(sample_peaks)
        if not sample_hashes:
            print("  No hashes generated for sample.")
            return {"match_found": False, "error": "No hashes generated"}
            
        if verbose:
            print(f"  Sample generated {len(sample_peaks)} peaks and {len(sample_hashes)} hashes.")
        
        # 5. Find Matches in Database
        potential_matches = defaultdict(list)  # {track_id: [list_of_time_diffs], ...}
        match_count = 0
        
        for sample_hash, sample_anchor_time in sample_hashes:
            if sample_hash in self.database_index:
                db_entries = self.database_index[sample_hash]
                for track_id, db_anchor_time in db_entries:
                    # Calculate time difference (offset) in frames
                    delta_t = db_anchor_time - sample_anchor_time
                    potential_matches[track_id].append(delta_t)
                    match_count += 1
        
        if verbose:
            print(f"  Found {match_count} potential hash matches across {len(potential_matches)} tracks.")
            
        if not potential_matches:
            print("  No potential track matches found.")
            return {"match_found": False, "error": "No matches found in database"}
        
        # 6. Score Matches using Histogramming
        best_match_track_id = None
        best_match_score = 0
        best_match_offset_frames = 0
        best_match_histogram = None
        
        for track_id, time_diffs in potential_matches.items():
            # Skip if too few matches
            if len(time_diffs) < self.min_matches_for_hit:
                continue
            
            # Create a histogram of time differences
            # Use a bin width that's reasonable for frame differences
            hist, bin_edges = np.histogram(time_diffs, bins=100)
            
            # Find the peak (most common time difference)
            max_bin = np.argmax(hist)
            max_count = hist[max_bin]
            bin_center = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
            
            # Check if the peak is significant enough
            if max_count > best_match_score and max_count >= self.min_histogram_peak:
                best_match_score = max_count
                best_match_track_id = track_id
                best_match_offset_frames = bin_center
                best_match_histogram = (hist, bin_edges)
                
            if verbose:
                track_name = self.track_metadata.get(track_id, f"Unknown (ID: {track_id})")
                print(f"    Track: {track_name}, Matches: {len(time_diffs)}, Best bin: {max_count}")
        
        # Plot the best histogram if requested
        if plot and best_match_histogram is not None:
            hist, bin_edges = best_match_histogram
            plt.figure(figsize=(10, 4))
            plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge')
            plt.title(f"Time Offset Histogram - {self.track_metadata.get(best_match_track_id, 'Unknown')}")
            plt.xlabel("Time offset (frames)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()
        
        # 7. Return Result
        process_time = time.time() - start_time
        
        if best_match_track_id is not None:
            track_name = self.track_metadata.get(best_match_track_id, "Unknown Track")
            
            # Convert offset from frames to seconds
            offset_seconds = best_match_offset_frames * self.hop_length / self.sample_rate
            
            if verbose:
                print(f"\n--- Match Found! ---")
                print(f"  Track: {track_name} (ID: {best_match_track_id})")
                print(f"  Score: {best_match_score} (matching hashes with consistent offset)")
                # Format time as minutes:seconds
                min_offset = int(abs(offset_seconds) // 60)
                sec_offset = int(abs(offset_seconds) % 60)
                time_str = f"{min_offset}:{sec_offset:02d}"
                print(f"  Estimated Offset: {offset_seconds:.2f} seconds ({time_str}) into the track.")
                print(f"  Processing time: {process_time:.3f} seconds")
            
            # Format offset as mm:ss
            min_offset = int(abs(offset_seconds) // 60)
            sec_offset = int(abs(offset_seconds) % 60)
            time_str = f"{min_offset}:{sec_offset:02d}"
            
            return {
                "match_found": True,
                "track_id": best_match_track_id,
                "track_name": track_name,
                "score": best_match_score,
                "offset_frames": best_match_offset_frames,
                "offset_seconds": offset_seconds,
                "offset_formatted": time_str,
                "processing_time": process_time
            }
        else:
            if verbose:
                print("\n--- No confident match found. ---")
                print(f"  Processing time: {process_time:.3f} seconds")
                
            return {
                "match_found": False,
                "processing_time": process_time,
                "error": "No confident match"
            }
    
    def save_database(self, filename='database/fingerprint_db.pkl'):
        """
        Save the fingerprint database to a file
        
        Parameters:
        -----------
        filename : str
            Path to save the database file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        database = {
            'database_index': dict(self.database_index),
            'track_metadata': self.track_metadata
        }
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Database saved to {filename}")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def load_database(self, filename='database/fingerprint_db.pkl'):
        """
        Load the fingerprint database from a file
        
        Parameters:
        -----------
        filename : str
            Path to the database file
            
        Returns:
        --------
        success : bool
            Whether the database was loaded successfully
        """
        try:
            with open(filename, 'rb') as f:
                database = pickle.load(f)
            
            self.database_index = defaultdict(list, database['database_index'])
            self.track_metadata = database['track_metadata']
            
            print(f"Loaded database with {len(self.track_metadata)} tracks and {sum(len(v) for v in self.database_index.values())} hashes.")
            return True
        except Exception as e:
            print(f"Error loading database: {e}")
            return False
    
    def generate_sample(self, file_path, start_time, duration, output_path=None, add_noise=False, noise_level=0.05):
        """
        Generate a sample from an audio file
        
        Parameters:
        -----------
        file_path : str
            Path to the audio file
        start_time : float
            Start time in seconds
        duration : float
            Duration of the sample in seconds
        output_path : str, optional
            Path to save the sample file
        add_noise : bool
            Whether to add noise to the sample
        noise_level : float
            Level of noise to add (0.0 - 1.0)
            
        Returns:
        --------
        sample_path : str
            Path to the generated sample file
        """
        # Load audio
        y, sr = self._load_audio(file_path)
        if y is None:
            print(f"Failed to load audio file: {file_path}")
            return None
        
        # Calculate start and end samples
        start_sample = int(start_time * sr)
        end_sample = min(int((start_time + duration) * sr), len(y))
        
        if start_sample >= len(y) or end_sample <= start_sample:
            print(f"Invalid time range for sample: {start_time}-{start_time + duration} seconds")
            return None
        
        # Extract sample
        sample = y[start_sample:end_sample]
        
        # Add noise if requested
        if add_noise and noise_level > 0:
            noise = np.random.normal(0, noise_level * np.std(sample), len(sample))
            sample = sample + noise
            # Ensure the result is within [-1, 1]
            sample = np.clip(sample, -1.0, 1.0)
        
        # Set default output path if not provided
        if output_path is None:
            # Create samples directory if it doesn't exist
            if not os.path.exists('samples'):
                os.makedirs('samples')
                
            # Generate a filename
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = f"samples/{base_name}_sample_{start_time:.0f}s_{duration:.0f}s.wav"
        
        # Save the sample
        try:
            librosa.output.write_wav(output_path, sample, sr)
            print(f"Sample saved to {output_path}")
        except Exception as e:
            print(f"Error saving sample: {e}")
            # Try alternative method
            try:
                from scipy.io import wavfile
                wavfile.write(output_path, sr, (sample * 32767).astype(np.int16))
                print(f"Sample saved to {output_path} (alternative method)")
            except Exception as e2:
                print(f"Error saving sample (alternative method): {e2}")
                return None
        
        return output_path
    
    def visualize_fingerprints(self, audio_path, duration=None):
        """
        Visualize the fingerprinting process for an audio file
        
        Parameters:
        -----------
        audio_path : str
            Path to the audio file
        duration : float, optional
            Duration in seconds to visualize (from the beginning)
        """
        # Load audio
        y, sr = self._load_audio(audio_path)
        if y is None:
            print(f"Failed to load audio file: {audio_path}")
            return
        
        # Limit duration if specified
        if duration is not None and duration > 0:
            y = y[:int(duration * sr)]
        
        # Compute spectrogram
        spectrogram = self._compute_spectrogram(y)
        if spectrogram is None:
            print("Failed to compute spectrogram")
            return
        
        # Find peaks
        peaks = self._find_peaks(spectrogram)
        if not peaks:
            print("No peaks found")
            return
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        
        # Plot the spectrogram
        plt.subplot(2, 1, 1)
        librosa.display.specshow(
            librosa.amplitude_to_db(spectrogram, ref=np.max),
            y_axis='log', x_axis='time', sr=sr, hop_length=self.hop_length
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        
        # Plot the peaks
        plt.subplot(2, 1, 2)
        librosa.display.specshow(
            librosa.amplitude_to_db(spectrogram, ref=np.max),
            y_axis='log', x_axis='time', sr=sr, hop_length=self.hop_length
        )
        
        # Convert peak coordinates to time and frequency
        times = [p[0] * self.hop_length / sr for p in peaks]
        freqs = [sr * p[1] / self.n_fft for p in peaks]
        
        plt.scatter(times, freqs, s=20, c='r', alpha=0.8)
        
        # Plot some example target zones
        num_anchors = min(5, len(peaks))
        for i in range(0, len(peaks), len(peaks) // num_anchors):
            anchor = peaks[i]
            anchor_time = anchor[0] * self.hop_length / sr
            anchor_freq = sr * anchor[1] / self.n_fft
            
            # Target zone boundaries
            max_target_time = (anchor[0] + self.target_dt) * self.hop_length / sr
            min_target_freq = sr * max(0, anchor[1] - self.target_df) / self.n_fft
            max_target_freq = sr * (anchor[1] + self.target_df) / self.n_fft
            
            # Plot target zone
            plt.plot(
                [anchor_time, max_target_time, max_target_time, anchor_time, anchor_time],
                [min_target_freq, min_target_freq, max_target_freq, max_target_freq, min_target_freq],
                'g-', alpha=0.4
            )
        
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constellation Map with Target Zones')
        
        plt.tight_layout()
        plt.show()