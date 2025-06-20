#!/usr/bin/env python3
"""
Statistics Updater Runner for Audio Fingerprinting Application

This script runs the stats_updater.main() function in a continuous loop
every 5 minutes. It's designed to be the primary process in the worker container.

Features:
- Robust error handling to prevent crashes
- Graceful shutdown on SIGTERM/SIGINT
- Detailed logging for monitoring
- 5-minute intervals between runs
"""

import logging
import time
import signal
import sys
from datetime import datetime

# Import the main function from our stats updater module
from src.stats_updater import main as run_stats_aggregation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stats_runner')

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    signal_name = signal.Signals(signum).name
    logger.info(f"📶 Received {signal_name} signal. Initiating graceful shutdown...")
    shutdown_requested = True

def main():
    """Main loop that runs stats aggregation every 5 minutes"""
    global shutdown_requested
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("🚀 Statistics updater worker starting...")
    logger.info("⏰ Will run stats aggregation every 5 minutes")
    logger.info("🛑 Send SIGTERM or SIGINT to stop gracefully")
    
    iteration_count = 0
    
    while not shutdown_requested:
        iteration_count += 1
        start_time = datetime.now()
        
        try:
            logger.info(f"📊 Starting stats aggregation run #{iteration_count}")
            
            # Run the actual stats aggregation
            run_stats_aggregation()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"✅ Stats aggregation completed successfully in {duration:.2f} seconds")
            
        except KeyboardInterrupt:
            logger.info("⌨️  Keyboard interrupt received. Shutting down...")
            break
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"❌ Stats aggregation failed after {duration:.2f} seconds: {e}", exc_info=True)
            logger.info("🔄 Will retry in 5 minutes...")
        
        # Wait 5 minutes before next run (unless shutdown requested)
        if not shutdown_requested:
            logger.info("😴 Sleeping for 5 minutes until next run...")
            
            # Sleep in small intervals to allow for responsive shutdown
            for _ in range(300):  # 300 seconds = 5 minutes
                if shutdown_requested:
                    break
                time.sleep(1)
    
    logger.info("👋 Statistics updater worker stopped gracefully")
    logger.info(f"📈 Total aggregation runs completed: {iteration_count}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"💥 Fatal error in stats runner: {e}", exc_info=True)
        sys.exit(1)