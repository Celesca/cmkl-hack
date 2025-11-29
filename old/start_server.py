#!/usr/bin/env python3
"""
Startup script for DynamicGroundingDINO API
Supports both development and production modes with automatic worker scaling
"""

import os
import sys
import multiprocessing
import subprocess
import argparse
from pathlib import Path

def get_optimal_workers():
    """Calculate optimal number of workers based on system resources"""
    cpu_count = multiprocessing.cpu_count()
    
    # For AI workloads, we typically use fewer workers than CPU cores
    # because each worker will be CPU and memory intensive
    if cpu_count <= 2:
        return 1
    elif cpu_count <= 4:
        return 2
    elif cpu_count <= 8:
        return min(4, cpu_count - 1)
    else:
        return min(6, cpu_count // 2)

def run_development():
    """Run in development mode with auto-reload"""
    print("🚀 Starting DynamicGroundingDINO API in DEVELOPMENT mode...")
    print("📝 Features: Auto-reload, Debug logging, Single worker")
    
    cmd = [
        "uvicorn",
        "server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--log-level", "debug",
        "--access-log"
    ]
    
    subprocess.run(cmd)

def run_production(workers=None, max_requests=None):
    """Run in production mode with Gunicorn"""
    workers = workers or get_optimal_workers()
    max_requests = max_requests or 1000
    
    print(f"🚀 Starting DynamicGroundingDINO API in PRODUCTION mode...")
    print(f"👥 Workers: {workers}")
    print(f"🔄 Max requests per worker: {max_requests}")
    print(f"⚡ Worker class: uvicorn.workers.UvicornWorker")
    
    # Set environment variables for Gunicorn config
    os.environ["WORKERS"] = str(workers)
    os.environ["MAX_REQUESTS"] = str(max_requests)
    
    cmd = [
        "gunicorn",
        "server:app",
        "-c", "gunicorn.conf.py"
    ]
    
    subprocess.run(cmd)

def run_with_custom_config(config_file):
    """Run with custom Gunicorn configuration"""
    print(f"🚀 Starting with custom config: {config_file}")
    
    cmd = [
        "gunicorn",
        "server:app",
        "-c", config_file
    ]
    
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="DynamicGroundingDINO API Server")
    parser.add_argument(
        "--mode", 
        choices=["dev", "prod", "custom"], 
        default="dev",
        help="Server mode (default: dev)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        help="Number of worker processes (production mode only)"
    )
    parser.add_argument(
        "--max-requests", 
        type=int, 
        default=1000,
        help="Maximum requests per worker before restart (production mode only)"
    )
    parser.add_argument(
        "--config", 
        help="Custom Gunicorn configuration file (custom mode only)"
    )
    
    args = parser.parse_args()
    
    # Check if required files exist
    if not Path("server.py").exists():
        print("❌ Error: server.py not found!")
        sys.exit(1)
    
    if args.mode == "prod" and not Path("gunicorn.conf.py").exists():
        print("❌ Error: gunicorn.conf.py not found!")
        sys.exit(1)
    
    # Print environment info
    print("=" * 60)
    print("🔍 DynamicGroundingDINO API Server")
    print("=" * 60)
    print(f"🖥️  CPU cores: {multiprocessing.cpu_count()}")
    print(f"🎯 Recommended workers: {get_optimal_workers()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📁 Working directory: {os.getcwd()}")
    print("=" * 60)
    
    if args.mode == "dev":
        run_development()
    elif args.mode == "prod":
        run_production(args.workers, args.max_requests)
    elif args.mode == "custom":
        if not args.config:
            print("❌ Error: --config required for custom mode!")
            sys.exit(1)
        run_with_custom_config(args.config)

if __name__ == "__main__":
    main()
