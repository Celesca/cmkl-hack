"""
Direct Video URL Downloader Utility
Handles direct video URL downloading (no YouTube support)
"""

import tempfile
import os
from typing import Tuple
import logging
import requests

logger = logging.getLogger(__name__)

def download_video_from_url(url: str) -> Tuple[str, dict]:
    """
    Download video from direct URL with comprehensive error handling.
    
    Args:
        url: Direct video URL (no YouTube URLs supported)
        
    Returns:
        Tuple of (temp_file_path, video_info)
        
    Raises:
        Exception: If download fails or URL is not supported
    """
    logger.info(f"📥 Processing video URL: {url}")
    
    # Check for YouTube URLs and reject them
    youtube_patterns = ['youtube.com', 'youtu.be', 'm.youtube.com']
    if any(pattern in url.lower() for pattern in youtube_patterns):
        raise Exception(
            "YouTube URLs are not supported due to access restrictions.\n\n"
            "� Please use one of these alternatives:\n"
            "1. Use a direct video file URL (e.g., .mp4, .avi, .mov files)\n"
            "2. Upload your video file directly using the /video_action/detect/upload endpoint\n"
            "3. Host your video file on a server and provide the direct URL"
        )
    
    try:
        logger.info(f"� Downloading video from direct URL: {url}")
        
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'video/*,*/*;q=0.9',
            'Accept-Encoding': 'identity',
            'Connection': 'keep-alive'
        }
        
        response = requests.get(url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()
        
        # Validate content type
        content_type = response.headers.get('content-type', '').lower()
        content_length = response.headers.get('content-length')
        
        logger.info(f"� Content-Type: {content_type}")
        logger.info(f"� Content-Length: {content_length}")
        
        # Validate that this looks like a video file
        video_types = ['video/', 'application/octet-stream']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
        
        is_video_content = any(video_type in content_type for video_type in video_types)
        is_video_url = any(ext in url.lower() for ext in video_extensions)
        
        if not is_video_content and not is_video_url:
            logger.warning(f"⚠️ URL may not be a video file. Content-Type: {content_type}")
            logger.warning("This might cause issues during video processing.")
        
        # Determine file extension from URL or content type
        file_extension = '.mp4'  # default
        for ext in video_extensions:
            if ext in url.lower():
                file_extension = ext
                break
        
        # Create temporary file with appropriate extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_path = temp_file.name
            
            # Download with progress tracking
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress for large files (every MB)
                    if content_length and downloaded % (1024 * 1024) == 0:
                        progress = (downloaded / int(content_length)) * 100
                        logger.info(f"📥 Downloaded: {downloaded} bytes ({progress:.1f}%)")
        
        # Verify file was created and has content
        if not os.path.exists(temp_path):
            raise Exception("Download completed but file was not created")
            
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            raise Exception("Download completed but file is empty")
        
        # Check for minimum reasonable file size (1KB)
        if file_size < 1024:
            logger.warning(f"⚠️ Downloaded file is very small ({file_size} bytes). This might not be a valid video file.")
        
        logger.info(f"✅ Video downloaded successfully: {temp_path} ({file_size} bytes)")
        
        video_info = {
            'title': f'Direct Video Download',
            'source_url': url,
            'content_type': content_type,
            'file_size': file_size,
            'file_extension': file_extension,
            'download_method': 'direct_url'
        }
        
        return temp_path, video_info
        
    except requests.RequestException as e:
        logger.error(f"❌ Failed to download from direct URL: {e}")
        
        # Provide more specific error messages
        error_msg = str(e)
        if "404" in error_msg:
            error_msg = "Video file not found (404). Please check the URL."
        elif "403" in error_msg:
            error_msg = "Access denied (403). The video file may be restricted."
        elif "timeout" in error_msg.lower():
            error_msg = "Download timed out. The video file may be too large or the server is slow."
        
        raise Exception(f"Direct URL download failed: {error_msg}")
        
    except Exception as e:
        logger.error(f"❌ Unexpected error during direct download: {e}")
        raise Exception(f"Download failed: {str(e)}")
