#!/usr/bin/env python3
"""YouTube video downloader for the dubbing pipeline.

This script downloads videos from YouTube for processing with the dubbing pipeline.
It supports downloading single videos or entire playlists.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add the src directory to the path so we can use the logger
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils.logger import logger, setup_logger

try:
    from yt_dlp import YoutubeDL
except ImportError:
    print("Error: yt-dlp is not installed. Please install it with: pip install yt-dlp")
    sys.exit(1)

def download_video(url, output_dir=None, quality="720p", format="mp4", audio_only=False):
    """Download a video from YouTube.
    
    Args:
        url: YouTube URL
        output_dir: Directory to save the video in. If None, a default is used.
        quality: Video quality (default: 720p)
        format: Video format (default: mp4)
        audio_only: If True, only download audio
        
    Returns:
        str: Path to the downloaded video
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = Path("youtube_videos")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading video from {url}")
    start_time = time.time()
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': f'bestvideo[height<={quality[:-1]}][ext={format}]+bestaudio/best[height<={quality[:-1]}]' if not audio_only else 'bestaudio/best',
        'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
        'restrictfilenames': True,  # Restrict filenames to ASCII
        'noplaylist': True,         # Download single video, not playlist
        'quiet': False,
        'no_warnings': False,
        'ignoreerrors': False,
        'progress_hooks': [lambda d: logger.info(f"Progress: {d.get('_percent_str', '0%')} of {d.get('filename', 'unknown')}")],
    }
    
    # If audio only, adjust format and postprocessors
    if audio_only:
        ydl_opts.update({
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        })
    
    try:
        # Create a YoutubeDL instance and download the video
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Get file path from info dict, handle both playlist and single video
            if 'entries' in info:
                # Playlist
                downloaded_files = []
                for entry in info['entries']:
                    if entry:
                        title = entry.get('title', 'unknown')
                        ext = entry.get('ext', format)
                        filename = f"{title}.{ext}"
                        # For audio only, the extension might be changed
                        if audio_only:
                            filename = f"{title}.wav"
                        downloaded_files.append(str(output_dir / filename))
                duration = time.time() - start_time
                logger.info(f"Downloaded {len(downloaded_files)} videos in {duration:.2f} seconds to {output_dir}")
                return downloaded_files
            else:
                # Single video
                title = info.get('title', 'unknown')
                ext = info.get('ext', format)
                filename = f"{title}.{ext}"
                # For audio only, the extension might be changed
                if audio_only:
                    filename = f"{title}.wav"
                duration = time.time() - start_time
                logger.info(f"Download completed in {duration:.2f} seconds: {filename}")
                return str(output_dir / filename)
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        raise

def main():
    """Parse command line arguments and download videos."""
    parser = argparse.ArgumentParser(description="Download videos from YouTube")
    
    parser.add_argument("url", type=str, help="YouTube URL (video or playlist)")
    parser.add_argument("--output_dir", type=str, default="youtube_videos", 
                        help="Directory to save videos in")
    parser.add_argument("--quality", type=str, default="720p", 
                        choices=["360p", "480p", "720p", "1080p", "1440p", "2160p"],
                        help="Video quality")
    parser.add_argument("--format", type=str, default="mp4", 
                        choices=["mp4", "webm", "mkv"],
                        help="Video format")
    parser.add_argument("--audio_only", action="store_true", 
                        help="Download audio only (in WAV format)")
    parser.add_argument("--playlist", action="store_true", 
                        help="Download all videos in the playlist")
    
    args = parser.parse_args()
    
    # Configure special logger for this script
    download_logger = setup_logger("youtube_downloader", "logs/youtube_download.log")
    
    try:
        # Update YoutubeDL options for playlist
        if args.playlist:
            logger.info(f"Downloading playlist: {args.url}")
        
        # Download the video(s)
        result = download_video(
            args.url,
            output_dir=args.output_dir,
            quality=args.quality,
            format=args.format,
            audio_only=args.audio_only
        )
        
        if isinstance(result, list):
            logger.info(f"Downloaded {len(result)} files:")
            for file_path in result:
                logger.info(f"  - {file_path}")
        else:
            logger.info(f"Downloaded file: {result}")
        
        return 0
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main()) 