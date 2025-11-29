"""
Video Action Detection FastAPI Server
Clean implementation without queue/consumer/producer patterns
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import os
import sys
from datetime import datetime
import tempfile
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import video action detection model
try:
    from video_action_model import ActionDetector
    VIDEO_ACTION_AVAILABLE = True
    logger.info("Video action detection model imported successfully")
except ImportError as e:
    VIDEO_ACTION_AVAILABLE = False
    ActionDetector = None
    logger.warning(f"Video action detection not available: {e}")
except Exception as e:
    VIDEO_ACTION_AVAILABLE = False
    ActionDetector = None
    logger.error(f"Unexpected error importing video action model: {e}")

# Import Qwen3-VL action detector
try:
    from qwen3_vl import Qwen3VLActionDetector
    QWEN_VL_AVAILABLE = True
    logger.info("Qwen3-VL action detector imported successfully")
except ImportError as e:
    QWEN_VL_AVAILABLE = False
    Qwen3VLActionDetector = None
    logger.warning(f"Qwen3-VL action detector not available: {e}")
except Exception as e:
    QWEN_VL_AVAILABLE = False
    Qwen3VLActionDetector = None
    logger.error(f"Unexpected error importing Qwen3-VL: {e}")

# Import Grounding DINO object detection model
try:
    from model import ModelManager, DynamicGroundingDINO
    GROUNDING_DINO_AVAILABLE = True
    logger.info("Grounding DINO model imported successfully")
except ImportError as e:
    GROUNDING_DINO_AVAILABLE = False
    ModelManager = None
    DynamicGroundingDINO = None
    logger.warning(f"Grounding DINO model not available: {e}")
except Exception as e:
    GROUNDING_DINO_AVAILABLE = False
    ModelManager = None
    DynamicGroundingDINO = None
    logger.error(f"Unexpected error importing Grounding DINO: {e}")

# Import Factory Alert Engine
try:
    from factory_alert_engine import FactoryAlertEngine, FactoryAlert, AnalysisResult
    FACTORY_ALERT_AVAILABLE = True
    logger.info("Factory Alert Engine imported successfully")
except ImportError as e:
    FACTORY_ALERT_AVAILABLE = False
    FactoryAlertEngine = None
    logger.warning(f"Factory Alert Engine not available: {e}")
except Exception as e:
    FACTORY_ALERT_AVAILABLE = False
    FactoryAlertEngine = None
    logger.error(f"Unexpected error importing Factory Alert Engine: {e}")

# Import YOLO Person Counter
try:
    from yolo_person_counter import YOLOPersonCounter, get_yolo_counter, YOLO_AVAILABLE
    logger.info("YOLO Person Counter imported successfully")
except ImportError as e:
    YOLO_AVAILABLE = False
    YOLOPersonCounter = None
    get_yolo_counter = None
    logger.warning(f"YOLO Person Counter not available: {e}")
except Exception as e:
    YOLO_AVAILABLE = False
    YOLOPersonCounter = None
    get_yolo_counter = None
    logger.error(f"Unexpected error importing YOLO Person Counter: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Video Action Detection API",
    description="Video action recognition API using BLIP and Sentence Transformers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global video action detector
video_action_detector = None

if VIDEO_ACTION_AVAILABLE:
    try:
        logger.info("Initializing video action detector...")
        video_action_detector = ActionDetector(
            person_weight=0.2,
            action_weight=0.7,
            context_weight=0.1,
            similarity_threshold=0.5,
            action_threshold=0.4
        )
        logger.info("Video action detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize video action detector: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        video_action_detector = None

# Global Grounding DINO model manager
grounding_dino_model = None

if GROUNDING_DINO_AVAILABLE:
    try:
        logger.info("Initializing Grounding DINO model manager...")
        model_manager = ModelManager()
        grounding_dino_model = model_manager.get_model()
        logger.info("Grounding DINO model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Grounding DINO model: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        grounding_dino_model = None


# ============================================================================
# Pydantic Models
# ============================================================================

class VideoActionRequest(BaseModel):
    """Request model for video action detection from URL"""
    video_url: str = Field(..., description="URL of the video to analyze")
    prompt: str = Field(..., description="Action description/prompt (e.g., 'person running', 'person jumping')")
    person_weight: Optional[float] = Field(0.2, ge=0.0, le=1.0, description="Weight for person detection component")
    action_weight: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Weight for action detection component")
    context_weight: Optional[float] = Field(0.1, ge=0.0, le=1.0, description="Weight for context detection component")
    similarity_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Overall similarity threshold")
    action_threshold: Optional[float] = Field(0.4, ge=0.0, le=1.0, description="Action-specific threshold")
    return_timeline: Optional[bool] = Field(True, description="Whether to return timeline visualization")


class VideoActionDetection(BaseModel):
    """Single video action detection result"""
    timestamp: float
    frame_idx: int
    confidence: float
    blip_description: str
    similarity_scores: Dict[str, float]
    passed: bool


class VideoActionSegment(BaseModel):
    """Video action segment"""
    start_time: float
    end_time: float
    confidence: float
    frame_count: int
    action_label: str
    detections: List[VideoActionDetection]


class VideoActionResponse(BaseModel):
    """Response model for video action detection results"""
    success: bool
    job_id: str
    video_path: Optional[str] = None
    prompt: str
    action_verb: str
    timestamp: str
    video_duration: float
    stats: Dict[str, Any]
    passed_detections: List[VideoActionDetection]
    segments: List[VideoActionSegment]
    timeline_visualization: Optional[Dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    video_action_available: bool
    detector_loaded: bool
    message: str


# ============================================================================
# Grounding DINO Object Detection Pydantic Models
# ============================================================================

class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: float
    height: float


class Detection(BaseModel):
    """Single detection result"""
    id: int
    label: str
    confidence: float
    bounding_box: BoundingBox


class ImageSize(BaseModel):
    """Image dimensions"""
    width: int
    height: int


class DetectionThresholds(BaseModel):
    """Detection thresholds"""
    box_threshold: float
    text_threshold: float


class Visualization(BaseModel):
    """Visualization data"""
    image_base64: str
    format: str


class DetectionRequest(BaseModel):
    """Request model for URL-based detection"""
    image_url: str = Field(..., description="URL of the image to analyze")
    text_queries: List[str] = Field(..., description="Text queries for object detection")
    box_threshold: Optional[float] = Field(0.4, ge=0.0, le=1.0, description="Confidence threshold for bounding boxes")
    text_threshold: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Confidence threshold for text matching")
    return_visualization: Optional[bool] = Field(True, description="Whether to return visualization image")


class DetectionResponse(BaseModel):
    """Response model for detection results"""
    success: bool
    num_detections: int
    detections: List[Detection]
    image_size: Optional[ImageSize] = None
    queries: Optional[List[str]] = None
    thresholds: Optional[DetectionThresholds] = None
    visualization: Optional[Visualization] = None
    error: Optional[str] = None


# ============================================================================
# Qwen3-VL Pydantic Models
# ============================================================================

class QwenActionRequest(BaseModel):
    """Request model for Qwen3-VL video action detection from URL"""
    video_url: str = Field(..., description="URL of the video to analyze")
    action_prompt: str = Field("running", description="Action to detect (e.g., 'running', 'walking', 'jumping')")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    frame_sample_rate: Optional[int] = Field(1, ge=1, le=10, description="Frames to analyze per second")
    ollama_url: Optional[str] = Field("http://localhost:11434", description="Ollama API URL")
    model_name: Optional[str] = Field("qwen2.5-vl", description="Qwen VL model name in Ollama")


class QwenActionDetectionResult(BaseModel):
    """Single frame detection result from Qwen3-VL"""
    timestamp: float
    frame_idx: int
    confidence: float
    description: str
    action_detected: bool


class QwenActionSegment(BaseModel):
    """Action segment detected by Qwen3-VL"""
    start_time: float
    end_time: float
    duration: float
    confidence: float
    frame_count: int
    action_label: str


class QwenActionResponse(BaseModel):
    """Response model for Qwen3-VL action detection"""
    success: bool
    job_id: str
    video_path: Optional[str] = None
    action_prompt: str
    timestamp: str
    video_info: Dict[str, Any]
    stats: Dict[str, Any]
    segments: List[QwenActionSegment]
    error: Optional[str] = None


# ============================================================================
# YOLO Person Counting Pydantic Models
# ============================================================================

class PersonBoundingBox(BaseModel):
    """Bounding box for detected person"""
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int


class PersonDetectionResult(BaseModel):
    """Single person detection"""
    person_id: int
    confidence: float
    bounding_box: PersonBoundingBox


class FramePersonCount(BaseModel):
    """Person count for a single frame"""
    frame_idx: int
    timestamp: float
    person_count: int
    detections: List[PersonDetectionResult]


class PersonCountTimeline(BaseModel):
    """Timeline data for person counting"""
    timestamps: List[float]
    counts: List[int]
    max_count: int
    min_count: int
    avg_count: float
    duration: float


class PersonCountRequest(BaseModel):
    """Request for person counting from URL"""
    video_url: str = Field(..., description="URL of the video to analyze")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="YOLO detection confidence threshold")
    frame_sample_rate: Optional[int] = Field(1, ge=1, le=30, description="Process every Nth frame")
    return_video: Optional[bool] = Field(True, description="Whether to return annotated video")
    return_base64: Optional[bool] = Field(False, description="Return video as base64 instead of file path")


class PersonCountResponse(BaseModel):
    """Response for person counting"""
    success: bool
    job_id: str
    video_path: Optional[str] = None
    timestamp: str
    video_info: Dict[str, Any]
    statistics: Dict[str, Any]
    timeline: Optional[PersonCountTimeline] = None
    frame_counts: Optional[List[FramePersonCount]] = None
    output_video_path: Optional[str] = None
    output_video_base64: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================

def download_video_from_url(url: str):
    """
    Download video from direct URL.
    
    Args:
        url: Direct video URL (no YouTube URLs supported)
        
    Returns:
        Tuple of (temp_file_path, video_info)
    """
    logger.info(f"Processing video URL: {url}")
    
    # Check for YouTube URLs and reject them
    youtube_patterns = ['youtube.com', 'youtu.be', 'm.youtube.com']
    if any(pattern in url.lower() for pattern in youtube_patterns):
        raise Exception(
            "YouTube URLs are not supported.\n"
            "Please use a direct video file URL or upload your video file."
        )
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'video/*,*/*;q=0.9',
            'Accept-Encoding': 'identity',
            'Connection': 'keep-alive'
        }
        
        response = requests.get(url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        content_length = response.headers.get('content-length')
        
        logger.info(f"Content-Type: {content_type}")
        logger.info(f"Content-Length: {content_length}")
        
        # Determine file extension
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
        file_extension = '.mp4'
        for ext in video_extensions:
            if ext in url.lower():
                file_extension = ext
                break
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_path = temp_file.name
            
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
                    downloaded += len(chunk)
        
        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            raise Exception("Download completed but file is empty")
        
        logger.info(f"Video downloaded: {temp_path} ({file_size} bytes)")
        
        video_info = {
            'title': 'Direct Video Download',
            'source_url': url,
            'content_type': content_type,
            'file_size': file_size,
            'file_extension': file_extension
        }
        
        return temp_path, video_info
        
    except requests.RequestException as e:
        logger.error(f"Failed to download from URL: {e}")
        raise Exception(f"Download failed: {str(e)}")


# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Video Action Detection API starting...")
    if video_action_detector:
        logger.info("Video action detector is ready")
    else:
        logger.warning("Video action detector is not available")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    detector_status = "ok" if video_action_detector else "error"
    status_text = "✅ Ready" if video_action_detector else "❌ Not Available"
    qwen_status = "ok" if QWEN_VL_AVAILABLE else "error"
    qwen_status_text = "✅ Ready" if QWEN_VL_AVAILABLE else "❌ Not Available"
    dino_status = "ok" if grounding_dino_model else "error"
    dino_status_text = "✅ Ready" if grounding_dino_model else "❌ Not Available"
    factory_status = "ok" if FACTORY_ALERT_AVAILABLE else "error"
    factory_status_text = "✅ Ready" if FACTORY_ALERT_AVAILABLE else "❌ Not Available"
    yolo_status = "ok" if YOLO_AVAILABLE else "error"
    yolo_status_text = "✅ Ready" if YOLO_AVAILABLE else "❌ Not Available"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Detection API</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }}
            h2 {{ color: #333; margin-top: 30px; }}
            .endpoint {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; background: #f8f9fa; }}
            .endpoint-qwen {{ border-left-color: #8b5cf6; }}
            .endpoint-dino {{ border-left-color: #10b981; }}
            .method {{ font-weight: bold; color: #007acc; }}
            .method-qwen {{ color: #8b5cf6; }}
            .method-dino {{ color: #10b981; }}
            .status {{ padding: 10px; border-radius: 5px; margin: 15px 0; display: inline-block; margin-right: 10px; }}
            .status-ok {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }}
            .status-error {{ background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
            pre {{ background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            a {{ color: #007acc; }}
            .models-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 AI Detection API</h1>
            <p>Object detection and video action recognition using multiple AI models</p>
            
            <div class="models-row">
                <div class="status status-{dino_status}">
                    <strong>Grounding DINO:</strong> {dino_status_text}
                </div>
                <div class="status status-{detector_status}">
                    <strong>BLIP Detector:</strong> {status_text}
                </div>
                <div class="status status-{qwen_status}">
                    <strong>Qwen3-VL (Ollama):</strong> {qwen_status_text}
                </div>
                <div class="status status-{yolo_status}">
                    <strong>YOLOv8 Counter:</strong> {yolo_status_text}
                </div>
                <div class="status status-{factory_status}">
                    <strong>Factory Alert Engine:</strong> {factory_status_text}
                </div>
            </div>
            
            <h2>🎯 Grounding DINO Object Detection</h2>
            <p>Zero-shot object detection with text-guided queries</p>
            
            <div class="endpoint endpoint-dino">
                <span class="method method-dino">POST</span> <strong>/detect</strong>
                <p>Detect objects in image from URL</p>
            </div>
            
            <div class="endpoint endpoint-dino">
                <span class="method method-dino">POST</span> <strong>/detect/upload</strong>
                <p>Detect objects in uploaded image file</p>
            </div>
            
            <div class="endpoint endpoint-dino">
                <span class="method method-dino">GET</span> <strong>/detect/status</strong>
                <p>Get Grounding DINO system status</p>
            </div>
            
            <h2>🧠 BLIP + Sentence Transformers Endpoints</h2>
            <p>Video action detection using BLIP image captioning and semantic similarity</p>
            
            <div class="endpoint">
                <span class="method">POST</span> <strong>/video_action/detect</strong>
                <p>Detect actions in video from URL</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <strong>/video_action/detect/upload</strong>
                <p>Detect actions in uploaded video file</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/video_action/status</strong>
                <p>Get BLIP detector system status</p>
            </div>
            
            <h2>🤖 Qwen3-VL (Ollama) Endpoints</h2>
            <p>Vision-Language Model action detection using Qwen3-VL via Ollama</p>
            
            <div class="endpoint endpoint-qwen">
                <span class="method method-qwen">POST</span> <strong>/qwen/detect</strong>
                <p>Detect actions in video from URL using Qwen3-VL</p>
            </div>
            
            <div class="endpoint endpoint-qwen">
                <span class="method method-qwen">POST</span> <strong>/qwen/detect/upload</strong>
                <p>Detect actions in uploaded video file using Qwen3-VL</p>
            </div>
            
            <div class="endpoint endpoint-qwen">
                <span class="method method-qwen">GET</span> <strong>/qwen/status</strong>
                <p>Get Qwen3-VL and Ollama status</p>
            </div>
            
            <h2>👥 YOLO Person Counting Endpoints</h2>
            <p>Fast person detection and counting using YOLOv8 nano with annotated video output</p>
            
            <div class="endpoint" style="border-left-color: #ef4444;">
                <span class="method" style="color: #ef4444;">POST</span> <strong>/person_count/detect</strong>
                <p>Count persons in video from URL</p>
            </div>
            
            <div class="endpoint" style="border-left-color: #ef4444;">
                <span class="method" style="color: #ef4444;">POST</span> <strong>/person_count/upload</strong>
                <p>Count persons in uploaded video file</p>
            </div>
            
            <div class="endpoint" style="border-left-color: #ef4444;">
                <span class="method" style="color: #ef4444;">GET</span> <strong>/person_count/status</strong>
                <p>Get YOLO person counting status</p>
            </div>
            
            <h2>🏭 Factory Alert Engine Endpoints</h2>
            <p>Smart factory video analysis with safety, efficiency, quality, and compliance alerts</p>
            
            <div class="endpoint" style="border-left-color: #f59e0b;">
                <span class="method" style="color: #f59e0b;">POST</span> <strong>/factory/alerts/detect/upload</strong>
                <p>Analyze factory video and generate structured alerts</p>
            </div>
            
            <div class="endpoint" style="border-left-color: #f59e0b;">
                <span class="method" style="color: #f59e0b;">GET</span> <strong>/factory/alerts/status</strong>
                <p>Get Factory Alert Engine status</p>
            </div>
            
            <h2>🔧 System Endpoints</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/health</strong>
                <p>Health check endpoint</p>
            </div>
            
            <h3>📚 Documentation</h3>
            <ul>
                <li><a href="/docs">Interactive API Documentation (Swagger UI)</a></li>
                <li><a href="/redoc">API Documentation (ReDoc)</a></li>
            </ul>
            
            <h3>🚀 Quick Start</h3>
            
            <h4>Object Detection (Grounding DINO):</h4>
            <pre>curl -X POST "http://localhost:8000/detect" \\
     -H "Content-Type: application/json" \\
     -d '{{"image_url": "https://example.com/image.jpg", "text_queries": ["person", "car", "dog"]}}'</pre>
            
            <h4>Object Detection (Upload):</h4>
            <pre>curl -X POST "http://localhost:8000/detect/upload" \\
     -F "file=@your_image.jpg" \\
     -F "text_queries=person, car, dog"</pre>
            
            <h4>Video Action Detection (BLIP):</h4>
            <pre>curl -X POST "http://localhost:8000/video_action/detect/upload" \\
     -F "file=@your_video.mp4" \\
     -F "prompt=person running"</pre>
            
            <h4>Video Action Detection (Qwen3-VL):</h4>
            <pre>curl -X POST "http://localhost:8000/qwen/detect/upload" \\
     -F "file=@your_video.mp4" \\
     -F "action_prompt=running"</pre>
            
            <h4>Person Counting (YOLO):</h4>
            <pre>curl -X POST "http://localhost:8000/person_count/upload" \\
     -F "file=@your_video.mp4" \\
     -F "confidence_threshold=0.5" \\
     -F "return_video=true"</pre>
            
            <h4>Factory Alert Detection:</h4>
            <pre>curl -X POST "http://localhost:8000/factory/alerts/detect/upload" \\
     -F "file=@factory_video.mp4" \\
     -F "zones=zone_a,zone_b" \\
     -F "alert_types=safety,efficiency,quality"</pre>
        </div>
    </body>
    </html>
    """
    return html_content


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    detector_loaded = video_action_detector is not None
    
    return {
        "status": "healthy" if detector_loaded else "degraded",
        "video_action_available": VIDEO_ACTION_AVAILABLE,
        "detector_loaded": detector_loaded,
        "message": "API is running and video action detector is loaded" if detector_loaded else "API is running but video action detector failed to load"
    }


@app.get("/video_action/status")
async def get_video_action_status():
    """Get video action detection system status"""
    return {
        "video_action_available": VIDEO_ACTION_AVAILABLE,
        "detector_loaded": video_action_detector is not None,
        "supported_formats": ["mp4", "avi", "mov", "mkv", "webm"],
        "features": {
            "action_detection": True,
            "timeline_visualization": True,
            "custom_weights": True,
            "parallel_processing": True
        },
        "model_info": {
            "blip_model": "Salesforce/blip-image-captioning-large",
            "similarity_model": "all-MiniLM-L6-v2"
        } if VIDEO_ACTION_AVAILABLE else None
    }


@app.post("/video_action/detect", response_model=VideoActionResponse)
async def detect_video_action_from_url(request: VideoActionRequest):
    """
    Detect actions in video from URL using BLIP and sentence transformers
    
    - **video_url**: URL of the video to analyze (direct URLs only, no YouTube)
    - **prompt**: Action description (e.g., 'person running', 'person jumping')
    - **person_weight**: Weight for person detection component (0.0-1.0)
    - **action_weight**: Weight for action detection component (0.0-1.0)
    - **context_weight**: Weight for context detection component (0.0-1.0)
    - **similarity_threshold**: Overall similarity threshold (0.0-1.0)
    - **action_threshold**: Action-specific threshold (0.0-1.0)
    - **return_timeline**: Whether to return timeline visualization data
    """
    if not VIDEO_ACTION_AVAILABLE or video_action_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Video action detection is not available. Required dependencies may be missing."
        )
    
    temp_video_path = None
    try:
        # Download video from URL
        logger.info(f"Downloading video from URL: {request.video_url}")
        temp_video_path, video_info = download_video_from_url(request.video_url)
        logger.info(f"Video downloaded: {video_info.get('title', 'Unknown')}")
        
        # Create ActionDetector with custom weights
        detector = ActionDetector(
            person_weight=request.person_weight,
            action_weight=request.action_weight,
            context_weight=request.context_weight,
            similarity_threshold=request.similarity_threshold,
            action_threshold=request.action_threshold
        )
        
        # Process video
        results = detector.process_video(
            video_path=temp_video_path,
            prompt=request.prompt,
            save_files=request.return_timeline
        )
        
        # Clean up temporary file
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        
        # Convert results to response format
        passed_detections = []
        for detection_data in results.get('passed_detections_only', []):
            passed_detections.append(VideoActionDetection(**detection_data))
        
        segments = []
        for segment_data in results.get('segments', []):
            segment_detections = [VideoActionDetection(**d) for d in segment_data['detections']]
            segment_data['detections'] = segment_detections
            segments.append(VideoActionSegment(**segment_data))
        
        response_data = VideoActionResponse(
            success=True,
            job_id=results['job_id'],
            video_path=request.video_url,
            prompt=results['prompt'],
            action_verb=results['action_verb'],
            timestamp=results['timestamp'],
            video_duration=results['video_duration'],
            stats=results['stats'],
            passed_detections=passed_detections,
            segments=segments,
            timeline_visualization=results.get('timeline_data') if request.return_timeline else None
        )
        
        logger.info(f"Video action detection completed. Job ID: {results['job_id']}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video action detection failed: {e}")
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video action detection failed: {str(e)}"
        )


@app.post("/video_action/detect/upload", response_model=VideoActionResponse)
async def detect_video_action_from_upload(
    file: UploadFile = File(..., description="Video file to analyze"),
    prompt: str = Form(..., description="Action description (e.g., 'person running')"),
    person_weight: float = Form(0.2, description="Weight for person detection component"),
    action_weight: float = Form(0.7, description="Weight for action detection component"),
    context_weight: float = Form(0.1, description="Weight for context detection component"),
    similarity_threshold: float = Form(0.5, description="Overall similarity threshold"),
    action_threshold: float = Form(0.4, description="Action-specific threshold"),
    return_timeline: bool = Form(True, description="Whether to return timeline visualization")
):
    """
    Detect actions in uploaded video file using BLIP and sentence transformers
    
    - **file**: Video file (MP4, AVI, MOV, etc.)
    - **prompt**: Action description (e.g., 'person running', 'person jumping')
    - **person_weight**: Weight for person detection component (0.0-1.0)
    - **action_weight**: Weight for action detection component (0.0-1.0) 
    - **context_weight**: Weight for context detection component (0.0-1.0)
    - **similarity_threshold**: Overall similarity threshold (0.0-1.0)
    - **action_threshold**: Action-specific threshold (0.0-1.0)
    - **return_timeline**: Whether to return timeline visualization
    """
    if not VIDEO_ACTION_AVAILABLE or video_action_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Video action detection is not available. Required dependencies may be missing."
        )
    
    temp_video_path = None
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a video"
            )
        
        # Validate parameters
        if not (0.0 <= person_weight <= 1.0):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="person_weight must be between 0.0 and 1.0")
        if not (0.0 <= action_weight <= 1.0):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="action_weight must be between 0.0 and 1.0")
        if not (0.0 <= context_weight <= 1.0):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="context_weight must be between 0.0 and 1.0")
        if not (0.0 <= similarity_threshold <= 1.0):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="similarity_threshold must be between 0.0 and 1.0")
        if not (0.0 <= action_threshold <= 1.0):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="action_threshold must be between 0.0 and 1.0")
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_video_path = temp_file.name
            contents = await file.read()
            temp_file.write(contents)
            temp_file.flush()
        
        logger.info(f"Video uploaded and saved to: {temp_video_path}")
        
        # Create ActionDetector with custom weights
        detector = ActionDetector(
            person_weight=person_weight,
            action_weight=action_weight,
            context_weight=context_weight,
            similarity_threshold=similarity_threshold,
            action_threshold=action_threshold
        )
        
        # Process video
        results = detector.process_video(
            video_path=temp_video_path,
            prompt=prompt,
            save_files=return_timeline
        )
        
        # Clean up temporary file
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        
        # Convert results to response format
        passed_detections = []
        for detection_data in results.get('passed_detections_only', []):
            passed_detections.append(VideoActionDetection(**detection_data))
        
        segments = []
        for segment_data in results.get('segments', []):
            segment_detections = [VideoActionDetection(**d) for d in segment_data['detections']]
            segment_data['detections'] = segment_detections
            segments.append(VideoActionSegment(**segment_data))
        
        response_data = VideoActionResponse(
            success=True,
            job_id=results['job_id'],
            video_path=file.filename,
            prompt=results['prompt'],
            action_verb=results['action_verb'],
            timestamp=results['timestamp'],
            video_duration=results['video_duration'],
            stats=results['stats'],
            passed_detections=passed_detections,
            segments=segments,
            timeline_visualization=results.get('timeline_data') if return_timeline else None
        )
        
        logger.info(f"Video action detection completed. Job ID: {results['job_id']}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video upload action detection failed: {e}")
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video action detection failed: {str(e)}"
        )


# ============================================================================
# Qwen3-VL Video Action Detection Endpoints
# ============================================================================

@app.post("/qwen/detect", response_model=QwenActionResponse)
async def detect_action_qwen_from_url(request: QwenActionRequest):
    """
    Detect actions in video using Qwen3-VL via Ollama
    
    - **video_url**: URL of the video to analyze (direct URLs only)
    - **action_prompt**: Action to detect (e.g., 'running', 'walking', 'jumping')
    - **confidence_threshold**: Minimum confidence to consider action detected (0.0-1.0)
    - **frame_sample_rate**: How many frames to analyze per second
    - **ollama_url**: Ollama API URL (default: http://localhost:11434)
    - **model_name**: Qwen VL model name in Ollama
    """
    if not QWEN_VL_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qwen3-VL detector is not available. Check qwen3_vl.py module."
        )
    
    temp_video_path = None
    try:
        # Download video from URL
        logger.info(f"Downloading video from URL: {request.video_url}")
        temp_video_path, video_info = download_video_from_url(request.video_url)
        logger.info(f"Video downloaded: {video_info.get('title', 'Unknown')}")
        
        # Create Qwen3-VL detector with custom settings
        detector = Qwen3VLActionDetector(
            ollama_url=request.ollama_url,
            model_name=request.model_name,
            confidence_threshold=request.confidence_threshold,
            frame_sample_rate=request.frame_sample_rate
        )
        
        # Process video
        results = detector.process_video(
            video_path=temp_video_path,
            action_prompt=request.action_prompt,
            parallel=False,
            save_results=False
        )
        
        # Clean up temporary file
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        
        # Convert segments to response format
        segments = []
        for seg_data in results.get('segments', []):
            segments.append(QwenActionSegment(
                start_time=seg_data['start_time'],
                end_time=seg_data['end_time'],
                duration=seg_data['duration'],
                confidence=seg_data['confidence'],
                frame_count=seg_data['frame_count'],
                action_label=seg_data['action_label']
            ))
        
        response_data = QwenActionResponse(
            success=True,
            job_id=results['job_id'],
            video_path=request.video_url,
            action_prompt=results['action_prompt'],
            timestamp=results['timestamp'],
            video_info=results['video_info'],
            stats=results['stats'],
            segments=segments
        )
        
        logger.info(f"Qwen3-VL detection completed. Job ID: {results['job_id']}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Qwen3-VL detection failed: {e}")
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Qwen3-VL detection failed: {str(e)}"
        )


@app.post("/qwen/detect/upload", response_model=QwenActionResponse)
async def detect_action_qwen_from_upload(
    file: UploadFile = File(..., description="Video file to analyze"),
    action_prompt: str = Form("running", description="Action to detect"),
    confidence_threshold: float = Form(0.5, description="Minimum confidence threshold"),
    frame_sample_rate: int = Form(1, description="Frames to analyze per second"),
    ollama_url: str = Form("http://localhost:11434", description="Ollama API URL"),
    model_name: str = Form("qwen2.5-vl", description="Qwen VL model name")
):
    """
    Detect actions in uploaded video using Qwen3-VL via Ollama
    
    - **file**: Video file (MP4, AVI, MOV, etc.)
    - **action_prompt**: Action to detect (e.g., 'running', 'walking', 'jumping')
    - **confidence_threshold**: Minimum confidence to consider action detected
    - **frame_sample_rate**: How many frames to analyze per second
    - **ollama_url**: Ollama API URL
    - **model_name**: Qwen VL model name in Ollama
    """
    if not QWEN_VL_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qwen3-VL detector is not available. Check qwen3_vl.py module."
        )
    
    temp_video_path = None
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a video"
            )
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_video_path = temp_file.name
            contents = await file.read()
            temp_file.write(contents)
            temp_file.flush()
        
        logger.info(f"Video uploaded and saved to: {temp_video_path}")
        
        # Create Qwen3-VL detector
        detector = Qwen3VLActionDetector(
            ollama_url=ollama_url,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            frame_sample_rate=frame_sample_rate
        )
        
        # Process video
        results = detector.process_video(
            video_path=temp_video_path,
            action_prompt=action_prompt,
            parallel=False,
            save_results=False
        )
        
        # Clean up temporary file
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        
        # Convert segments to response format
        segments = []
        for seg_data in results.get('segments', []):
            segments.append(QwenActionSegment(
                start_time=seg_data['start_time'],
                end_time=seg_data['end_time'],
                duration=seg_data['duration'],
                confidence=seg_data['confidence'],
                frame_count=seg_data['frame_count'],
                action_label=seg_data['action_label']
            ))
        
        response_data = QwenActionResponse(
            success=True,
            job_id=results['job_id'],
            video_path=file.filename,
            action_prompt=results['action_prompt'],
            timestamp=results['timestamp'],
            video_info=results['video_info'],
            stats=results['stats'],
            segments=segments
        )
        
        logger.info(f"Qwen3-VL detection completed. Job ID: {results['job_id']}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Qwen3-VL upload detection failed: {e}")
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Qwen3-VL detection failed: {str(e)}"
        )


@app.get("/qwen/status")
async def get_qwen_status():
    """Get Qwen3-VL detection system status"""
    ollama_status = "unknown"
    available_models = []
    
    # Check Ollama connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            ollama_status = "connected"
            models = response.json().get('models', [])
            available_models = [m.get('name', '') for m in models]
        else:
            ollama_status = "error"
    except requests.exceptions.ConnectionError:
        ollama_status = "disconnected"
    except Exception as e:
        ollama_status = f"error: {str(e)}"
    
    return {
        "qwen_vl_available": QWEN_VL_AVAILABLE,
        "ollama_status": ollama_status,
        "available_models": available_models,
        "recommended_models": ["qwen2.5-vl", "qwen2-vl", "llava"],
        "supported_formats": ["mp4", "avi", "mov", "mkv", "webm"],
        "features": {
            "action_detection": True,
            "temporal_segmentation": True,
            "custom_action_prompts": True,
            "adjustable_confidence": True
        }
    }


# ============================================================================
# YOLO Person Counting Endpoints
# ============================================================================

@app.post("/person_count/detect", response_model=PersonCountResponse)
async def count_persons_from_url(request: PersonCountRequest):
    """
    Count persons in video from URL using YOLO
    
    - **video_url**: URL of the video to analyze (direct URLs only)
    - **confidence_threshold**: YOLO detection confidence (0.0-1.0)
    - **frame_sample_rate**: Process every Nth frame
    - **return_video**: Whether to generate annotated output video
    - **return_base64**: Return video as base64 string
    """
    if not YOLO_AVAILABLE or get_yolo_counter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="YOLO Person Counter is not available. Install with: pip install ultralytics"
        )
    
    temp_video_path = None
    try:
        # Download video from URL
        logger.info(f"Downloading video from URL: {request.video_url}")
        temp_video_path, video_info = download_video_from_url(request.video_url)
        logger.info(f"Video downloaded: {video_info.get('title', 'Unknown')}")
        
        # Get YOLO counter
        counter = get_yolo_counter(confidence_threshold=request.confidence_threshold)
        if counter is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to initialize YOLO counter"
            )
        
        # Process video
        result = counter.process_video(
            video_path=temp_video_path,
            output_video=request.return_video,
            frame_sample_rate=request.frame_sample_rate,
            return_base64=request.return_base64
        )
        
        # Clean up input video
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        
        # Build timeline
        timeline = PersonCountTimeline(
            timestamps=[fc.timestamp for fc in result.frame_counts],
            counts=[fc.person_count for fc in result.frame_counts],
            max_count=result.max_persons,
            min_count=result.min_persons,
            avg_count=result.avg_persons,
            duration=result.duration
        )
        
        # Build frame counts (limit to avoid huge responses)
        frame_counts = []
        sample_interval = max(1, len(result.frame_counts) // 100)  # Max 100 frames in response
        for i, fc in enumerate(result.frame_counts):
            if i % sample_interval == 0:
                detections = []
                for det in fc.detections:
                    detections.append(PersonDetectionResult(
                        person_id=det.person_id,
                        confidence=det.confidence,
                        bounding_box=PersonBoundingBox(
                            x1=det.bbox[0],
                            y1=det.bbox[1],
                            x2=det.bbox[2],
                            y2=det.bbox[3],
                            center_x=det.center[0],
                            center_y=det.center[1]
                        )
                    ))
                frame_counts.append(FramePersonCount(
                    frame_idx=fc.frame_idx,
                    timestamp=fc.timestamp,
                    person_count=fc.person_count,
                    detections=detections
                ))
        
        job_id = f"yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        response_data = PersonCountResponse(
            success=True,
            job_id=job_id,
            video_path=request.video_url,
            timestamp=datetime.now().isoformat(),
            video_info={
                "total_frames": result.total_frames,
                "fps": result.fps,
                "duration": result.duration
            },
            statistics={
                "max_persons": result.max_persons,
                "min_persons": result.min_persons,
                "avg_persons": round(result.avg_persons, 2),
                "frames_analyzed": len(result.frame_counts)
            },
            timeline=timeline,
            frame_counts=frame_counts,
            output_video_path=result.output_video_path,
            output_video_base64=result.output_video_base64
        )
        
        logger.info(f"Person counting completed. Job ID: {job_id}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Person counting failed: {e}")
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Person counting failed: {str(e)}"
        )


@app.post("/person_count/upload", response_model=PersonCountResponse)
async def count_persons_from_upload(
    file: UploadFile = File(..., description="Video file to analyze"),
    confidence_threshold: float = Form(0.5, description="YOLO detection confidence"),
    frame_sample_rate: int = Form(1, description="Process every Nth frame"),
    return_video: bool = Form(True, description="Generate annotated output video"),
    return_base64: bool = Form(False, description="Return video as base64")
):
    """
    Count persons in uploaded video using YOLO
    
    - **file**: Video file (MP4, AVI, MOV, etc.)
    - **confidence_threshold**: YOLO detection confidence (0.0-1.0)
    - **frame_sample_rate**: Process every Nth frame (1-30)
    - **return_video**: Whether to generate annotated output video
    - **return_base64**: Return video as base64 string
    
    Returns:
    - Person count per frame
    - Timeline with max/min/avg counts
    - Annotated video with bounding boxes (optional)
    """
    if not YOLO_AVAILABLE or get_yolo_counter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="YOLO Person Counter is not available. Install with: pip install ultralytics"
        )
    
    temp_video_path = None
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a video"
            )
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_video_path = temp_file.name
            contents = await file.read()
            temp_file.write(contents)
            temp_file.flush()
        
        logger.info(f"Video uploaded and saved to: {temp_video_path}")
        
        # Get YOLO counter
        counter = get_yolo_counter(confidence_threshold=confidence_threshold)
        if counter is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to initialize YOLO counter"
            )
        
        # Process video
        result = counter.process_video(
            video_path=temp_video_path,
            output_video=return_video,
            frame_sample_rate=frame_sample_rate,
            return_base64=return_base64
        )
        
        # Clean up input video
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        
        # Build timeline
        timeline = PersonCountTimeline(
            timestamps=[fc.timestamp for fc in result.frame_counts],
            counts=[fc.person_count for fc in result.frame_counts],
            max_count=result.max_persons,
            min_count=result.min_persons,
            avg_count=result.avg_persons,
            duration=result.duration
        )
        
        # Build frame counts (limit to avoid huge responses)
        frame_counts = []
        sample_interval = max(1, len(result.frame_counts) // 100)
        for i, fc in enumerate(result.frame_counts):
            if i % sample_interval == 0:
                detections = []
                for det in fc.detections:
                    detections.append(PersonDetectionResult(
                        person_id=det.person_id,
                        confidence=det.confidence,
                        bounding_box=PersonBoundingBox(
                            x1=det.bbox[0],
                            y1=det.bbox[1],
                            x2=det.bbox[2],
                            y2=det.bbox[3],
                            center_x=det.center[0],
                            center_y=det.center[1]
                        )
                    ))
                frame_counts.append(FramePersonCount(
                    frame_idx=fc.frame_idx,
                    timestamp=fc.timestamp,
                    person_count=fc.person_count,
                    detections=detections
                ))
        
        job_id = f"yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        response_data = PersonCountResponse(
            success=True,
            job_id=job_id,
            video_path=file.filename,
            timestamp=datetime.now().isoformat(),
            video_info={
                "total_frames": result.total_frames,
                "fps": result.fps,
                "duration": result.duration
            },
            statistics={
                "max_persons": result.max_persons,
                "min_persons": result.min_persons,
                "avg_persons": round(result.avg_persons, 2),
                "frames_analyzed": len(result.frame_counts)
            },
            timeline=timeline,
            frame_counts=frame_counts,
            output_video_path=result.output_video_path,
            output_video_base64=result.output_video_base64
        )
        
        logger.info(f"Person counting completed. Job ID: {job_id}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Person counting upload failed: {e}")
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Person counting failed: {str(e)}"
        )


@app.get("/person_count/status")
async def get_person_count_status():
    """Get YOLO person counting system status"""
    return {
        "yolo_available": YOLO_AVAILABLE,
        "model_info": {
            "name": "YOLOv8n (nano)",
            "type": "Object Detection",
            "class": "Person (COCO class 0)"
        } if YOLO_AVAILABLE else None,
        "supported_formats": ["mp4", "avi", "mov", "mkv", "webm"],
        "features": {
            "person_detection": True,
            "bounding_boxes": True,
            "confidence_scores": True,
            "annotated_video_output": True,
            "base64_video_output": True,
            "timeline_statistics": True
        }
    }


# ============================================================================
# Grounding DINO Object Detection Endpoints
# ============================================================================

@app.post("/detect", response_model=DetectionResponse)
async def detect_objects_from_url(request: DetectionRequest):
    """
    Detect objects in image from URL using Grounding DINO
    
    - **image_url**: URL of the image to analyze
    - **text_queries**: List of text descriptions of objects to detect
    - **box_threshold**: Confidence threshold for bounding boxes (0.0 to 1.0)
    - **text_threshold**: Confidence threshold for text matching (0.0 to 1.0)
    - **return_visualization**: Whether to return visualization image as base64
    """
    if not GROUNDING_DINO_AVAILABLE or grounding_dino_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Grounding DINO model is not available. Check model.py module."
        )
    
    try:
        # Process detection
        result = grounding_dino_model.process_detection(
            image_source=request.image_url,
            text_queries=request.text_queries,
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold,
            return_visualization=request.return_visualization
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Detection failed")
            )
        
        # Convert to response format
        detections = []
        for det in result.get("detections", []):
            detections.append(Detection(
                id=det["id"],
                label=det["label"],
                confidence=det["confidence"],
                bounding_box=BoundingBox(**det["bounding_box"])
            ))
        
        response_data = DetectionResponse(
            success=True,
            num_detections=result["num_detections"],
            detections=detections,
            image_size=ImageSize(**result["image_size"]) if result.get("image_size") else None,
            queries=result.get("queries"),
            thresholds=DetectionThresholds(**result["thresholds"]) if result.get("thresholds") else None,
            visualization=Visualization(**result["visualization"]) if result.get("visualization") else None
        )
        
        logger.info(f"Object detection completed. Found {result['num_detections']} objects.")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Object detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Object detection failed: {str(e)}"
        )


@app.post("/detect/upload", response_model=DetectionResponse)
async def detect_objects_from_upload(
    file: UploadFile = File(..., description="Image file to analyze"),
    text_queries: str = Form(..., description="Comma-separated text queries for object detection"),
    box_threshold: float = Form(0.4, description="Confidence threshold for bounding boxes"),
    text_threshold: float = Form(0.3, description="Confidence threshold for text matching"),
    return_visualization: bool = Form(True, description="Whether to return visualization image")
):
    """
    Detect objects in uploaded image file using Grounding DINO
    
    - **file**: Image file (JPEG, PNG, etc.)
    - **text_queries**: Comma-separated text descriptions of objects to detect
    - **box_threshold**: Confidence threshold for bounding boxes (0.0 to 1.0)
    - **text_threshold**: Confidence threshold for text matching (0.0 to 1.0)
    - **return_visualization**: Whether to return visualization image as base64
    """
    if not GROUNDING_DINO_AVAILABLE or grounding_dino_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Grounding DINO model is not available. Check model.py module."
        )
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Read image file
        contents = await file.read()
        
        # Parse text queries
        queries_list = [q.strip() for q in text_queries.split(",") if q.strip()]
        if not queries_list:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one text query is required"
            )
        
        # Validate thresholds
        if not (0.0 <= box_threshold <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="box_threshold must be between 0.0 and 1.0"
            )
        
        if not (0.0 <= text_threshold <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="text_threshold must be between 0.0 and 1.0"
            )
        
        # Process detection
        result = grounding_dino_model.process_detection(
            image_source=contents,
            text_queries=queries_list,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            return_visualization=return_visualization
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Detection failed")
            )
        
        # Convert to response format
        detections = []
        for det in result.get("detections", []):
            detections.append(Detection(
                id=det["id"],
                label=det["label"],
                confidence=det["confidence"],
                bounding_box=BoundingBox(**det["bounding_box"])
            ))
        
        response_data = DetectionResponse(
            success=True,
            num_detections=result["num_detections"],
            detections=detections,
            image_size=ImageSize(**result["image_size"]) if result.get("image_size") else None,
            queries=result.get("queries"),
            thresholds=DetectionThresholds(**result["thresholds"]) if result.get("thresholds") else None,
            visualization=Visualization(**result["visualization"]) if result.get("visualization") else None
        )
        
        logger.info(f"Object detection completed. Found {result['num_detections']} objects.")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload object detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Object detection failed: {str(e)}"
        )


@app.get("/detect/status")
async def get_detect_status():
    """Get Grounding DINO object detection system status"""
    return {
        "grounding_dino_available": GROUNDING_DINO_AVAILABLE,
        "model_loaded": grounding_dino_model is not None,
        "supported_formats": ["jpg", "jpeg", "png", "webp", "bmp"],
        "features": {
            "zero_shot_detection": True,
            "text_guided_detection": True,
            "bounding_boxes": True,
            "visualization": True,
            "thai_language_support": True
        },
        "model_info": {
            "model_id": "rziga/mm_grounding_dino_large_all",
            "device": grounding_dino_model.device if grounding_dino_model else None
        } if GROUNDING_DINO_AVAILABLE else None
    }


# ============================================================================
# Factory Alert Detection Endpoints
# ============================================================================

class FactoryAlertModel(BaseModel):
    """Factory alert model for API response"""
    alert_id: str
    alert_type: str
    severity: str
    zone_id: str
    title: str
    description: str
    person_count: int
    action_detected: str
    confidence: float
    detected_at: str
    frame_timestamp: float
    recommended_action: str
    metadata: Dict[str, Any] = {}


class FactoryAnalysisSummary(BaseModel):
    """Summary of factory analysis"""
    total_frames: int
    frames_analyzed: int
    average_person_count: float
    action_distribution: Dict[str, int]
    alert_counts: Dict[str, int]
    critical_alerts: int
    high_alerts: int
    medium_alerts: int
    low_alerts: int


class FactoryAnalysisRequest(BaseModel):
    """Request model for factory video analysis"""
    video_url: str = Field(..., description="URL of the video to analyze")
    zone_id: Optional[str] = Field("default_zone", description="Zone identifier for alerts")
    actions_to_detect: Optional[List[str]] = Field(
        None, 
        description="Specific actions to detect: running, idle, falling, fighting, sleeping, working, standing, sitting"
    )
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence for alerts")
    frame_sample_rate: Optional[int] = Field(1, ge=1, le=10, description="Frames to analyze per second")
    person_count_threshold: Optional[int] = Field(5, ge=1, description="Max persons before crowding alert")
    ollama_url: Optional[str] = Field("http://localhost:11434", description="Ollama API URL")
    model_name: Optional[str] = Field("qwen2.5-vl", description="Qwen VL model name")


class FactoryAnalysisResponse(BaseModel):
    """Response model for factory video analysis"""
    success: bool
    job_id: str
    video_duration: float
    frames_analyzed: int
    total_persons_detected: int
    alerts: List[FactoryAlertModel]
    summary: FactoryAnalysisSummary
    timestamp: str
    error: Optional[str] = None


@app.post("/factory/analyze", response_model=FactoryAnalysisResponse)
async def analyze_factory_video_from_url(request: FactoryAnalysisRequest):
    """
    Analyze factory video and generate structured alerts
    
    - **video_url**: URL of the video to analyze
    - **zone_id**: Zone identifier for alerts (e.g., 'assembly_line_1', 'warehouse_a')
    - **actions_to_detect**: List of actions to look for (running, idle, falling, etc.)
    - **confidence_threshold**: Minimum confidence to generate alerts
    - **frame_sample_rate**: Frames to analyze per second (1-10)
    - **person_count_threshold**: Max persons before generating crowding alert
    """
    if not FACTORY_ALERT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Factory Alert Engine is not available. Check factory_alert_engine.py module."
        )
    
    temp_video_path = None
    try:
        # Download video from URL
        logger.info(f"Downloading video from URL: {request.video_url}")
        temp_video_path, video_info = download_video_from_url(request.video_url)
        logger.info(f"Video downloaded: {video_info.get('title', 'Unknown')}")
        
        # Create Factory Alert Engine
        engine = FactoryAlertEngine(
            ollama_url=request.ollama_url,
            model_name=request.model_name,
            confidence_threshold=request.confidence_threshold,
            frame_sample_rate=request.frame_sample_rate,
            person_count_threshold=request.person_count_threshold,
            zone_id=request.zone_id
        )
        
        # Analyze video
        result = engine.analyze_video(
            video_path=temp_video_path,
            zone_id=request.zone_id,
            actions_to_detect=request.actions_to_detect
        )
        
        # Clean up temporary file
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        
        # Convert to response format
        from dataclasses import asdict
        
        alerts = [FactoryAlertModel(**asdict(a)) for a in result.alerts]
        summary = FactoryAnalysisSummary(**result.summary)
        
        response_data = FactoryAnalysisResponse(
            success=True,
            job_id=result.job_id,
            video_duration=result.video_duration,
            frames_analyzed=result.frames_analyzed,
            total_persons_detected=result.total_persons_detected,
            alerts=alerts,
            summary=summary,
            timestamp=result.timestamp
        )
        
        logger.info(f"Factory analysis completed. Job ID: {result.job_id}, Alerts: {len(alerts)}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Factory analysis failed: {e}")
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Factory analysis failed: {str(e)}"
        )


@app.post("/factory/analyze/upload", response_model=FactoryAnalysisResponse)
async def analyze_factory_video_from_upload(
    file: UploadFile = File(..., description="Video file to analyze"),
    zone_id: str = Form("default_zone", description="Zone identifier for alerts"),
    actions_to_detect: str = Form(
        "", 
        description="Comma-separated actions to detect: running, idle, falling, fighting, sleeping, working"
    ),
    confidence_threshold: float = Form(0.5, description="Minimum confidence for alerts"),
    frame_sample_rate: int = Form(1, description="Frames to analyze per second"),
    person_count_threshold: int = Form(5, description="Max persons before crowding alert"),
    ollama_url: str = Form("http://localhost:11434", description="Ollama API URL"),
    model_name: str = Form("qwen2.5-vl", description="Qwen VL model name")
):
    """
    Analyze uploaded factory video and generate structured alerts
    
    Returns structured JSON alerts for:
    - Safety alerts (running, falling, fighting)
    - Efficiency alerts (idle, working patterns)
    - Capacity alerts (overcrowding)
    - Compliance alerts (sleeping, violations)
    """
    if not FACTORY_ALERT_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Factory Alert Engine is not available. Check factory_alert_engine.py module."
        )
    
    temp_video_path = None
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a video"
            )
        
        # Parse actions to detect
        if actions_to_detect.strip():
            actions_list = [a.strip() for a in actions_to_detect.split(",") if a.strip()]
        else:
            actions_list = None  # Detect all actions
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_video_path = temp_file.name
            contents = await file.read()
            temp_file.write(contents)
            temp_file.flush()
        
        logger.info(f"Video uploaded and saved to: {temp_video_path}")
        
        # Create Factory Alert Engine
        engine = FactoryAlertEngine(
            ollama_url=ollama_url,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            frame_sample_rate=frame_sample_rate,
            person_count_threshold=person_count_threshold,
            zone_id=zone_id
        )
        
        # Analyze video
        result = engine.analyze_video(
            video_path=temp_video_path,
            zone_id=zone_id,
            actions_to_detect=actions_list
        )
        
        # Clean up temporary file
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        
        # Convert to response format
        from dataclasses import asdict
        
        alerts = [FactoryAlertModel(**asdict(a)) for a in result.alerts]
        summary = FactoryAnalysisSummary(**result.summary)
        
        response_data = FactoryAnalysisResponse(
            success=True,
            job_id=result.job_id,
            video_duration=result.video_duration,
            frames_analyzed=result.frames_analyzed,
            total_persons_detected=result.total_persons_detected,
            alerts=alerts,
            summary=summary,
            timestamp=result.timestamp
        )
        
        logger.info(f"Factory analysis completed. Job ID: {result.job_id}, Alerts: {len(alerts)}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Factory upload analysis failed: {e}")
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Factory analysis failed: {str(e)}"
        )


@app.get("/factory/status")
async def get_factory_status():
    """Get Factory Alert Engine status"""
    ollama_status = "unknown"
    available_models = []
    
    # Check Ollama connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            ollama_status = "connected"
            models = response.json().get('models', [])
            available_models = [m.get('name', '') for m in models]
        else:
            ollama_status = "error"
    except requests.exceptions.ConnectionError:
        ollama_status = "disconnected"
    except Exception as e:
        ollama_status = f"error: {str(e)}"
    
    return {
        "factory_alert_available": FACTORY_ALERT_AVAILABLE,
        "ollama_status": ollama_status,
        "available_models": available_models,
        "supported_actions": [
            "running", "falling", "idle", "standing", "sitting", 
            "fighting", "sleeping", "working", "crowded"
        ],
        "alert_types": ["safety", "efficiency", "quality", "capacity", "compliance", "maintenance"],
        "severity_levels": ["critical", "high", "medium", "low"],
        "supported_formats": ["mp4", "avi", "mov", "mkv", "webm"]
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Action Detection API")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info("Starting Video Action Detection API...")
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )