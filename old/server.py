from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import uvicorn
import io
import base64
from PIL import Image
import logging
import os
import sys
from datetime import datetime
import tempfile
import shutil
import requests

from model import ModelManager, DynamicGroundingDINO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [PID:%(process)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Import video action detection model
try:
    from video_action_model import ActionDetector
    from youtube_downloader import download_video_from_url
    VIDEO_ACTION_AVAILABLE = True
    logger.info("Video action detection model and video downloader imported successfully")
except ImportError as e:
    VIDEO_ACTION_AVAILABLE = False
    ActionDetector = None
    download_video_from_url = None
    logger.warning(f"Video action detection not available: {e}")
    logger.warning("This may be due to missing dependencies. Check that opencv-python, sentence-transformers, scikit-learn, and spacy are installed.")
except Exception as e:
    VIDEO_ACTION_AVAILABLE = False
    ActionDetector = None
    logger.error(f"Unexpected error importing video action model: {e}")
    logger.error("This may be due to missing system dependencies or model initialization issues.")

# Worker process identification
WORKER_ID = os.getpid()
logger.info(f"Worker process started with PID: {WORKER_ID}")

# Check if queue mode is enabled
ENABLE_QUEUE = os.getenv("ENABLE_QUEUE", "true").lower() == "true"
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "team06-mq")

# Import queue worker only if queue is enabled
task_manager = None
TaskManager = None

if ENABLE_QUEUE:
    try:
        from queue_worker_rabbitmq import task_manager, TaskManager
        logger.info("RabbitMQ queue worker imported successfully")
    except ImportError as e:
        logger.warning(f"Failed to import RabbitMQ queue worker: {e}. Queue functionality will be disabled.")
        ENABLE_QUEUE = False

# Initialize FastAPI app
app = FastAPI(
    title="DynamicGroundingDINO & Video Action Detection API",
    description="Zero-shot object detection and video action recognition API with RabbitMQ queue support for production scaling",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model manager
model_manager = ModelManager()

# Global video action detector manager
video_action_detector = None

if VIDEO_ACTION_AVAILABLE:
    try:
        logger.info("Attempting to initialize video action detector...")
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
        logger.error("This may be due to missing model files or insufficient memory")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        video_action_detector = None
else:
    logger.warning("Video action detector not available - skipping initialization")


# Pydantic models for request/response
class DetectionRequest(BaseModel):
    """Request model for URL-based detection"""
    image_url: str = Field(..., description="URL of the image to analyze")
    text_queries: Union[str, List[str]] = Field(..., description="Text queries for object detection")
    box_threshold: Optional[float] = Field(0.4, ge=0.0, le=1.0, description="Confidence threshold for bounding boxes")
    text_threshold: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Confidence threshold for text matching")
    return_visualization: Optional[bool] = Field(True, description="Whether to return visualization image")
    async_processing: Optional[bool] = Field(False, description="Whether to process asynchronously using queue")
    priority: Optional[int] = Field(5, ge=0, le=9, description="Task priority (0-9, higher is more priority)")


class AsyncDetectionRequest(BaseModel):
    """Request model for async detection operations"""
    image_url: str = Field(..., description="URL of the image to analyze")
    text_queries: Union[str, List[str]] = Field(..., description="Text queries for object detection")
    box_threshold: Optional[float] = Field(0.4, ge=0.0, le=1.0, description="Confidence threshold for bounding boxes")
    text_threshold: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Confidence threshold for text matching")
    return_visualization: Optional[bool] = Field(True, description="Whether to return visualization image")
    priority: Optional[int] = Field(5, ge=0, le=9, description="Task priority (0-9, higher is more priority)")


class TaskSubmissionResponse(BaseModel):
    """Response for task submission"""
    task_id: str
    status: str
    message: str
    estimated_completion: Optional[str] = None


class QueueStatusResponse(BaseModel):
    """Response for queue status"""
    status: str
    active_tasks: int
    scheduled_tasks: int
    reserved_tasks: int
    workers: List[str]
    timestamp: str


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


class Thresholds(BaseModel):
    """Detection thresholds"""
    box_threshold: float
    text_threshold: float


class Visualization(BaseModel):
    """Visualization data"""
    image_base64: str
    format: str


class DetectionResponse(BaseModel):
    """Response model for detection results"""
    success: bool
    num_detections: int
    detections: List[Detection]
    image_size: Optional[ImageSize] = None
    queries: Optional[List[str]] = None
    thresholds: Optional[Thresholds] = None
    visualization: Optional[Visualization] = None
    error: Optional[str] = None


class TaskStatusResponse(BaseModel):
    """Response for task status check"""
    task_id: str
    status: str
    progress: Optional[int] = None
    stage: Optional[str] = None
    message: Optional[str] = None
    result: Optional[DetectionResponse] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    message: str


# Video Action Detection Models
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
    timeline_visualization: Optional[str] = None
    error: Optional[str] = None


class VideoActionUploadRequest(BaseModel):
    """Request model for video action detection upload parameters"""
    prompt: str
    person_weight: Optional[float] = 0.2
    action_weight: Optional[float] = 0.7
    context_weight: Optional[float] = 0.1
    similarity_threshold: Optional[float] = 0.5
    action_threshold: Optional[float] = 0.4
    return_timeline: Optional[bool] = True


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    try:
        logger.info(f"Worker {WORKER_ID}: Loading DynamicGroundingDINO model...")
        model_manager.get_model()
        logger.info(f"Worker {WORKER_ID}: Model loaded successfully!")
    except Exception as e:
        logger.error(f"Worker {WORKER_ID}: Failed to load model: {e}")
        raise e


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info(f"Worker {WORKER_ID}: Shutting down...")
    try:
        # Cleanup model resources if needed
        if model_manager.is_model_loaded():
            logger.info(f"Worker {WORKER_ID}: Cleaning up model resources...")
        logger.info(f"Worker {WORKER_ID}: Shutdown complete")
    except Exception as e:
        logger.error(f"Worker {WORKER_ID}: Error during shutdown: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    queue_status = "enabled" if (ENABLE_QUEUE and task_manager) else "disabled"
    queue_available = ENABLE_QUEUE and task_manager is not None
    
    # Prepare code examples (can't use backslashes directly in f-strings)
    sync_example = '''curl -X POST "http://localhost:8000/detect" \\
     -H "Content-Type: application/json" \\
     -d '{
       "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
       "text_queries": ["cat", "remote", "person"],
       "async_processing": false
     }\\' '''
    
    async_example = '''# Submit task
curl -X POST "http://localhost:8000/detect/async" \\
     -H "Content-Type: application/json" \\
     -d '{
       "image_url": "http://images.cocodataset.org/val2017/000000039769.jpg",
       "text_queries": ["cat", "remote", "person"],
       "priority": 7
     }\\' 

# Check task status
curl -X GET "http://localhost:8000/task/{task_id}"'''
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DynamicGroundingDINO API</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            .endpoint {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; background: #f5f5f5; }}
            .method {{ font-weight: bold; color: #007acc; }}
            .queue-status {{ padding: 10px; border-radius: 5px; margin: 15px 0; }}
            .queue-enabled {{ background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }}
            .queue-disabled {{ background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }}
            pre {{ background: #f0f0f0; padding: 10px; border-radius: 5px; }}
            .note {{ background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 10px; border-radius: 5px; margin: 15px 0; }}
        </style>
    </head>
    <body>
        <h1>🔍 DynamicGroundingDINO API v2.0</h1>
        <p>Zero-shot object detection API using Grounding DINO model with RabbitMQ queue support</p>
        
        <div class="queue-status queue-{'enabled' if queue_available else 'disabled'}">
            <strong>Queue Processing:</strong> {queue_status.upper()}
            {f'<br><small>RabbitMQ Host: {RABBITMQ_HOST}</small>' if queue_available else ''}
            {f'<br><small>⚠️ Queue worker module not available - using synchronous processing only</small>' if ENABLE_QUEUE and not task_manager else ''}
        </div>
        
        {'''<div class="note">
            <strong>Note:</strong> Queue functionality is disabled. Only synchronous processing is available.
            <br>For production deployment with queue support, ensure RabbitMQ is properly configured.
        </div>''' if not queue_available else ''}
        
        <h3>🚀 Detection Endpoints</h3>
        <div class="endpoint">
            <span class="method">POST</span> <strong>/detect</strong> - Detect objects from image URL (sync{'/async' if queue_available else ''})
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <strong>/detect/upload</strong> - Detect objects from uploaded image (sync{'/async' if queue_available else ''})
        </div>
        
        {f'''<div class="endpoint">
            <span class="method">POST</span> <strong>/detect/async</strong> - Submit async detection task
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <strong>/detect/async/upload</strong> - Submit async detection task with upload
        </div>''' if queue_available else ''}
        
        {f'''<h3>📊 Queue Management</h3>
        <div class="endpoint">
            <span class="method">GET</span> <strong>/task/{{task_id}}</strong> - Get task status and result
        </div>
        
        <div class="endpoint">
            <span class="method">DELETE</span> <strong>/task/{{task_id}}</strong> - Cancel task
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/queue/status</strong> - Get queue status and statistics
        </div>''' if queue_available else ''}
        
        <h3>🔧 System Endpoints</h3>
        <div class="endpoint">
            <span class="method">GET</span> <strong>/health</strong> - Check API health and model status
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/model/info</strong> - Get model information
        </div>
        
        <h3>📚 Documentation</h3>
        <ul>
            <li><a href="/docs">Interactive API Documentation (Swagger UI)</a></li>
            <li><a href="/redoc">API Documentation (ReDoc)</a></li>
        </ul>
        
        <h3>🚀 Quick Start Examples</h3>
        
        <h4>Synchronous Detection:</h4>
        <pre>{sync_example}</pre>
        
        {f'<h4>Asynchronous Detection:</h4><pre>{async_example}</pre>' if queue_available else ''}
    </body>
    </html>
    """
    return html_content


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model_loaded = model_manager.is_model_loaded()
        video_action_available = VIDEO_ACTION_AVAILABLE
        video_action_loaded = video_action_detector is not None
        
        # Log video action status for debugging
        logger.info(f"Health check - Video action available: {video_action_available}, loaded: {video_action_loaded}")
        
        if model_loaded:
            status_msg = f"API is running and model is loaded (Worker PID: {WORKER_ID})"
            if video_action_available and video_action_loaded:
                status_msg += " - Video action detection available"
            elif video_action_available and not video_action_loaded:
                status_msg += " - Video action detection failed to initialize"
            else:
                status_msg += " - Video action detection not available"
                
            return HealthResponse(
                status="healthy",
                model_loaded=True,
                message=status_msg
            )
        else:
            return HealthResponse(
                status="loading",
                model_loaded=False,
                message=f"API is running but model is still loading (Worker PID: {WORKER_ID})"
            )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            model_loaded=False,
            message=f"Health check failed: {str(e)} (Worker PID: {WORKER_ID})"
        )


@app.post("/detect", response_model=Union[DetectionResponse, TaskSubmissionResponse])
async def detect_objects_from_url(request: DetectionRequest):
    """
    Detect objects in image from URL - supports both sync and async processing
    
    - **image_url**: URL of the image to analyze
    - **text_queries**: Text descriptions of objects to detect (string or list of strings)
    - **box_threshold**: Confidence threshold for bounding boxes (0.0 to 1.0)
    - **text_threshold**: Confidence threshold for text matching (0.0 to 1.0)
    - **return_visualization**: Whether to return visualization image as base64
    - **async_processing**: Whether to process asynchronously using queue (if enabled)
    - **priority**: Task priority for async processing (0-9, higher is more priority)
    """
    try:
        # Check if async processing is requested and queue is enabled
        if request.async_processing and ENABLE_QUEUE and task_manager:
            # Submit to queue
            task_id = task_manager.submit_detection_task(
                image_data=request.image_url,
                image_type="url",
                text_queries=request.text_queries,
                box_threshold=request.box_threshold,
                text_threshold=request.text_threshold,
                return_visualization=request.return_visualization,
                priority=request.priority
            )
            
            return TaskSubmissionResponse(
                task_id=task_id,
                status="submitted",
                message="Task submitted for async processing",
                estimated_completion=None
            )
        
        # Synchronous processing
        if request.async_processing and (not ENABLE_QUEUE or not task_manager):
            logger.warning("Async processing requested but queue is disabled or unavailable, processing synchronously")
        
        # Get model instance
        model = model_manager.get_model()
        
        # Process detection
        result = model.process_detection(
            image_source=request.image_url,
            text_queries=request.text_queries,
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold,
            return_visualization=request.return_visualization
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
        return DetectionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )


@app.post("/detect/upload", response_model=Union[DetectionResponse, TaskSubmissionResponse])
async def detect_objects_from_upload(
    file: UploadFile = File(..., description="Image file to analyze"),
    text_queries: str = Form(..., description="Comma-separated text queries for object detection"),
    box_threshold: float = Form(0.4, description="Confidence threshold for bounding boxes"),
    text_threshold: float = Form(0.3, description="Confidence threshold for text matching"),
    return_visualization: bool = Form(True, description="Whether to return visualization image"),
    async_processing: bool = Form(False, description="Whether to process asynchronously using queue"),
    priority: int = Form(5, description="Task priority (0-9, higher is more priority)")
):
    """
    Detect objects in uploaded image file - supports both sync and async processing
    
    - **file**: Image file (JPEG, PNG, etc.)
    - **text_queries**: Comma-separated text descriptions of objects to detect
    - **box_threshold**: Confidence threshold for bounding boxes (0.0 to 1.0)
    - **text_threshold**: Confidence threshold for text matching (0.0 to 1.0)
    - **return_visualization**: Whether to return visualization image as base64
    - **async_processing**: Whether to process asynchronously using queue (if enabled)
    - **priority**: Task priority for async processing (0-9, higher is more priority)
    """
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
        
        # Check if async processing is requested and queue is enabled
        if async_processing and ENABLE_QUEUE and task_manager:
            # Submit to queue
            task_id = task_manager.submit_detection_task(
                image_data=contents,
                image_type="bytes",
                text_queries=queries_list,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                return_visualization=return_visualization,
                priority=priority
            )
            
            return TaskSubmissionResponse(
                task_id=task_id,
                status="submitted",
                message="Task submitted for async processing",
                estimated_completion=None
            )
        
        # Synchronous processing
        if async_processing and (not ENABLE_QUEUE or not task_manager):
            logger.warning("Async processing requested but queue is disabled or unavailable, processing synchronously")
        
        # Get model instance
        model = model_manager.get_model()
        
        # Process detection
        result = model.process_detection(
            image_source=contents,
            text_queries=queries_list,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            return_visualization=return_visualization
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
        return DetectionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )


@app.post("/detect/async", response_model=TaskSubmissionResponse)
async def submit_async_detection_url(request: AsyncDetectionRequest):
    """
    Submit async detection task for image URL
    
    - **image_url**: URL of the image to analyze
    - **text_queries**: Text descriptions of objects to detect
    - **box_threshold**: Confidence threshold for bounding boxes (0.0 to 1.0)
    - **text_threshold**: Confidence threshold for text matching (0.0 to 1.0)
    - **return_visualization**: Whether to return visualization image as base64
    - **priority**: Task priority (0-9, higher is more priority)
    """
    if not ENABLE_QUEUE or not task_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue processing is disabled or unavailable. Use synchronous endpoints instead."
        )
    
    try:
        task_id = task_manager.submit_detection_task(
            image_data=request.image_url,
            image_type="url",
            text_queries=request.text_queries,
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold,
            return_visualization=request.return_visualization,
            priority=request.priority
        )
        
        return TaskSubmissionResponse(
            task_id=task_id,
            status="submitted",
            message="Task submitted for async processing"
        )
        
    except Exception as e:
        logger.error(f"Failed to submit async task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit task: {str(e)}"
        )


@app.post("/detect/async/upload", response_model=TaskSubmissionResponse)
async def submit_async_detection_upload(
    file: UploadFile = File(..., description="Image file to analyze"),
    text_queries: str = Form(..., description="Comma-separated text queries for object detection"),
    box_threshold: float = Form(0.4, description="Confidence threshold for bounding boxes"),
    text_threshold: float = Form(0.3, description="Confidence threshold for text matching"),
    return_visualization: bool = Form(True, description="Whether to return visualization image"),
    priority: int = Form(5, description="Task priority (0-9, higher is more priority)")
):
    """
    Submit async detection task for uploaded image file
    
    - **file**: Image file (JPEG, PNG, etc.)
    - **text_queries**: Comma-separated text descriptions of objects to detect
    - **box_threshold**: Confidence threshold for bounding boxes (0.0 to 1.0)
    - **text_threshold**: Confidence threshold for text matching (0.0 to 1.0)
    - **return_visualization**: Whether to return visualization image as base64
    - **priority**: Task priority (0-9, higher is more priority)
    """
    if not ENABLE_QUEUE or not task_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue processing is disabled or unavailable. Use synchronous endpoints instead."
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
        
        task_id = task_manager.submit_detection_task(
            image_data=contents,
            image_type="bytes",
            text_queries=queries_list,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            return_visualization=return_visualization,
            priority=priority
        )
        
        return TaskSubmissionResponse(
            task_id=task_id,
            status="submitted",
            message="Task submitted for async processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit async upload task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit task: {str(e)}"
        )


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get status and result of a task
    
    - **task_id**: ID of the task to check
    """
    if not ENABLE_QUEUE or not task_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue processing is disabled or unavailable."
        )
    
    try:
        result = task_manager.get_task_result(task_id)
        
        # Convert result to response format
        response_data = {
            "task_id": task_id,
            "status": result["status"],
            "progress": result.get("progress"),
            "stage": result.get("stage"),
            "message": result.get("message"),
            "error": result.get("error")
        }
        
        # Add result if completed successfully
        if result["status"] == "completed" and "result" in result:
            response_data["result"] = DetectionResponse(**result["result"])
        
        return TaskStatusResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )


@app.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a task
    
    - **task_id**: ID of the task to cancel
    """
    if not ENABLE_QUEUE or not task_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue processing is disabled or unavailable."
        )
    
    try:
        result = task_manager.cancel_task(task_id)
        return result
        
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}"
        )


@app.get("/queue/status", response_model=QueueStatusResponse)
async def get_queue_status():
    """
    Get queue status and statistics
    """
    if not ENABLE_QUEUE or not task_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue processing is disabled or unavailable."
        )
    
    try:
        result = task_manager.get_queue_status()
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return QueueStatusResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue status: {str(e)}"
        )


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    try:
        model_loaded = model_manager.is_model_loaded()
        video_action_loaded = video_action_detector is not None
        
        info = {
            "grounding_dino": {
                "model_loaded": model_loaded,
                "worker_pid": WORKER_ID
            },
            "video_action": {
                "available": VIDEO_ACTION_AVAILABLE,
                "model_loaded": video_action_loaded
            },
            "worker_info": {
                "process_id": WORKER_ID,
                "python_version": sys.version.split()[0]
            }
        }
        
        if model_loaded:
            model = model_manager.get_model()
            info["grounding_dino"].update({
                "device": model.device,
                "model_id": "onnx-community/grounding-dino-tiny-ONNX"
            })
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


# ============================================================================
# VIDEO ACTION DETECTION ENDPOINTS
# ============================================================================

@app.post("/video_action/detect", response_model=VideoActionResponse)
async def detect_video_action_from_url(request: VideoActionRequest):
    """
    Detect actions in video from URL using BLIP and sentence transformers
    
    - **video_url**: URL of the video to analyze
    - **prompt**: Action description (e.g., 'person running', 'person jumping')
    - **person_weight**: Weight for person detection component (0.0-1.0)
    - **action_weight**: Weight for action detection component (0.0-1.0)
    - **context_weight**: Weight for context detection component (0.0-1.0)
    - **similarity_threshold**: Overall similarity threshold (0.0-1.0)
    - **action_threshold**: Action-specific threshold (0.0-1.0)
    - **return_timeline**: Whether to return timeline visualization
    """
    if not VIDEO_ACTION_AVAILABLE or video_action_detector is None:
        logger.error(f"Video action detection unavailable - VIDEO_ACTION_AVAILABLE: {VIDEO_ACTION_AVAILABLE}, video_action_detector: {video_action_detector is not None}")
        if not VIDEO_ACTION_AVAILABLE:
            logger.error("Video action module import failed - check dependencies")
        if video_action_detector is None:
            logger.error("Video action detector initialization failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Video action detection is not available. Required dependencies may be missing."
        )
    
    try:
        # Download video from URL (direct URLs only, no YouTube support)
        logger.info(f"Downloading video from URL: {request.video_url}")
        temp_video_path, video_info = download_video_from_url(request.video_url)
        logger.info(f"� Video downloaded: {video_info.get('title', 'Unknown')}")
        
        # Create temporary ActionDetector with custom weights
        temp_detector = ActionDetector(
            person_weight=request.person_weight,
            action_weight=request.action_weight,
            context_weight=request.context_weight,
            similarity_threshold=request.similarity_threshold,
            action_threshold=request.action_threshold
        )
        
        # Process video - no file saving needed
        results = temp_detector.process_video(
            video_path=temp_video_path,
            prompt=request.prompt,
            save_files=False
        )
        
        # Clean up temporary file
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
        
        logger.info(f"Video action detection completed successfully. Job ID: {results['job_id']}")
        return response_data
        
    except requests.RequestException as e:
        logger.error(f"Failed to download video: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download video from URL: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Video action detection failed: {e}")
        # Clean up temporary file if it exists
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video action detection failed: {str(e)}"
        )


@app.post("/video_action/detect", response_model=VideoActionResponse)
async def detect_video_action_from_url(request: VideoActionRequest):
    """
    Detect actions in video from URL using BLIP and sentence transformers
    
    - **video_url**: URL of the video to analyze
    - **prompt**: Action description (e.g., 'person running', 'person jumping')
    - **person_weight**: Weight for person detection component (0.0-1.0)
    - **action_weight**: Weight for action detection component (0.0-1.0)
    - **context_weight**: Weight for context detection component (0.0-1.0)
    - **similarity_threshold**: Overall similarity threshold (0.0-1.0)
    - **action_threshold**: Action-specific threshold (0.0-1.0)
    - **return_timeline**: Whether to return timeline visualization
    """
    if not VIDEO_ACTION_AVAILABLE or video_action_detector is None:
        logger.error(f"Video action detection unavailable - VIDEO_ACTION_AVAILABLE: {VIDEO_ACTION_AVAILABLE}, video_action_detector: {video_action_detector is not None}")
        if not VIDEO_ACTION_AVAILABLE:
            logger.error("Video action module import failed - check dependencies")
        if video_action_detector is None:
            logger.error("Video action detector initialization failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Video action detection is not available. Required dependencies may be missing."
        )
    
    try:
        # Validate parameters
        if not (0.0 <= request.person_weight <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="person_weight must be between 0.0 and 1.0"
            )
        
        if not (0.0 <= request.action_weight <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="action_weight must be between 0.0 and 1.0"
            )
        
        if not (0.0 <= request.context_weight <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="context_weight must be between 0.0 and 1.0"
            )
        
        if not (0.0 <= request.similarity_threshold <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="similarity_threshold must be between 0.0 and 1.0"
            )
        
        if not (0.0 <= request.action_threshold <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="action_threshold must be between 0.0 and 1.0"
            )
        
        # Download video from URL (direct URLs only, no YouTube support)
        logger.info(f"Downloading video from URL: {request.video_url}")
        temp_video_path, video_info = download_video_from_url(request.video_url)
        logger.info(f"� Video downloaded: {video_info.get('title', 'Unknown')}")
        
        # Create temporary ActionDetector with custom weights
        temp_detector = ActionDetector(
            person_weight=request.person_weight,
            action_weight=request.action_weight,
            context_weight=request.context_weight,
            similarity_threshold=request.similarity_threshold,
            action_threshold=request.action_threshold
        )
        
        # Process video - no file saving needed
        results = temp_detector.process_video(
            video_path=temp_video_path,
            prompt=request.prompt,
            save_files=False
        )
        
        # Clean up temporary file
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
        
        logger.info(f"Video action detection completed successfully. Job ID: {results['job_id']}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video action detection failed: {e}")
        # Clean up temporary file if it exists
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
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
        logger.error(f"Video action detection unavailable - VIDEO_ACTION_AVAILABLE: {VIDEO_ACTION_AVAILABLE}, video_action_detector: {video_action_detector is not None}")
        if not VIDEO_ACTION_AVAILABLE:
            logger.error("Video action module import failed - check dependencies")
        if video_action_detector is None:
            logger.error("Video action detector initialization failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Video action detection is not available. Required dependencies may be missing."
        )
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a video"
            )
        
        # Validate parameters
        if not (0.0 <= person_weight <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="person_weight must be between 0.0 and 1.0"
            )
        
        if not (0.0 <= action_weight <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="action_weight must be between 0.0 and 1.0"
            )
        
        if not (0.0 <= context_weight <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="context_weight must be between 0.0 and 1.0"
            )
        
        if not (0.0 <= similarity_threshold <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="similarity_threshold must be between 0.0 and 1.0"
            )
        
        if not (0.0 <= action_threshold <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="action_threshold must be between 0.0 and 1.0"
            )
        
        # Save uploaded file to temporary location with proper binary handling
        temp_video_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_video_path = temp_file.name
                # Read file content as bytes to avoid UTF-8 decoding issues
                contents = await file.read()
                temp_file.write(contents)
                temp_file.flush()  # Ensure data is written to disk
            
            logger.info(f"Video uploaded and saved to: {temp_video_path}")
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process uploaded file: {str(e)}"
            )
        
        # Create temporary ActionDetector with custom weights
        temp_detector = ActionDetector(
            person_weight=person_weight,
            action_weight=action_weight,
            context_weight=context_weight,
            similarity_threshold=similarity_threshold,
            action_threshold=action_threshold
        )
        
        # Process video - no file saving needed
        results = temp_detector.process_video(
            video_path=temp_video_path,
            prompt=prompt,
            save_files=False
        )
        
        # Clean up temporary file
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
        
        logger.info(f"Video action detection completed successfully. Job ID: {results['job_id']}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video upload action detection failed: {e}")
        # Clean up temporary file if it exists
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video action detection failed: {str(e)}"
        )

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
            "similarity_model": "all-MiniLM-L6-v2",
            "nlp_available": VIDEO_ACTION_AVAILABLE
        } if VIDEO_ACTION_AVAILABLE else None
    }


def create_app():
    """Application factory for use with Gunicorn"""
    return app


if __name__ == "__main__":
    # Development mode - single worker with auto-reload
    import argparse
    
    parser = argparse.ArgumentParser(description="DynamicGroundingDINO API")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (development only)")
    
    args = parser.parse_args()
    
    logger.info("Starting in development mode...")
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        workers=args.workers if not args.reload else 1  # Reload doesn't work with multiple workers
    )
