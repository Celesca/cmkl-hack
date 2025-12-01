# VALZ - Video-based AI for Low-cost Zero-shot Processing

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A comprehensive video and image AI analysis API featuring multiple state-of-the-art models for zero-shot detection, action recognition, and factory monitoring.**

</div>

## 🚀 Features

VALZ provides a unified FastAPI server with multiple AI capabilities:

| Feature | Model | Description |
|---------|-------|-------------|
| **Object Detection** | Grounding DINO | Zero-shot text-guided object detection |
| **Video Action Detection** | BLIP + Sentence Transformers | Semantic action recognition in videos |
| **Vision-Language Analysis** | Qwen3-VL (via Ollama) | Advanced VLM-based video understanding |
| **Person Counting** | YOLOv8 Nano | Fast real-time person detection and tracking |
| **Factory Alerts** | Multi-model Pipeline | Smart factory monitoring with structured alerts |

## 📋 API Endpoints

### 🎯 Grounding DINO Object Detection
- `POST /detect` - Detect objects in image from URL
- `POST /detect/upload` - Detect objects in uploaded image
- `GET /detect/status` - Get Grounding DINO system status

### 🧠 BLIP Video Action Detection
- `POST /video_action/detect` - Detect actions in video from URL
- `POST /video_action/detect/upload` - Detect actions in uploaded video
- `GET /video_action/status` - Get BLIP detector status

### 🤖 Qwen3-VL (Ollama) Analysis
- `POST /qwen/detect` - Analyze video from URL using Qwen3-VL
- `POST /qwen/detect/upload` - Analyze uploaded video using Qwen3-VL
- `GET /qwen/status` - Get Qwen3-VL and Ollama status

### 👥 YOLO Person Counting
- `POST /person_count/detect` - Count persons in video from URL
- `POST /person_count/upload` - Count persons in uploaded video
- `GET /person_count/status` - Get YOLO person counting status

### 🏭 Factory Alert Engine
- `POST /factory/analyze` - Analyze factory video from URL
- `POST /factory/analyze/upload` - Analyze uploaded factory video
- `GET /factory/status` - Get Factory Alert Engine status

### 🔧 System Endpoints
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API Documentation (Swagger UI)
- `GET /redoc` - API Documentation (ReDoc)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)
- [Ollama](https://ollama.ai/) (for Qwen3-VL features)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Celesca/cmkl-hack.git
   cd cmkl-hack
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install Ollama and Qwen model**
   ```bash
   # Install Ollama from https://ollama.ai/
   ollama pull qwen2.5-vl
   ```

## 🚀 Quick Start

### Start the Server
```bash
python server.py --port 8000
```

Or with auto-reload for development:
```bash
python server.py --port 8000 --reload
```

### Access the API
- **API Root**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📖 Usage Examples

### Object Detection (Grounding DINO)
```bash
curl -X POST "http://localhost:8000/detect" \
     -H "Content-Type: application/json" \
     -d '{"image_url": "https://example.com/image.jpg", "text_queries": ["person", "car", "dog"]}'
```

### Object Detection (Upload)
```bash
curl -X POST "http://localhost:8000/detect/upload" \
     -F "file=@your_image.jpg" \
     -F "text_queries=person, car, dog"
```

### Video Action Detection (BLIP)
```bash
curl -X POST "http://localhost:8000/video_action/detect/upload" \
     -F "file=@your_video.mp4" \
     -F "prompt=person running"
```

### Video Action Detection (Qwen3-VL)
```bash
curl -X POST "http://localhost:8000/qwen/detect/upload" \
     -F "file=@your_video.mp4" \
     -F "action_prompt=running"
```

### Person Counting (YOLO)
```bash
curl -X POST "http://localhost:8000/person_count/upload" \
     -F "file=@your_video.mp4" \
     -F "confidence_threshold=0.5" \
     -F "return_video=true"
```

### Factory Alert Detection
```bash
curl -X POST "http://localhost:8000/factory/analyze/upload" \
     -F "file=@factory_video.mp4" \
     -F "zone_id=assembly_line_1" \
     -F "actions_to_detect=running,idle,falling"
```

## 🏭 Factory Alert Types

The Factory Alert Engine can detect and generate alerts for:

| Alert Type | Severity | Actions Detected |
|------------|----------|------------------|
| **Safety** | Critical/High | Running, Falling, Fighting |
| **Efficiency** | Medium | Idle, Working patterns |
| **Capacity** | High | Overcrowding |
| **Compliance** | Medium/Low | Sleeping, Violations |

## 📁 Project Structure

```
VALZ/
├── server.py                 # Main FastAPI server
├── model.py                  # Grounding DINO model manager
├── video_action_model.py     # BLIP-based action detection
├── qwen3_vl.py               # Qwen3-VL action detector
├── qwen3_vl_obdetection.py   # Qwen3-VL object detection
├── factory_alert_engine.py   # Factory monitoring engine
├── yolo_person_counter.py    # YOLOv8 person counting
├── yolov8n.pt                # YOLOv8 nano weights
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## ⚙️ Configuration

### BLIP Action Detection Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `person_weight` | 0.2 | Weight for person detection component |
| `action_weight` | 0.7 | Weight for action detection component |
| `context_weight` | 0.1 | Weight for context detection component |
| `similarity_threshold` | 0.5 | Overall similarity threshold |
| `action_threshold` | 0.4 | Action-specific threshold |

### Qwen3-VL Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.5 | Minimum confidence threshold |
| `frame_sample_rate` | 1 | Frames to analyze per second |
| `ollama_url` | http://localhost:11434 | Ollama API URL |
| `model_name` | qwen2.5-vl | Qwen VL model name |

### YOLO Person Counting Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.5 | YOLO detection confidence |
| `frame_sample_rate` | 1 | Process every Nth frame |
| `return_video` | true | Generate annotated output video |

## 🔌 Supported Formats

- **Video**: MP4, AVI, MOV, MKV, WebM, FLV, WMV
- **Image**: JPEG, PNG, WebP, BMP

## 📊 Model Information

| Model | Source | Purpose |
|-------|--------|---------|
| Grounding DINO | `rziga/mm_grounding_dino_large_all` | Zero-shot object detection |
| BLIP | `Salesforce/blip-image-captioning-large` | Image captioning |
| Sentence Transformers | `all-MiniLM-L6-v2` | Semantic similarity |
| YOLOv8 Nano | Ultralytics | Fast person detection |
| Qwen2.5-VL | Ollama | Vision-language understanding |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Model hub
- [Ultralytics](https://ultralytics.com/) - YOLOv8 implementation
- [Ollama](https://ollama.ai/) - Local LLM serving
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) - Zero-shot detection

---

<div align="center">
  Made with ❤️ for the CMKL Hackathon
</div>
