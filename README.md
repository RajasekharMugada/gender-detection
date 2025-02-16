# Gender Detection from Live Audio

A Python application that detects gender from live audio input using deep learning.

## Prerequisites

### System Dependencies

For Linux (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev
```

For macOS:
```bash
brew install portaudio
```

### Python Dependencies

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip pip install -e .
```

## Usage

### Running the Application

```bash
./run_server.sh
```

Or directly using uvicorn:

```bash
uvicorn src.gender_detection.api.main:app --reload
```

The server will start at http://localhost:8000 and you can access:
API docs at http://localhost:8000/docs
Health check at http://localhost:8000/api/v1/health
Gender detection endpoint at http://localhost:8000/api/v1/detect-gender

### Running Tests

```bash
pytest tests/
```
