#!/bin/bash
export PYTHONPATH=$PYTHONPATH:src


# Default values
PORT=8000
HOST="0.0.0.0"
RELOAD="--reload"

# Start the server
echo -e "Running server on http://$HOST:$PORT"
uvicorn src.gender_detection.api.main:app --host $HOST --port $PORT $RELOAD