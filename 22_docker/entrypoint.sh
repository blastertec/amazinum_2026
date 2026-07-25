#!/bin/sh
set -e

# first argument decides what runs (default "serve" comes from CMD)
case "$1" in
  serve)
    exec uvicorn main:app --host 0.0.0.0 --port 8000
    ;;
  forecast)
    shift
    exec python cli.py "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
