#!/usr/bin/env bash

# Health
curl http://127.0.0.1:8000/health

# Search
curl -X POST "http://127.0.0.1:8000/search?top_k=10&unique_originals=true" \
  -F "file=@query.jpg"
