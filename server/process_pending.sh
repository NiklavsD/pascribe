#!/bin/bash
# Check for pending transcript triggers
PENDING_DIR="/home/nik/clawd/projects/pascribe/server/pending"
if [ ! -d "$PENDING_DIR" ] || [ -z "$(ls -A "$PENDING_DIR" 2>/dev/null)" ]; then
    exit 0
fi
# Output pending files for OpenClaw to pick up
for f in "$PENDING_DIR"/*.trigger; do
    [ -f "$f" ] && echo "PENDING: $(cat "$f")"
done
