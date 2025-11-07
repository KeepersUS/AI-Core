#!/bin/bash
# Unix/Linux shell script to run the annotation tool
# Make executable with: chmod +x run_annotation_tool.sh

echo "Starting Image Annotation Tool..."
echo

python3 annotation_tool.py

# Keep terminal open on error
if [ $? -ne 0 ]; then
    echo "Press Enter to close..."
    read
fi

