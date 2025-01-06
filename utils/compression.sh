#!/bin/bash

# Directories
IMAGES_DIR="./images"
COMPRESSED_DIR="./compressed"
LOGS_DIR="./logs"

# Create compressed and logs directories if they don't exist
mkdir -p "$COMPRESSED_DIR"
mkdir -p "$LOGS_DIR"

# Loop through each image in the images directory
for IMAGE in "$IMAGES_DIR"/*; do
    # Get the base name of the image file
    BASENAME=$(basename "$IMAGE")
    
    # Define the recompressed image path
    RECOMPRESSED_IMAGE="$COMPRESSED_DIR/recompressed_$BASENAME"

    # Compress the image to 2MB
    convert "$IMAGE" -define jpeg:extent=2MB "$RECOMPRESSED_IMAGE"
    
    # Define the log file path
    LOG_FILE="$LOGS_DIR/psnr_${BASENAME%.*}.log"
    
    # Calculate PSNR and log the stats
    ffmpeg -i "$IMAGE" -i "$RECOMPRESSED_IMAGE" -lavfi psnr="stats_file=$LOG_FILE" -f null -
done

echo "Compression and logging completed."