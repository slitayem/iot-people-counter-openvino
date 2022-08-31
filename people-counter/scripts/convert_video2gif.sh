#!/bin/sh
# Convert video file to GIF format
#  e.g:  ./convert_video2gif.sh input_video.mov output_filename.gif 720 30

INPUT_FILE=$1
OUTPUT_FILE=$2
SCALE=$3
FPS=$4
filters="fps=$FPS,scale=$SCALE:-1:flags=lanczos"

palette="/tmp/palette.png"
filters="fps=$FPS,scale=$SCALE:-1:flags=lanczos"
echo "Converting to video to GIF format ..."
ffmpeg -v warning -i "$INPUT_FILE" -vf "$filters,palettegen" -y "$palette"
ffmpeg -v warning -i "$INPUT_FILE" -i $palette -lavfi "$filters [x]; [x][1:v] paletteuse" -y "$OUTPUT_FILE"

# Compress the image
echo "Compressing the GIF image ..."
mogrify -layers "optimize" -fuzz 7% "$OUTPUT_FILE"