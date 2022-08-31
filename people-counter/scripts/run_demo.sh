#!/bin/bash
set -o pipefail

cecho(){
	# colored echo function
	RED="\033[0;31m"
	GREEN="\033[0;32m"
	YELLOW="\033[1;33m"
	NC="\033[0m" # No Color
	CYAN="\033[36m"
	printf "${!1}${2} ${NC}\n"
}

WORKSPACE=${HOME}/app
SRC_DIR=${WORKSPACE}/people-counter
# current script folder
DIR="$(dirname "${BASH_SOURCE[0]}")"
$DIR/start_components.sh
sleep 5

MODEL_NAME="pedestrian-detection-adas-0002"
DATA_TYPE="FP32"
DEVICE="CPU"
THRESHOLD="0.8"

cecho "GREEN" "Running the people counter service ..."
python3 $SRC_DIR/src/main.py -i ${WORKSPACE}/resources/Pedestrian_Detect_2_1_1.mp4 \
	-m ${WORKSPACE}/models/$MODEL_NAME/$DATA_TYPE/$MODEL_NAME.xml -d $DEVICE \
	-pt $THRESHOLD -n $MODEL_NAME -p $DATA_TYPE | \
	ffmpeg -v warning -f rawvideo -pixel_format rgb24 \
	-video_size 768x432 -framerate 24 -i - -threads 2 -c copy  \
	http://0.0.0.0:3004/fac.ffm
