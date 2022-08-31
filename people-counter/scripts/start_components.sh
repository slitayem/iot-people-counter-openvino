#!/bin/bash
set -o pipefail

# Script for starting all the components of the application

create_dir_if_not_exist(){
	DIR=$1
	if [ ! -d "$DIR" ]; then
		mkdir $DIR
fi
}

cecho(){
	RED="\033[0;31m"
	GREEN="\033[0;32m"
	YELLOW="\033[1;33m"
	NC="\033[0m" # No Color
	CYAN="\033[36m"
	printf "${!1}${2} ${NC}\n"
}

cecho "CYAN" "Starting all the application components: node server, web ui, ffmpeg server ..."

WORKSPACE="/home/openvino/app/people-counter"
LOGS_DIR="/home/openvino/app/logs"

create_dir_if_not_exist $LOGS_DIR

pushd $WORKSPACE/webservice/server/node-server
nohup node ./server.js > ${LOGS_DIR}/mqtt-server.out 2> \
	$LOGS_DIR/mqtt-server.err < /dev/null &
popd

pushd $WORKSPACE/webservice/ui
yes | nohup npm run dev  > ${LOGS_DIR}/webservice-ui.out 2> \
	$LOGS_DIR/webservice-ui.err < /dev/null &
popd

pushd $WORKSPACE
nohup ffserver -f ./ffmpeg/server.conf  > ${LOGS_DIR}/ffserver.out 2> \
	$LOGS_DIR/ffserver.err < /dev/null &
popd