#!/bin/bash

FILE='/Users/wiktor/Downloads/05_Beyond_That_Hill'

echo $FILE

#desired clip length
LEN='29'

#find original clip duration
DURATION=$( ffprobe -i $FILE.mp3 -show_entries format=duration -v quiet -of csv='p=0')
DURATION=${DURATION%.*}

# the start of the clip will be in the middle of the song
let START="$DURATION / 2  - $LEN / 2 "

#end is start + len
let END="$START + $LEN"

echo $START
echo $END

# extract 29 second long sample from the middle of the track and encode it in
# MagnaTagATune format
ffmpeg \
	-i "$FILE".mp3 \
	-ac 1 \
	-filter:a "atempo=1.0" \
	-ss "$START" \
	-to "$END" \
	-ar 16000 \
	-ab 32k \
	-map_metadata -1 \
	-codec:a libmp3lame \
	-map 0:a \
	-y "$FILE"_clip.mp3 \
	-write_xing 0
	

	
	
	
	
	

