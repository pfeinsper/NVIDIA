run: sign-detector road-follower

setup:
	sudo apt install v4l-utils
	sudo apt install v4l2loopback-dkms
	sudo apt install mosquitto
	sudo apt-get install libcanberra-gtk-module
	sudo apt install python3-pip
	sudo -H pip3 install --upgrade pip
	pip3 install -r requirements.txt

broker:
	mosquitto -p 1884
	
sign-detector:
	python3 sign_detector.py

road-follower:
	python3 road_follower.py

subscriber:
	python3 subscriber.py

create-devices:
	sudo modprobe v4l2loopback devices=4

video-loop:
	ffmpeg -re -stream_loop 30 -i video1.mp4 -vf format=yuv420p -f v4l2 /dev/video5


clone-video:
	ffmpeg -f video4linux2 -i /dev/video0 -codec copy -f v4l2 /dev/video3 -codec copy -f v4l2 /dev/video4
