build:
	docker build -t watts .

run:
	docker run -i -t \
		--shm-size=8G \
		-p 8265:8265 \
		--mount type=bind,source=`pwd`/logs,target=/home/ray/logs \
		--mount type=bind,source=`pwd`/snapshots,target=/home/ray/snapshots \
		watts python poet_distributed.py
