#!/bin/bash

# or can you this if above not working
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 730335254649.dkr.ecr.eu-north-1.amazonaws.com
docker pull 730335254649.dkr.ecr.eu-north-1.amazonaws.com/yt-comments-sentiment-analysis:latest

# Check if the container 'my-app' is running
if [ "$(docker ps -q -f name=my-app)" ]; then
    # Stop the running container
    docker stop my-app
fi

# Check if the container 'my-app' exists (stopped or running)
if [ "$(docker ps -aq -f name=my-app)" ]; then
    # Remove the container if it exists
    docker rm my-app
fi
docker run -d -p 8000:8000 -e DAGSHUB_PAT=d99c9da88fe635eb0297d593a60bc91b40b73dec --name my-app 730335254649.dkr.ecr.eu-north-1.amazonaws.com/yt-comments-sentiment-analysis:latest

echo "Container Start Successfully"