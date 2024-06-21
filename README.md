

Build the docker image using the Docker file

docker build -t maroonvillage-rag-system-aws .

Run the docker container

docker run -d -p 8005:8005 maroonvillage-rag-system

docker run -d -p 11434:11434 ollama/ollama

docker run -d -p 8000:8000 server



Execute commands on the running container


docker exec -it [container_id] [command-to-run]

####################################################


Containers: 

ollama - using port 11434

Chroma Server - using port 8000

rag system - using port 8005

#####################################################

sample requests:

http://0.0.0.0:8005/ask/what+are+the+four+categories+of+the+MANAGE+function+of+RMF



docker run -d  -p 8000:8000 -e ALLOW_RESET=True <image_name>
