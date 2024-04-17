import docker

# Create a Docker client
client = docker.from_env()

# Remove all containers
containers = client.containers.list()
for container in containers:
    container.remove(force=True)

# Remove all images
images = client.images.list()
for image in images:
    client.images.remove(image.id, force=True)
