# 1. Base Image: Start with an official Python image.
FROM python:3.11-slim

# 2. Working Directory: Set the context for subsequent commands.
WORKDIR /app

# 3. Copy and Install Dependencies:
# This is done in a separate step to leverage Docker's layer caching.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Graphviz, the critical system-level dependency. Need Root Level Acess
USER root
RUN apt-get update && apt-get install -y graphviz

# 4. Copy Application Code: Copy the rest of your application into the container.
COPY . .

# 5. Expose Port: Inform Docker that the container listens on port 8086.
EXPOSE 8086

# 6. Run Command: Specify the command to run when the container starts.
CMD ["python", "mindmap_mcp_server.py"]