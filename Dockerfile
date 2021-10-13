# Create the compile image
FROM python:3.7-slim AS compile-image

# Create and use virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency-related items
COPY requirements.txt ./requirements.txt
COPY setup.py ./setup.py
COPY src/ ./src/

# Install dependencies
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    pip3 install -e .

# Create the build image
FROM python:3.7-slim AS build-image

# Create user and copy virtual environment
RUN useradd --create-home bot
USER bot
COPY --from=compile-image --chown=bot /opt/venv /opt/venv

RUN pip3 install nltk && \
    python3 -m nltk.downloader popular

# Make sure to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /home/bot/

# Copy chatbot related items
COPY global_env.sh ./global_env.sh
COPY data/ ./data/
COPY exec/ ./exec/
COPY src/ ./src/

# Set global environment variables
# ENTRYPOINT ["/bin/bash", "-c", "source global_env.sh"]
ENTRYPOINT ["/bin/bash", "-c"]

# Run the chatbot
CMD ["source global_env.sh && sh $CHATBOT"]


# build and run from command line (Make sure dockerhub is running on Mac before running this):
# docker build -t chatbot . && docker run -it chatbot:latest


