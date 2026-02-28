FROM postgres:16

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    postgresql-server-dev-16 \
    && rm -rf /var/lib/apt/lists/*

# Install pgvector
RUN git clone https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install

# Enable extension automatically
COPY init.sql /docker-entrypoint-initdb.d/

EXPOSE 5432