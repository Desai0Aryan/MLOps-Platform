#!/bin/sh
# Runs once, when the Postgres data volume is first created.
# The default database (POSTGRES_DB=mlops) holds the prediction log; MLflow gets its own database.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'EOSQL'
SELECT 'CREATE DATABASE mlflow' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
EOSQL
