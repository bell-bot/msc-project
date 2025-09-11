# Builder
FROM alpine/git:latest as builder

WORKDIR /app
COPY . .
RUN git submodule update --init --recursive

#Final Image
FROM python:3.12-slim

WORKDIR /app

#Project files
COPY --from=builder /app/pyproject.toml .
COPY --from=builder /app/src ./src
COPY --from=builder /app/circuits ./circuits
COPY --from=builder /app/experiments ./experiments
COPY --from=builder /app/tests ./tests
COPY --from=builder /app/histograms ./histograms
COPY --from=builder /app/results ./results

RUN pip install .