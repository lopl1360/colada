.PHONY: help build up down shell run

help:
	@echo "Makefile commands:"
	@echo "  make build       - Build the Docker image using docker-compose"
	@echo "  make up          - Start the container using docker-compose"
	@echo "  make down        - Stop and remove the container"
	@echo "  make shell       - Start the container and open a shell"
	@echo "  make run SYMBOL= - Run the trading bot for a symbol (e.g., make run SYMBOL=AAPL)"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

shell:
	docker-compose exec trading-bot bash

run:
	 run trading-bot python cli.py alpaca $(SYMBOL)
