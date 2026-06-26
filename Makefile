.PHONY: help train-v1 train-v2 compare error-analysis show-clahe
export PYTHONUTF8 = 1

help:
	@echo "Comandos disponíveis:"
	@echo "  make train-v2       - Treina o modelo V2"
	@echo "  make train-v1       - Treina o modelo V1"
	@echo "  make compare        - Roda a comparação V1 vs V2"
	@echo "  make error-analysis - Roda a análise de erros com Grad-CAM"
	@echo "  make show-clahe     - Gera as comparações visuais do CLAHE"

train-v2:
	uv run python main.py train-v2

train-v1:
	uv run python main.py train-v1

compare:
	uv run python main.py compare

error-analysis:
	uv run python main.py error-analysis

show-clahe:
	uv run python main.py show-clahe
