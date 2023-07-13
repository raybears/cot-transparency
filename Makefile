.PHONY: hooks
hooks:
	pre-commit install --overwrite --install-hooks --hook-type pre-commit --hook-type post-checkout --hook-type pre-push
	git checkout

.PHONY: format
check: hooks
	pre-commit run -a --hook-stage commit

.PHONY: env
env: 
	pip install -r requirements.txt
