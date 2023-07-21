.PHONY: hooks
hooks:
	pre-commit install --overwrite --install-hooks --hook-type pre-commit --hook-type post-checkout --hook-type pre-push
	git checkout

.PHONY: check
check: hooks
	SKIP=no-commit-to-branch pre-commit run -a --hook-stage commit

.PHONY: check-python
check-python:
	@python3 --version | awk '{split($$2, a, "."); if (a[1] < 3 || (a[1] == 3 && a[2] < 10)) {print "Python 3.10 or higher is required"; exit 1}}'

.PHONY: env
env: check-python
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt')"