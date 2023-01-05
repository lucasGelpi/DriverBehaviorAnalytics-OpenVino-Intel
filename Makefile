REPO_FOLDER=$(PWD)
PYTHON=python3
PIP=pip3
VIRTUAL_ENV = $(REPO_FOLDER)/venv

download_models:
	./Scripts/download_model.sh face-detection-retail-0005

virtualenv:
	@echo "Creating and activating Virtual Environment"
	$(REPO_FOLDER)/Scripts/config_virtual_env.sh $(PIP) $(PYTHON) $(VIRTUAL_ENV)

run:
	$(PYTHON) $(REPO_FOLDER)/dba_dummy.py

start: download_models virtualenv run