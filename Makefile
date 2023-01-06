REPO_FOLDER=$(PWD)
PYTHON=python3
PIP=pip3
VIRTUAL_ENV = $(REPO_FOLDER)/venv

download_models:
	@echo "---------------------------------------------------------------------"
	@echo "Downloading models"
	./Scripts/download_model.sh face-detection-retail-0005
	@echo "---------------------------------------------------------------------"

virtualenv:
	@echo "---------------------------------------------------------------------"
	@echo "Creating and activating Virtual Environment"
	$(REPO_FOLDER)/Scripts/config_virtual_env.sh $(PIP) $(PYTHON) $(VIRTUAL_ENV)
	@echo "---------------------------------------------------------------------"

run:
	@echo "---------------------------------------------------------------------"
	@echo "Running Use Case"
	@echo "---------------------------------------------------------------------"
	$(PYTHON) $(REPO_FOLDER)/dba_dummy.py

start: download_models virtualenv run