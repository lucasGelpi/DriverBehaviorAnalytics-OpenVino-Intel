REPO_FOLDER=$(PWD)
PYTHON=python3
PIP=pip3
VIRTUAL_ENV = $(REPO_FOLDER)/venv

download_models:
	@echo "-----------------------------------------------------------"
	@echo "Downloading models"
	@echo "-----------------------------------------------------------"
	./Scripts/download_model.sh face-detection-retail-0005

run_proyect:
	@echo "-----------------------------------------------------------"
	@echo "Creating and activating Virtual Environment"
	@echo "-----------------------------------------------------------"
	$(REPO_FOLDER)/Scripts/config_virtual_env.sh $(PIP) $(PYTHON) $(VIRTUAL_ENV)

start:
	download_models run_proyect