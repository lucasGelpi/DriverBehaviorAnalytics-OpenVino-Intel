REPO_FOLDER=$(PWD)
PYTHON=python3
PIP=pip3
VIRTUAL_ENV = $(REPO_FOLDER)/venv

download_models:
	./Scripts/download_det_model.sh face-detection-retail-0005
	./Scripts/download_reid_model.sh face-reidentification-retail-0095

run_proyect:
	@echo "-----------------------------------------------------------"
	@echo "Creating and activating Virtual Environment"
	@echo "-----------------------------------------------------------"
	$(REPO_FOLDER)/Scripts/config_virtual_env.sh $(PIP) $(PYTHON) $(VIRTUAL_ENV)

start: download_models run_proyect