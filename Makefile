USE_CASE="DBA-Dummy"
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

-include src/common.mk
-include src/middleware.mk

update:
	# Clone or update Submodules
	git checkout $(BRANCH)
	git pull origin $(BRANCH) && git submodule update --init --recursive

download_models::
	sudo -A $(REPO_FOLDER)/src/scripts/download_dlib_model.sh
	./src/scripts/download_model.sh $(USE_CASE) python face-detection-retail-0004
	./src/scripts/download_model.sh $(USE_CASE) python head-pose-estimation-adas-0001
	./src/scripts/download_model.sh $(USE_CASE) python face-reidentification-retail-0095
