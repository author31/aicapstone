.PHONY: install install-dev test build-isaaclab launch-isaaclab check-isaaclab-gpu submodules submodules-pull
IMAGE ?= leisaac-isaaclab:latest
DOCKERFILE ?= Dockerfile

submodules:
	git submodule update --init --recursive

submodules-pull:
	git submodule update --remote --recursive

install: submodules
	uv sync

install-dev: submodules
	uv sync --extra dev

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra dev pytest tests/test_repo_layout.py

build-isaaclab: submodules
	docker build -f $(DOCKERFILE) -t $(IMAGE) .

GPU ?= all
CONTAINER_NAME ?= isaaclab

launch-isaaclab: build-isaaclab
	@set -e; \
	xhost +local:root >/dev/null || true; \
	trap 'xhost -local:root >/dev/null || true' EXIT; \
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		--gpus '"device=$(GPU)"' \
		--net=host \
		--ipc=host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-v $(shell pwd):/workspace/aicapstone \
		-v /workspace/aicapstone/.venv \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
		-v /etc/vulkan/icd.d:/etc/vulkan/icd.d:ro \
		-e DISPLAY=$$DISPLAY \
		-e OMNI_KIT_ACCEPT_EULA=Y \
		-e PRIVACY_CONSENT=Y \
		-e QT_X11_NO_MITSHM=1 \
		-e NVIDIA_VISIBLE_DEVICES=$(GPU) \
		-e NVIDIA_DRIVER_CAPABILITIES=graphics,display,utility,compute \
		$(IMAGE) \
		bash -lc '\
			set -e; \
			echo "== GPU check =="; \
			nvidia-smi || true; \
			echo "== Vulkan ICD candidates =="; \
			ls -l /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null || true; \
			unset VK_ICD_FILENAMES; \
			for icd in \
				/usr/share/vulkan/icd.d/nvidia_icd.json \
				/etc/vulkan/icd.d/nvidia_icd.json; do \
				if [ -f "$$icd" ]; then \
					export VK_ICD_FILENAMES="$$icd"; \
					echo "Using Vulkan ICD: $$VK_ICD_FILENAMES"; \
					break; \
				fi; \
			done; \
			if [ -z "$${VK_ICD_FILENAMES:-}" ]; then \
				echo "WARNING: No NVIDIA Vulkan ICD found."; \
				echo "Check nvidia-container-toolkit / driver installation."; \
			fi; \
			for lib in libGLU.so.1 libXt.so.6 libX11.so.6 libvulkan.so.1; do \
				if ! ldconfig -p | grep -q "$$lib"; then \
					echo "Missing $$lib in image." >&2; \
					exit 1; \
				fi; \
			done; \
			cd /workspace/aicapstone; \
			exec /bin/bash \
		'

check-isaaclab-gpu:
	@docker run --rm \
		--device nvidia.com/gpu=all \
		-e ACCEPT_EULA=Y \
		-e NVIDIA_VISIBLE_DEVICES=all \
		-e NVIDIA_DRIVER_CAPABILITIES=all \
		-e __GLX_VENDOR_LIBRARY_NAME=nvidia \
		$(IMAGE) \
		bash -lc ' \
			set -e; \
			for icd in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do \
				if [ -f "$$icd" ]; then \
					export VK_ICD_FILENAMES="$$icd"; \
					echo "Using NVIDIA Vulkan ICD: $$VK_ICD_FILENAMES"; \
					break; \
				fi; \
			done; \
			if [ -z "$${VK_ICD_FILENAMES:-}" ]; then \
				echo "NVIDIA Vulkan ICD was not found under /etc/vulkan/icd.d or /usr/share/vulkan/icd.d" >&2; \
				exit 1; \
			fi; \
			nvidia-smi; \
			ldconfig -p | grep "libGLU.so.1"; \
			ldconfig -p | grep "libXt.so.6"; \
			python -c "import torch; print(\"torch cuda available:\", torch.cuda.is_available()); print(\"torch cuda device:\", torch.cuda.get_device_name(0))"; \
			ls -l /etc/vulkan/icd.d /usr/share/vulkan/icd.d 2>/dev/null || true; \
			'
			vulkaninfo --summary \
