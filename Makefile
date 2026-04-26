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

launch-isaaclab: build-isaaclab
	@set -e; \
	xhost +local:root >/dev/null; \
	trap 'xhost -local:root >/dev/null' EXIT; \
	docker run --rm -it \
		--name isaaclab \
		--runtime=nvidia \
		--gpus all \
		--net=host \
		--ipc=host \
		-v $(shell pwd):/workspace/aicapstone \
		-v /workspace/aicapstone/.venv \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		-e DISPLAY=$$DISPLAY \
		-e OMNI_KIT_ACCEPT_EULA=Y \
		-e PRIVACY_CONSENT=Y \
		-e QT_X11_NO_MITSHM=1 \
		-e NVIDIA_VISIBLE_DEVICES=all \
		-e NVIDIA_DRIVER_CAPABILITIES=graphics,display,utility,compute \
		$(IMAGE) \
		bash -lc ' \
			set -e; \
			echo "== GPU check =="; \
			nvidia-smi || true; \
			echo "== Vulkan ICD candidates =="; \
			ls -l /etc/vulkan/icd.d /usr/share/vulkan/icd.d 2>/dev/null || true; \
			unset VK_ICD_FILENAMES; \
			for icd in \
				/usr/share/vulkan/icd.d/nvidia_icd.json \
				/etc/vulkan/icd.d/nvidia_icd.json \
				/usr/share/vulkan/icd.d/nvidia_layers.json \
				/etc/vulkan/icd.d/nvidia_layers.json; do \
				if [ -f "$$icd" ]; then \
					export VK_ICD_FILENAMES="$$icd"; \
					echo "Using Vulkan ICD: $$VK_ICD_FILENAMES"; \
					break; \
				fi; \
			done; \
			if [ -z "$${VK_ICD_FILENAMES:-}" ]; then \
				echo "No NVIDIA Vulkan ICD JSON found in container."; \
				echo "Check host nvidia-container-toolkit installation."; \
			fi; \
			for lib in libGLU.so.1 libXt.so.6; do \
				if ! ldconfig -p | grep -q "$$lib"; then \
					echo "$$lib is missing from the image." >&2; \
					exit 1; \
				fi; \
			done; \
			cd /workspace/aicapstone; \
			exec /bin/bash \
		'

check-isaaclab-gpu:
	@docker run --rm \
		--gpus all \
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
