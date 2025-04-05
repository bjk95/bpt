#!/bin/bash
pip3 install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu126 --python-version 311 --only-binary=:all: --target gh200-2025-02-18-3/lib64/python3.11/site-packages/
pip install -r requirements.txt