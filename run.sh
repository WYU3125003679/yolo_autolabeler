#!/usr/bin/env bash
# 本地启动器：用 yolo 虚拟环境的 Python 运行当前目录下的 autolabeler 包。
# 用法: ./run.sh <workspace> [--no-browser] [--host ...] [--port ...]
cd "$(dirname "$0")"
exec /home/ananan/.conda/envs/yolo/bin/python -m autolabeler "$@"
