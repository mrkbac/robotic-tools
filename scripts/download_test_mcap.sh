#!/usr/bin/env bash

cd $(dirname "$0")/data
git clone https://github.com/foxglove/ros-foxglove-bridge-benchmark-assets.git data --depth 1
