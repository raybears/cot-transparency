#!/bin/bash
SRC_DIR="$(dirname "$(readlink -f "$0")")"

pushd $SRC_DIR

# URL for the raw content
URL="https://raw.githubusercontent.com/deepmind/AQuA/master/dev.json"

# Use curl to download
curl -LJO $URL
popd