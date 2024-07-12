#!/bin/bash

project_path=$(realpath "$(dirname "$0")")/..

git config core.hooksPath "$project_path"/scripts/hooks