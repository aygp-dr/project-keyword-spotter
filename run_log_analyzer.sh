#!/bin/bash
cd "$(dirname "$0")"
source /home/aygp-dr/.cache/pypoetry/virtualenvs/project-keyword-spotter-*/bin/activate
python -m project_keyword_spotter.log_analyzer
