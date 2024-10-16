@echo
python -m venv .venv
call .venv/Scripts/activate
pip install -r requirements.txt
pip install numpy==1.26.4
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
pause
pip list
pause