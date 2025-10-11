Quick start

1) Put these files in your project folder.
2) (Optional) Edit requirements.txt if you want to add/remove packages.
3) Create & install everything into a new virtualenv:
   chmod +x setup_venv.sh
   ./setup_venv.sh ~/MastersAI/course1-env "course1-env" "Python (Course1)"

4) Activate the environment before you work:
   source ~/MastersAI/course1-env/bin/activate

5) Start JupyterLab and pick the kernel named "Python (Course1)":
   jupyter-lab

Notes
- The 'os' library is built into Python—do not add it to requirements.
- rpy2 requires R to be installed: brew install r
- If you later add packages, freeze them to a lock file for sharing:
   pip freeze > requirements-lock.txt
