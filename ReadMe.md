## Flask Tensorflow backend served with uWSGI

# How to install  
1. Create a Venv with python3 -m venv ./[myenvname]
2. Activate the venv with source [myenvname]/bin/activate
3. Install the requirements.txt with pip install -r requirements.txt
4. uWSGI should have been installed by pip
5. Run uwsgi app.ini or [myenvname]/bin/uwsgi app.ini
6. All set