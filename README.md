# my_streamlit
test streamlit deploy


# prepare for run
pip install -r requirements.txt

# setting the folder with following step in Drive send the request access to this Drive:
https://drive.google.com/drive/folders/17qHpIHABuO6pZlq9WJ9GjOKzm9MwsSaV?usp=share_link

# step:
1. create .env and .streamlit/secrets.toml file with copy file content in drive<br>
2. activate virtual environment<br>
    if use the default venv >> On terminal >>pip install -r requirements.txt<br>
    if create new on terminal >>> 1. python -m venv .venv<br>
                      2. for windows OS >> .venv/Scripts/activate<br>
                         for macOS >> source .venv/bin/activate<br>
                      3. pip install -r requirements.txt<br>
3. On terminal >> streamlit run Welcome.py