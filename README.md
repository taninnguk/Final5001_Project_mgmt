# Powered by
1. 6720422002 กมลพรรณ กันทะวงศ์ <br>
2. 6720422008 ณัฐดนัย คำเผ่า <br>
3. 6720422009 ธนิน ตราบชั่วกัลปาว์ <br>
4. 6724022013 เตชสิทธิ์ กันทะใจ <br>
5. 6720422015 จิระพงษ์ ศิริโชติธนาวงษ์ <br>

# Streamlit (Notice: Snowflake will expire on 09/12/2025)
Streamlit app deploy on <a href="https://final5001-project-mgmt.streamlit.app">https://final5001-project-mgmt.streamlit.app</a>

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
