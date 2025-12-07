import duckdb
import streamlit as st
import pandas as pd
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from openai import OpenAI

@st.cache_resource
def get_duck() -> duckdb.DuckDBPyConnection:
    """Return a shared DuckDB connection persisted on disk."""
    return duckdb.connect("cache.duckdb")


@st.cache_data(ttl=600, show_spinner=False)
def refresh_cache() -> bool:
    """Pull data from Snowflake and materialize into DuckDB tables."""
    con = get_duck()
    sf = st.connection("snowflake")
    project_df = sf.query("SELECT * FROM FINAL_PROJECT;", ttl=300)
    invoice_df = sf.query("SELECT * FROM FINAL_INVOICE;", ttl=300)
    meta_df = sf.query("SELECT * FROM COLUMN_META;", ttl=300)

    con.register("project_df", project_df)
    con.execute("CREATE OR REPLACE TABLE project AS SELECT * FROM project_df")

    con.register("invoice_df", invoice_df)
    con.execute("CREATE OR REPLACE TABLE invoice AS SELECT * FROM invoice_df")

    if meta_df is not None and not meta_df.empty:
        con.register("meta_df", meta_df)
        con.execute("CREATE OR REPLACE TABLE column_meta AS SELECT * FROM meta_df")
    return True


@st.cache_data(ttl=120, show_spinner=False)
def load_cached_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Project/Invoice data from DuckDB cache."""
    con = get_duck()
    try:
        project = con.execute("SELECT * FROM project").df()
    except Exception:
        project = pd.DataFrame()
    try:
        invoice = con.execute("SELECT * FROM invoice").df()
    except Exception:
        invoice = pd.DataFrame()
    return project, invoice


@st.cache_data(ttl=600, show_spinner=False)
def load_cached_meta() -> pd.DataFrame:
    """Load column metadata from DuckDB cache."""
    con = get_duck()
    try:
        return con.execute("SELECT * FROM column_meta").df()
    except Exception:
        return pd.DataFrame(columns=["Table_name", "Field_name", "Description"])


@st.cache_data(ttl=1800, show_spinner=False)
def load_cached_pmbok() -> pd.DataFrame:
    """Load PMBOK chunks from DuckDB cache (if any)."""
    con = get_duck()
    try:
        return con.execute("SELECT * FROM pmbok_chunks ORDER BY chunk_index").df()
    except Exception:
        return pd.DataFrame(columns=["chunk_index", "text"])


@st.cache_data(ttl=1800, show_spinner=False)
def load_cached_pmbok_vectors() -> pd.DataFrame:
    """Load PMBOK vector embeddings from DuckDB cache (if any)."""
    con = get_duck()
    try:
        return con.execute("SELECT * FROM pmbok_vectors ORDER BY chunk_index").df()
    except Exception:
        return pd.DataFrame(columns=["chunk_index", "text", "embedding"])


def load_env_key(key: str, env_path: Path = Path(".env")) -> Optional[str]:
    if key in os.environ:
        return os.environ[key]
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return None


def _get_openrouter_api_key() -> Optional[str]:
    """Return OpenRouter API key from secrets or environment."""
    try:
        return st.secrets.get("api", {}).get("OPENROUTER_API_KEY")
    except Exception:
        return load_env_key("OPENROUTER_API_KEY")

####Settings for AI Chart Summary using OpenRouter GPT-OSS 20B####
def ai_chart_summary(title: str, df: pd.DataFrame, hint: str, key: str, meta_text: str = "", model: str = "openai/gpt-5-mini") -> None:
    """
    Render a button to summarize a chart via OpenRouter GPT-OSS 20B.
    Shows output in a collapsible expander.
    """
    model_used = "openai/gpt-5-mini"
    state_key = f"ai_summary_{key}"
    if st.button(f"🤖 AI summarize: {title}", key=key, use_container_width=True):
        api_key = _get_openrouter_api_key()
        if not api_key:
            st.error("OpenRouter client is not available. Set OPENROUTER_API_KEY in environment/.env.")
            return
        data_preview = "No data"
        if df is not None and not df.empty:
            data_preview = df.head(50).to_csv(index=False)
        system_prompt = (
            "คุณเป็นนักวิเคราะห์ข้อมูลที่สรุปผลกระชับเป็นภาษาไทยเท่านั้น "
            "สรุปกราฟด้านล่างเป็น bullet 2-4 ข้อ ระบุแนวโน้ม จุดสูง/ต่ำ ความเสี่ยง และข้อเสนอแนะที่เป็นไปได้ "
            "ถ้าข้อมูลไม่พอให้บอกอย่างตรงไปตรงมา"
            "ให้คำแนะนำโดยอิงจาก PMBOK 7th Edition"
            "ถ้าได้รับข้อมูลเป็นภาษาไทย ให้ตอบกลับเป็นภาษาไทย ถ้าได้รับข้อมูลเป็นภาษาอังกฤษ ให้ตอบกลับเป็นภาษาอังกฤษ"
        )
        if meta_text:
            system_prompt += f"\n\nข้อมูล schema/คำอธิบายคอลัมน์:\n{meta_text}"
        user_prompt = (
            f"หัวข้อกราฟ: {title}\n"
            f"บริบทกราฟ: {hint}\n"
            f"ข้อมูล (CSV แถวตัวอย่าง):\n{data_preview}\n"
            "ช่วยสรุปข้อมูลกราฟนี้เป็น bullet ภาษาไทย"
        )
        try:
            summary = _cached_ai_summary(model_used, api_key, system_prompt, user_prompt)
            st.session_state[state_key] = summary
        except Exception as exc:  # noqa: BLE001
            st.error(f"AI summary failed: {exc}")
    if state_key in st.session_state:
        with st.expander(f"ดูสรุป AI ({model_used})", expanded=False):
            st.info(st.session_state[state_key])


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_ai_summary(model: str, api_key: str, system_prompt: str, user_prompt: str) -> str:
    """Cacheable call to OpenRouter for chart summarization."""
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content
