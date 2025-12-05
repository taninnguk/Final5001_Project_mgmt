import duckdb
import streamlit as st
import pandas as pd
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

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