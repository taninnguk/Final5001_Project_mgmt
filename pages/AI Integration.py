from typing import Any, Dict, Generator, List, Optional, Tuple
import gzip
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
from ollama import chat
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from data_cache import (
    load_cached_data,
    load_cached_meta,
    refresh_cache,
    load_env_key,
    load_cached_pmbok,
    load_cached_pmbok_vectors,
    get_duck,
)

from add_record_form import render_invoice_form, render_project_form
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

st.set_page_config(page_title="AI Assistant (Project & Invoice)", page_icon="🤖", layout="wide")

PROJECT_WORKFLOW = (
    "Project workflow sequence: "
    "1) Prepare document Focus, 2) Procurement Focus, 3) Fabrication Focus, "
    "4) Final inspection, 5) Shipping, 6) Final Document (no delay considered), "
    "7) Completed (no delay considered)."
)

PMBOK_GUIDELINE = (
    "Follow PMBOK 7th principles: be risk-aware, schedule/cost conscious, and stakeholder-focused. "
    "Highlight scope, timeline, budget, quality, risk, communication, and change control. "
    "Give concise, action-oriented advice; if data is missing, say so. "
    "ตอบเป็นภาษาไทยถ้าคำถามเป็นภาษาไทย และตอบเป็นอังกฤษถ้าคำถามเป็นอังกฤษ."
)

OPENROUTER_API_KEY = st.secrets["api"]["OPENROUTER_API_KEY"] # or load_env_key("OPENROUTER_API_KEY")
openrouter_client = None
if OPENROUTER_API_KEY:
    try:
        openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    except Exception:
        openrouter_client = None

# -----------------------------
# Data loading (same sources as dashboards)
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_project_invoice() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Load project/invoice data from DuckDB cache (refreshed from Snowflake)."""
    refresh_cache()
    project_raw, invoice_raw = load_cached_data()
    meta: Dict[str, str] = {"project_source": "duckdb_cache", "invoice_source": "duckdb_cache"}
    return clean_project(project_raw), clean_invoice(invoice_raw), meta


@st.cache_data(ttl=600, show_spinner=False)
def load_column_meta() -> pd.DataFrame:
    """Load column descriptions from COLUMN_META via DuckDB cache."""
    meta_df = load_cached_meta()
    if meta_df is None or meta_df.empty:
        return pd.DataFrame(columns=["Table_name", "Field_name", "Description"])
    meta_df = meta_df.rename(columns=str.strip)
    return meta_df


def clean_project(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df.rename(columns={"Q'ty": "Qty"}, inplace=True)
    df = df.dropna(how="all")

    date_cols = [
        "PO Date",
        "Original Delivery Date",
        "Estimated shipdate",
        "Actual shipdate",
        "Waranty end",
    ]
    numeric_cols = [
        "Project year",
        "Order number",
        "Project Value",
        "Balance",
        "Progress",
        "Number of Status",
        "Max LD",
        "Max LD Amount",
        "Extra cost",
        "Change order amount",
        "Storage fee amount",
        "Days late",
        "Qty",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Progress" in df.columns:
        df["Progress"] = df["Progress"].clip(lower=0, upper=1)
    return df


def clean_invoice(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df.rename(columns={"Currency unit ": "Currency unit"}, inplace=True)
    df = df.dropna(how="all")
    numeric_cols = [
        "Project year",
        "SEQ",
        "Total amount",
        "Percentage of amount",
        "Invoice value",
        "Plan Delayed",
        "Actual Delayed",
        "Claim Plan 2025",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    date_cols = [
        "Invoice plan date",
        "Issued Date",
        "Invoice due date",
        "Plan payment date",
        "Expected Payment date",
        "Actual Payment received date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def ensure_pmbok_cached() -> int:
    """
    Ensure PMBOK chunks exist in DuckDB cache.
    If missing, pull from Snowflake stage, chunk, and store.
    Returns number of cached chunks.
    """
    con = get_duck()
    try:
        existing = con.execute("SELECT COUNT(*) FROM pmbok_chunks").fetchone()[0]
        if existing and existing > 0:
            return int(existing)
    except Exception:
        pass

    if PdfReader is None:
        return 0

    try:
        session = st.connection("snowflake").session()
    except Exception:
        return 0

    cache_dir = Path(tempfile.gettempdir()) / "pmbok_stage_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_pdf = cache_dir / "PMBOK.pdf"
    targets = [
        "@MY_STAGE/PMBOK.pdf",
        "@MY_STAGE/PMBOK.pdf.gz",
        "@MY_STAGE/PMBOK 7th Edition.pdf",
        "@MY_STAGE/PMBOK 7th Edition.pdf.gz",
    ]
    for target in targets:
        try:
            session.file.get(target, str(cache_dir))
            break
        except Exception:
            continue
    gz_file = next(cache_dir.glob("PMBOK*.pdf.gz"), None)
    if gz_file and not local_pdf.exists():
        try:
            with gzip.open(gz_file, "rb") as src, open(local_pdf, "wb") as dst:
                shutil.copyfileobj(src, dst)
        except Exception:
            pass
        finally:
            gz_file.unlink(missing_ok=True)
    if not local_pdf.exists():
        pdf_found = next(cache_dir.glob("PMBOK*.pdf"), None)
        if pdf_found:
            try:
                pdf_found.rename(local_pdf)
            except Exception:
                local_pdf = pdf_found
    if not local_pdf.exists():
        return 0

    try:
        reader = PdfReader(str(local_pdf))
        pages_text = [p.extract_text() or "" for p in reader.pages]
        full_text = "\n".join([t for t in pages_text if t])
        chunks: List[str] = []
        chunk_size = 1200
        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i : i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        if not chunks:
            return 0
        df = pd.DataFrame({"chunk_index": range(len(chunks)), "text": chunks})
        con.register("pmbok_chunks_df", df)
        con.execute("CREATE OR REPLACE TABLE pmbok_chunks AS SELECT * FROM pmbok_chunks_df")
        return len(chunks)
    except Exception:
        return 0


@st.cache_resource
def get_default_embedder() -> Optional[OpenAIEmbeddings]:
    api_key = None
    api_key = load_env_key("OPENAI_API_KEY") or api_key
    try:
        api_key = api_key or st.secrets.get("api", {}).get("OPENAI_API_KEY")
    except Exception:
        pass
    try:
        api_key = api_key or st.secrets.get("api", {}).get("OPENROUTER_API_KEY")
    except Exception:
        pass
    api_key = api_key or load_env_key("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAIEmbeddings(api_key=api_key)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def load_pmbok_chunks() -> List[str]:
    """Ensure PMBOK chunks exist, then load from DuckDB cache."""
    count = ensure_pmbok_cached()
    cached = load_cached_pmbok()
    if cached is None or cached.empty:
        st.warning("ไม่พบ PMBOK chunks ใน DuckDB cache")
        return []
    return cached.sort_values("chunk_index")["text"].dropna().astype(str).tolist()


@st.cache_data(ttl=1800, show_spinner=False)
def load_pmbok_vectors() -> pd.DataFrame:
    """
    Ensure PMBOK vectors exist; if missing, build them (Snowflake -> DuckDB).
    Returns DataFrame with chunk_index, text, embedding.
    """
    existing = load_cached_pmbok_vectors()
    if existing is not None and not existing.empty:
        return existing
    # Build vectors from chunks if available; otherwise fetch chunks first
    chunks_df = load_cached_pmbok()
    if chunks_df is None or chunks_df.empty:
        ensure_pmbok_cached()
        chunks_df = load_cached_pmbok()
    if chunks_df is None or chunks_df.empty:
        return pd.DataFrame(columns=["chunk_index", "text", "embedding"])

    api_key = load_env_key("OPENAI_API_KEY")
    try:
        api_key = api_key or st.secrets.get("api", {}).get("OPENAI_API_KEY")
    except Exception:
        pass
    try:
        api_key = api_key or st.secrets.get("api", {}).get("OPENROUTER_API_KEY")
    except Exception:
        pass
    api_key = api_key or load_env_key("OPENROUTER_API_KEY")
    if not api_key:
        return pd.DataFrame(columns=["chunk_index", "text", "embedding"])

    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        resp = client.embeddings.create(
            model="openai/text-embedding-3-small",
            input=chunks_df["text"].astype(str).tolist(),
            encoding_format="float",
        )
        vectors = [d.embedding for d in resp.data]
        chunks_df = chunks_df.copy()
        chunks_df["embedding"] = vectors
        con = get_duck()
        con.register("pmbok_vec_df", chunks_df)
        con.execute("CREATE OR REPLACE TABLE pmbok_vectors AS SELECT * FROM pmbok_vec_df")
        return chunks_df
    except Exception:
        return pd.DataFrame(columns=["chunk_index", "text", "embedding"])


# -----------------------------
# Simple RAG helpers
# -----------------------------
def row_to_snippet(row: pd.Series, kind: str) -> str:
    if kind == "project":
        parts = [
            f"Project: {row.get('Project', '')}",
            f"Customer: {row.get('Customer', '')}",
            f"Engineer: {row.get('Project Engineer', '')}",
            f"Order: {row.get('Order number', '')}",
            f"Status: {row.get('Status', '')}",
            f"Progress: {row.get('Progress', 0):.0%}" if pd.notna(row.get("Progress")) else "Progress: n/a",
            f"Value: {row.get('Project Value', '')}",
            f"Balance: {row.get('Balance', '')}",
            f"Phrase: {row.get('Project Phrase', '')}",
        ]
    else:
        parts = [
            f"Customer: {row.get('Customer', '')}",
            f"Engineer: {row.get('Project Engineer', '')}",
            f"Order: {row.get('Sale order No.', '')}",
            f"Invoice value: {row.get('Invoice value', '')}",
            f"Payment status: {row.get('Payment Status', '')}",
            f"Plan date: {row.get('Invoice plan date', '')}",
            f"Issued: {row.get('Issued Date', '')}",
        ]
    return " | ".join(str(p) for p in parts if p)


def build_corpus(
    project_df: pd.DataFrame,
    invoice_df: pd.DataFrame,
    domain: str,
    include_pmbok: bool,
    pmbok_chunks: List[str],
    include_workflow: bool = True,
    limit: int = 200,
):
    docs = []
    if domain in {"project", "both"}:
        sample = project_df.head(limit)
        for _, row in sample.iterrows():
            docs.append({"source": "project", "text": row_to_snippet(row, "project")})
    if domain in {"invoice", "both"}:
        sample = invoice_df.head(limit)
        for _, row in sample.iterrows():
            docs.append({"source": "invoice", "text": row_to_snippet(row, "invoice")})
    if include_pmbok and pmbok_chunks:
        for chunk in pmbok_chunks[:50]:  # cap chunks for efficiency
            docs.append({"source": "pmbok", "text": chunk})
    if include_workflow:
        docs.append({"source": "workflow", "text": PROJECT_WORKFLOW})
    return docs


def meta_text_for_domain(meta_df: pd.DataFrame, domain: str) -> str:
    if meta_df is None or meta_df.empty:
        return ""
    cols = {c.strip().lower(): c for c in meta_df.columns}
    if not {"table_name", "field_name", "description"}.issubset(set(cols)):
        return ""  # ไม่มีคอลัมน์ครบก็ข้ามไป
    tbl_col = cols["table_name"]; fld_col = cols["field_name"]; desc_col = cols["description"]

    wanted_tables = ["final_project", "project", "final_invoice", "invoice"]
    if domain == "project":
        wanted_tables = ["final_project", "project"]
    elif domain == "invoice":
        wanted_tables = ["final_invoice", "invoice"]

    meta_df = meta_df.copy()
    meta_df[tbl_col] = meta_df[tbl_col].astype(str).str.lower().str.strip()
    meta_df[fld_col] = meta_df[fld_col].astype(str).str.strip()
    meta_df[desc_col] = meta_df[desc_col].astype(str).str.strip()
    filtered = meta_df[meta_df[tbl_col].isin(wanted_tables)]
    lines = [f"{r[fld_col]}: {r[desc_col]}" for _, r in filtered.head(80).iterrows()]
    return "\n".join(lines)



def rank_docs(query: str, docs: List[Dict[str, str]], top_k: int = 10):
    # Simple keyword overlap score
    tokens = set(query.lower().split())
    scored = []
    for doc in docs:
        words = set(doc["text"].lower().split())
        score = len(tokens & words)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    # Always keep at least one PMBOK chunk if available and nothing matches
    top = [doc for score, doc in scored[:top_k] if score > 0]
    if not top:
        top = [doc for _, doc in scored[: max(1, top_k // 3)]]  # fallback few docs
    return top


def retrieve_pmbok_vectors(query: str, vectors_df: pd.DataFrame, embedder: Optional[OpenAIEmbeddings], top_k: int = 3) -> List[Dict[str, str]]:
    if embedder is None or vectors_df is None or vectors_df.empty:
        return []
    try:
        q_vec = np.array(embedder.embed_query(query), dtype=float)
    except Exception:
        return []
    try:
        mat = np.vstack(vectors_df["embedding"].apply(np.array))
    except Exception:
        return []
    # cosine similarity
    q_norm = np.linalg.norm(q_vec) + 1e-9
    m_norm = np.linalg.norm(mat, axis=1) + 1e-9
    sims = (mat @ q_vec) / (m_norm * q_norm)
    top_idx = sims.argsort()[::-1][:top_k]
    results = []
    for idx in top_idx:
        row = vectors_df.iloc[idx]
        results.append({"source": "pmbok_vector", "text": str(row.get("text", "")), "score": float(sims[idx])})
    return results


def call_model_stream(question: str, context: List[Dict[str, str]], model_choice: str, meta_text: str) -> Generator[str, None, None]:
    ctx_block = "\n".join([f"- ({d['source']}) {d['text']}" for d in context])
    system_prompt = (
        "You are an expert in project management (PMP/PMBOK) and an assistant for project/invoice data. "
        f"{PMBOK_GUIDELINE} "
        f"Always consider this project workflow: {PROJECT_WORKFLOW} "
        "Answer with enough detail (3-5 sentences) using only the provided context; include key numbers/status when available. "
        "If unsure, say you do not have that information."
    )
    if meta_text:
        system_prompt += "\n\nField metadata (use to interpret columns):\n" + meta_text
    user_prompt = f"Context:\n{ctx_block}\n\nQuestion: {question}"

    if model_choice.startswith("ollama"):
        for chunk in chat(
            model="gemma3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        ):
            yield chunk.get("message", {}).get("content") if isinstance(chunk, dict) else ""
    else:
        if openrouter_client is None:
            raise RuntimeError("OpenRouter client unavailable: set OPENROUTER_API_KEY in .env")
        model_id = "x-ai/grok-4.1-fast:free" if model_choice == "grok_openrouter" else model_choice
        extra_body = {"reasoning": {"enabled": True}} if model_choice == "grok_openrouter" else None
        stream = openrouter_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            extra_body=extra_body,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


# -----------------------------
# UI
# -----------------------------
st.title("AI Assistant for Project & Invoice")
st.caption("ถาม-ตอบเกี่ยวกับโครงการ/ใบแจ้งหนี้ ระบบจะค้นหาบริบท (RAG) และสตรีมคำตอบแบบละเอียด")

nav_cols = st.columns(3)
with nav_cols[0]:
    st.page_link("pages/project.py", label="📊 Project dashboard")
with nav_cols[1]:
    st.page_link("pages/Invoice.py", label="🧾 Invoice dashboard")
with nav_cols[2]:
    st.page_link("pages/CRM.py", label="📈 CRM dashboard")
with st.popover("➕ Add project record", use_container_width=True):
    render_project_form(form_key="ai_add_project_form")
with st.popover("➕ Add invoice record", use_container_width=True):
    render_invoice_form(form_key="ai_add_invoice_form")

try:
    project_df, invoice_df, meta = load_project_invoice()
    column_meta_df = load_column_meta()
    pmbok_chunks = load_pmbok_chunks()
    st.success(
        f"Data ready (Project: {meta.get('project_source','?')}, Invoice: {meta.get('invoice_source','?')}, PMBOK chunks: {len(pmbok_chunks)})",
        icon="✅",
    )
except Exception as exc:  # noqa: BLE001
    st.error(f"โหลดข้อมูลไม่สำเร็จ: {exc}", icon="🚫")
    st.stop()

MODEL_OPTIONS = [
    ("ollama_gemma3", "Ollama gemma3 (local)"),
    ("grok_openrouter", "Grok (OpenRouter)"),
    ("amazon/nova-2-lite-v1:free", "Amazon Nova 2 Lite (OpenRouter)"),
    ("tngtech/tng-r1t-chimera:free", "TNG R1T Chimera (OpenRouter)"),
    ("nvidia/nemotron-nano-12b-v2-vl:free", "NVIDIA Nemotron Nano 12B (OpenRouter)"),
    ("openai/gpt-oss-20b:free", "OpenAI GPT-OSS 20B (OpenRouter)"),
    ("tngtech/deepseek-r1t2-chimera:free", "Deepseek R1T2 Chimera (OpenRouter)"),
    ("qwen/qwen3-235b-a22b:free", "Qwen3 235B A22B (OpenRouter)"),
]

model_choice = st.selectbox(
    "Model",
    [m[0] for m in MODEL_OPTIONS],
    format_func=lambda v: dict(MODEL_OPTIONS).get(v, v),
    help="เลือกโมเดล: ใช้ Ollama หรือ OpenRouter (ต้องมี OPENROUTER_API_KEY ใน .env สำหรับ OpenRouter)",
)
# Fix data source to Snowflake FINAL_PROJECT + FINAL_INVOICE (no selection needed)
domain = "both"
pmbok_use = st.checkbox("ใช้ข้อมูลจาก PMBOK (PDF) เป็นบริบทเสริม", value=True)

# Initialize question state
if "question_box" not in st.session_state:
    st.session_state["question_box"] = st.session_state.get("ai_question_prefill", "")

st.markdown("**Quick prompts**")
prompt_cols = st.columns(4)
quick_prompts = [
    "Project ไหน Delay และต้องเร่งให้ทันกำหนดส่ง?",
    "Invoice ไหนจ่ายช้า/Overdue ต้องตามลูกค้า?",
    "ยอด Invoice ที่ Paid แล้วปีนี้รวมเท่าไหร่?",
    "สรุปความเสี่ยงหลักของ Project มูลค่าสูงสุด 3 อันดับ",
]
for col, p in zip(prompt_cols, quick_prompts):
    if col.button(p, use_container_width=True):
        st.session_state["ai_question_prefill"] = p
        st.session_state["question_box"] = p

question = st.text_area(
    "ถามคำถาม",
    key="question_box",
    placeholder="เช่น สถานะออเดอร์ 182xxxx เป็นอย่างไร? หรือ Invoice ของ Customer X อยู่ที่สถานะอะไร?",
    height=120,
)
if st.button("Ask AI", type="primary", disabled=not question.strip()):
    with st.spinner("กำลังค้นหาและตอบ..."):
        corpus = build_corpus(project_df, invoice_df, domain, pmbok_use, pmbok_chunks, include_workflow=True)
        context = rank_docs(question, corpus, top_k=8)
        pmbok_vector_context: List[Dict[str, str]] = []
        if pmbok_use:
            pmbok_vectors_df = load_pmbok_vectors()
            embedder = get_default_embedder()
            pmbok_vector_context = retrieve_pmbok_vectors(question, pmbok_vectors_df, embedder, top_k=3)
            context.extend(pmbok_vector_context)
        try:
            meta_cols = set(column_meta_df.columns.str.lower())
            if meta_cols >= {"table_name", "field_name", "description"}:
                meta_text = meta_text_for_domain(column_meta_df, domain)
            else:
                meta_text = ""
        except Exception:
            meta_text = ""
        try:
            chosen = model_choice  # pass through actual selection
            stream = call_model_stream(question, context, chosen, meta_text)
            st.subheader("Answer:")
            st.write_stream(stream)
            with st.expander("ดูบริบทที่ใช้ตอบ (context)"):
                for idx, doc in enumerate(context, 1):
                    st.markdown(f"{idx}. **{doc['source']}** — {doc['text']}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"เรียกโมเดลไม่สำเร็จ: {exc}")
elif not question.strip():
    st.info("พิมพ์คำถามก่อน แล้วกด Ask AI")
