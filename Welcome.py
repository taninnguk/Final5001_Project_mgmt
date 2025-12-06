import streamlit as st
import pandas as pd
import plotly.express as px
import gzip
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from data_cache import refresh_cache, load_cached_data, get_duck, load_env_key
from langchain_openai import OpenAIEmbeddings

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

st.set_page_config(page_title="Welcome", page_icon="👋", layout="wide")


def render_welcome() -> None:
    st.title("Welcome 👋")
    st.caption("ภาพรวมฟีเจอร์หลักและ Executive summary สำหรับผู้ใช้ใหม่")

    cols = st.columns(4)
    with cols[0]:
        st.page_link("pages/project.py", label="📊 Project dashboard")
    with cols[1]:
        st.page_link("pages/Invoice.py", label="🧾 Invoice dashboard")
    with cols[2]:
        st.page_link("pages/CRM.py", label="CRM dashboard")
    with cols[3]:
        st.page_link("pages/AI Integration.py", label="🤖 AI assistant")

    st.markdown("## Background")
    st.caption("แผนที่จุดพิกัดผู้ผลิต (สีตาม Product) จาก FINAL_PROJECT; hover เพื่อดูผู้ผลิต/สินค้า")
    geo_col = st.container()
    with geo_col:
        project_geo = load_project_geo()
        if project_geo is None:
            st.info("ยังไม่สามารถแสดงแผนที่ได้: ต้องมีคอลัมน์ Manufactured by หรือข้อมูลประเทศ/พิกัด")
        elif project_geo.empty:
            st.info("ไม่มีข้อมูลผู้ผลิตให้แสดงบนแผนที่")
        else:
            fig = px.scatter_mapbox(
                project_geo,
                lat="Latitude",
                lon="Longitude",
                color="Product",
                size="Qty",
                hover_name="Country",
                hover_data={"Manufactured by": True, "Qty": True, "Product": True},
                size_max=15,
                zoom=1,
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig.update_layout(
                mapbox_style="carto-positron",
                height=520,
                margin=dict(l=0, r=0, t=20, b=0),
                legend_title_text="Product",
            )
            st.plotly_chart(fig, use_container_width=True)
    st.caption("หมายเหตุ: พิกัดบางจุดอาจมาจากการแมปประเทศโดยประมาณ หากไม่มีข้อมูลละติจูด/ลองจิจูดในตาราง FINAL_PROJECT")
    overview_data = """
    แดชบอร์ดนี้ใช้ข้อมูลจาก Snowflake (FINAL_PROJECT / FINAL_INVOICE) ผ่าน DuckDB cache เพื่อให้ดูภาพรวมธุรกิจในที่เดียว:
    - Project: มูลค่า/คงเหลือ (Balance), ความคืบหน้า, Top orders, สถานะส่งมอบ
    - Invoice: แผน/รับเงินจริง, สถานะชำระ (Paid/Aging/Not invoiced), คาดการณ์กระแสเงินสด
    - CRM & AI: มุมมองลูกค้า/พฤติกรรมการจ่าย และถาม-ตอบเชิงบริบทด้วย AI

    แผนที่ด้านบนมาจาก FINAL_PROJECT แสดงแหล่งผลิตตามประเทศ (Manufactured by) และแยกสีตาม Product (Heater, Control Panel, Vessel) เพื่อดูการกระจายฐานการผลิตและความเสี่ยงซัพพลายเชื่อมโยงกับการส่งมอบ/การออกบิล
    """
    st.markdown(overview_data)

    st.markdown("## Objectives")
    st.write(
        """
        ระบบนี้ช่วยให้คุณติดตามสถานะโครงการและใบแจ้งหนี้ได้ครบวงจร พร้อมผู้ช่วย AI สำหรับการถาม-ตอบเชิงบริบท:
        - **Project dashboard**: ดูมูลค่าโครงการ, ยอดคงเหลือ, ความคืบหน้า, top orders, พายแบ่งตามวิศวกร/ลูกค้า, และปริมาณสินค้าตามผู้ผลิต
        - **Invoice dashboard**: ดูมูลค่าใบแจ้งหนี้, สถานะการชำระเงิน, การวางแผน/รับเงินรายเดือน, และการเชื่อมโยงกับข้อมูลโครงการ
        - **CRM dashboard** : ติดตามใบเสร็จและลูกค้า เพื่อบริหารจัดความสัมพันธ์กับลูกค้า
        - **AI assistant**: ถาม-ตอบเรื่องโครงการ/ใบแจ้งหนี้ด้วยข้อมูลที่มี (RAG) พร้อมความรู้ PMBOK และ workflow ของโปรเจกต์
        """
    )

    st.markdown("## How to use")
    st.write(
        """
        1) ไปที่ **Project dashboard** เพื่อดูภาพรวมมูลค่าและความคืบหน้า เลือกกรองตามวิศวกร/ลูกค้า/ปี/โปรเจกต์ได้
        2) ไปที่ **Invoice dashboard** เพื่อติดตามมูลค่าใบแจ้งหนี้, สถานะการจ่ายเงิน และแผน/รับจริงรายเดือ
        3) ใช้ **CRM dashboard** เพื่อติดตามลูกค้า และมอบสิทธิประโยชน์ที่เหมาะสมให้กับลูกค้าประจำ และวิเคราะห์หาประเด็นที่เกิดขึ้นจากลูกค้าที่ไม่กลับมาซื้อซ้ำ
        4) ใช้ **AI assistant** เพื่อถามคำถามเชิงวิเคราะห์ เช่น โปรเจกต์ที่ Delay หรือใบแจ้งหนี้ที่ต้องเร่ง ตามข้อมูลล่าสุด
        """
    )

    st.markdown("## Quick tips")
    st.write(
        """
        - ใช้ตัวกรองด้านซ้ายของแต่ละหน้าลดรายการให้ตรงกับสิ่งที่สนใจ
        - กดปุ่ม **Add record** บนหน้า Project/Invoice เพื่อเพิ่มข้อมูลใหม่ (เชื่อมกับ Snowflake โดยตรง)
        - ใน AI assistant สามารถเลือกใช้ข้อมูล Project/Invoice หรือรวมกัน และเปิดใช้ความรู้ PMBOK ได้
        """
    )
    st.success("พร้อมใช้งาน: เลือกลิงก์ด้านบนเพื่อเริ่มสำรวจข้อมูลหรือถาม AI ได้ทันที", icon="✅")


    st.caption("โหลดข้อมูลจาก DuckDB cache (Snowflake → DuckDB) หลังจากส่วน static แสดงผลแล้ว")
    with st.spinner("กำลังเตรียมข้อมูลจาก Snowflake ผ่าน DuckDB cache..."):
        refresh_cache()
        project_df_cache, invoice_df_cache = load_cached_data()
        pmbok_chunks = ensure_pmbok_cached()
        pmbok_vectors = ensure_pmbok_vectors_cached()
        project_geo = load_project_geo()
    st.caption(
        f"Project rows: {len(project_df_cache)} | Invoice rows: {len(invoice_df_cache)} "
        f"| PMBOK chunks: {pmbok_chunks} | PMBOK vectors: {pmbok_vectors}"
    )


@st.cache_data(ttl=1800, show_spinner=False)
def ensure_pmbok_cached() -> int:
    """
    Pull PMBOK PDF from Snowflake stage into DuckDB cache as chunks (for RAG).
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
    last_err = None
    for target in targets:
        try:
            session.file.get(target, str(cache_dir))
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
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
        chunks = []
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


@st.cache_data(ttl=1800, show_spinner=False)
def ensure_pmbok_vectors_cached() -> int:
    """
    Embed PMBOK chunks into vectors and store in DuckDB (pmbok_vectors).
    Returns number of vectors cached.
    """
    con = get_duck()
    try:
        existing = con.execute("SELECT COUNT(*) FROM pmbok_vectors").fetchone()[0]
        if existing and existing > 0:
            return int(existing)
    except Exception:
        pass

    chunks_df = con.execute("SELECT chunk_index, text FROM pmbok_chunks ORDER BY chunk_index").df()
    if chunks_df is None or chunks_df.empty:
        return 0

    api_key = None
    try:
        api_key = st.secrets.get("api", {}).get("OPENROUTER_API_KEY")  # reuse OpenRouter key for embeddings
    except Exception:
        api_key = None
    api_key = api_key or load_env_key("OPENROUTER_API_KEY")
    if not api_key:
        return 0

    try:
        embedder = OpenAIEmbeddings(api_key=api_key)
        vectors = embedder.embed_documents(chunks_df["text"].astype(str).tolist())
        chunks_df["embedding"] = vectors
        con.register("pmbok_vec_df", chunks_df)
        con.execute("CREATE OR REPLACE TABLE pmbok_vectors AS SELECT * FROM pmbok_vec_df")
        return len(chunks_df)
    except Exception:
        return 0


@st.cache_data(ttl=300, show_spinner=False)
def load_project_geo() -> Optional[pd.DataFrame]:
    """
    Load manufacturing locations from DuckDB cache (FINAL_PROJECT); derive lat/lon from country if missing.
    Returns row-level points with lat/lon, Product, Manufactured by, Qty, Country.
    """
    try:
        refresh_cache()
        project_df, _ = load_cached_data()
    except Exception:
        return None
    if project_df is None or project_df.empty:
        return pd.DataFrame()

    df = project_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    manu_col = None
    for candidate in ["Manufactured by", "Manufacturer", "manufactured_by"]:
        if candidate in df.columns:
            manu_col = candidate
            break
    if manu_col is None:
        return None

    # Qty cleanup
    if "Qty" in df.columns:
        df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").fillna(1)
    else:
        df["Qty"] = 1

    # Country -> (country, iso3, lat, lon)
    country_map = {
        "japan": ("Japan", "JPN", 36.2048, 138.2529),
        "usa": ("United States", "USA", 37.0902, -95.7129),
        "united states": ("United States", "USA", 37.0902, -95.7129),
        "china": ("China", "CHN", 35.8617, 104.1954),
        "germany": ("Germany", "DEU", 51.1657, 10.4515),
        "thailand": ("Thailand", "THA", 15.87, 100.9925),
        "korea": ("South Korea", "KOR", 36.5, 127.8),
        "south korea": ("South Korea", "KOR", 36.5, 127.8),
        "vietnam": ("Vietnam", "VNM", 14.0583, 108.2772),
        "malaysia": ("Malaysia", "MYS", 4.2105, 101.9758),
        "singapore": ("Singapore", "SGP", 1.3521, 103.8198),
        "taiwan": ("Taiwan", "TWN", 23.6978, 120.9605),
        "india": ("India", "IND", 20.5937, 78.9629),
        "spain": ("Spain", "ESP", 40.4637, -3.7492),
        "espana": ("Spain", "ESP", 40.4637, -3.7492),
    }

    # Use existing coordinates if present; otherwise map by country
    lat_col = "Latitude" if "Latitude" in df.columns else ("lat" if "lat" in df.columns else None)
    lon_col = "Longitude" if "Longitude" in df.columns else ("lon" if "lon" in df.columns else None)

    if lat_col and lon_col:
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        df = df.dropna(subset=[lat_col, lon_col])
        df.rename(columns={lat_col: "Latitude", lon_col: "Longitude"}, inplace=True)
        df["Country"] = df.get("Country", df.get(manu_col, ""))
        df["iso3"] = df.get("iso3", "")
    else:
        df["country_norm"] = df[manu_col].astype(str).str.lower().map(country_map)
        df = df.dropna(subset=["country_norm"])
        if df.empty:
            return pd.DataFrame()
        df[["Country", "iso3", "Latitude", "Longitude"]] = pd.DataFrame(
            df["country_norm"].tolist(), index=df.index
        )
        df = df.dropna(subset=["Latitude", "Longitude"])

    if "Product" not in df.columns:
        df["Product"] = "Product"

    return df[["Latitude", "Longitude", "Product", manu_col, "Qty", "Country", "iso3"]].rename(
        columns={manu_col: "Manufactured by"}
    )


# Navigation setup (do not include this file as a page source to avoid recursion)
current_page = st.navigation(
    [
        st.Page(render_welcome, title="Welcome", icon="👋", default=True),
        st.Page("pages/project.py", title="Project", icon="📊"),
        st.Page("pages/Invoice.py", title="Invoice", icon="🧾"),
        st.Page("pages/CRM.py", title="CRM", icon="📈"),
        st.Page("pages/AI Integration.py", title="AI Integration", icon="🤖"),
    ]
)
current_page.run()
