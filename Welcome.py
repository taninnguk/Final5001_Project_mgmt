import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Optional

st.set_page_config(page_title="Welcome", page_icon="👋", layout="wide")


def render_welcome() -> None:
    st.title("Welcome to the Project & Invoice Hub")
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

    st.markdown("## Executive summary")
    st.write(
        """
        ระบบนี้ช่วยให้คุณติดตามสถานะโครงการและใบแจ้งหนี้ได้ครบวงจร พร้อมผู้ช่วย AI สำหรับการถาม-ตอบเชิงบริบท:
        - **Project dashboard**: ดูมูลค่าโครงการ, ยอดคงเหลือ, ความคืบหน้า, top orders, พายแบ่งตามวิศวกร/ลูกค้า, และปริมาณสินค้าตามผู้ผลิต
        - **Invoice dashboard**: ดูมูลค่าใบแจ้งหนี้, สถานะการชำระเงิน, การวางแผน/รับเงินรายเดือน, และการเชื่อมโยงกับข้อมูลโครงการ
        - **AI assistant**: ถาม-ตอบเรื่องโครงการ/ใบแจ้งหนี้ด้วยข้อมูลที่มี (RAG) พร้อมความรู้ PMBOK และ workflow ของโปรเจกต์
        - **CRM dashboard** : ติดตามใบเสร็จและลูกค้า เพื่อบริหารจัดความสัมพันธ์กับลูกค้า
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
        - กดปุ่ม **Add record** บนหน้า Project/Invoice เพื่อเพิ่มข้อมูลใหม่ (เชื่อมกับ Google Sheets/Excel)
        - ใน AI assistant สามารถเลือกใช้ข้อมูล Project/Invoice หรือรวมกัน และเปิดใช้ความรู้ PMBOK ได้
        """
    )

    st.success("พร้อมใช้งาน: เลือกลิงก์ด้านบนเพื่อเริ่มสำรวจข้อมูลหรือถาม AI ได้ทันที", icon="✅")

    st.markdown("## Where we manufacture (preview)")
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


@st.cache_data(ttl=300, show_spinner=False)
def load_project_geo() -> Optional[pd.DataFrame]:
    """
    Load manufacturing locations from Snowflake FINAL_PROJECT; derive lat/lon from country if missing.
    Returns row-level points with lat/lon, Product, Manufactured by, Qty, Country.
    """
    try:
        conn = st.connection("snowflake")
        df = conn.query("SELECT * FROM FINAL_PROJECT;", ttl=300)
    except Exception:
        return None
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
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
