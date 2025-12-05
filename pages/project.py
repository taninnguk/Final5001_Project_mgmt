import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from add_record_form import render_project_form
from data_cache import load_cached_data, refresh_cache, load_cached_meta, load_env_key
from openai import OpenAI

OPENROUTER_API_KEY = load_env_key("OPENROUTER_API_KEY")
openrouter_client = None
if OPENROUTER_API_KEY:
    try:
        openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    except Exception:
        openrouter_client = None

st.set_page_config(page_title="Project Management", page_icon="📊", layout="wide")


def fmt_m(value: float) -> str:
    if value is None or pd.isna(value):
        return "0"
    return f"{value/1_000_000:,.2f} M"

def metric_card(label: str, value: str, fg: str = "#0f172a", bg: str = "#f5f7fb") -> str:
    """Return HTML for a simple metric card."""
    return f"""
    <div style="
        padding: 12px 14px;
        border-radius: 10px;
        background: {bg};
        border: 1px solid #e0e4ef;
        margin: 6px 0;
    ">
        <div style="font-size: 12px; color: #475569; margin-bottom: 4px;">{label}</div>
        <div style="font-size: 22px; font-weight: 700; color: {fg}; line-height: 1.2;">{value}</div>
    </div>
    """


def ai_chart_summary(title: str, df: pd.DataFrame, hint: str, key: str, meta_text: str = "") -> None:
    """
    Render a button that asks AI to summarize a chart based on its data.
    Keeps the latest summary in session_state until page refresh/leave.
    """
    state_key = f"ai_summary_{key}"
    if st.button(f"🤖 AI summarize: {title}", key=key, use_container_width=True):
        if openrouter_client is None:
            st.error("OpenRouter client is not available. Set OPENROUTER_API_KEY in environment/.env.")
            return
        data_preview = "No data"
        if df is not None and not df.empty:
            data_preview = df.head(50).to_csv(index=False)
        system_prompt = (
            "คุณเป็นนักวิเคราะห์ข้อมูลที่สรุปผลกระชับเป็นภาษาไทยเท่านั้น "
            "สรุปกราฟด้านล่างเป็น bullet 2-4 ข้อ ระบุแนวโน้ม จุดสูง/ต่ำ ความเสี่ยง และข้อเสนอแนะที่เป็นไปได้ "
            "ถ้าข้อมูลไม่พอให้บอกอย่างตรงไปตรงมา"
        )
        if meta_text:
            system_prompt += f"\n\nข้อมูล schema/คำอธิบายคอลัมน์:\n{meta_text}"
        user_prompt = (
            f"หัวข้อกราฟ: {title}\n"
            f"บริบทกราฟ: {hint}\n"
            f"ข้อมูล (CSV แถวตัวอย่าง):\n{data_preview}\n"
            "ช่วยสรุปข้อมูลกราฟนี้เป็น bullet ภาษาไทย"
        )
        with st.spinner("กำลังสรุปด้วย AI..."):
            try:
                resp = openrouter_client.chat.completions.create(
                    model="tngtech/deepseek-r1t2-chimera:free",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                st.session_state[state_key] = resp.choices[0].message.content
            except Exception as exc:  # noqa: BLE001
                st.error(f"AI summary failed: {exc}")
    if state_key in st.session_state:
        st.info(st.session_state[state_key])


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


def meta_text_for_project(meta_df: pd.DataFrame) -> str:
    """
    Convert COLUMN_META rows to a compact text block for the AI prompt (project tables only).
    Expected columns: Table_name, Field_name, Description.
    """
    if meta_df is None or meta_df.empty:
        return ""
    cols = {c.strip().lower(): c for c in meta_df.columns}
    if not {"table_name", "field_name", "description"}.issubset(set(cols)):
        return ""
    tbl_col = cols["table_name"]
    fld_col = cols["field_name"]
    desc_col = cols["description"]

    meta_df = meta_df.copy()
    meta_df[tbl_col] = meta_df[tbl_col].astype(str).str.lower().str.strip()
    meta_df[fld_col] = meta_df[fld_col].astype(str).str.strip()
    meta_df[desc_col] = meta_df[desc_col].astype(str).str.strip()
    filtered = meta_df[meta_df[tbl_col].isin(["final_project", "project"])]
    lines = [f"{r[fld_col]}: {r[desc_col]}" for _, r in filtered.head(80).iterrows()]
    return "\n".join(lines)


try:
    refresh_cache()
    project_df_raw, invoice_df_raw = load_cached_data()
    project_df = clean_project(project_df_raw)
    invoice_df = clean_invoice(invoice_df_raw)
    meta_df = load_cached_meta()
    project_meta_text = meta_text_for_project(meta_df)
    data_source = "duckdb_cache"
except Exception as exc:  # noqa: BLE001
    data_source = "error"
    st.title("Project Management Dashboard")
    st.error(
        f"Data could not be loaded.\n\n{exc}",
        icon="🚫",
    )
    st.stop()

st.title("Project Management Dashboard")
if data_source == "duckdb_cache":
    st.caption("❄️ Using DuckDB cache from Snowflake (FINAL_PROJECT / FINAL_INVOICE)")
else:
    st.caption("🚫 Data not loaded")

nav_cols = st.columns(4)
with nav_cols[0]:
    st.page_link("pages/Invoice.py", label="Go to Invoice dashboard", icon="🧾")
with nav_cols[1]:
    st.page_link("pages/project.py", label="Stay on Project dashboard", icon="📊")
with nav_cols[2]:
    st.page_link("pages/CRM.py", label="Go to CRM dashboard", icon="📈")
with nav_cols[3]:
    with st.popover("➕ Add project record", use_container_width=True):
        render_project_form(form_key="project_add_form")

with st.sidebar:
    st.header("Filters")
    engineer_filter = st.multiselect(
        "Project engineer",
        sorted(project_df["Project Engineer"].dropna().unique()),
        default=[],
    )
    project_filter = st.multiselect(
        "Project",
        sorted(project_df["Project"].dropna().unique()),
        default=[],
    )
    year_filter = st.multiselect(
        "Project year",
        sorted(project_df["Project year"].dropna().unique()),
        default=[],
    )
    status_filter = st.multiselect(
        "Status",
        sorted(project_df["Status"].dropna().unique()),
        default=[],
    )
    phrase_filter = st.multiselect(
        "Project phrase",
        sorted(project_df["Project Phrase"].dropna().unique()),
    )
    customer_filter = st.multiselect(
        "Customer",
        sorted(project_df["Customer"].dropna().unique()),
    )

filtered = project_df.copy()
if engineer_filter:
    filtered = filtered[filtered["Project Engineer"].isin(engineer_filter)]
if project_filter:
    filtered = filtered[filtered["Project"].isin(project_filter)]
if year_filter:
    filtered = filtered[filtered["Project year"].isin(year_filter)]
if status_filter:
    filtered = filtered[filtered["Status"].isin(status_filter)]
if phrase_filter:
    filtered = filtered[filtered["Project Phrase"].isin(phrase_filter)]
if customer_filter:
    filtered = filtered[filtered["Customer"].isin(customer_filter)]

if filtered.empty:
    st.warning("No records match the current filters.")
    st.stop()

total_value = filtered["Project Value"].sum()
balance_sum = filtered["Balance"].sum()
avg_progress_pct = filtered["Progress"].mean()
avg_progress_pct = 0 if pd.isna(avg_progress_pct) else avg_progress_pct * 100
order_count = filtered["Order number"].nunique()

product_counts = {}
for product_name in ["Control Panel", "Heater", "Vessel"]:
    product_counts[product_name] = (
        filtered.loc[filtered["Product"].str.contains(product_name, case=False, na=False), "Qty"]
        .sum()
    )

status_totals = filtered["Status"].value_counts()

st.markdown("## Portfolio overview")
st.caption("สรุปมูลค่า คงเหลือ ความคืบหน้า และจำนวนออเดอร์ พร้อมยอดหน่วยสินค้าและสถานะในมุมมองเดียว")
summary_top = st.columns(4)
summary_top[0].markdown(metric_card("Sum of Project Value", fmt_m(total_value), fg="#2563eb"), unsafe_allow_html=True)
summary_top[1].markdown(metric_card("Sum of Balance", fmt_m(balance_sum), fg="#dc2626"), unsafe_allow_html=True)
summary_top[2].markdown(metric_card("Avg. Progress", f"{avg_progress_pct:,.0f}%", fg="#0ea5e9"), unsafe_allow_html=True)
summary_top[3].markdown(metric_card("Orders", int(order_count), fg="#0f172a"), unsafe_allow_html=True)

summary_bottom = st.columns(6)
summary_bottom[0].markdown(metric_card("Control Panel", int(product_counts.get("Control Panel", 0)), fg="#0f172a"), unsafe_allow_html=True)
summary_bottom[1].markdown(metric_card("Heater", int(product_counts.get("Heater", 0)), fg="#0f172a"), unsafe_allow_html=True)
summary_bottom[2].markdown(metric_card("Vessel", int(product_counts.get("Vessel", 0)), fg="#0f172a"), unsafe_allow_html=True)
summary_bottom[3].markdown(metric_card("Delayed", int(status_totals.get("Delayed", 0)), fg="#dc2626", bg="#fff2f2"), unsafe_allow_html=True)
summary_bottom[4].markdown(metric_card("On track", int(status_totals.get("On track", 0)), fg="#15803d", bg="#ecfdf3"), unsafe_allow_html=True)
summary_bottom[5].markdown(metric_card("Shipped", int(status_totals.get("Shipped", 0)), fg="#0ea5e9", bg="#f0f9ff"), unsafe_allow_html=True)

st.divider()

st.markdown("## Delivery & progress")
st.caption("ภาพรวมยอดมูลค่า/คงเหลือต่อออเดอร์ และเกจความคืบหน้าที่สรุปเฉลี่ยทุกโครงการ")
val_col_left, val_col_right = st.columns(2)

with val_col_left:
    st.caption("Top 20 orders by value (stacked with balance)")
    order_summary = (
        filtered.groupby("Order number", dropna=True)
        .agg(
            {
                "Project Value": "sum",
                "Balance": "sum",
                "Project": lambda x: x.dropna().iloc[0] if not x.dropna().empty else "",
            }
        )
        .reset_index()
        .sort_values("Project Value", ascending=False)
        .head(20)
    )
    if not order_summary.empty:
        order_summary["Order display"] = order_summary["Order number"].astype("string").fillna("").str.strip()
        long_orders = order_summary.melt(
            id_vars=["Order number", "Order display"],
            value_vars=["Project Value", "Balance"],
            var_name="Metric",
            value_name="Amount",
        )
        order_fig = px.bar(
            long_orders,
            x="Amount",
            y="Order display",
            color="Metric",
            orientation="h",
            labels={"Amount": "Amount", "Order display": "Order number"},
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        order_fig.update_traces(hovertemplate="<b>Order %{y}</b><br>%{fullData.name}: %{x:,.0f}", marker_line_width=0.6)
        order_fig.update_layout(
            showlegend=True,
            margin=dict(l=10, r=10, t=30, b=10),
            height=420,
            yaxis=dict(
                title="Order number",
                tickfont=dict(size=13),
                type="category",
                categoryorder="array",
                categoryarray=order_summary["Order display"].tolist()[::-1],
            ),
            xaxis=dict(title="Amount", tickfont=dict(size=12)),
            bargap=0.2,
        )
        st.plotly_chart(order_fig, use_container_width=True)
        ai_chart_summary(
            "Top 20 orders by value (with balance)",
            order_summary[["Order display", "Project Value", "Balance"]],
            "Each row is an order; Project Value and Balance are amounts.",
            key="ai_order_summary",
            meta_text=project_meta_text,
        )
    else:
        st.info("No order number data to display.")

with val_col_right:
    st.caption("Average progress (ทุกโครงการที่กรอง)")
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_progress_pct,
            number={"suffix": "%", "valueformat": ".0f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 50], "color": "#f4d6d6"},
                    {"range": [50, 80], "color": "#f9e9c5"},
                    {"range": [80, 100], "color": "#d6f4da"},
                ],
            },
        )
    )
    gauge.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=20))
    st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
    st.caption("ค่าความคืบหน้าเฉลี่ยหลังกรองข้อมูล ใช้ดูภาพรวมการส่งมอบ")
    status_cols = st.columns(3)
    status_cols[0].markdown(metric_card("Delayed", int(status_totals.get("Delayed", 0)), fg="#dc2626", bg="#fff2f2"), unsafe_allow_html=True)
    status_cols[1].markdown(metric_card("On track", int(status_totals.get("On track", 0)), fg="#15803d", bg="#ecfdf3"), unsafe_allow_html=True)
    status_cols[2].markdown(metric_card("Shipped", int(status_totals.get("Shipped", 0)), fg="#0ea5e9", bg="#f0f9ff"), unsafe_allow_html=True)
    status_df = pd.DataFrame(
        {
            "Metric": ["Avg progress %", "Delayed count", "On track count", "Shipped count"],
            "Value": [
                round(avg_progress_pct, 2),
                int(status_totals.get("Delayed", 0)),
                int(status_totals.get("On track", 0)),
                int(status_totals.get("Shipped", 0)),
            ],
        }
    )
    ai_chart_summary(
        "Progress gauge and status counts",
        status_df,
        "Gauge shows average progress percentage; counts show number of projects per status.",
        key="ai_progress_gauge",
        meta_text=project_meta_text,
    )

st.divider()

st.markdown("## Mix by owner/customer")
st.caption("สัดส่วนมูลค่าโครงการแยกตามวิศวกรผู้ดูแลและลูกค้า")
pie_col1, pie_col2 = st.columns(2)
with pie_col1:
    engineer_value = (
        filtered.groupby("Project Engineer", as_index=False)["Project Value"]
        .sum()
        .sort_values("Project Value", ascending=False)
    )
    if not engineer_value.empty:
        eng_fig = px.pie(
            engineer_value,
            names="Project Engineer",
            values="Project Value",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        eng_fig.update_traces(hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f}<br>%{percent}")
        eng_fig.update_layout(height=420)
        st.plotly_chart(eng_fig, use_container_width=True)
        ai_chart_summary(
            "Project value by engineer",
            engineer_value,
            "Sum of project value by project engineer; values are currency amounts.",
            key="ai_engineer_pie",
            meta_text=project_meta_text,
        )
    else:
        st.info("No engineer data.")
with pie_col2:
    customer_value = (
        filtered.groupby("Customer", as_index=False)["Project Value"]
        .sum()
        .sort_values("Project Value", ascending=False)
    )
    if not customer_value.empty:
        cust_fig = px.pie(
            customer_value,
            names="Customer",
            values="Project Value",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        cust_fig.update_traces(hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f}<br>%{percent}")
        cust_fig.update_layout(height=420)
        st.plotly_chart(cust_fig, use_container_width=True)
        ai_chart_summary(
            "Project value by customer",
            customer_value,
            "Sum of project value by customer; values are currency amounts.",
            key="ai_customer_pie",
            meta_text=project_meta_text,
        )
    else:
        st.info("No customer data.")

st.divider()

st.markdown("## Operations & status")
st.caption("ซัพพลายเชน: จำนวนสินค้าแยกผู้ผลิต/สินค้า, พร้อมสถานะและคำสำคัญของโครงการ")
table_col_left, table_col_right = st.columns(2)

with table_col_left:
    st.caption("Manufactured by / Product (sum of Qty)")
    qty_by_manu = (
        filtered.groupby(["Manufactured by", "Product"], as_index=False)["Qty"]
        .sum()
        .sort_values("Qty", ascending=False)
    )
    if not qty_by_manu.empty:
        manu_fig = px.bar(
            qty_by_manu,
            x="Qty",
            y="Manufactured by",
            color="Product",
            orientation="h",
            labels={"Qty": "Units", "Manufactured by": "Manufacturer"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        manu_fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>%{y}<br>Qty: %{x:,.0f}")
        manu_fig.update_traces(customdata=qty_by_manu[["Product"]])
        manu_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=420)
        st.plotly_chart(manu_fig, use_container_width=True)
        ai_chart_summary(
            "Units by manufacturer and product",
            qty_by_manu,
            "Sum of quantity (Qty) grouped by manufacturer and product; sorted by Qty.",
            key="ai_manu_qty",
            meta_text=project_meta_text,
        )
    else:
        st.info("No manufacturing data.")

with table_col_right:
    total_status_rows = len(filtered)
    metric_a, metric_b = st.columns(2)
    metric_a.metric("Status rows", total_status_rows)
    metric_b.metric("Orders", order_count)

    phrase_counts = filtered["Project Phrase"].value_counts().rename_axis("Project Phrase").reset_index(name="Count")
    st.caption("Project phrases (top keywords)")
    if not phrase_counts.empty:
        phrase_fig = px.bar(
            phrase_counts.sort_values("Count").tail(15),
            x="Count",
            y="Project Phrase",
            orientation="h",
            labels={"Count": "Projects"},
            color="Count",
            color_continuous_scale="Greens",
        )
        phrase_fig.update_traces(hovertemplate="<b>%{y}</b><br>Projects: %{x}")
        phrase_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=420)
        st.plotly_chart(phrase_fig, use_container_width=True)
        ai_chart_summary(
            "Project phrases (keywords)",
            phrase_counts,
            "Counts of project phrases/keywords; top 15 shown.",
            key="ai_phrase_counts",
            meta_text=project_meta_text,
        )
    else:
        st.info("No phrase data.")

st.markdown("## Project details")
st.caption("ตารางโครงการแบบละเอียด ใช้กรองด้านซ้ายเพื่อลดรายการและโฟกัสเฉพาะที่สนใจ")
display_cols = [
    "Project",
    "Customer",
    "Project Engineer",
    "Project year",
    "Status",
    "Project Phrase",
    "Product",
    "Order number",
    "Progress",
    "Estimated shipdate",
    "Actual shipdate",
    "Balance",
    "Project Value",
]
existing_cols = [c for c in display_cols if c in filtered.columns]
st.dataframe(filtered[existing_cols].sort_values("Project"), use_container_width=True)
