import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI

from add_record_form import render_invoice_form
from data_cache import load_cached_data, refresh_cache, load_cached_meta, load_env_key

st.set_page_config(page_title="Invoice Dashboard", page_icon="🧾", layout="wide")

OPENROUTER_API_KEY = st.secrets.get("api", {}).get("OPENROUTER_API_KEY") if hasattr(st, "secrets") else None
OPENROUTER_API_KEY = OPENROUTER_API_KEY or load_env_key("OPENROUTER_API_KEY")
openrouter_client = None
if OPENROUTER_API_KEY:
    try:
        openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    except Exception:
        openrouter_client = None


def fmt_m(value: float) -> str:
    if value is None or pd.isna(value):
        return "0"
    return f"{value/1_000_000:,.2f} M"


def format_dates_for_display(df: pd.DataFrame, date_columns: list[str], fmt: str = "%Y-%m-%d") -> pd.DataFrame:
    """Return a copy with date columns rendered without time."""
    formatted = df.copy()
    for col in date_columns:
        if col in formatted.columns:
            formatted[col] = pd.to_datetime(formatted[col], errors="coerce").dt.strftime(fmt)
            formatted[col] = formatted[col].fillna("")
    return formatted


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


def normalize_order_number(value) -> str:
    """Convert order numbers to comparable strings for joining."""
    if pd.isna(value):
        return ""
    # Handle numbers that may come as floats (e.g., 1234.0)
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return ""
        try:
            return str(int(value))
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value).strip()


def combine_columns(df: pd.DataFrame, primary: str, secondary: str) -> pd.Series:
    """Return primary column with fallback to secondary, safely handling missing columns."""
    primary_series = df[primary] if primary in df else pd.Series([None] * len(df))
    secondary_series = df[secondary] if secondary in df else pd.Series([None] * len(df))
    return primary_series.combine_first(secondary_series)



try:
    refresh_cache()
    project_df_raw, invoice_df_raw = load_cached_data()
    project_df = clean_project(project_df_raw)
    invoice_df = clean_invoice(invoice_df_raw)
except Exception as exc:  # noqa: BLE001
    st.title("Invoice Dashboard")
    st.error(f"Data could not be loaded.\n\n{exc}", icon="🚫")
    st.stop()

st.title("Invoice Dashboard")
st.caption(
    "❄️ Using DuckDB cache from Snowflake (FINAL_PROJECT / FINAL_INVOICE)"
)
nav_cols = st.columns(4)
with nav_cols[0]:
    st.page_link("pages/project.py", label="Go to Project dashboard", icon="📊")
with nav_cols[1]:
    st.page_link("pages/Invoice.py", label="Stay on Invoice dashboard", icon="🧾")
with nav_cols[2]:
    st.page_link("pages/CRM.py", label="Go to CRM dashboard", icon="📈")
with nav_cols[3]:
    with st.popover("➕ Add invoice record", use_container_width=True):
        render_invoice_form(form_key="invoice_add_form")

# Sync invoice rows with project metadata for richer visuals.
invoice_df["Order number"] = invoice_df.get("Sale order No.", pd.Series(dtype=object)).apply(normalize_order_number)
project_df["Order number"] = project_df["Order number"].apply(normalize_order_number)
project_lookup_cols = ["Order number", "Project", "Customer", "Project Value", "Balance", "Project Engineer", "Status", "Progress"]
project_lookup = project_df[[c for c in project_lookup_cols if c in project_df.columns]].drop_duplicates(subset="Order number")
merged = invoice_df.merge(project_lookup, on="Order number", how="left", suffixes=("", "_project"))

# Unify key text columns for filtering.
merged["Project Engineer Combined"] = combine_columns(merged, "Project Engineer", "Project Engineer_project")
merged["Customer Combined"] = combine_columns(merged, "Customer", "Customer_project")
merged["Project Combined"] = combine_columns(merged, "Project", "Project_project")

with st.sidebar:
    st.header("Filters")
    engineer_filter = st.multiselect(
        "Project engineer",
        sorted([e for e in merged["Project Engineer Combined"].dropna().unique()]),
    )
    project_filter = st.multiselect(
        "Project",
        sorted([p for p in merged["Project Combined"].dropna().unique()]),
    )
    year_filter = st.multiselect(
        "Project year",
        sorted([int(y) for y in merged["Project year"].dropna().unique() if pd.notna(y)]),
    )
    customer_filter = st.multiselect(
        "Customer",
        sorted([c for c in merged["Customer Combined"].dropna().unique()]),
    )
    payment_filter = st.multiselect(
        "Payment status",
        sorted([s for s in merged["Payment Status"].dropna().unique()]),
    )

filtered = merged.copy()
if engineer_filter:
    filtered = filtered[filtered["Project Engineer Combined"].isin(engineer_filter)]
if project_filter:
    filtered = filtered[filtered["Project Combined"].isin(project_filter)]
if year_filter:
    filtered = filtered[filtered["Project year"].isin(year_filter)]
if customer_filter:
    filtered = filtered[filtered["Customer Combined"].isin(customer_filter)]
if payment_filter:
    filtered = filtered[filtered["Payment Status"].isin(payment_filter)]

if filtered.empty:
    st.warning("No invoice records match the current filters.")
    st.stop()

total_invoice_value = filtered["Invoice value"].sum()
# De-duplicate matched projects by order number to avoid double-counting project value/balance
project_for_metrics = (
    filtered[["Order number", "Project Value", "Balance"]]
    .dropna(subset=["Order number"])
    .drop_duplicates(subset="Order number")
)
total_project_value = project_for_metrics["Project Value"].sum()
coverage_pct = 0 if total_project_value == 0 else (total_invoice_value / total_project_value) * 100
balance_total = project_for_metrics["Balance"].sum()

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Total invoiced", fmt_m(total_invoice_value))
metric_col2.metric("Project value (matched)", fmt_m(total_project_value))
metric_col3.metric("Coverage vs project", f"{coverage_pct:,.1f}%")
metric_col4.metric("Balance (matched projects)", fmt_m(balance_total))

dist_left, dist_right = st.columns(2)

with dist_left:
    dist_left.subheader("Payment status")
    payment_counts = filtered["Payment Status"].value_counts()
    if not payment_counts.empty:
        pay_fig = px.pie(
            payment_counts.rename_axis("Payment Status").reset_index(name="Count"),
            names="Payment Status",
            values="Count",
            hole=0.4,
        )
        pay_fig.update_traces(hovertemplate="<b>%{label}</b><br>Count: %{value}")
        st.plotly_chart(pay_fig, use_container_width=True)
    else:
        st.info("No payment status data.")


with dist_right:
    dist_right.subheader("Invoice distribution by owner/year")
    year_status = (
        filtered.dropna(subset=["Project year", "Payment Status"])
        .groupby(["Project year", "Payment Status"])["Invoice value"]
        .sum()
        .reset_index()
    )
    if not year_status.empty:
        year_fig = px.bar(
            year_status,
            x="Project year",
            y="Invoice value",
            color="Payment Status",
            barmode="stack",
            labels={"Invoice value": "Invoice value", "Project year": "Year"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        year_fig.update_traces(hovertemplate="<b>Year %{x}</b><br>Status: %{customdata[0]}<br>Invoice: %{y:,.0f}")
        year_fig.update_traces(customdata=year_status[["Payment Status"]])
        year_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=420)
        st.plotly_chart(year_fig, use_container_width=True)
    else:
        st.info("No year/payment status data.")



st.subheader("Invoice plan vs actual (monthly)")
monthly_plan = (
    filtered.dropna(subset=["Invoice plan date"])
    .assign(month=lambda df: df["Invoice plan date"].dt.to_period("M").dt.to_timestamp())
    .groupby("month")["Invoice value"]
    .sum()
    .reset_index()
    .rename(columns={"Invoice value": "Planned"})
)
monthly_actual = (
    filtered.dropna(subset=["Actual Payment received date"])
    .assign(month=lambda df: df["Actual Payment received date"].dt.to_period("M").dt.to_timestamp())
    .groupby("month")["Invoice value"]
    .sum()
    .reset_index()
    .rename(columns={"Invoice value": "Actual"})
)
monthly_actual_status = (
    filtered.dropna(subset=["Actual Payment received date"])
    .assign(month=lambda df: df["Actual Payment received date"].dt.to_period("M").dt.to_timestamp())
    .groupby(["month", "Payment Status"])["Invoice value"]
    .sum()
    .reset_index()
)
monthly = pd.merge(monthly_plan, monthly_actual, on="month", how="outer").fillna(0).sort_values("month")
if not monthly.empty:
    monthly["month_str"] = monthly["month"].dt.strftime("%Y-%m")
    monthly_actual_status["month_str"] = monthly_actual_status["month"].dt.strftime("%Y-%m")

    palette = {
        "Paid": px.colors.qualitative.Set2[1],
        "Invoiced": px.colors.qualitative.Set2[2] if len(px.colors.qualitative.Set2) > 2 else "#a78bfa",
        "Planned": px.colors.qualitative.Set2[3] if len(px.colors.qualitative.Set2) > 3 else "#22c55e",
        "Overdue": "#ef4444",
        "": "#94a3b8",
    }

    # Actual as stacked bars by Payment Status
    bar_fig = px.bar(
        monthly_actual_status,
        x="month_str",
        y="Invoice value",
        color="Payment Status",
        labels={"Invoice value": "Invoice value", "month_str": "Month", "Payment Status": "Status"},
        color_discrete_map=palette,
    )
    for trace in bar_fig.data:
        trace.update(
            hovertemplate="<b>%{x}</b><br>Status: %{customdata[0]}<br>Actual: %{y:,.0f}",
            customdata=monthly_actual_status[["Payment Status"]],
            marker_line_width=0.6,
        )

    # Planned as line
    planned_df = monthly[["month_str", "Planned"]]
    line_trace = px.line(
        planned_df,
        x="month_str",
        y="Planned",
        labels={"Planned": "Invoice value", "month_str": "Month"},
        color_discrete_sequence=[px.colors.qualitative.Set2[0]],
    ).data[0]
    line_trace.update(
        name="Planned",
        legendgroup="Planned",
        hovertemplate="<b>%{x}</b><br>Planned: %{y:,.0f}",
        line=dict(width=2.4),
        marker=dict(size=7, symbol="circle"),
    )

    # Combine traces
    monthly_fig = px.line()  # empty fig
    for trace in bar_fig.data:
        monthly_fig.add_trace(trace)
    monthly_fig.add_trace(line_trace)

    monthly_fig.update_layout(
        legend=dict(title=None),
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(tickangle=-45, tickfont=dict(size=12), categoryorder="category ascending", title="Month"),
        yaxis=dict(tickfont=dict(size=12), title="Invoice value"),
        bargap=0.15,
    )
    st.plotly_chart(monthly_fig, use_container_width=True)
    ai_chart_summary(
        "Planned vs actual invoice (monthly)",
        monthly_actual_status,
        "Stacked bars show actual invoice value by Payment Status each month; line shows planned invoice value.",
        key="ai_invoice_monthly",
    )
else:
    st.info("No monthly plan or actual data to chart.")

st.subheader("Invoice details (joined with projects)")
display_cols = [
    "Project Combined",
    "Order number",
    "Customer Combined",
    "Project Engineer Combined",
    "Project year",
    "Payment Status",
    "Invoice plan date",
    "Actual Payment received date",
    "Invoice value",
    "Claim Plan 2025",
    "Project Value",
    "Balance",
]
existing_cols = [c for c in display_cols if c in filtered.columns]
table_df = filtered[existing_cols].rename(
    columns={
        "Project Combined": "Project",
        "Customer Combined": "Customer",
        "Project Engineer Combined": "Project Engineer",
    }
)
sort_col = "Invoice plan date" if "Invoice plan date" in table_df.columns else None
if sort_col:
    table_df = table_df.sort_values(sort_col)
table_df = format_dates_for_display(
    table_df,
    [
        "Invoice plan date",
        "Issued Date",
        "Invoice due date",
        "Plan payment date",
        "Expected Payment date",
        "Actual Payment received date",
    ],
)
st.dataframe(table_df, use_container_width=True, height=420)
