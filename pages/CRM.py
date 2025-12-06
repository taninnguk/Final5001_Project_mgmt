import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt
from data_cache import load_cached_data, refresh_cache, load_cached_meta, ai_chart_summary


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="CRM Invoice Dashboard",
    layout="wide"
)

def clean_invoice(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    date_cols = [
        "Invoice plan date",
        "Invoice Issued Date",
        "Invoice due date",
        "Plan payment date",
        "Expected Payment date",
        "Actual Payment received date",
    ]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
    money_cols = ["Total amount", "Invoice value"]
    for c in money_cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(" ", "", regex=False)
                .str.replace("-", "0", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    today = pd.Timestamp.today().normalize()
    if "Expected Payment date" in df.columns:
        df["days_to_expected"] = (df["Expected Payment date"] - today).dt.days
        df["is_overdue"] = df["days_to_expected"] < 0
    else:
        df["days_to_expected"] = pd.NA
        df["is_overdue"] = False
    if {"Actual Payment received date", "Expected Payment date"}.issubset(df.columns):
        df["days_diff_actual_expected"] = (
            df["Actual Payment received date"] - df["Expected Payment date"]
        ).dt.days
    df["Payment Status"] = df["Payment Status"].astype(str).str.strip()
    df["status_lower"] = df["Payment Status"].str.lower()
    df["is_paid"] = df["status_lower"].str.startswith("paid")
    return df


def format_dates_for_display(df: pd.DataFrame, date_columns: list[str], fmt: str = "%Y-%m-%d") -> pd.DataFrame:
    """Return a copy with date columns rendered without time."""
    formatted = df.copy()
    for col in date_columns:
        if col in formatted.columns:
            formatted[col] = pd.to_datetime(formatted[col], errors="coerce").dt.strftime(fmt)
            formatted[col] = formatted[col].fillna("")
    return formatted


# Header section
st.title("CRM Invoice Dashboard")
st.caption("❄️ Using DuckDB cache from Snowflake (FINAL_INVOICE)")
nav_cols = st.columns(4)
with nav_cols[0]:
    st.page_link("pages/project.py", label="📊 Go to Project dashboard")
with nav_cols[1]:
    st.page_link("pages/Invoice.py", label="🧾 Go to Invoice dashboard")
with nav_cols[2]:
    st.page_link("pages/CRM.py", label="📈 Stay on CRM dashboard")
with nav_cols[3]:
    with st.popover("➕ Add invoice record", use_container_width=True):
        # Reuse the add_record_form invoice form
        from add_record_form import render_invoice_form

        render_invoice_form(form_key="crm_add_invoice_form")

# ---------------------------------------------------
# LOAD & PREPARE DATA
# ---------------------------------------------------
refresh_cache()
project_df_raw, df = load_cached_data()
df = clean_invoice(df)

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.header("Filters")

# Filter ตาม Payment Status (CRM อยากเลือกดูสถานะต่าง ๆ ได้)
payment_statuses = sorted(df["Payment Status"].dropna().unique())
status_selected = st.sidebar.multiselect(
    "Payment Status",
    payment_statuses,
    default=[],
    help="ถ้าไม่เลือกจะแสดงทุกสถานะ",
)

df_filtered = df.copy()
if status_selected:
    df_filtered = df_filtered[df_filtered["Payment Status"].isin(status_selected)]

# Optional: filter ตามลูกค้า
customers = ["All"] + sorted(df_filtered["Customer"].dropna().unique())
customer_selected = st.sidebar.selectbox("Customer", customers, index=0)

if customer_selected != "All":
    df_filtered = df_filtered[df_filtered["Customer"] == customer_selected]

# ---------------------------------------------------
# TOP SUMMARY KPI : Aging & Unpaid Customers
# ---------------------------------------------------
st.markdown("## Summary (Current Filters)")

df_current = df_filtered.copy()

# 1) Total Aging Amount (เฉพาะสถานะ Aging)
aging_mask = df_current["status_lower"] == "aging"
total_aging_amount = df_current.loc[aging_mask, "Invoice value"].sum()

# 2) จำนวนลูกค้าที่มี invoice ยังไม่ชำระ (ไม่ใช่ Paid)
unpaid_mask = ~df_current["is_paid"]
n_unpaid_customers = df_current.loc[unpaid_mask, "Customer"].nunique()

k1, k2 = st.columns(2)

with k1:
    st.metric(
        "Total Aging Amount",
        f"{total_aging_amount:,.0f}"
    )

with k2:
    st.metric(
        "Customers with Unpaid Invoices",
        f"{n_unpaid_customers:,d}"
    )


st.markdown("---")
st.header("Project Value Overview (Total amount)")

# ใช้ข้อมูลหลังกรอง (df_filtered) เพื่อให้สอดคล้องกับ filter ด้านซ้าย
if df_filtered.empty:
    st.info("No data for Project Value overview with current filters.")
else:
    # รวมมูลค่าโครงการตาม Customer
    cust_val = (
        df_filtered
        .groupby("Customer", as_index=False)["Total amount"]
        .sum()
        .sort_values("Total amount", ascending=False)
    )

    # รวมมูลค่าโครงการตาม Project Engineer
    eng_val = (
        df_filtered
        .groupby("Project Engineer", as_index=False)["Total amount"]
        .sum()
        .sort_values("Total amount", ascending=False)
    )


    c1, c2 = st.columns(2)

    with c1:
        st.subheader("💰 Customer Lifetime Value (CLV)")
        chart_cust = (
            alt.Chart(cust_val)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Total amount:Q",
                    title="Total amount",
                    axis=alt.Axis(format=",.0f"),
                ),
                y=alt.Y("Customer:N", sort="-x", title="Customer"),
                tooltip=[
                    alt.Tooltip("Customer:N", title="Customer"),
                    alt.Tooltip("Total amount:Q", title="Total amount", format=",.0f"),
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(chart_cust, use_container_width=True)
        ai_chart_summary(
            "Customer lifetime value (total amount)",
            cust_val,
            "Bar chart of total amount per customer (current filters).",
            key="ai_crm_cust_val",
        )


    with c2:
        st.subheader("👷‍♂️ Total Project Value by Project Engineer")
        chart_eng = (
            alt.Chart(eng_val)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Total amount:Q",
                    title="Total amount",
                    axis=alt.Axis(format=",.0f"),
                ),
                y=alt.Y("Project Engineer:N", sort="-x", title="Project Engineer"),
                tooltip=[
                    alt.Tooltip("Project Engineer:N", title="Project Engineer"),
                    alt.Tooltip("Total amount:Q", title="Total amount", format=",.0f"),
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(chart_eng, use_container_width=True)
        ai_chart_summary(
            "Project value by engineer",
            eng_val,
            "Bar chart of total amount per project engineer (current filters).",
            key="ai_crm_eng_val",
        )









# ---------------------------------------------------
# SECTION 2: CUSTOMER PAYMENT BEHAVIOR (FAST vs SLOW)
# ---------------------------------------------------
st.markdown("---")
st.header("Customer Payment Behavior (Fast vs Slow Payers)")

# ใช้ข้อมูลเฉพาะ invoice ที่จ่ายแล้ว + มี Expected & Actual Payment
df_behavior = df[
    df["is_paid"]
    & df["Expected Payment date"].notna()
    & df["Actual Payment received date"].notna()
].copy()

if df_behavior.empty:
    st.info("No paid invoices with Expected & Actual payment dates to analyze.")
else:
    # ถ้าบาง invoice ยังไม่ได้คำนวณ days_diff_actual_expected ให้ทำซ้ำอีกครั้ง (กันพลาด)
    df_behavior["days_diff_actual_expected"] = (
        df_behavior["Actual Payment received date"] - df_behavior["Expected Payment date"]
    ).dt.days

    # คำนวณเฉลี่ยต่อลูกค้า
    behavior = (
        df_behavior.groupby("Customer")["days_diff_actual_expected"]
        .mean()
        .reset_index()
        .rename(columns={"days_diff_actual_expected": "avg_days_diff"})
    )

    # ยิ่ง avg_days_diff น้อย (ติดลบมาก) = จ่ายเร็ว
    best = behavior.sort_values("avg_days_diff").set_index("Customer").head(3)
    worst = behavior.sort_values("avg_days_diff", ascending=False).set_index("Customer").head(3)

    # แปลงคำอธิบาย
    def describe_behavior(d):
        if pd.isna(d):
            return ""
        d = float(d)
        if d < 0:
            return f"Pays {abs(d):.1f} days earlier on average"
        elif d == 0:
            return "Pays on time"
        else:
            return f"Pays {d:.1f} days late on average"

    best["Behavior"] = best["avg_days_diff"].apply(describe_behavior)
    worst["Behavior"] = worst["avg_days_diff"].apply(describe_behavior)

    b1, b2 = st.columns(2)

    with b1:
        st.subheader("⭐ Top Fast Payers")
        st.caption("ลูกค้าที่จ่ายเร็วกว่าคาด (ค่าติดลบมาก = ดี)")
        st.dataframe(
            best.rename(columns={"avg_days_diff": "Avg days (Actual - Expected)"}),
            use_container_width=True,
        )

    with b2:
        st.subheader("⚠️ Top Slow Payers")
        st.caption("ลูกค้าที่จ่ายช้ากว่าคาด (ค่าสูงกว่า 0 = เสี่ยง)")
        st.dataframe(
            worst.rename(columns={"avg_days_diff": "Avg days (Actual - Expected)"}),
            use_container_width=True,
        )



# ---------------------------------------------------
# SECTION 4: Customer Lifetime Value (CLV)
# ---------------------------------------------------
st.markdown("---")
st.header("💰 Customer Lifetime Value (CLV)")

df_clv = df.copy()

# สร้าง project_year ถ้ายังไม่มี
if "project_year" not in df_clv.columns:
    if "Invoice Issued Date" in df_clv.columns:
        df_clv["project_year"] = df_clv["Invoice Issued Date"].dt.year
    else:
        df_clv["project_year"] = df_clv["invoice_date"].dt.year

# กันเคสไม่มี Total amount
if df_clv["Total amount"].isna().all():
    st.info("Total amount is missing for all rows. Cannot compute CLV.")
else:
    # รวมข้อมูลต่อ Customer
    clv = (
        df_clv.groupby("Customer")
        .agg(
            total_project_value=("Total amount", "sum"),
            total_invoice_value=("Invoice value", "sum"),
            n_invoices=("Invoice value", "count"),
            first_year=("project_year", "min"),
            last_year=("project_year", "max"),
        )
        .reset_index()
    )

    # สร้างช่วงปีความสัมพันธ์
    clv["years_span"] = (clv["last_year"] - clv["first_year"]).abs() + 1

    # ค่าเฉลี่ยต่อปี
    clv["avg_yearly_value"] = clv["total_project_value"] / clv["years_span"]

    # 🔧 แปลงปีให้เป็นเลขจำนวนเต็ม (ไม่มีทศนิยม)
    int_cols = ["first_year", "last_year", "years_span"]
    for col in int_cols:
        clv[col] = clv[col].fillna(np.nan).astype("Int64")

    # เรียงตามมูลค่าโปรเจค
    clv_top = clv.sort_values("total_project_value", ascending=False)

    TOP_N = 15
    clv_chart_data = clv_top.head(TOP_N)

    st.subheader(f"Top {TOP_N} Customers by Total Project Value")

    # กราฟ CLV
    chart_clv = (
        alt.Chart(clv_chart_data)
        .mark_bar()
        .encode(
            x=alt.X(
                "total_project_value:Q",
                title="Total Project Value",
                axis=alt.Axis(format=",.0f"),
            ),
            y=alt.Y("Customer:N", sort="-x"),
            tooltip=[
                alt.Tooltip("Customer:N"),
                alt.Tooltip("total_project_value:Q", format=",.0f"),
                alt.Tooltip("total_invoice_value:Q", format=",.0f"),
                alt.Tooltip("n_invoices:Q"),
                alt.Tooltip("first_year:Q", format="d"),
                alt.Tooltip("last_year:Q", format="d"),
                alt.Tooltip("avg_yearly_value:Q", format=",.0f"),
            ],
        )
        .properties(height=450)
    )

    st.altair_chart(chart_clv, use_container_width=True)
    ai_chart_summary(
        "Top customers by total project value",
        clv_chart_data,
        "Bar chart of top customers by total project value (CLV view).",
        key="ai_crm_clv",
    )

    
# ---------------------------------------------------
# SECTION 1: INVOICE OVERVIEW (CRM VIEW)
# ---------------------------------------------------




    # ---------------------------------------------------      
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
    existing_cols = [c for c in display_cols if c in df_filtered.columns]
    table_df = df_filtered[existing_cols].rename(
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

def format_dates_for_display(df_in: pd.DataFrame, date_cols):
    df_out = df_in.copy()
    for c in date_cols:
        if c in df_out.columns:
            df_out[c] = pd.to_datetime(df_out[c], errors="coerce").dt.date
    return df_out

# ---------------------------------------------------
# SECTION 1: INVOICE DETAILS (JOINED VIEW)
# ---------------------------------------------------
st.title("CRM Invoice Overview")

if df_filtered.empty:
    st.warning("No data for current filter selection.")
else:
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

    # เลือกเฉพาะคอลัมน์ที่มีจริงใน df_filtered
    existing_cols = [c for c in display_cols if c in df_filtered.columns]

    if not existing_cols:
        st.info("No matching columns found in current data.")
    else:
        table_df = df_filtered[existing_cols].rename(
            columns={
                "Project Combined": "Project",
                "Customer Combined": "Customer",
                "Project Engineer Combined": "Project Engineer",
            }
        )

        # sort ตาม Invoice plan date ถ้ามี
        sort_col = "Invoice plan date" if "Invoice plan date" in table_df.columns else None
        if sort_col:
            table_df = table_df.sort_values(sort_col)

        # แปลงวันที่ให้ตัดเวลาออก
        table_df = format_dates_for_display(
            table_df,
            [
                "Invoice plan date",
                "Invoice Issued Date",
                "Invoice due date",
                "Plan payment date",
                "Expected Payment date",
                "Actual Payment received date",
            ],
        )

        # จัด format ตัวเลข (ถ้ามีคอลัมน์เหล่านี้)
        fmt_cols = {}
        for num_col in ["Invoice value", "Claim Plan 2025", "Project Value", "Balance"]:
            if num_col in table_df.columns:
                fmt_cols[num_col] = "{:,.0f}".format

        if fmt_cols:
            styled_table = table_df.style.format(fmt_cols, na_rep="")
            st.dataframe(styled_table, use_container_width=True, height=420)
        else:
            st.dataframe(table_df, use_container_width=True, height=420)

