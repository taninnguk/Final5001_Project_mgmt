import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt
from data_cache import load_cached_data, refresh_cache, load_cached_meta, load_env_key


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
    best = behavior.sort_values("avg_days_diff").head(3)
    worst = behavior.sort_values("avg_days_diff", ascending=False).head(3)

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

    
# ---------------------------------------------------
# SECTION 1: INVOICE OVERVIEW (CRM VIEW)
# ---------------------------------------------------

if df_filtered.empty:
    st.warning("No data for current filter selection.")
else:
    # เลือก column สำหรับแสดงภาพรวม CRM
    overview_cols = [
        "Customer",
        "Sale order No.",
        "Invoice value",
        "Invoice Issued Date",
        "Expected Payment date",
        "Payment Status",
        "days_to_expected",
    ]

    display_df = df_filtered[overview_cols].copy()

    # แปลงชื่อ column ให้คนอ่านเข้าใจง่าย
    display_df = display_df.rename(columns={
        "Sale order No.": "Sale Order No.",
        "Invoice value": "Invoice Value",
        "Invoice Issued Date": "Invoice Issued Date",
        "Expected Payment date": "Expected Payment Date",
        "days_to_expected": "Days to Expected Payment",
        "is_overdue": "Overdue?",
    })

    for c in ["Invoice Issued Date", "Expected Payment Date"]:
        display_df[c] = pd.to_datetime(display_df[c], errors="coerce").dt.date

    # สร้างคำอธิบายข้อความ เช่น "Due in 5 days" / "Overdue by 3 days"
    def describe_due(days):
        if pd.isna(days):
            return ""
        days = int(days)
        if days > 0:
            return f"Due in {days} days"
        elif days == 0:
            return "Due today"
        else:
            return f"Overdue by {-days} days"

    display_df["Due Status"] = display_df["Days to Expected Payment"].apply(describe_due)

    # จัด format ตัวเลข Invoice Value ให้มี comma
    format_dict = {
        "Invoice Value": "{:,.0f}".format,
        "Days to Expected Payment": "{:+.0f}".format,  # ให้เห็น +/- วัน
    }

    # สร้าง style ให้แถวที่ overdue เป็นสีแดง
    def highlight_overdue(row):
        try:
            days = float(row["Days to Expected Payment"])
        except Exception:
            days = 0

        if days < 0:
            return ["color: red; font-weight: bold"] * len(row)
        else:
            return [""] * len(row)



    styled = display_df.style.format(format_dict, na_rep="").apply(
        highlight_overdue, axis=1
    )

    st.subheader("Invoice list by Customer")
    st.caption(
        "สีแดง = วันนี้เกิน Expected Payment Date แล้ว (เกินเครดิตเทอม) | "
        "Days to Expected Payment เป็นจำนวนวันจากวันนี้ถึงวันที่คาดว่าจะได้รับเงิน"
    )
    st.dataframe(styled, use_container_width=True)



