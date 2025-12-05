import streamlit as st

st.set_page_config(page_title="Snowflake check")

st.title("Snowflake stage check")
st.caption("ตรวจสอบไฟล์ PMBOK ใน @MYSTAGE และตัวอย่างอ่านตาราง FINAL_PROJECT/FINAL_INVOICE")

try:
    conn = st.connection("snowflake")  # streamlit connection
    session = conn.session()           # Snowpark session
except Exception as exc:  # noqa: BLE001
    st.error(f"สร้างการเชื่อมต่อ Snowflake ไม่สำเร็จ: {exc}")
    st.stop()

# Check PMBOK file in stage
try:
    listing = session.sql("LIST @MY_STAGE").collect()
    st.subheader("PMBOK in @MY_STAGE")
    if listing:
        st.table(listing)
    else:
        st.warning("ไม่พบไฟล์ใน @MY_STAGE")
except Exception as exc:  # noqa: BLE001
    st.error(f"LIST @MY_STAGE ล้มเหลว: {exc}")

# Sample data from tables (small head)
try:
    st.subheader("FINAL_PROJECT (ตัวอย่าง 5 แถว)")
    proj_df = conn.query("SELECT * FROM FINAL_PROJECT LIMIT 5;", ttl=60)
    st.dataframe(proj_df)
except Exception as exc:  # noqa: BLE001
    st.error(f"อ่าน FINAL_PROJECT ไม่สำเร็จ: {exc}")

try:
    st.subheader("FINAL_INVOICE (ตัวอย่าง 5 แถว)")
    inv_df = conn.query("SELECT * FROM FINAL_INVOICE LIMIT 5;", ttl=60)
    st.dataframe(inv_df)
except Exception as exc:  # noqa: BLE001
    st.error(f"อ่าน FINAL_INVOICE ไม่สำเร็จ: {exc}")
