import snowflake.snowpark as snowpark
from snowflake.snowpark import Session
import pandas as pd
from snowflake.snowpark.files import SnowflakeFile
import re
import sqlalchemy

# --- PART 1: Your Transformation Logic ---
def main(session: snowpark.Session):
    # 1. Define the path to your file in the stage
    # Ensure this stage exists in the Database/Schema defined in connection_parameters
    stage_path = "@MY_STAGE/Final.xlsx"

    print(f"Reading file from {stage_path}...")

    # 2. Open the file using SnowflakeFile
    with SnowflakeFile.open(stage_path, 'rb') as f:
        # Read ALL sheets
        sheets_dict = pd.read_excel(f, sheet_name=None)

    created_tables = []

    # 3. Loop through each sheet and write to a separate table
    for sheet_name, df in sheets_dict.items():
        # Clean the table name (remove spaces/special chars)
        clean_suffix = re.sub(r'[^a-zA-Z0-9]', '_', sheet_name).upper()
        table_name = f"FINAL_{clean_suffix}"

        print(f"Writing sheet '{sheet_name}' to table '{table_name}'...")

        # Write to Snowflake
        session.write_pandas(
            df, 
            table_name, 
            auto_create_table=True, 
            overwrite=True
        )
        created_tables.append(table_name)

    return f"Success! Tables created: {', '.join(created_tables)}"

# --- PART 2: Connection Setup (Only needed for local execution) ---
if __name__ == "__main__":
    # Define your connection parameters
    connection_parameters = {
        "account": "YELBYGL-ZD54853",  # e.g., 'xy12345.us-east-1'
        "user": "nguk32909",
        "password": "Yongnguk11760.",
        "role": "ACCOUNTADMIN",           # e.g., 'SYSADMIN'
        "warehouse": "STREAMLIT_WH", # e.g., 'COMPUTE_WH'
        "database": "FINAL_5001",         # The DB where @MY_STAGE is located
        "schema": "RAW_DATA"        # The Schema where @MY_STAGE is located
    }

    # Create the session
    print("Connecting to Snowflake...")
    session = Session.builder.configs(connection_parameters).create()
    
    # Run the main function
    result = main(session)
    print(result)
    
    # Close the session
    session.close()