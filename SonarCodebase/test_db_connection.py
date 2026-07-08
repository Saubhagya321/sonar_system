"""Standalone script to verify SQL Server connectivity before wiring DB_CONNECTION_STRING into any project config."""
from sqlalchemy import create_engine, text

DB_CONNECTION_STRING = (
    "mssql+pyodbc://localhost/Q3_Recommendation"
    "?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)

def main():
    print(f"Connecting to: {DB_CONNECTION_STRING}")
    engine = create_engine(DB_CONNECTION_STRING)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT @@VERSION AS version"))
        print("Connected successfully.")
        print(result.fetchone()[0])

        tables = conn.execute(text(
            "SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_SCHEMA, TABLE_NAME"
        )).fetchall()
        print(f"\nFound {len(tables)} tables in Q3_Recommendation:")
        for schema, name in tables:
            print(f"  {schema}.{name}")

if __name__ == "__main__":
    main()
