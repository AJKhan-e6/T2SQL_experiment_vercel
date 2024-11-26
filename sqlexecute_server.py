from fastapi import FastAPI, HTTPException, Form
from e6data_python_connector import Connection
from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables from .env file
load_dotenv()

app = FastAPI()
PORT = 8011

# Configuration for the e6data connector
E6DATA_CONFIG = {
    "host": os.getenv("E6DATA_HOST"),
    "port": int(os.getenv("E6DATA_PORT", 80)),  # Default to 80 if not set
    "username": os.getenv("E6DATA_USERNAME"),
    "password": os.getenv("E6DATA_PASSWORD"),
    "database": os.getenv("E6DATA_DATABASE"),
    "catalog_name": os.getenv("E6DATA_CATALOG_NAME"),
}


@app.post("/execute-sql")
async def execute_sql(query: str = Form(...)):
    """
    Endpoint to execute a SQL query using e6data-python-connector and fetch the first 150 rows.
    The query is sent as a form input.
    """
    # Establish connection to e6data
    try:
        conn = Connection(
            host=E6DATA_CONFIG["host"],
            port=E6DATA_CONFIG["port"],
            username=E6DATA_CONFIG["username"],
            database=E6DATA_CONFIG["database"],
            password=E6DATA_CONFIG["password"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection error: {str(e)}")

    # Execute the query and fetch results
    try:
        cursor = conn.cursor(catalog_name=E6DATA_CONFIG["catalog_name"])
        cursor.execute(query)
        rows = cursor.fetchmany(150)  # Fetch first 150 rows
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query execution error: {str(e)}")

    # Format results as a list of dictionaries for JSON response
    result = []
    if rows:
        columns = [desc[0] for desc in cursor.description]  # Get column names
        for row in rows:
            result.append(dict(zip(columns, row)))

    return {"rows": result}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        proxy_headers=True,
        workers=1,
    )