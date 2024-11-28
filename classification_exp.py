import re
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Constants
assert 'OPENAI_API_KEY' in os.environ, "Please set the OPENAI_API_KEY environment variable."

PROMPT_FILE_PATH = 'query_classification_prompt.txt'

# Load the initial prompt
def load_prompt(file_path):
    with open(file_path, 'r') as prompt_file:
        return prompt_file.read()

prompt = load_prompt(PROMPT_FILE_PATH)

# Initialize the OpenAI LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=500)

# Define the function to build the prompt template dynamically
def build_prompt_template(prompt_content):
    return ChatPromptTemplate.from_messages([
        ("system", f"{prompt_content}"),
        ("user", "Given the following SQL query, classify it into one of the categories. Strictly follow the output format defined in the prompt."
                 "If 'other', suggest a new category.\n\n"
                 "SQL Query:\n{sql_query}\n\n"
                 "Tables Involved:\n{list_all_tables}\n\n"
                 "Schema Details:\n{schema_details}\n\n")
    ])

# Initialize the prompt template
prompt_template = build_prompt_template(prompt)

# Filtering the relevant parts from the schema 
def filter_schema(schema_text, tables):
    # Extract table names from fully qualified names in tables list
    clean_table_names = {table.split(".")[-1] for table in tables}

    # Build regex patterns to identify table and column definitions
    table_pattern = re.compile(r'^\s*"(\w+)":\w+:\d+', re.MULTILINE)
    # Updated regex pattern to capture data types with optional parentheses, including decimals
    column_pattern = re.compile(r'^\s*{\s*"(\w+)":([\w()0-9,]+)\s*}', re.MULTILINE)

    filtered_schema = ""
    current_table = None
    include_table = False

    for line in schema_text.splitlines():
        # Match catalog or database lines to add them directly
        if not line.startswith('\t') and not line.startswith(' '):
            filtered_schema += line + "\n"
            continue

        # Match table line and check if it should be included
        table_match = table_pattern.match(line)
        if table_match:
            table_name = table_match.group(1)
            include_table = table_name in clean_table_names
            if include_table:
                filtered_schema += line + "\n"
            current_table = table_name
            continue

        # Include column lines only if the current table is included
        if include_table and current_table:
            column_match = column_pattern.match(line)
            if column_match:
                filtered_schema += line + "\n"

    return filtered_schema

# Function to parse the LLM response into structured elements
def parse_response_content(content):
    try:
        result = {
            "Type": re.search(r"- Type: (.+?)(?:\n|$)", content).group(1).strip() if re.search(r"- Type: (.+?)(?:\n|$)", content) else "",
            "Stakeholders": re.search(r"- Stakeholders: (.+?)(?:\n|$)", content).group(1).strip() if re.search(r"- Stakeholders: (.+?)(?:\n|$)", content) else "",
            "Thought Process": re.search(r"- Thought Process: (.+?)(?=\n- |$)", content, re.DOTALL).group(1).strip() if re.search(r"- Thought Process: (.+?)(?=\n- |$)", content, re.DOTALL) else "",
            "Major Tables": re.search(r"- Major Tables:\n(.+?)(?=\n- |$)", content, re.DOTALL).group(1).strip() if re.search(r"- Major Tables:\n(.+?)(?=\n- |$)", content, re.DOTALL) else "",
            "Other Category": re.search(r"- Other: (.+?)(?:\n|$)", content).group(1).strip() if re.search(r"- Other: (.+?)(?:\n|$)", content) else ""
        }
        return result
    except Exception as e:
        print(f"Error parsing response content: {e}, Content: {content}")
        return {key: "" for key in ["Type", "Stakeholders", "Thought Process", "Major Tables", "Other Category"]}

def update_txt_file(file_path, new_category):
    """
    Updates the .txt file to add the new category just before the closing
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the last line with closing triple quotes
    closing_triple_quotes_index = None
    for idx, line in enumerate(reversed(lines)):
        if line.strip() == '"""':
            closing_triple_quotes_index = len(lines) - 1 - idx
            break

    # Insert the new category just before the closing triple quotes
    if closing_triple_quotes_index is not None:
        lines.insert(closing_triple_quotes_index, f"- {new_category}\n")
    else:
        # If no closing triple quotes are found, append the new category at the end
        lines.append(f"- {new_category}\n")

    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

# Read the CSV file
df = pd.read_csv('example_csv/processed_results (16).csv')

# Initialize lists to store the results
results = {"Filtered Schema": [], "Type": [], "Stakeholders": [], "Thought Process": [], "Major Tables": [], "Other Category": []}

# Process each SQL query and fetch data
for index, row in df.iterrows():
    sql_query = row.get('Original_Query', "")
    list_all_tables = eval(row.get('list_of_tables', "[]")) if pd.notna(row.get('list_of_tables')) else []
    schema = row.get('schema', "")
    filtered_schema = filter_schema(schema, list_all_tables)

    # Format the prompt
    formatted_prompt = prompt_template.format_messages(
        sql_query=sql_query,
        list_all_tables=list_all_tables,
        schema_details=filtered_schema,
    )

    # Invoke LLM and handle response
    try:
        response = llm.invoke(formatted_prompt)
        print(f"\n\n Response Content:\n{response.content}")
        parsed = parse_response_content(response.content)

        # Check if the category is 'Other'
        if parsed.get("Type", "").lower() == "other":
            new_category = parsed.get("Other Category", "").strip()
            if new_category:  # Only if there's a new category
                update_txt_file(PROMPT_FILE_PATH, new_category)

                # Reload the prompt dynamically
                updated_prompt = load_prompt(PROMPT_FILE_PATH)
                prompt_template = build_prompt_template(updated_prompt)

        results["Filtered Schema"].append(filtered_schema)
        for key in results.keys():
            if key != "Filtered Schema":
                results[key].append(parsed.get(key, ""))
    except Exception as e:
        print(f"Error during LLM invocation for index {index}: {e}")
        results["Filtered Schema"].append(filtered_schema)
        for key in results.keys():
            if key != "Filtered Schema":
                results[key].append("")

# Add the results as new columns in the DataFrame
for key, values in results.items():
    df[key] = values

# Ensure 'Filtered Schema' is the first new column
original_columns = list(df.columns[:-len(results)])
new_columns = list(results.keys())
output_columns = original_columns + new_columns
df = df[output_columns]

# Save the updated DataFrame back to the CSV file
df.to_csv('example_csv/classification_output.csv', index=False)