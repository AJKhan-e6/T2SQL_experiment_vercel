import os
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable in your .env file.")

# Define the path to your input/output CSV files and the prompt file
PROMPT_FILE_PATH = 'sqltonl_prompt.txt'
INPUT_CSV_PATH = 'example_csv/classification_output.csv'
OUTPUT_CSV_PATH = 'example_csv/classification_nl_output.csv'

# Function to load the prompt from a text file
def load_prompt(file_path):
    with open(file_path, 'r') as prompt_file:
        return prompt_file.read()

# Load the prompt content
prompt_content = load_prompt(PROMPT_FILE_PATH)

# Initialize the OpenAI LLM
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0.2, max_tokens=500)

# Function to build the prompt template dynamically
def build_prompt_template(prompt_content):
    return ChatPromptTemplate.from_messages([
        ("system", prompt_content),
        ("user", "Given the following information, generate a natural language question a user might input "
                 "into a text-to-SQL mechanism to retrieve this query.\n\n"
                 "SQL Query:\n{sql_query}\n\n"
                 "Query Type:\n{query_type}\n\n"
                 "Major Stakeholder:\n{stakeholder}\n\n"
                 "Thought Process:\n{thought_process}")
    ])

# Build the prompt template
prompt_template = build_prompt_template(prompt_content)

# Read the input CSV into a DataFrame
df = pd.read_csv(INPUT_CSV_PATH)

# Check if required columns are present
required_columns = ['SQL Query', 'Type', 'Stakeholders', 'Thought Process']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Initialize new columns in the DataFrame
df['Natural Language Query'] = ''
df['Alternatives'] = ''
df['Steps'] = ''


# Function to extract sections using regex
def extract_sections(response_content):
    try:
        # Print the response content for debugging
        print(f"Debug: Response Content:\n{response_content}")

        # Initialize variables
        nl_query = ""
        alternatives = []
        steps = []

        # Initialize flags
        in_alternatives = False
        in_steps = False

        # Split the content into lines
        lines = response_content.splitlines()

        # Loop through lines and process based on section markers
        for line in lines:
            stripped_line = line.strip()

            # Detect Natural Language Query
            if stripped_line.startswith("- Natural Language Query:"):
                nl_query = stripped_line.split(":", 1)[1].strip().strip("“”\"")
                in_alternatives = False
                in_steps = False  # Reset flags
            
            # Detect Alternatives section
            elif stripped_line.startswith("- Alternatives:"):
                in_alternatives = True
                in_steps = False  # Switch to Alternatives section
            
            # Detect Steps section
            elif stripped_line.startswith("- Steps:"):
                in_alternatives = False
                in_steps = True  # Switch to Steps section
            
            # Collect alternative lines
            elif in_alternatives and stripped_line.startswith("-"):
                alternative = stripped_line.lstrip("- ").strip("“”\"")
                alternatives.append(alternative)
            
            # Collect step lines
            elif in_steps and re.match(r"\d+\.\s", stripped_line):
                step = stripped_line.split(" ", 1)[1].strip("“”\"")
                steps.append(step)

        # Join alternatives and steps into multiline strings
        alternatives = "\n".join(alternatives)
        steps = "\n".join(steps)

        print(f"Extracted Natural Language Query: {nl_query}")
        print(f"Extracted Alternatives: {alternatives}")
        print(f"Extracted Steps: {steps}")

        return nl_query, alternatives, steps

    except Exception as e:
        print(f"Error extracting sections: {e}")
        return "", "", ""

# Function to generate natural language question for each row
def generate_natural_language_question(row):
    formatted_prompt = prompt_template.format_messages(
        sql_query=row['SQL Query'],
        query_type=row['Type'],
        stakeholder=row['Stakeholders'],
        thought_process=row['Thought Process']
    )
    
    try:
        response = llm.invoke(formatted_prompt)
        response_content = response.content.strip()
        print(f"\n\nResponse Content:\n{response_content}")

        # Extract sections using the new parsing logic
        nl_query, alternatives, steps = extract_sections(response_content)

        # Return extracted sections as a Series
        return pd.Series([nl_query, alternatives, steps])
    
    except Exception as e:
        print(f"Error processing row: {e}")
        return pd.Series(["", "", ""])

# Apply the function to each row and populate the new columns
df[['Natural Language Query', 'Alternatives', 'Steps']] = df.apply(generate_natural_language_question, axis=1)

# Save the updated DataFrame to an output CSV
df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"Processing completed. Output saved to {OUTPUT_CSV_PATH}")