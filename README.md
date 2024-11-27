This is a demo/prototype project for testing natural language to sql capabilities and displaying them on vercel's nextjs interface. 

# T2SQL Backend

To begin, first initialise the environment file with the following variables

```bash
OPENAI_API_KEY=""
THEIRSTACK_API_KEY=""
E6DATA_HOST=you can get this from clusters -> connection details
E6DATA_PORT=80
E6DATA_USERNAME=your e6data email ID
E6DATA_PASSWORD=this is an access token. You can generate one by going to your user settings in e6data and creating a new token
E6DATA_DATABASE=
E6DATA_CATALOG_NAME=
```

Also install the required python libraries using
```bash
pip install -r requirements.txt
```

There are already compiled files in the *example_csv* folder. If you want to introduce a new input, go through the following code flow:
1. First, initialise the input file by putting it though the *classification_exp* file. This code generates the type/classification of SQL queries based on their business use case.
2. Next, generate natural questions of those sql queries using *sqltonl* file. This file generates questions which a user could input into a t2sql system in order to get the sql query.

Hence, you are prepped for using the system. 
Start the two server files - 

```bash
python t2sql_server.py
python executesql_server.py
```

These two files will start two services on the ports 8010 (t2sql service) and 8011 (query execution in e6data service)

The input to 8010 will look a post request with the body
```json
{
    "question":"What are the customer creation times and dates in the 'America/Los_Angeles' timezone for March 12, 2023?"
}
```

While the input to 8011 will be a form with the field 'query'. This field should have a sql query. 

# Vercel NextJS Frontend


[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fvercel-labs%2Fnatural-language-postgres&env=OPENAI_API_KEY&envDescription=Learn%20more%20about%20how%20to%20get%20the%20API%20Keys%20for%20the%20application&envLink=https%3A%2F%2Fgithub.com%2Fvercel-labs%2Fnatural-language-postgres%2Fblob%2Fmain%2F.env.example&demo-title=Natural%20Language%20Postgres&demo-description=Query%20PostgreSQL%20database%20using%20natural%20language%20and%20visualize%20results%20with%20Next.js%20and%20AI%20SDK.&demo-url=https%3A%2F%2Fnatural-language-postgres.vercel.app&stores=%5B%7B%22type%22%3A%22postgres%22%7D%5D)

This project is a Next.js application that allows users to query a PostgreSQL database using natural language and visualize the results. It's powered by the AI SDK by Vercel and uses OpenAI's GPT-4o model to translate natural language queries into SQL.

## Features

- Natural Language to SQL: Users can input queries in plain English, which are then converted to SQL using AI.
- Data Visualization: Results are displayed in both table and chart formats, with the chart type automatically selected based on the data.
- Query Explanation: Users can view the full SQL query and get an AI-generated explanation of each part of the query.

## Technology Stack

- Next.js for the frontend and API routes
- AI SDK by Vercel for AI integration
- OpenAI's GPT-4o for natural language processing
- PostgreSQL for data storage
- Vercel Postgres for database hosting
- Framer Motion for animations
- ShadowUI for UI components
- Tailwind CSS for styling
- Recharts for data visualization

## How It Works

1. The user enters a natural language query about unicorn companies.
2. The application uses GPT-4 to generate an appropriate SQL query.
3. The SQL query is executed against the PostgreSQL database.
4. Results are displayed in a table format.
5. An AI-generated chart configuration is created based on the data.
6. The results are visualized using the generated chart configuration.
7. Users can toggle between table and chart views.
8. Users can request an explanation of the SQL query, which is also generated by AI.

## Data

The database contains information about unicorn companies, including:

- Company name
- Valuation
- Date joined (unicorn status)
- Country
- City
- Industry
- Select investors

This data is based on CB Insights' list of unicorn companies.

## Getting Started

To get the project up and running, follow these steps:

1. Install dependencies:

   ```bash
   pnpm install
   ```

2. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

3. Add your OpenAI API key and PostgreSQL connection string to the `.env` file:

   ```
   OPENAI_API_KEY=your_api_key_here
   POSTGRES_URL="..."
   POSTGRES_PRISMA_URL="..."
   POSTGRES_URL_NO_SSL="..."
   POSTGRES_URL_NON_POOLING="..."
   POSTGRES_USER="..."
   POSTGRES_HOST="..."
   POSTGRES_PASSWORD="..."
   POSTGRES_DATABASE="..."
   ```
4. Download the dataset:
  - Go to https://www.cbinsights.com/research-unicorn-companies
  - Download the unicorn companies dataset
  - Save the file as `unicorns.csv` in the root of your project

5. Seed the database:
   ```bash
   pnpm run seed
   ```

6. Start the development server:
   ```bash
   pnpm run dev
   ```

Your project should now be running on [http://localhost:3000](http://localhost:3000).

## Deployment

The project is set up for easy deployment on Vercel. Use the "Deploy with Vercel" button in the repository to create your own instance of the application.

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fvercel-labs%2Fnatural-language-postgres&env=OPENAI_API_KEY&envDescription=Learn%20more%20about%20how%20to%20get%20the%20API%20Keys%20for%20the%20application&envLink=https%3A%2F%2Fgithub.com%2Fvercel-labs%2Fnatural-language-postgres%2Fblob%2Fmain%2F.env.example&demo-title=Natural%20Language%20Postgres&demo-description=Query%20PostgreSQL%20database%20using%20natural%20language%20and%20visualize%20results%20with%20Next.js%20and%20AI%20SDK.&demo-url=https%3A%2F%2Fnatural-language-postgres.vercel.app&stores=%5B%7B%22type%22%3A%22postgres%22%7D%5D)


## Learn More

To learn more about the technologies used in this project, check out the following resources:

- [Next.js Documentation](https://nextjs.org/docs)
- [AI SDK](https://sdk.vercel.ai/docs)
- [OpenAI](https://openai.com/)
- [Vercel Postgres powered by Neon](https://vercel.com/docs/storage/vercel-postgres)
- [Framer Motion](https://www.framer.com/motion/)
- [ShadcnUI](https://ui.shadcn.com/)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Recharts](https://recharts.org/en-US/)
