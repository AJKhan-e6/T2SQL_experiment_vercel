This is a demo/prototype project for testing natural language to sql capabilities and displaying them on vercel's nextjs interface. 

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
