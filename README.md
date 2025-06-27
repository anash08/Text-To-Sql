# Text-to-SQL with LangChain and Groq

This project provides a **Streamlit-based web application** that allows users to interact with an SQLite database using natural language queries. It leverages the LangChain library and Groq's language model to convert plain English questions into SQL queries, execute them, and display results in a table, chart, and natural language summary.


## Features

- **Upload SQLite Database**: Upload an SQLite database file to the application.
- **Natural Language Queries**: Enter plain English queries to retrieve data from the database.
- **Real-time Results**: Get results from your database based on your queries.
- **Table Display**: Results are shown in a table with column names.
- **Conditional Charting**: If your query requests a chart (bar, line, or pie), the app will generate and display it automatically.
- **Natural Language Summary**: The app provides a human-friendly summary of the results using the LLM.

## Installation

To set up and run this application locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/anash08/Text-To-Sql.git
cd 
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root of the project and add your Groq API key:

```
GROQ_API_KEY=your-groq-api-key-here
```

### 5. Run the Application

```bash
streamlit run app.py
```

## Usage

1. **Open the Application**: The Streamlit app will open in your browser.
2. **Upload Database**: Use the file upload component to upload an SQLite database file.
3. **Query the Database**: Enter your question in plain English and click "Run Query" to see the results.
4. **View Results**: Results are shown as a table. If your question requests a chart, a relevant chart will be displayed. A natural language summary is also provided.

## Example Queries

- "How many rows are there?"
- "What is the total revenue by month?"
- "Which product line had the largest revenue?"
- "Which day of the week has the best average ratings for each branch?"
- "Show a bar chart of total revenue by country."
- "Plot a pie chart of sales by product category."

## Notes
- This app uses [LangChain](https://python.langchain.com/) and [Groq](https://groq.com/) for LLM-powered SQL generation and summarization.
- All charting is handled automatically based on your query and the LLM's decision.
 

---

**Powered by Groq LLM and LangChain**

