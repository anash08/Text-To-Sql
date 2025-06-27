import streamlit as st
import sqlite3
import re
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from graph_utils import ChartAdvisor
from llm_provider import get_llm

class CleanSQLDatabase(SQLDatabase):
    """Custom SQLDatabase that cleans SQL queries before execution and stores last query."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_query = None
    def run(self, command, fetch="all", include_columns=False, **kwargs):
        cleaned_command = clean_sql_query(command)
        self.last_query = cleaned_command
        return super().run(cleaned_command, fetch, include_columns, **kwargs)

def clean_sql_query(sql_query):
    """Clean SQL query by removing markdown formatting and extra whitespace"""
    if not sql_query:
        return sql_query
    
    # Remove markdown code blocks
    sql_query = re.sub(r'```sql\s*', '', sql_query)
    sql_query = re.sub(r'```\s*$', '', sql_query)

    # Remove leading/trailing whitespace
    sql_query = sql_query.strip()
    
    return sql_query

def get_column_names(db_path, sql_query):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query + " LIMIT 1")
        colnames = [desc[0] for desc in cursor.description]
        conn.close()
        return colnames
    except Exception:
        return None

def try_parse_table(result, db_path, last_query=None):
    """Try to parse the result as a table and return a DataFrame if possible."""
    if not isinstance(result, str) and isinstance(result, list) and result:
        # Try to get column names from the last query using sqlite3
        colnames = None
        if last_query:
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                # Remove LIMIT if present, add LIMIT 1 for efficiency
                query = last_query.strip().rstrip(';')
                if "limit" not in query.lower():
                    query += " LIMIT 1"
                cursor.execute(query)
                colnames = [desc[0] for desc in cursor.description]
                conn.close()
            except Exception:
                colnames = None
        if not colnames:
            colnames = [f"col_{i+1}" for i in range(len(result[0]))]
        try:
            df = pd.DataFrame(result, columns=colnames)
            return df
        except Exception:
            return pd.DataFrame(result)
    return None

def format_results(result):
    """Format the database results in a more readable way (for text fallback)"""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        if not result:
            return "No results found."
        if result and isinstance(result[0], tuple):
            formatted = "Query Results:\n"
            formatted += "-" * 50 + "\n"
            for i, row in enumerate(result, 1):
                formatted += f"Row {i}: {row}\n"
            return formatted
        return f"Results: {result}"
    return str(result)

def upload_database(uploaded_file):
    """Handle database upload and return the file path"""
    if uploaded_file is not None:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return uploaded_file.name
    return None

def parse_chain_output(chain_output):
    # chain_output is a dict with keys: 'result', 'intermediate_steps'
    # We want to extract the SQL result and the LLM's final answer
    sql_result = None
    llm_answer = None
    if isinstance(chain_output, dict):
        # Try to extract from intermediate_steps
        steps = chain_output.get('intermediate_steps', [])
        # Heuristic: the SQL result is usually a string or list in the steps
        for step in steps:
            if isinstance(step, list):
                sql_result = step
            elif isinstance(step, str) and step.startswith('[') and step.endswith(']'):
                try:
                    sql_result = eval(step)
                except Exception:
                    pass
        # The final answer is usually in 'result' key
        llm_answer = chain_output.get('result', None)
    return sql_result, llm_answer

def query_sql_db(query, db_path):
    """Run SQL query on the database"""
    if db_path is None:
        return None, None, None

    try:
        # Connect to the uploaded SQLite database using our custom class
        input_db = CleanSQLDatabase.from_uri(f"sqlite:///{db_path}")
        
        # Create an instance of the ChatGroq model
        groq_llm = get_llm()
        
        # Use the LLM instance in the SQLDatabaseChain with proper configuration
        db_chain = SQLDatabaseChain.from_llm(
            groq_llm, 
            input_db, 
            verbose=False,
            return_sql=False,  # Don't return the SQL query
            return_direct=True,  # Return the direct results from the database
            use_query_checker=True,  # Enable query checker to fix SQL issues
            return_intermediate_steps=True  # Enable intermediate steps
        )
        
        # Execute the query using invoke
        chain_output = db_chain.invoke({"query": query})
        sql_result, llm_answer = parse_chain_output(chain_output)
        last_query = input_db.last_query
        return sql_result, llm_answer, last_query
    except Exception as e:
        return None, f"An error occurred: {str(e)}\n\nPlease try rephrasing your question or check if the database contains the relevant data.", None

def get_natural_language_summary(user_query, sql_result):
    llm = get_llm()
    prompt = f"""
Given the following user question and the SQL result, write a concise, clear, and human-friendly summary in natural language. Do not return a table or code, or any types of chart or graph, just a readable answer.

User question: {user_query}
SQL result: {sql_result}
"""
    response_obj = llm.invoke(prompt)
    return response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

class DatabaseHandler:
    """
    Handles database connection, query cleaning, and result parsing.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_column_names(self, sql_query: str):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql_query + " LIMIT 1")
            colnames = [desc[0] for desc in cursor.description]
            conn.close()
            return colnames
        except Exception:
            return None

    def try_parse_table(self, result, last_query=None):
        if not isinstance(result, str) and isinstance(result, list) and result:
            colnames = None
            if last_query:
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    query = last_query.strip().rstrip(';')
                    if "limit" not in query.lower():
                        query += " LIMIT 1"
                    cursor.execute(query)
                    colnames = [desc[0] for desc in cursor.description]
                    conn.close()
                except Exception:
                    colnames = None
            if not colnames:
                colnames = [f"col_{i+1}" for i in range(len(result[0]))]
            try:
                df = pd.DataFrame(result, columns=colnames)
                return df
            except Exception:
                return pd.DataFrame(result)
        return None

# Streamlit app
def main():
    st.set_page_config(
        page_title="Text-to-SQL Query Interface",
        page_icon="🗄️",
        layout="wide"
    )
    
    st.title("🗄️ Text-to-SQL Query Interface")
    st.markdown("Upload your SQLite database and ask questions in plain English to get SQL query results.")

    # Initialize session state
    if 'db_path' not in st.session_state:
        st.session_state.db_path = None
    
    # Sidebar for database upload
    with st.sidebar:
        st.header("📁 Database Upload")
        uploaded_file = st.file_uploader(
            "Choose a SQLite database file",
            type=['sqlite', 'db'],
            help="Upload your SQLite database file (.sqlite or .db)"
        )
        
        if uploaded_file is not None:
            db_path = upload_database(uploaded_file)
            st.session_state.db_path = db_path
            st.success(f"✅ Database uploaded: {uploaded_file.name}")
        else:
            st.session_state.db_path = None
            st.info("Please upload a database file to get started.")

    # Main content area
    if st.session_state.db_path:
        st.header("🔍 Query Your Database")
        
        # Query input
        query = st.text_area(
            "Enter your question in plain English:",
            placeholder="e.g., Which customers are from the United States and how many invoices have they made?",
            height=100,
            help="Ask questions about your data in natural language"
        )
        
        # Query button
        if st.button("🚀 Run Query", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Processing your query..."):
                    sql_result, llm_answer, last_query = query_sql_db(query, st.session_state.db_path)
                
                # Try to display as a table if possible
                db_handler = DatabaseHandler(st.session_state.db_path)
                df = db_handler.try_parse_table(sql_result, last_query)
                if df is not None and not df.empty:
                    st.subheader("📊 SQL Query Results (Table)")
                    st.dataframe(df, use_container_width=True)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=df.to_csv(index=False),
                        file_name="query_results.csv",
                        mime="text/csv"
                    )
                    # --- Chart logic ---
                    chart_advisor = ChartAdvisor()
                    chart_advisor.advise_and_plot(query, df)
                else:
                    st.subheader("📊 SQL Query Results (Text)")
                    st.text_area("Query Results:", value=format_results(sql_result), height=300, disabled=True)
                    st.download_button(
                        label="📥 Download Results (Text)",
                        data=format_results(sql_result),
                        file_name="query_results.txt",
                        mime="text/plain"
                    )
                if llm_answer:
                    st.subheader("🧠 LLM Final Answer")
                    nl_summary = get_natural_language_summary(query, llm_answer)
                    st.info(nl_summary)
            else:
                st.warning("Please enter a question to query the database.")
    else:
        st.info("👆 Please upload a database file from the sidebar to start querying.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>Powered by Groq LLM and LangChain</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()