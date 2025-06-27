import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import re
import json
from typing import Optional, Dict
from llm_provider import get_llm

class ChartAdvisor:
    """
    Handles LLM-based chart suggestion and plotting for a DataFrame.
    """
    def __init__(self):
        self.llm = get_llm()

    def get_chart_instruction(self, user_query: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        Ask the LLM if a chart should be shown, and if so, which type and columns.
        """
        prompt = f"""
You are an AI assistant. Given the following user question and the columns of a pandas DataFrame, decide if a chart should be shown.
Only suggest a chart if the user question contains words like 'chart', 'plot', 'graph', 'visualize', or 'draw'.
If not, always return "show_chart": false.

User question: {user_query}
DataFrame columns: {list(df.columns)}

Respond with ONLY a single JSON object, no explanation, no markdown, no extra text:
{{
  "show_chart": true/false,
  "chart_type": "bar"/"pie"/"line"/null,
  "x": "column_name or null",
  "y": "column_name or null",
  "label": "column_name or null"  // for pie chart
}}
"""
        response_obj = self.llm.invoke(prompt)
        response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
        st.write("LLM chart instruction response:", response)
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            st.info("Could not parse chart instruction from LLM (no JSON found).")
            return None
        try:
            chart_info = json.loads(match.group())
            return chart_info
        except Exception as e:
            st.info(f"Could not parse chart instruction from LLM (invalid JSON): {e}")
            st.text_area("Raw LLM response", value=response, height=200)
            return None

    def plot_chart(self, chart_info: Dict, df: pd.DataFrame):
        """
        Plot the chart in Streamlit based on LLM's chart_info.
        """
        if not chart_info or not chart_info.get("show_chart"):
            return
        chart_type = chart_info.get("chart_type")
        x = chart_info.get("x")
        y = chart_info.get("y")
        label = chart_info.get("label")
        st.subheader(f"📈 Chart: {chart_type.title()}" if chart_type else "📈 Chart")
        try:
            if chart_type == "bar" and x and y:
                st.bar_chart(df[[x, y]].set_index(x))
            elif chart_type == "line" and x and y:
                st.line_chart(df[[x, y]].set_index(x))
            elif chart_type == "pie" and label and y:
                fig, ax = plt.subplots()
                ax.pie(df[y], labels=df[label], autopct='%1.1f%%')
                ax.axis('equal')
                st.pyplot(fig)
            else:
                st.info("Chart type or columns not recognized or not enough data to plot.")
        except Exception as e:
            st.info(f"Could not generate chart: {e}")

    def advise_and_plot(self, user_query: str, df: pd.DataFrame):
        """
        Main entry: get chart instruction and plot if needed.
        """
        chart_info = self.get_chart_instruction(user_query, df)
        self.plot_chart(chart_info, df)

def get_chart_instruction_and_plot(user_query, df):
    if df is None or df.empty:
        return
    
    prompt = f"""
You are an AI assistant. Given the following user question and the columns of a pandas DataFrame, decide if a chart should be shown.
Only suggest a chart if the user question contains words like 'chart', 'plot', 'graph', 'visualize', or 'draw'.
If not, always return "show_chart": false.

User question: {user_query}
DataFrame columns: {list(df.columns)}

Respond with ONLY a single JSON object, no explanation, no markdown, no extra text:
{{
  "show_chart": true/false,
  "chart_type": "bar"/"pie"/"line"/null,
  "x": "column_name or null",
  "y": "column_name or null",
  "label": "column_name or null"  // for pie chart
}}
"""
    llm = get_llm()
    response_obj = llm.invoke(prompt)
    response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
    st.write("LLM chart instruction response:", response)
    
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        st.info("Could not parse chart instruction from LLM (no JSON found).")
        return
    try:
        chart_info = json.loads(match.group())
    except Exception as e:
        st.info(f"Could not parse chart instruction from LLM (invalid JSON): {e}")
        st.text_area("Raw LLM response", value=response, height=200)
        return
    if not chart_info.get("show_chart"):
        return
    chart_type = chart_info.get("chart_type")
    x = chart_info.get("x")
    y = chart_info.get("y")
    label = chart_info.get("label")
    st.subheader(f"📈 Chart: {chart_type.title()}" if chart_type else "📈 Chart")
    try:
        if chart_type == "bar" and x and y:
            st.bar_chart(df[[x, y]].set_index(x))
        elif chart_type == "line" and x and y:
            st.line_chart(df[[x, y]].set_index(x))
        elif chart_type == "pie" and label and y:
            fig, ax = plt.subplots()
            ax.pie(df[y], labels=df[label], autopct='%1.1f%%')
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.info("Chart type or columns not recognized or not enough data to plot.")
    except Exception as e:
        st.info(f"Could not generate chart: {e}") 