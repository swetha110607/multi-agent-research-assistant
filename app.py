import streamlit as st
import requests
import time

# page config
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔍",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/research"

# header
st.title("Multi-Agent Research Assistant")
st.markdown("Enter any topic and get a structured research report powered by multiple AI agents.")

# input
topic = st.text_input("Research topic", placeholder="e.g. impact of LLMs on software engineering jobs")

# optional upload pdf
uploaded_pdf = st.file_uploader("Optional: upload a PDF to include in the research", type=["pdf"])

# button
if st.button("Generate Report", type="primary"):
    if not topic.strip():
        st.error("Please enter a topic")
    else:
        # progress indications
        progress_placeholder = st.empty()

        steps = [
            "🔍 Searching the web....",
            "📄 Reading uploaded document..." if uploaded_pdf else None,
            "📝 Summarizing findings....",
            "✅ Fact-checking claims....",
            "📄 Writing final report...."
        ]
        steps = [s for s in steps if s]  # remove none if no pdf is uploaded

        # fake sequential progress while running real requests in background
        with st.spinner("Running multi-agent pipeline...."):
            for step in steps:
                progress_placeholder.info(step)
                time.sleep(1.5)

            try:
                data = {"topic": topic}
                files = {}
                if uploaded_pdf:
                    files["pdf"] = (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")
                response = requests.post(API_URL, data=data, files=files if files else None, timeout=180)

                if response.status_code == 200:
                    result = response.json()
                    progress_placeholder.success("✅ Report generated successfully!")

                    if result.get("used_pdf"):
                        st.caption("📄 This report includes insights from your uploaded document")

                    st.divider()
                    st.subheader(f"Report: {result['topic']}")
                    st.markdown(result['final_report'])

                    # download option
                    st.download_button(
                        label="Download report as text",
                        data=result['final_report'],
                        file_name=f"research_report_{topic[:30]}.txt",
                        mime="text/plain"
                    )

                else:
                    progress_placeholder.error(f"Something went wrong: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                progress_placeholder.error("Could not connect to the backend. Make sure your FastAPI server is running on port 8000.")
            except requests.exceptions.Timeout:
                progress_placeholder.error("Request timed out.")