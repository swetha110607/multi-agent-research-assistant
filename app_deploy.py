import streamlit as st
import time
import sys
import os

# Add src folder to path so we can import directly
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from orchestrator import orchestrator

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Multi-Agent Research Assistant")
st.markdown("Enter any topic and get a structured research report powered by multiple AI agents.")

topic = st.text_input("Research topic", placeholder="e.g. impact of LLMs on software engineering jobs")
uploaded_pdf = st.file_uploader("Optional: upload a PDF to include in the research", type=["pdf"])

if st.button("Generate Report", type="primary"):
    if not topic.strip():
        st.error("Please enter a topic first.")
    else:
        progress_placeholder = st.empty()

        steps = [
            "🔍 Searching the web...",
            "📄 Reading uploaded document..." if uploaded_pdf else None,
            "📝 Summarizing findings...",
            "✅ Fact-checking claims...",
            "📄 Writing final report..."
        ]
        steps = [s for s in steps if s]

        with st.spinner("Running multi-agent pipeline..."):
            for step in steps:
                progress_placeholder.info(step)
                time.sleep(1.5)

            try:
                pdf_path = None
                if uploaded_pdf:
                    os.makedirs("temp_uploads", exist_ok=True)
                    pdf_path = f"temp_uploads/{uploaded_pdf.name}"
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_pdf.getvalue())

                result = orchestrator(topic, pdf_path=pdf_path)

                progress_placeholder.success("✅ Report generated successfully!")

                if result.get("used_pdf"):
                    st.caption("📄 This report includes insights from your uploaded document")

                st.divider()
                st.subheader(f"Report: {result['topic']}")
                st.markdown(result['final_report'])

                st.download_button(
                    label="Download report as text",
                    data=result['final_report'],
                    file_name=f"research_report_{topic[:30]}.txt",
                    mime="text/plain"
                )

                if pdf_path and os.path.exists(pdf_path):
                    os.remove(pdf_path)

            except Exception as e:
                progress_placeholder.error(f"Something went wrong: {str(e)}")