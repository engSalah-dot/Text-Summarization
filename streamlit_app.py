
import requests
import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Pegasus Text Summarizer",
    page_icon="📝",
    layout="centered",
)

API_URL = "http://127.0.0.1:8000/predict"

# -----------------------------
# Initialize Session State
# -----------------------------
if "summary" not in st.session_state:
    st.session_state.summary = ""

if "last_input" not in st.session_state:
    st.session_state.last_input = ""

# -----------------------------
# User Interface
# -----------------------------
st.title("Text Summarizer")
st.caption("AI-powered text summarization using Pegasus")

st.markdown(
    """
    Enter any English text below, and the model will generate
    a shorter summary while preserving the main ideas.
    """
)

text = st.text_area(
    "Your text",
    height=250,
    placeholder="Paste your text here...",
)

if st.button("Summarize", type="primary", use_container_width=True):

    if not text.strip():
        st.warning("Please enter some text first.")

    else:
        try:
            with st.spinner("Generating your summary..."):

                response = requests.post(
                    API_URL,
                    params={"text": text.strip()},
                    timeout=180,
                )

            if response.ok:
                result = response.json()
                summary = result.get("summary", "")

                if summary:
                    st.session_state.summary = summary
                    st.session_state.last_input = text.strip()
                    st.success("Summary generated successfully!")
                else:
                    st.error("The API returned an empty summary.")

            else:
                try:
                    error_detail = response.json().get(
                        "detail", response.text
                    )
                except ValueError:
                    error_detail = response.text

                st.error(
                    f"API Error ({response.status_code}): "
                    f"{error_detail}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. "
                "Please start the backend first."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The request timed out. "
                "Try again with shorter text."
            )

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

# -----------------------------
# Display Summary
# -----------------------------
if st.session_state.summary:
    st.divider()
    st.subheader("Generated Summary")
    st.write(st.session_state.summary)

    st.download_button(
        label="Download Summary",
        data=st.session_state.summary,
        file_name="summary.txt",
        mime="text/plain",
        use_container_width=True,
    )
