import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Task 5: Auto Tagging Support Tickets", page_icon="🎫", layout="wide")
st.title("🎫 Task 5: Auto Tagging Support Tickets Using LLM")
st.write("Automatically classify free-text support tickets into structured categories using Zero-Shot and Few-Shot Learning.")

# --- 2. AUTHENTICATION SIDEBAR ---
st.sidebar.header("🔑 Authentication")
groq_api_key = st.sidebar.text_input("Enter Groq API Key (gsk_...)", type="password")

# --- 3. DEFINING CATEGORIES ---
SUPPORT_CATEGORIES = [
    "Technical Support (Bug/Crash)",
    "Billing & Payments",
    "Account Recovery / Login Issue",
    "Feature Request",
    "Product Inquiry / General Questions"
]

# --- 4. PROMPT ENGINEERING TEMPLATES ---
zero_shot_system_prompt = (
    "You are an expert customer support routing AI. Your task is to analyze the text of a support ticket "
    "and classify it by assigning the top 3 most probable tags from the allowed categories list.\n\n"
    "Allowed Categories:\n{categories}\n\n"
    "Output Requirements:\n"
    "Provide exactly the top 3 categories ranked by relevance. "
    "Format the response strictly as a numbered list with confidence scores (High/Medium/Low), like this:\n"
    "1. Category Name - Confidence\n"
    "2. Category Name - Confidence\n"
    "3. Category Name - Confidence\n"
    "Do not write any introductory or concluding text."
)
zero_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", zero_shot_system_prompt),
    ("human", "Ticket Content:\n{ticket_text}")
])

few_shot_system_prompt = (
    "You are an expert customer support routing AI. Your task is to analyze the text of a support ticket "
    "and classify it by assigning the top 3 most probable tags from the allowed categories list.\n\n"
    "Allowed Categories:\n{categories}\n\n"
    "Here are a few examples:\n\n"
    "Example 1:\nTicket Content: 'I am trying to checkout but my credit card keeps getting declined.'\n"
    "Output:\n1. Billing & Payments - High\n2. Technical Support (Bug/Crash) - Medium\n3. Product Inquiry / General Questions - Low\n\n"
    "Example 2:\nTicket Content: 'The app keeps freezing and crashing every time I open the dashboard.'\n"
    "Output:\n1. Technical Support (Bug/Crash) - High\n2. Product Inquiry / General Questions - Low\n3. Feature Request - Low\n\n"
    "Now, classify the following ticket following the exact same ranking and format structure."
)
few_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", few_shot_system_prompt),
    ("human", "Ticket Content:\n{ticket_text}")
])

# --- 5. USER INTERFACE ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Input Support Ticket")
    ticket_input = st.text_area(
        "Paste the customer support message here:",
        height=150,
        placeholder="Example: I paid for the premium plan yesterday but my profile still shows free tier..."
    )
    technique = st.radio(
        "Select Classification Technique:",
        ["Zero-Shot Learning (Direct Instructions)", "Few-Shot Learning (With Examples Context)"],
        horizontal=True
    )

with col2:
    st.subheader("🏷️ Available System Tags")
    for cat in SUPPORT_CATEGORIES:
        st.markdown(f"- `{cat}`")

# --- 6. EXECUTION AND EVALUATION ---
if st.button("🏷️ Run Auto-Tagging Classifier"):
    if not groq_api_key:
        st.error("Please enter your Groq API Key in the sidebar first!")
    elif not ticket_input.strip():
        st.warning("Please enter a valid ticket message first.")
    else:
        with st.spinner("Analyzing text patterns and ranking tags..."):
            try:
                # Initialize Groq LLM dynamically inside call to prevent initialization freeze
                llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0.1)
                
                selected_prompt = zero_shot_prompt if "Zero-Shot" in technique else few_shot_prompt
                chain = selected_prompt | llm | StrOutputParser()
                
                response = chain.invoke({
                    "categories": "\n".join([f"- {c}" for c in SUPPORT_CATEGORIES]),
                    "ticket_text": ticket_input
                })
                
                st.success("Analysis Complete! Top 3 Probable Tags:")
                st.info(response)
                
            except Exception as e:
                st.error(f"API Error occurred: {str(e)}")