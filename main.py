import os
import asyncio
import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone


# ---- Config & Styling ----
PAGE_TITLE = "Ask Empress"
BG_COLOR = "#FAF5F0"        # soft beige background (approx)
PRIMARY_PURPLE = "#5D2E8C"  # deep purple (approx)
ACCENT_GOLD = "#C6A15A"     # gold accent (approx)
CARD_BG = "#FFFFFF"

st.set_page_config(page_title=PAGE_TITLE, layout="wide", initial_sidebar_state="expanded")




# Apply nest_asyncio patch
nest_asyncio.apply()

# Load environment variables
load_dotenv()


# Get API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check for missing API keys
if not PINECONE_API_KEY:
    st.error("Pinecone API key is missing. Please set it in your .env file.")
    st.stop()
if not GOOGLE_API_KEY:
    st.error("Google API key is missing. Please set it in your .env file.")
    st.stop()





# CSS to approximate Empress look
st.markdown(
    f"""
    <style>
    :root {{
      --empress-bg: {BG_COLOR};
      --empress-purple: {PRIMARY_PURPLE};
      --empress-gold: {ACCENT_GOLD};
      --card-bg: {CARD_BG};
    }}
    .block-container{{background-color:var(--empress-bg); padding:30px 40px;}}
    .logo{{font-weight:800; color:var(--empress-purple); font-size:20px;}}
    .nav-links a{{color:var(--empress-purple); text-decoration:none; margin-right:14px; font-weight:600}}
    .hero-title{{color:var(--empress-purple); font-size:38px; font-weight:700; margin-bottom:6px;}}
    .hero-sub{{color:#5a4a6f; font-size:16px; margin-bottom:12px;}}
    .card{{background:var(--card-bg); border-radius:12px; padding:18px; box-shadow:0 8px 24px rgba(90,70,100,0.06);}}
    .small-muted{{color:#6b6b6b; font-size:14px}}
    .btn-empress{{
        background: linear-gradient(90deg,var(--empress-purple), #7a47b0);
        color: #fff;
        padding: 8px 18px;
        border-radius: 10px;
        border: none;
    }}
    footer{{color:#555; padding-top:20px;}}
    </style>
    """,
    unsafe_allow_html=True,
)







# Initialize Pinecone and embedding model
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("empress")
embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Define system prompt template
system_prompt_template = """
Your name is Ask Empress Chatbot and Ask Empress Ambassador. You are a Post Menopausal Health and Wellness Expert. Answer questions very comprehensively and elaborately, stating managements and product recommendation where necessary. Use the following information to answer the user's question and recommend post menopausal wellness products based on user's questions:

{doc_content}

Provide very comprehensively and elaborately, stating managements and product recommendation where necessary based on the provided information and your expertise.
"""

def generate_response(question):
    """Generate a response using Pinecone retrieval and Gemini 2.0 Flash."""
    
    
    # Embed the user's question
    query_embed = embed_model.embed_query(question)
    query_embed = [float(val) for val in query_embed]  # Ensure standard floats
    
    # Query Pinecone for relevant documents - MODIFIED: top_k=3
    results = pinecone_index.query(
        vector=query_embed,
        top_k=5,  # CHANGED from 2 to 3
        include_values=False,
        include_metadata=True
    )
    
    # Extract document contents - MODIFIED: Added terminal printing
    doc_contents = []
    print("\n" + "="*50)
    print(f"RETRIEVED DOCUMENTS FOR: '{question}'")
    for i, match in enumerate(results.get('matches', [])):
        text = match['metadata'].get('text', '')
        doc_contents.append(text)
        print(f"\nDOCUMENT {i+1}:\n{text}\n")
    print("="*50 + "\n")
    
    doc_content = "\n".join(doc_contents).replace('{', '{{').replace('}', '}}') if doc_contents else "No additional information found."
    
    # Format the system prompt with retrieved content
    formatted_prompt = system_prompt_template.format(doc_content=doc_content)
    
    # Rebuild chat history from session state
    chat_history = ChatMessageHistory()
    lc_chat_history = []
    for msg in st.session_state.chat_history:
        for msg in st.session_state.chat_history:
         if msg["role"] == "user":
            lc_chat_history.append(HumanMessage(content=msg["content"]))
         elif msg["role"] == "assistant":
            lc_chat_history.append(AIMessage(content=msg["content"]))
    
    # Initialize memory with chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=chat_history,
        return_messages=True
    )
    
    # Create the conversation prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(formatted_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )
    
    # Initialize Gemini 2.0 Flash model with explicit client
    chat = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    
    # Create the conversation chain
    conversation = RunnableSequence(prompt, chat)
    
    # Generate the response
    res = conversation.invoke({
        "question": question,
        "chat_history": lc_chat_history  # Pass the history here
    }) # synchronous call

    # safe extraction — adapt if your wrapper uses a different key
    return res.content
# Streamlit app layout remains unchanged
st.title("Ask Empress")
st.write("Ask your Post Menopausal Wellness questions and receive expert medical advice.")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Do you have questions about your perimenopause / menopause wellness? Ask me anything!"}
    ]

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
user_input = st.chat_input("Type your message here....")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("Thinking..."):
        response = generate_response(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})







# ---- Top nav / header (mock) ----
nav_col1, nav_col2 = st.columns([1, 3])
with nav_col1:
    st.markdown(f'<div class="logo">EMPRESS<br><span style="font-size:12px;color:#8a6fb5">NATURALS</span></div>', unsafe_allow_html=True)
with nav_col2:
    st.markdown(
        '<div style="display:flex; align-items:center; justify-content:space-between;">'
        '<div class="nav-links"><a href="#">Shop</a><a href="#">Best Sellers</a><a href="#">Blog</a><a href="#">About</a></div>'
        '<div><button class="btn-empress" onclick="">Shop</button></div>'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ---- Hero section ----
hero_left, hero_right = st.columns([2, 1])
with hero_left:
    st.markdown('<div class="hero-title">THE AM & PM SERUM</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Resynchronizing and restoring your skin with naturally sourced ingredients designed for post-menopausal skin.</div>', unsafe_allow_html=True)
    if st.button("SHOP NOW", key="hero_shop"):
        st.info("Link to shop (configure your target URL).")

with hero_right:
    # placeholder image (replace with your hosted product image)
    st.image("https://images.unsplash.com/photo-1542831371-d531d36971e6?w=800", width=320, caption="AM & PM Serum")

st.markdown("---")

# ---- Middle cards ----
c1, c2, c3 = st.columns([1,1,1])
with c1:
    st.markdown('<div class="card"><h3>OUR MISSION</h3><p class="small-muted">Empowering women through perimenopause & menopause with radiant, natural skincare.</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown(
    """
    <div class="card">
        <h3>ELIZABETH'S STORY</h3>
        <p class="small-muted">
            Founder-led journey from hormonal imbalance to hormone-conscious care. 
            <a href="#">Read more</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

with c3:
    st.markdown('<div class="card"><h3>EVERYDAY ESSENTIALS</h3><p class="small-muted">Curated serums and cleansers to support mature skin. <a href="#">Shop the Edit</a></p></div>', unsafe_allow_html=True)

st.markdown("---")

# ---- Blog preview and chat column ----
blog_col, chat_col = st.columns([2, 1])
with blog_col:
    st.subheader("Empress Blog")
    st.markdown("- Self Care Massage for Anti-Aging | Menopause Skincare Guide (Aug 11, 2025)")
    st.markdown("- Best Skincare Products for Menopausal Skin (Aug 4, 2025)")
    st.markdown("- Hydrating Ingredients for Menopause (Aug 1, 2025)")

with chat_col:
    st.subheader("Ask Empress — Chatbot")
    st.write("Ask your post-menopausal wellness questions and receive expert advice.")

st.markdown("---")

# ---- Footer ----
f1, f2, f3 = st.columns([2,1,1])
with f1:
    st.markdown("**Empress Naturals** — cruelty free · clean ingredients · woman-founded")
with f2:
    st.markdown("Contact: hello@empressnaturals.co")
with f3:
    st.markdown("© Empress Naturals")

# Developer note
st.info("Developer notes: replace placeholder images with real product assets, add real shop links, and ensure .env contains PINECONE_API_KEY & GOOGLE_API_KEY. Install dependencies before running.")