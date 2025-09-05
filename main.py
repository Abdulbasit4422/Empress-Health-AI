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

# Initialize Pinecone and embedding model
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("empress")
embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Define system prompt template
system_prompt_template = """
You are Ask Empress, a trusted Post-Menopausal Health and Wellness Expert. 
Your role is to provide users with clear, empathetic, and deeply informative answers to their questions. 

When responding:
1. **Be comprehensive and well-structured** – organize your response into clear sections, Use the format below when you think it is best, and adopt better ones when need arise
   - Overview
   - Causes or Contributing Factors 
   - Management & Lifestyle Recommendations 
   - Evidence-based Product Recommendations (from the provided context where available) 
   - When to Seek Professional Help (if relevant).
   - Disclaimer to consult a doctor when necessary.

these format above are just to guide you, you can always adjust it as the case may be.

2. Ground your advice in the retrieved knowledge below as much as possible. If no relevant information is available, rely on your medical expertise but be transparent about it.

3. Personalize your response to the user’s concern, showing empathy and reassurance in a more feminine tone. 

4. Always mention appropriate Empress Naturals products (from the retrieved content) if they can help the user’s situation, and explain why they are useful.

5. Avoid short or generic answers – aim for depth, clarity, elaborate, comprehensive, well structured and practical guidance, with good fonts and relevants emojis. 

Retrieved information for context:
{doc_content}
provide users with clear, elaborate, comprehensive, well structured, outlined, empathetic, and deeply informative answers to their questions with good fonts and relevants emojis.
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