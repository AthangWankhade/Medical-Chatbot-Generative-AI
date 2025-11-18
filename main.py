import os
import uuid  # Import uuid to generate unique chat IDs

from dotenv import load_dotenv
from flask import (Flask, jsonify, redirect, render_template, request, session,
                   url_for)
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import \
    create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- 1. Initialization and Setup ---
load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get(
    'FLASK_SECRET_KEY', 'a_very_secret_key_fallback')

# --- 2. Load Environment Variables ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# --- 3. Initialize Embeddings and Vector Store (Singleton) ---
print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
index_name = "medibot"
print("Connecting to Pinecone...")
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 3})

# --- 4. Initialize LLM (Singleton) ---
print("Initializing LLM...")
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.5,
    max_tokens=500,
    convert_system_message_to_human=True
)

# --- 5. Define Prompts and RAG Chain (Singleton) ---
# Prompt 1: Rephrase question based on history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Prompt 2: Answer the question using retrieved documents
try:
    from src.prompt import system_prompt
except ImportError:
    # --- PROMPT UPDATED HERE ---
    system_prompt = (
        "You are a medical health assistant designed to help users understand their symptoms. "
        "Using only the retrieved medical context provided, assess the user's symptoms, suggest possible diagnoses, "
        "recommend appropriate next steps including treatments or medications when supported by the context, "
        "and clearly advise when the user should seek in-person evaluation or consult a specific type of healthcare professional (for example: primary care physician, emergency department, cardiologist, dermatologist, etc.). "
        "Base your reasoning strictly on the given context; if the information is insufficient or absent, explicitly say 'I don't know' or that the context is insufficient. "
        "Always encourage users to consult a qualified healthcare professional for diagnosis and treatment confirmation. "
        "Keep responses concise and limited to three sentences."
        "\n\n"
        "Context:\n"
        "{context}"
    )
    # --- END OF UPDATE ---
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain)
print("RAG chain created successfully.")

# --- 6. Helper Function for Session Management ---


def get_chat_summaries():
    """Generates a list of chat summaries from the session."""
    if 'chats' not in session:
        session['chats'] = {}

    summaries = []
    # Sort by chat_id (which is a timestamp) to get newest first
    sorted_chat_ids = sorted(session['chats'].keys(), reverse=True)

    for chat_id in sorted_chat_ids:
        messages = session['chats'][chat_id]
        if not messages:
            title = "New Chat"
        else:
            # Use the first user message as the title
            title = messages[0]['content'][:30] + "..."
        summaries.append({'id': chat_id, 'title': title})
    return summaries


def get_current_chat_id():
    """Gets the current chat ID, creating a new chat if none exists."""
    if 'chats' not in session:
        session['chats'] = {}

    current_chat_id = session.get('current_chat_id')

    # If no current chat or current chat was deleted, create a new one
    if not current_chat_id or current_chat_id not in session['chats']:
        # Use a timestamp for a sortable, unique ID
        new_chat_id = f"chat_{uuid.uuid4()}"
        session['chats'][new_chat_id] = []
        session['current_chat_id'] = new_chat_id
        session.modified = True

    return session['current_chat_id']

# --- 7. Define Flask Routes ---


@app.route("/")
def index():
    """
    Load the main chat page. It will either load the
    current chat or create a new one if none exists.
    """
    current_chat_id = get_current_chat_id()
    return redirect(url_for('load_chat', chat_id=current_chat_id))


@app.route("/chat/<chat_id>")
def load_chat(chat_id):
    """
    Load a specific chat conversation.
    """
    if 'chats' not in session or chat_id not in session['chats']:
        # If chat_id is invalid, redirect to a new chat
        return redirect(url_for('index'))

    # Set the current chat ID
    session['current_chat_id'] = chat_id

    # Get history for this chat
    chat_history = session['chats'][chat_id]

    # Get summaries for all chats
    chat_summaries = get_chat_summaries()

    return render_template(
        "chat.html",
        chat_history=chat_history,
        chat_summaries=chat_summaries,
        current_chat_id=chat_id
    )


@app.route("/new_chat")
def new_chat():
    """
    Create a new, empty chat and load it.
    """
    # Use a timestamp for a sortable, unique ID
    new_chat_id = f"chat_{uuid.uuid4()}"
    if 'chats' not in session:
        session['chats'] = {}

    session['chats'][new_chat_id] = []
    session['current_chat_id'] = new_chat_id
    session.modified = True

    print(f"New chat session started: {new_chat_id}")
    return redirect(url_for('load_chat', chat_id=new_chat_id))


@app.route("/get", methods=["POST"])
def chat():
    """
    Handle the incoming chat message for the CURRENT chat.
    """
    try:
        json_body = request.get_json()
        if not json_body or 'msg' not in json_body:
            return jsonify({"error": "No 'msg' key found in JSON payload."}), 400

        user_message_content = json_body["msg"]

        # Get the current chat ID from the session
        current_chat_id = session.get('current_chat_id')
        if not current_chat_id or current_chat_id not in session.get('chats', {}):
            return jsonify({"error": "No active chat session found."}), 400

        # Get the history for the *current* chat
        current_chat_history = session['chats'][current_chat_id]

        # Convert session history (list of dicts) to LangChain message objects
        chat_history_messages = []
        for msg in current_chat_history:
            if msg['role'] == 'user':
                chat_history_messages.append(
                    HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                chat_history_messages.append(AIMessage(content=msg['content']))

        # Invoke the RAG chain
        print(
            f"Invoking chain for chat {current_chat_id} with: {user_message_content}")
        response = rag_chain.invoke({
            "input": user_message_content,
            "chat_history": chat_history_messages
        })

        answer = response.get("answer", "Sorry, I couldn't find an answer.")

        # Update session history for the current chat
        session['chats'][current_chat_id].append(
            {'role': 'user', 'content': user_message_content})
        session['chats'][current_chat_id].append(
            {'role': 'assistant', 'content': answer})
        session.modified = True

        print(f"Responding with: {answer}")
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Error in /get: {e}")
        return jsonify({"error": "Failed to process request", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
