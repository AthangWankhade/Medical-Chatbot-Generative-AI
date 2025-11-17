import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_classic.chains.combine_documents import \
    create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from langchain_pinecone import PineconeVectorStore

from src.helper import (download_hugging_face_embeddings, load_pdf_file,
                        text_split)
from src.prompt import *

load_dotenv()


app = Flask(__name__)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medibot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity", search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="models/gemini-2.5-flash",         # YES—this model name works here!
    api_key=OPENAI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    temperature=0.5,
    max_tokens=500
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


# route: supports GET, form POST, and JSON POST
@app.route("/get", methods=["GET", "POST"])
def chat():
    # Try multiple ways to find the incoming message
    msg = None

    if request.method == "POST":
        # 1) form data (typical HTML form submit)
        msg = request.form.get("msg")

        # 2) json body (fetch with Content-Type: application/json)
        if not msg:
            json_body = request.get_json(silent=True)
            if isinstance(json_body, dict):
                msg = json_body.get("msg")

        # 3) fallback to raw body (for debugging)
        if not msg and request.data:
            try:
                # try to decode as utf-8 string
                raw = request.data.decode("utf-8")
                if raw:
                    msg = raw
            except Exception:
                pass

    else:  # GET
        msg = request.args.get("msg")

    if not msg:
        # helpful error for debugging
        return jsonify({"error": "no 'msg' provided. send as form field 'msg' or JSON {\"msg\": \"...\"}"}), 400

    try:
        # Invoke your RAG chain (existing object)
        response = rag_chain.invoke({"input": msg})
        # response may be dict-like; handle conservatively
        answer = response.get("answer") if isinstance(response, dict) else str(response)
    except Exception as e:
        # return error (ok for dev). In production log instead of returning.
        return jsonify({"error": "chain invoke failed", "details": str(e)}), 500

    return jsonify({"answer": answer})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
