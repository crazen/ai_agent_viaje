import streamlit as st
import os, json, time
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
from pathlib import Path

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ Configure SUPABASE_URL e SUPABASE_KEY.")
    st.stop()

if not NVIDIA_API_KEY:
    st.error("❌ NVIDIA_API_KEY não configurada.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)


def get_current_user():
    params = st.query_params
    return params.get("user_id"), params.get("access_token")


def user_storage_dir(user_id: str):
    p = Path("user_data") / user_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_user_history(user_id: str):
    if not user_id:
        return []
    try:
        resp = (
            supabase.table("viajantes")
            .select("historico_chat")
            .eq("id", user_id)
            .execute()
        )
        if resp.data and resp.data[0].get("historico_chat"):
            return resp.data[0]["historico_chat"]
    except Exception as e:
        st.warning(f"Erro Supabase: {e}")

    p = user_storage_dir(user_id) / "chat_history.json"
    if p.exists():
        return json.loads(p.read_text("utf-8"))
    return []


def save_user_history(user_id: str, messages):
    if not user_id:
        return
    try:
        supabase.table("viajantes").update(
            {"historico_chat": messages}
        ).eq("id", user_id).execute()
    except Exception as e:
        st.warning(f"Erro Supabase: {e}")

    (user_storage_dir(user_id) / "chat_history.json").write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def carregar_docs(pasta="docs"):
    docs = []
    if not os.path.exists(pasta):
        return docs
    for arq in os.listdir(pasta):
        path = os.path.join(pasta, arq)
        if arq.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif arq.lower().endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs



def gerar_resposta(prompt_text: str):
    completion = client.chat.completions.create(
        model="deepseek-ai/deepseek-v3.1-terminus",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=8192,
        stream=True
    )

    resposta = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            resposta += chunk.choices[0].delta.content
    return resposta



st.set_page_config(page_title="Chat vIAje!", page_icon="🤖")
st.title("Chat vIAje!")

user_id, _ = get_current_user()
if not user_id:
    st.warning("Faça login com ?user_id=ID")
    st.stop()


# HISTÓRICO EM MEMÓRIA MANUAL
if "messages" not in st.session_state:
    st.session_state.messages = load_user_history(user_id)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


if "vectors" not in st.session_state:
    docs = carregar_docs("./docs")
    if docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)
        st.session_state.vectors = FAISS.from_documents(
            chunks,
            NVIDIAEmbeddings()
        )
    else:
        st.session_state.vectors = None


if user_input := st.chat_input("Digite sua pergunta..."):
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("🤖 Pensando..."):
        if st.session_state.vectors:
            docs = st.session_state.vectors.as_retriever().get_relevant_documents(user_input)
            context = "\n\n".join(d.page_content for d in docs)
            prompt = f"Contexto:\n{context}\n\nPergunta: {user_input}"
        else:
            prompt = user_input

        answer = gerar_resposta(prompt)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    save_user_history(user_id, st.session_state.messages)

    with st.chat_message("assistant"):
        st.markdown(answer)

