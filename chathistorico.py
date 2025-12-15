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
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("❌ Configure SUPABASE_URL e SUPABASE_KEY nas variáveis de ambiente.")
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
    user_id = params.get("user_id", None)
    token = params.get("access_token", None)
    return user_id, token


def user_storage_dir(user_id: str):
    base = Path("user_data")
    base.mkdir(exist_ok=True)
    user_dir = base / user_id
    user_dir.mkdir(exist_ok=True)
    return user_dir


def load_user_history(user_id: str):
    """Busca histórico JSON da coluna historico_chat em viajantes."""
    if not user_id:
        return []
    try:
        resp = (
            supabase.table("viajantes")
            .select("historico_chat")
            .eq("id", user_id)
            .execute()
        )
        data = resp.data
        if data and data[0].get("historico_chat"):
            return data[0]["historico_chat"]
    except Exception as e:
        st.warning(f"Erro ao buscar histórico no Supabase: {e}")

    p = user_storage_dir(user_id) / "chat_history.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []


def save_user_history(user_id: str, messages):
    """Atualiza campo historico_chat (jsonb) em viajantes."""
    if not user_id:
        return
    try:
        supabase.table("viajantes").update({
            "historico_chat": messages
        }).eq("id", user_id).execute()
    except Exception as e:
        st.warning(f"Erro ao salvar histórico no Supabase: {e}")

    p = user_storage_dir(user_id) / "chat_history.json"
    p.write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def faiss_path_for_user(user_id: str):
    return user_storage_dir(user_id) / "faiss_index.pkl"


def carregar_docs(pasta="docs"):
    documentos = []
    if not os.path.exists(pasta):
        return documentos
    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)
        if arquivo.lower().endswith(".pdf"):
            documentos.extend(PyPDFLoader(caminho).load())
        elif arquivo.lower().endswith(".txt"):
            documentos.extend(TextLoader(caminho, encoding="utf-8").load())
    return documentos


def gerar_resposta(prompt_text: str):
    """Chama o modelo deepseek-ai/deepseek-v3.1-terminus via NVIDIA API."""
    completion = client.chat.completions.create(
        model="deepseek-ai/deepseek-v3.1-terminus",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=8192,
        extra_body={"chat_template_kwargs": {"thinking": True}},
        stream=True
    )

    resposta = ""
    for chunk in completion:
        reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
        if reasoning:
            print(reasoning, end="")
        if chunk.choices[0].delta.content is not None:
            resposta += chunk.choices[0].delta.content
    return resposta



st.set_page_config(
    page_title="DeepSeek Chatbot Multiusuário",
    page_icon="🤖",
    layout="centered"
)
st.title("Chat vIAje!")

user_id, access_token = get_current_user()
if not user_id:
    st.warning("Você não está autenticado. Faça login com ?user_id=ID.")
    st.stop()

st.markdown(f"**Usuário atual:** `{user_id}`")


history_messages = load_user_history(user_id)

if "memory" not in st.session_state or st.session_state.get("user_for_memory") != user_id:
    mem = ConversationBufferMemory(return_messages=True)
    for m in history_messages:
        if m.get("role") == "user":
            mem.chat_memory.add_user_message(m["content"])
        elif m.get("role") == "assistant":
            mem.chat_memory.add_ai_message(m["content"])
    st.session_state.memory = mem
    st.session_state.user_for_memory = user_id


if "vectors" not in st.session_state or st.session_state.get("vectors_user") != user_id:
    emb = NVIDIAEmbeddings()
    st.session_state.embeddings = emb

    docs = carregar_docs("./docs")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    final_documents = splitter.split_documents(docs)

    faiss_file = faiss_path_for_user(user_id)
    if faiss_file.exists():
        try:
            st.session_state.vectors = FAISS.load_local(
                str(faiss_file),
                st.session_state.embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.warning(f"Erro ao carregar FAISS local. Recriando: {e}")
            st.session_state.vectors = FAISS.from_documents(
                final_documents,
                st.session_state.embeddings
            )
            st.session_state.vectors.save_local(str(faiss_file))
    else:
        if final_documents:
            st.session_state.vectors = FAISS.from_documents(
                final_documents,
                st.session_state.embeddings
            )
            st.session_state.vectors.save_local(str(faiss_file))
        else:
            st.session_state.vectors = None

    st.session_state.vectors_user = user_id


prompt = ChatPromptTemplate.from_template("""
Responda com base apenas no contexto:
<context>
{context}
<context>
Pergunta: {input}
""")

# Mostrar histórico (como chat)
for msg in st.session_state.memory.chat_memory.messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)


if user_input := st.chat_input("Digite sua pergunta..."):
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("🤖 Pensando..."):
        if st.session_state.vectors:
            retriever = st.session_state.vectors.as_retriever()
            docs = retriever.get_relevant_documents(user_input)

            context_text = "\n\n".join([d.page_content for d in docs])
            full_prompt = f"Contexto:\n{context_text}\n\nPergunta: {user_input}"

            start = time.process_time()
            answer = gerar_resposta(full_prompt)

            st.session_state.memory.chat_memory.add_user_message(user_input)
            st.session_state.memory.chat_memory.add_ai_message(answer)

            messages = []
            for m in st.session_state.memory.chat_memory.messages:
                role = "user" if m.type == "human" else "assistant"
                messages.append({"role": role, "content": m.content})

            save_user_history(user_id, messages)

            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(
                    f"⏱ Tempo de resposta: "
                    f"{time.process_time() - start:.2f} segundos"
                )

            with st.expander("🔎 Documentos semelhantes"):
                for doc in docs:
                    st.markdown(doc.page_content)
                    st.write("---")
        else:
            start = time.process_time()
            answer = gerar_resposta(user_input)

            st.session_state.memory.chat_memory.add_user_message(user_input)
            st.session_state.memory.chat_memory.add_ai_message(answer)

            messages = []
            for m in st.session_state.memory.chat_memory.messages:
                role = "user" if m.type == "human" else "assistant"
                messages.append({"role": role, "content": m.content})

            save_user_history(user_id, messages)

            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(
                    f"⏱ Tempo de resposta: "
                    f"{time.process_time() - start:.2f} segundos"
                )
