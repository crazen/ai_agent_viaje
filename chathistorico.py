import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# ---------------------------------------------------------
# CONFIGURAÇÕES INICIAIS
# ---------------------------------------------------------
load_dotenv()

# 🔐 Chaves de API
try:
    nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
except:
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

if not nvidia_api_key:
    st.error("⚠️ NVIDIA_API_KEY não encontrada. Configure nos Secrets do Streamlit.")
    st.stop()

if not supabase_url or not supabase_key:
    st.error("⚠️ SUPABASE_URL e SUPABASE_KEY não configuradas.")
    st.stop()

os.environ["NVIDIA_API_KEY"] = nvidia_api_key

# ---------------------------------------------------------
# CLIENTE SUPABASE
# ---------------------------------------------------------
supabase: Client = create_client(supabase_url, supabase_key)

# ---------------------------------------------------------
# FUNÇÕES DE BANCO (SUPABASE)
# ---------------------------------------------------------
def carregar_historico_db(usuario_id: int):
    """Carrega o histórico de chat de um usuário no Supabase."""
    try:
        response = (
            supabase.table("viajantes")
            .select("historico_chat")
            .eq("id", usuario_id)
            .execute()
        )
        data = response.data
        if data and data[0].get("historico_chat"):
            return data[0]["historico_chat"]
        return []
    except Exception as e:
        st.warning(f"Erro ao carregar histórico: {e}")
        return []

def salvar_historico_db(usuario_id: int, historico: list):
    """Atualiza o histórico de chat do usuário no Supabase."""
    try:
        supabase.table("viajantes").update({
            "historico_chat": historico
        }).eq("id", usuario_id).execute()
    except Exception as e:
        st.warning(f"Erro ao salvar histórico: {e}")

# ---------------------------------------------------------
# INICIALIZAR MEMÓRIA POR USUÁRIO
# ---------------------------------------------------------
def load_memory(usuario_id):
    if "memory" not in st.session_state:
        memory = ConversationBufferMemory(return_messages=True)
        historico = carregar_historico_db(usuario_id)
        for m in historico:
            memory.chat_memory.add_user_message(m["user"])
            memory.chat_memory.add_ai_message(m["ai"])
        st.session_state.memory = memory

def save_memory(usuario_id):
    """Salva o histórico atual no Supabase."""
    messages = st.session_state.memory.chat_memory.messages
    historico = []
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            historico.append({
                "user": messages[i].content,
                "ai": messages[i + 1].content
            })
    salvar_historico_db(usuario_id, historico)

# ---------------------------------------------------------
# CARREGAR DOCUMENTOS PDF/TXT
# ---------------------------------------------------------
def carregar_docs(pasta="docs"):
    documentos = []
    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)
        if arquivo.lower().endswith(".pdf"):
            documentos.extend(PyPDFLoader(caminho).load())
        elif arquivo.lower().endswith(".txt"):
            documentos.extend(TextLoader(caminho, encoding="utf-8").load())
    return documentos

# ---------------------------------------------------------
# CRIAR EMBEDDINGS E FAISS
# ---------------------------------------------------------
def vector_embedding():
    if not st.session_state.get("vectors"):
        st.session_state.embeddings = NVIDIAEmbeddings()
        docs = carregar_docs("./docs")
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        st.session_state.final_documents = splitter.split_documents(docs)
        if st.session_state.final_documents:
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )
            st.success("✅ Vector Store criado com sucesso!")
        else:
            st.warning("⚠ Nenhum documento válido encontrado em ./docs")

# ---------------------------------------------------------
# INTERFACE STREAMLIT
# ---------------------------------------------------------
st.title("🤖 Chat IA Personalizado (Histórico via Supabase)")

usuario_id = st.text_input("🧑‍💻 ID do usuário (viajantes.id):", value="1")

if not usuario_id.isdigit():
    st.warning("Insira um ID de usuário válido (número).")
    st.stop()

usuario_id = int(usuario_id)
llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")

# Carregar memória do Supabase
load_memory(usuario_id)

prompt = ChatPromptTemplate.from_template("""
Responda com base apenas no contexto:
<context>
{context}
<context>
Pergunta: {input}
""")

user_input = st.text_input("Digite sua pergunta:")

if st.button("📂 Criar Embeddings dos Documentos"):
    vector_embedding()

if user_input:
    if st.session_state.get("vectors"):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_input})
        answer = response['answer']

        st.session_state.memory.chat_memory.add_user_message(user_input)
        st.session_state.memory.chat_memory.add_ai_message(answer)
        save_memory(usuario_id)

        st.write(f"🤖 {answer}")
        st.write(f"⏱ {time.process_time() - start:.2f} segundos")

        with st.expander("🔎 Documentos semelhantes"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        conversation = ConversationChain(llm=llm, memory=st.session_state.memory)
        answer = conversation.predict(input=user_input)
        save_memory(usuario_id)
        st.write(f"🤖 {answer}")

# Exibir histórico
with st.expander("📝 Histórico de conversa do usuário"):
    for msg in st.session_state.memory.chat_memory.messages:
        role = "👤 Usuário" if msg.type == "human" else "🤖 Assistente"
        st.write(f"**{role}:** {msg.content}")
