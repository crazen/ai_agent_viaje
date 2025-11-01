import streamlit as st
import os
import json
import time
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# --------------------------------------
# CONFIGURAÇÕES INICIAIS
# --------------------------------------
load_dotenv()

try:
    nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
except:
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")

if nvidia_api_key:
    os.environ['NVIDIA_API_KEY'] = nvidia_api_key
else:
    st.error("⚠️ NVIDIA_API_KEY não encontrada.")
    st.stop()

# --------------------------------------
# BANCO DE DADOS
# --------------------------------------
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        port=os.getenv("DB_PORT", 5432),
        cursor_factory=RealDictCursor
    )

def carregar_historico_db(usuario_id):
    """Carrega histórico JSON de um usuário específico do banco."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT historico_chat FROM viajantes WHERE id = %s", (usuario_id,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result and result["historico_chat"]:
        return result["historico_chat"]
    return []

def salvar_historico_db(usuario_id, historico):
    """Atualiza o histórico JSON no banco de dados."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE viajantes SET historico_chat = %s WHERE id = %s",
        (json.dumps(historico, ensure_ascii=False), usuario_id)
    )
    conn.commit()
    cur.close()
    conn.close()

# --------------------------------------
# INICIALIZAR MEMÓRIA POR USUÁRIO
# --------------------------------------
def load_memory(usuario_id):
    if "memory" not in st.session_state:
        memory = ConversationBufferMemory(return_messages=True)
        historico = carregar_historico_db(usuario_id)
        for m in historico:
            memory.chat_memory.add_user_message(m["user"])
            memory.chat_memory.add_ai_message(m["ai"])
        st.session_state.memory = memory

def save_memory(usuario_id):
    """Salva o histórico atual no banco."""
    messages = st.session_state.memory.chat_memory.messages
    historico = []
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            historico.append({
                "user": messages[i].content,
                "ai": messages[i+1].content
            })
    salvar_historico_db(usuario_id, historico)

# --------------------------------------
# CARREGAR DOCUMENTOS PDF/TXT
# --------------------------------------
def carregar_docs(pasta="docs"):
    documentos = []
    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)
        if arquivo.lower().endswith(".pdf"):
            documentos.extend(PyPDFLoader(caminho).load())
        elif arquivo.lower().endswith(".txt"):
            documentos.extend(TextLoader(caminho, encoding="utf-8").load())
    return documentos

# --------------------------------------
# CRIAR EMBEDDINGS E FAISS
# --------------------------------------
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

# --------------------------------------
# INTERFACE STREAMLIT
# --------------------------------------
st.title("🤖 Nvidia NIM Chat por Usuário (com histórico no BD)")

usuario_id = st.text_input("ID do usuário (viajantes.id):", value="1")

if not usuario_id.isdigit():
    st.warning("Insira um ID de usuário válido (número).")
    st.stop()

usuario_id = int(usuario_id)
llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")

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
        role = "👤 Usuário" if msg.type == "human" else "Assistente"
        st.write(f"**{role}:** {msg.content}")
