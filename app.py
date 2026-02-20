import os
import sys
import time
import logging
import asyncio
import uuid
import threading
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# --- IMPORTS DO PROJETO ---
import LLM
import main
import ingestion

# --- IMPORTS DE BANCO DE DADOS E GRAFO ---
from qdrant_client import QdrantClient
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
import traceback

# =========================================================
# CONFIGURAÇÃO GERAL E CSS
# =========================================================
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], force=True)

st.set_page_config(
    page_title="Jurídico AI", 
    page_icon="⚖️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# CSS: Sidebar limpa + Botões alinhados à esquerda
st.markdown("""
<style>
    div[data-testid="stSidebar"] button {
        text-align: left !important;
        display: block;
        width: 100%;
        border: none;
        background: transparent;
        color: inherit;
    }
    div[data-testid="stSidebar"] button:hover {
        background-color: #f0f2f6;
        border: 1px solid #ccc;
    }
    div[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #ff4b4b !important;
        color: white !important;
        text-align: center !important;
        border: 1px solid #ff4b4b;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 0. VARIÁVEIS DE AMBIENTE
# =========================================================
QDRANT_URL = os.getenv("QDRANT_URL")
DB_URL = os.getenv("DB_URL")

if not QDRANT_URL: QDRANT_URL = "http://localhost:6333"
if not DB_URL: DB_URL = "postgresql://admin:admin@localhost:5432/juridico_db"

# =========================================================
# 1. FUNÇÕES DE BANCO DE DADOS
# =========================================================
async def init_db_tables():
    sql = """
    CREATE TABLE IF NOT EXISTS user_threads (
        thread_id TEXT PRIMARY KEY,
        user_id TEXT,
        title TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    async with AsyncConnectionPool(conninfo=DB_URL, kwargs={"autocommit": True}) as pool:
        async with pool.connection() as conn:
            await conn.execute(sql)

async def criar_nova_conversa_db(user_id, titulo, thread_id=None):
    if not thread_id:
        thread_id = str(uuid.uuid4())
    sql = "INSERT INTO user_threads (thread_id, user_id, title) VALUES (%s, %s, %s)"
    async with AsyncConnectionPool(conninfo=DB_URL, kwargs={"autocommit": True}) as pool:
        async with pool.connection() as conn:
            await conn.execute(sql, (thread_id, user_id, titulo))
    return thread_id

async def listar_conversas_db(user_id):
    sql = "SELECT thread_id, title FROM user_threads WHERE user_id = %s ORDER BY created_at DESC"
    async with AsyncConnectionPool(conninfo=DB_URL) as pool:
        async with pool.connection() as conn:
            cursor = await conn.execute(sql, (user_id,))
            return await cursor.fetchall()

async def carregar_historico_langgraph(thread_id):
    async with AsyncConnectionPool(conninfo=DB_URL, kwargs={"autocommit": True}) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = await checkpointer.aget(config)
        
        msgs_formatadas = []
        if checkpoint and "channel_values" in checkpoint:
            estado = checkpoint["channel_values"]
            if "chat_history" in estado:
                for msg_str in estado["chat_history"]:
                    if msg_str.startswith("User:"):
                        msgs_formatadas.append({"role": "user", "content": msg_str.replace("User: ", "")})
                    elif msg_str.startswith("AI:"):
                        msgs_formatadas.append({"role": "assistant", "content": msg_str.replace("AI: ", "")})
        return msgs_formatadas

async def atualizar_titulo_chat_db(thread_id, novo_titulo):
    sql = "UPDATE user_threads SET title = %s WHERE thread_id = %s"
    async with AsyncConnectionPool(conninfo=DB_URL, kwargs={"autocommit": True}) as pool:
        async with pool.connection() as conn:
            await conn.execute(sql, (novo_titulo, thread_id))

async def excluir_conversa_db(thread_id):
    sql = "DELETE FROM user_threads WHERE thread_id = %s"
    async with AsyncConnectionPool(conninfo=DB_URL, kwargs={"autocommit": True}) as pool:
        async with pool.connection() as conn:
            await conn.execute(sql, (thread_id,))

def gerar_titulo_inteligente_sync(primeira_pergunta):
    print(f"🤖 IA TÍTULO: Iniciando geração para: {primeira_pergunta[:15]}...")
    try:
        prompt = (
            f"Resuma a frase abaixo em um título de 3 a 5 palavras para um chat.\n"
            f"Regras: Sem aspas, sem ponto final, capitalize a primeira letra.\n"
            f"Frase: {primeira_pergunta}\n"
            f"Título:"
        )
        resposta = LLM.llm_haiku.complete(prompt)
        titulo_limpo = resposta.text.strip().replace('"', '').replace('.', '')
        print(f"✅ IA TÍTULO: Sucesso -> '{titulo_limpo}'")
        return titulo_limpo
    except Exception as e:
        print(f"❌ IA TÍTULO: Falhou! Erro: {e}")
        return f"Chat {time.strftime('%H:%M')}"

# =========================================================
# 2. BACKEND - RAG E SETUP
# =========================================================
@st.cache_resource
def carregar_engine_rag():
    try:
        asyncio.run(init_db_tables())
        Settings.embed_model = LLM.embed_model
        Settings.llm = LLM.llm_haiku
        client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
        vector_store = QdrantVectorStore(collection_name="leis_v3", client=client, enable_hybrid=False)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        return index.as_query_engine(similarity_top_k=5)
    except Exception as e:
        st.error(f"Erro ao carregar IA: {e}")
        return None

query_engine = carregar_engine_rag()

async def processar_chat(prompt_usuario, thread_id, pdf_bytes=None):
    async with AsyncConnectionPool(conninfo=DB_URL, max_size=10, kwargs={"autocommit": True}) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        
        workflow = main.create_workflow(query_engine, checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        
        estado_input = {
            "user_question": prompt_usuario,
            "file_bytes": pdf_bytes  
        }
        
        resultado = await workflow.ainvoke(estado_input, config=config)
        return resultado["final_response"]

# =========================================================
# 3. AUTENTICAÇÃO
# =========================================================
APP_USER = os.getenv("APP_USER", "admin")
APP_PASS = os.getenv("APP_PASSWORD", "admin")

def check_login(username, password):
    return username == APP_USER and password == APP_PASS

def login_page():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.title("🔐 Login")
        with st.form("login_form"):
            username = st.text_input("Usuário")
            password = st.text_input("Senha", type="password")
            if st.form_submit_button("Entrar", type="primary", use_container_width=True):
                if check_login(username, password):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("Credenciais inválidas.")

# =========================================================
# 4. MODAIS
# =========================================================
@st.dialog("✏️ Renomear Conversa")
def modal_renomear(thread_id, titulo_atual):
    novo_nome = st.text_input("Novo nome", value=titulo_atual)
    if st.button("Salvar", type="primary", use_container_width=True):
        asyncio.run(atualizar_titulo_chat_db(thread_id, novo_nome))
        st.rerun()

@st.dialog("🗑️ Tem certeza?")
def modal_excluir(thread_id):
    st.warning("A conversa será removida da lista.")
    if st.button("Sim, excluir", type="primary", use_container_width=True):
        asyncio.run(excluir_conversa_db(thread_id))
        st.session_state["current_thread_id"] = None
        st.session_state.messages = []
        st.rerun()

# =========================================================
# 5. UI - CHAT
# =========================================================
def render_chat_message(role, text):
    avatar = "🧑‍💼" if role == "user" else "⚖️"
    with st.chat_message(role, avatar=avatar):
        st.markdown(text)

def stream_text(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

def pagina_chat():
    user_id = st.session_state["username"]
    
    # --- 1. BUSCA HISTÓRICO DO BANCO ---
    try:
        conversas_db = asyncio.run(listar_conversas_db(user_id))
    except:
        conversas_db = []

    # --- 2. LÓGICA DE SELEÇÃO ---
    if "current_thread_id" not in st.session_state:
        st.session_state["current_thread_id"] = None

    thread_atual_id = st.session_state["current_thread_id"]
    
    titulo_atual = "Nova Conversa"
    if thread_atual_id:
        for cid, ctitle in conversas_db:
            if cid == thread_atual_id:
                titulo_atual = ctitle
                break
    else:
        st.session_state.messages = [] 

    # --- 3. SIDEBAR ---
    with st.sidebar:
        if st.button("➕ Nova Conversa", type="primary", use_container_width=True):
            st.session_state["current_thread_id"] = None
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.caption("Histórico")
        
        for cid, ctitle in conversas_db:
            label = f"💬 {ctitle}"
            if cid == thread_atual_id:
                label = f"👉 **{ctitle}**"
            
            if st.button(label, key=cid, use_container_width=True):
                st.session_state["current_thread_id"] = cid
                st.session_state.messages = asyncio.run(carregar_historico_langgraph(cid))
                st.rerun()

        st.markdown("---")
        if st.button("📚 Gestão de Leis", use_container_width=True):
            st.session_state["pagina_atual"] = "ingestao"
            st.rerun()
        if st.button("🔒 Sair", use_container_width=True):
            st.session_state["logged_in"] = False
            st.rerun()

    # --- 4. HEADER ---
    if thread_atual_id:
        col_tit, col_edit, col_del = st.columns([0.8, 0.1, 0.1])
        with col_tit: st.subheader(titulo_atual)
        with col_edit:
            if st.button("✏️", help="Renomear"): modal_renomear(thread_atual_id, titulo_atual)
        with col_del:
            if st.button("🗑️", help="Excluir"): modal_excluir(thread_atual_id)
    else:
        st.subheader("Nova Conversa")
        st.caption("Qual sua dúvida jurídica de hoje?")

    # --- 5. RENDERIZA MENSAGENS E CONTAINER ---
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            render_chat_message(msg["role"], msg["content"])

    # --- 6. INPUT COM UPLOAD NATIVO E PROCESSAMENTO ---
    # O accept_file adiciona um ícone de "clipe de papel" direto na barra do chat!
    if prompt_data := st.chat_input("Digite aqui sua dúvida...", accept_file=True, file_type=["pdf"]):
        
        # Como accept_file=True, o Streamlit retorna um objeto contendo .text e .files
        texto_usuario = prompt_data.text if hasattr(prompt_data, "text") else prompt_data
        arquivos = prompt_data.files if hasattr(prompt_data, "files") else []
        
        bytes_data = None
        nome_arquivo = None
        
        if arquivos and len(arquivos) > 0:
            arquivo_anexado = arquivos[0]
            bytes_data = arquivo_anexado.getvalue()
            nome_arquivo = arquivo_anexado.name

        # Verifica se pelo menos um texto ou um arquivo foi enviado
        if not texto_usuario and not bytes_data:
            st.stop()
            
        display_prompt = texto_usuario or "Analise o documento anexado."
        if nome_arquivo:
            display_prompt = f"📄 *[Arquivo Anexado: {nome_arquivo}]*\n\n{display_prompt}"
            
        st.session_state.messages.append({"role": "user", "content": display_prompt})
        
        with chat_container:
            render_chat_message("user", display_prompt)
        
        if not query_engine:
            st.error("IA Offline.")
            st.stop()

        flag_novo_chat = False
        if thread_atual_id is None:
            novo_id = str(uuid.uuid4())
            asyncio.run(criar_nova_conversa_db(user_id, "Nova Conversa...", thread_id=novo_id))
            st.session_state["current_thread_id"] = novo_id
            thread_atual_id = novo_id
            flag_novo_chat = True

        with chat_container:
            with st.chat_message("assistant", avatar="⚖️"):
                spinner_msg = f"Lendo o documento '{nome_arquivo}'..." if bytes_data else "Analisando legislação e jurisprudência..."
                
                with st.spinner(spinner_msg):
                    try:
                        resposta_final = asyncio.run(
                            processar_chat(texto_usuario, thread_atual_id, pdf_bytes=bytes_data)
                        )
                    except Exception as e:
                        st.error(f"Erro: {e}")
                        resposta_final = "Erro ao processar sua solicitação."

                if resposta_final:
                    st.write_stream(stream_text(resposta_final))
                    st.session_state.messages.append({"role": "assistant", "content": resposta_final})

        # --- Geração de Título em Background ---
        if flag_novo_chat or (len(st.session_state.messages) <= 2 and "Nova Conversa" in titulo_atual):
            def _worker_titulo(p_pergunta, p_thread_id):
                novo_tit = gerar_titulo_inteligente_sync(p_pergunta)
                if novo_tit:
                    asyncio.run(atualizar_titulo_chat_db(p_thread_id, novo_tit))
                    print(f"✅ [Background] Título atualizado: {novo_tit}")

            # Passa o texto extraído ao invés do objeto inteiro
            t = threading.Thread(target=_worker_titulo, args=(texto_usuario, thread_atual_id))
            t.start()
            
        st.rerun()

# =========================================================
# 6. GESTÃO DE LEIS
# =========================================================
def pagina_ingestao():
    st.sidebar.button("⬅️ Voltar ao Chat", on_click=lambda: st.session_state.update({"pagina_atual": "chat"}))
    st.header("📥 Gestão de Leis")
    st.divider()
    c_input, c_list = st.columns([0.4, 0.6], gap="medium")
    with c_input:
        st.subheader("Nova Lei")
        urls_txt = st.text_area("URLs:", height=200)
        if st.button("🚀 Processar", type="primary", use_container_width=True):
            if not urls_txt.strip(): st.warning("Vazio.")
            else:
                l_urls = [u.strip() for u in urls_txt.split('\n') if u.strip()]
                st.markdown("### Status")
                barra = st.progress(0, text="Iniciando...")
                log_exp = st.expander("Logs", expanded=True)
                with log_exp:
                    for up in ingestion.processar_urls_stream(l_urls):
                        val = min(max(up["progresso"], 0.0), 1.0)
                        barra.progress(val, text=f"{int(val*100)}%")
                        tipo, msg = up["tipo"], up["msg"]
                        color = "green" if tipo == "success" else "red" if tipo == "error" else "orange" if tipo == "warn" else "blue"
                        st.markdown(f":{color}[{msg}]")
                st.success("Fim!")
                time.sleep(1)
                st.rerun()
    with c_list:
        c1, c2 = st.columns([0.8, 0.2])
        with c1: st.subheader("Base Atual")
        with c2: 
            if st.button("🔄"): st.rerun()
        
        st.divider()

        with st.spinner("Carregando leis..."):
            urls = ingestion.listar_urls_no_banco()
        
        if urls:
            for i, u in enumerate(urls):
                col_url, col_btn = st.columns([0.85, 0.15])
                with col_url:
                    st.markdown(f"🔗 `{u}`")
                with col_btn:
                    if st.button("🗑️", key=f"del_{i}", help=f"Excluir {u}"):
                        with st.spinner("Removendo..."):
                            if ingestion.excluir_lei_no_banco(u):
                                st.toast(f"Lei removida: {u}", icon="✅")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Erro ao excluir.")
        else:
            st.info("A base de dados está vazia.")

# =========================================================
# 7. ROTEAMENTO
# =========================================================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "pagina_atual" not in st.session_state:
    st.session_state["pagina_atual"] = "chat"

if not st.session_state["logged_in"]:
    login_page()
else:
    if st.session_state["pagina_atual"] == "chat":
        pagina_chat()
    elif st.session_state["pagina_atual"] == "ingestao":
        pagina_ingestao()