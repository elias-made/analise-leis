import os
from qdrant_client import QdrantClient
import streamlit as st
import asyncio
import time
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

import logging
import sys

import LLM
import main
import ingestion
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

# =========================================================
# CONFIGURAÇÃO DE LOGS
# =========================================================
# Configura o logger raiz para aceitar mensagens de nível INFO
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout) # Força a saída para o terminal
    ],
    force=True # Sobrescreve configurações padrão do Streamlit/outras libs
)

# Opcional: Silenciar logs chatos de bibliotecas externas (httpx, qdrant, etc)
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("httpcore").setLevel(logging.WARNING)
# logging.getLogger("qdrant_client").setLevel(logging.WARNING)

url = os.getenv("QDRANT_URL")

# =========================================================
# 1. CONFIGURAÇÃO GERAL
# =========================================================
st.set_page_config(
    page_title="Jurídico AI", 
    page_icon="⚖️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. BACKEND (IA)
# =========================================================
@st.cache_resource
def carregar_sistema_ia():
    print("🚀 Iniciando sistema...")
    try:
        # Configurações
        Settings.embed_model = LLM.embed_model
        Settings.llm = LLM.llm_haiku

        client = QdrantClient(url=url, prefer_grpc=False)
        
        # Conexão Qdrant
        vector_store = QdrantVectorStore(
            collection_name="leis_v3", 
            # url=url,
            # api_key=None,
            client=client,
            enable_hybrid=False,
            # vector_name="text-dense"
        )
        
        # Criação do Grafo
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            vector_store_query_mode="default"
        )
        return main.create_workflow(query_engine)
        
    except Exception as e:
        print(f"❌ Erro ao carregar IA: {e}")
        return None

workflow = carregar_sistema_ia()

# =========================================================
# 3. AUTENTICAÇÃO (LOGIN)
# =========================================================
env_user = os.getenv("APP_USER", "admin")
env_pass = os.getenv("APP_PASSWORD", "admin")

def check_login(username, password):
    return username == env_user and password == env_pass

def login_page():
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.title("🔐 Acesso Restrito")
        st.markdown("Entre com suas credenciais.")
        
        with st.form("login_form"):
            username = st.text_input("Usuário")
            password = st.text_input("Senha", type="password")
            
            submit = st.form_submit_button("Entrar", type="primary", width="stretch")
            
            if submit:
                if check_login(username, password):
                    st.session_state["logged_in"] = True
                    st.rerun()
                else:
                    st.error("Usuário ou senha incorretos.")

# =========================================================
# 4. PÁGINAS DO SISTEMA
# =========================================================

def render_chat_message(role, text):
    avatar = "🧑‍💼" if role == "user" else "⚖️"
    with st.chat_message(role, avatar=avatar):
        st.markdown(text)

def pagina_chat():
    st.header("💬 Consultoria Jurídica")
    st.caption("Especialista em Microempresas e Legislação Geral.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Exibe histórico
    for msg in st.session_state.messages:
        render_chat_message(msg["role"], msg["content"])

    # Input do usuário
    if prompt := st.chat_input("Digite sua dúvida..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_chat_message("user", prompt)
        
        if not workflow:
            st.error("⚠️ Sistema de IA offline. Verifique o terminal.")
            return

        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("Analisando legislação..."):
                try:
                    # Contexto para a IA
                    hist = [f"{'User' if m['role']=='user' else 'AI'}: {m['content']}" 
                           for m in st.session_state.messages if m['content'] != prompt]
                    
                    estado = {"user_question": prompt, "chat_history": hist}
                    
                    # Chamada Async
                    res = asyncio.run(workflow.ainvoke(estado))
                    
                    resposta = res["final_response"]
                    st.markdown(resposta)
                    st.session_state.messages.append({"role": "assistant", "content": resposta})
                    
                except Exception as e:
                    st.error(f"Erro na execução: {str(e)}")

def pagina_ingestao():
    st.header("📥 Gestão de Leis")
    st.divider()
    
    col_input, col_list = st.columns([0.4, 0.6], gap="medium")
    
    with col_input:
        st.subheader("➕ Adicionar Nova Lei")
        urls_input = st.text_area("URLs (uma por linha):", height=200, placeholder="https://www.planalto.gov.br/...")
        
        if st.button("🚀 Iniciar Processamento", type="primary", width="stretch"):
            if not urls_input.strip():
                st.warning("O campo está vazio.")
            else:
                lista_urls = [u.strip() for u in urls_input.split('\n') if u.strip()]
                
                # Container de Progresso
                st.markdown("### 📊 Status")
                barra = st.progress(0, text="Iniciando...")
                log_box = st.expander("Logs Detalhados", expanded=True)
                
                with log_box:
                    for update in ingestion.processar_urls_stream(lista_urls):
                        val = min(max(update["progresso"], 0.0), 1.0)
                        
                        barra.progress(val, text=f"Progresso: {int(val*100)}%")
                        
                        tipo, msg = update["tipo"], update["msg"]
                        if tipo == "success": st.markdown(f":green[{msg}]")
                        elif tipo == "error": st.markdown(f":red[{msg}]")
                        elif tipo == "warn": st.markdown(f":orange[{msg}]")
                        else: st.text(msg)
                
                st.success("Concluído!")
                st.rerun()

    # --- DIREITA: LISTA ---
    with col_list:
        c1, c2 = st.columns([0.8, 0.2])
        with c1: st.subheader("📚 Leis na Base")
        with c2: 
            if st.button("🔄"): st.rerun()

        with st.spinner("Buscando dados..."):
            urls = ingestion.listar_urls_no_banco()
        
        if urls:
            df = pd.DataFrame(urls, columns=["Fonte Indexada"])
            st.dataframe(df, width="stretch", hide_index=True, height=450)
        else:
            st.info("Nenhuma lei cadastrada ainda.")

# =========================================================
# 5. CONTROLE DE FLUXO (ROTEAMENTO)
# =========================================================

# Inicializa estado de login
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Verifica login
if not st.session_state["logged_in"]:
    login_page()
else:
    # Mostra aplicação principal
    with st.sidebar:
        st.title("⚖️ Menu")
        pg = st.radio("Navegação", ["Chat", "Gestão de Leis"], label_visibility="collapsed")
        
        st.divider()
    
        if pg == "Chat":
            if st.button("🗑️ Limpar Chat", width="stretch"):
                st.session_state.messages = []
                st.rerun()
        
        st.divider()
        if st.button("🔒 Sair", width="stretch"):
            st.session_state["logged_in"] = False
            st.rerun()

    if pg == "Chat":
        pagina_chat()
    elif pg == "Gestão de Leis":
        pagina_ingestao()