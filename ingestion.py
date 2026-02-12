# ingestion.py
from qdrant_client import QdrantClient, models
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
import utils
import LLM
import os
import math # Importante para cálculos de lote
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "leis_v3"
url = os.getenv("QDRANT_URL")
# Timeout aumentado para evitar quedas em lotes grandes
client = QdrantClient(url=url, timeout=300) 

# 1. VERIFICAÇÃO
def verificar_se_url_existe(url_para_checar):
    try:
        if not client.collection_exists(COLLECTION_NAME): return False
        res = client.count(
            collection_name=COLLECTION_NAME,
            count_filter=models.Filter(must=[models.FieldCondition(key="url_geral", match=models.MatchValue(value=url_para_checar))])
        )
        return res.count > 0
    except Exception: return False

# 2. LISTAGEM
def listar_urls_no_banco():
    try:
        if not client.collection_exists(COLLECTION_NAME): return []
        unique_urls = set()
        next_offset = None
        while True:
            records, next_offset = client.scroll(
                collection_name=COLLECTION_NAME, limit=100, 
                with_payload=["url_geral"], with_vectors=False, offset=next_offset
            )
            for r in records:
                if r.payload.get("url_geral"): unique_urls.add(r.payload.get("url_geral"))
            if next_offset is None: break
        return list(unique_urls)
    except: return []

# 3. SALVAMENTO (Agora aceita lotes menores)
def run_ingestion_batch(documentos_batch):
    """
    Processa apenas um lote de documentos para permitir atualização da UI.
    """
    vector_store = QdrantVectorStore(
        collection_name=COLLECTION_NAME, 
        url=url,
        api_key="", # Se tiver api key, coloque os.getenv("QDRANT_API_KEY")
        enable_hybrid=False, 
        batch_size=64,
    )
    # StorageContext garante que estamos conectando ao Qdrant existente
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # show_progress=False pois nós controlamos o progresso no Streamlit
    VectorStoreIndex.from_documents(
        documentos_batch, 
        storage_context=storage_context, 
        show_progress=False 
    )

# 4. PROCESSAMENTO OTIMIZADO
def processar_urls_stream(lista_urls: list):
    Settings.embed_model = LLM.embed_model 
    Settings.llm = LLM.llm_haiku
    
    total_urls = len(lista_urls)
    fatia_por_url = 1.0 / total_urls
    
    # Definição de Pesos para Realismo
    # Verificação (5%) + Download (10%) + Fatiamento (5%) + Indexação (80%)
    PESO_SETUP = 0.20 
    PESO_INDEXACAO = 0.80
    
    # Tamanho do lote de documentos para atualização da barra
    BATCH_SIZE_DOCS = 20 

    for i, url in enumerate(lista_urls):
        base = i / total_urls # Onde começa esta URL na barra total (0.0 a 1.0)
        
        # --- ETAPA 1: VERIFICAÇÃO ---
        yield {
            "tipo": "info", 
            "msg": f"🔍 [{i+1}/{total_urls}] Verificando: {url}...", 
            "progresso": base + (fatia_por_url * 0.05)
        }
        
        if verificar_se_url_existe(url):
            yield {
                "tipo": "warn", 
                "msg": f"⏩ Já existe no banco: {url}", 
                "progresso": base + fatia_por_url
            }
            continue

        try:
            # --- ETAPA 2: DOWNLOAD ---
            yield {
                "tipo": "info", 
                "msg": f"📥 [{i+1}/{total_urls}] Baixando HTML...", 
                "progresso": base + (fatia_por_url * 0.15)
            }
            
            titulo, texto_bruto = utils.extract_html(url)
            
            if titulo == "Erro":
                yield {"tipo": "error", "msg": f"❌ Falha de conexão: {url}", "progresso": base + fatia_por_url}
                continue
            
            # --- ETAPA 3: FATIAMENTO (PREPARAÇÃO) ---
            yield {
                "tipo": "info", 
                "msg": f"🔪 [{i+1}/{total_urls}] Fatiando artigos...", 
                "progresso": base + (fatia_por_url * PESO_SETUP)
            }
            
            chunks_dict = utils.fatiar_por_artigos(texto_bruto, titulo, url)
            
            documentos_totais = []
            for item in chunks_dict:
                doc = Document(
                    text=item['conteudo'], metadata=item['metadata'],
                    excluded_llm_metadata_keys=['url_geral', 'tipo'],
                    excluded_embed_metadata_keys=['url_geral']
                )
                documentos_totais.append(doc)

            # --- ETAPA 4: INDEXAÇÃO EM LOTES (AQUI ESTÁ A MÁGICA) ---
            total_docs = len(documentos_totais)
            
            if total_docs > 0:
                # Calcula quantos loops vamos fazer
                num_batches = math.ceil(total_docs / BATCH_SIZE_DOCS)
                
                for b_idx in range(num_batches):
                    start = b_idx * BATCH_SIZE_DOCS
                    end = start + BATCH_SIZE_DOCS
                    lote_atual = documentos_totais[start:end]
                    
                    # Calcula o progresso dentro da fase de indexação
                    progresso_lote = (b_idx / num_batches)
                    # Traduz isso para a barra global
                    progresso_global = base + (fatia_por_url * PESO_SETUP) + (fatia_por_url * PESO_INDEXACAO * progresso_lote)
                    
                    yield {
                        "tipo": "info", 
                        "msg": f"🧠 [{i+1}/{total_urls}] Incorporando vetores: {start}/{total_docs} docs...", 
                        "progresso": progresso_global
                    }
                    
                    # Chama a função de ingestão apenas para esse pedaço
                    run_ingestion_batch(lote_atual)

                # --- ETAPA 5: CONCLUSÃO DESTA URL ---
                yield {
                    "tipo": "success", 
                    "msg": f"✅ Sucesso: {titulo} ({total_docs} partes)", 
                    "progresso": base + fatia_por_url
                }
            else:
                yield {"tipo": "warn", "msg": f"⚠️ Arquivo vazio ou sem artigos identificados: {url}", "progresso": base + fatia_por_url}
                
        except Exception as e:
            # Em caso de erro, imprime no console pra ajudar a debugar
            print(f"Erro detalhado na URL {url}: {e}")
            yield {"tipo": "error", "msg": f"🔥 Erro crítico: {str(e)[:100]}...", "progresso": base + fatia_por_url}

    # Garante 100% no final
    yield {"tipo": "complete", "msg": "Processo Finalizado!", "progresso": 1.0}

# Adicione ao final do ingestion.py

def excluir_lei_no_banco(url_para_excluir):
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="url_geral", 
                        match=models.MatchValue(value=url_para_excluir)
                    )
                ]
            ),
        )
        return True
    except Exception as e:
        print(f"❌ Erro ao excluir do Qdrant: {e}")
        return False