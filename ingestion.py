# ingestion.py
from qdrant_client import QdrantClient, models
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
import utils
import LLM
import os

COLLECTION_NAME = "leis_v2"
url = os.getenv("QDRANT_URL")
client = QdrantClient(url=url, timeout=120)

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

# 3. SALVAMENTO
def run_ingestion(documentos_llamaindex):
    vector_store = QdrantVectorStore(collection_name=COLLECTION_NAME, client=client, enable_hybrid=False)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(documentos_llamaindex, storage_context=storage_context, show_progress=False)

# 4. PROCESSAMENTO COM PROGRESSO DETALHADO (CORRIGIDO)
def processar_urls_stream(lista_urls: list):
    Settings.embed_model = LLM.embed_model 
    Settings.llm = LLM.llm_sonnet
    
    total_urls = len(lista_urls)
    # Tamanho da fatia da barra para cada URL (Ex: se são 2 urls, cada uma vale 0.5)
    fatia_por_url = 1.0 / total_urls
    
    for i, url in enumerate(lista_urls):
        # Base do progresso (onde começa essa URL na barra geral)
        base = i / total_urls
        
        # --- ETAPA 1: VERIFICAÇÃO (10% da fatia) ---
        yield {
            "tipo": "info", 
            "msg": f"🔍 [{i+1}/{total_urls}] Verificando: {url}...", 
            "progresso": base + (fatia_por_url * 0.1)
        }
        
        if verificar_se_url_existe(url):
            yield {
                "tipo": "warn", 
                "msg": f"⏩ Já existe no banco: {url}", 
                "progresso": base + fatia_por_url # Pula pro final da fatia
            }
            continue

        try:
            # --- ETAPA 2: DOWNLOAD (30% da fatia) ---
            yield {
                "tipo": "info", 
                "msg": f"📥 [{i+1}/{total_urls}] Baixando HTML...", 
                "progresso": base + (fatia_por_url * 0.3)
            }
            
            titulo, texto_bruto = utils.extract_html(url)
            
            if titulo == "Erro":
                yield {"tipo": "error", "msg": f"❌ Falha de conexão: {url}", "progresso": base + fatia_por_url}
                continue
            
            # --- ETAPA 3: FATIAMENTO (60% da fatia) ---
            yield {
                "tipo": "info", 
                "msg": f"🔪 [{i+1}/{total_urls}] Fatiando artigos...", 
                "progresso": base + (fatia_por_url * 0.6)
            }
            
            chunks_dict = utils.fatiar_por_artigos(texto_bruto, titulo, url)
            
            documentos = []
            for item in chunks_dict:
                doc = Document(
                    text=item['conteudo'], metadata=item['metadata'],
                    excluded_llm_metadata_keys=['url_geral', 'tipo'],
                    excluded_embed_metadata_keys=['url_geral']
                )
                documentos.append(doc)

            # --- ETAPA 4: INDEXAÇÃO (90% da fatia) ---
            if documentos:
                yield {
                    "tipo": "info", 
                    "msg": f"💾 [{i+1}/{total_urls}] Salvando {len(documentos)} vetores no Qdrant...", 
                    "progresso": base + (fatia_por_url * 0.9)
                }
                
                run_ingestion(documentos)
                
                # --- ETAPA 5: CONCLUSÃO DESTA URL (100% da fatia) ---
                yield {
                    "tipo": "success", 
                    "msg": f"✅ Sucesso: {titulo}", 
                    "progresso": base + fatia_por_url
                }
            else:
                yield {"tipo": "warn", "msg": f"⚠️ Arquivo vazio: {url}", "progresso": base + fatia_por_url}
                
        except Exception as e:
            yield {"tipo": "error", "msg": f"🔥 Erro crítico: {e}", "progresso": base + fatia_por_url}

    # Garante que fecha em 100% no final
    yield {"tipo": "complete", "msg": "Processo Finalizado!", "progresso": 1.0}