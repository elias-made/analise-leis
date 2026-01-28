# ingestion.py
from qdrant_client import QdrantClient, models # <--- Faltava o 'models'
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Seus imports locais
import utils
import LLM

# Configuração Global
COLLECTION_NAME = "leis_v2"
client = QdrantClient(url="http://localhost:6333", timeout=120)

# ==============================================================================
# 1. FUNÇÃO DE VERIFICAÇÃO (Fica aqui, pois mexe com Banco)
# ==============================================================================
def verificar_se_url_existe(url_para_checar):
    try:
        # Se a coleção nem existe, a URL com certeza não está lá
        if not client.collection_exists(COLLECTION_NAME):
            return False
        
        # Conta quantos documentos têm essa url_geral
        resultado = client.count(
            collection_name=COLLECTION_NAME,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="url_geral",
                        match=models.MatchValue(value=url_para_checar)
                    )
                ]
            )
        )
        return resultado.count > 0
    except Exception as e:
        print(f"⚠️ Erro ao verificar Qdrant: {e}")
        return False

# ==============================================================================
# 2. FUNÇÃO DE SALVAMENTO (Indexação)
# ==============================================================================
def run_ingestion(documentos_llamaindex):
    print(f"💾 Salvando {len(documentos_llamaindex)} vetores no Qdrant...")
    
    # Recria a conexão específica para o store
    vector_store = QdrantVectorStore(
        collection_name=COLLECTION_NAME,
        client=client,
        enable_hybrid=True, 
        batch_size=15,
        fastembed_sparse_model="Qdrant/bm25"
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documentos_llamaindex,
        storage_context=storage_context,
        show_progress=True
    )
    print("✅ Ingestão finalizada com sucesso.")

# ==============================================================================
# 3. WORKER DO BACKGROUND (A Mágica)
# ==============================================================================
def processar_urls_background(lista_urls: list):
    """
    Esta função roda em segundo plano. Ela baixa, fatia e salva.
    """
    print(f"🔄 BACKGROUND: Iniciando lista de {len(lista_urls)} URLs...")
    
    # Garante configuração na thread paralela
    Settings.embed_model = LLM.embed_model 
    
    lista_final_documentos = []

    for url in lista_urls:
        # CORREÇÃO AQUI: Chamamos a função local, não a do utils
        if verificar_se_url_existe(url):
            print(f"⏩ Pulando (Já existe): {url}")
            continue

        print(f"--- Processando: {url} ---")
        
        try:
            # B. Baixa e Limpa (Isso sim vem do utils)
            titulo, texto_bruto = utils.extract_html(url)
            
            if titulo == "Erro":
                print(f"❌ Pulei {url} devido a erro de conexão.")
                continue
            
            # C. Fatia
            chunks_dict = utils.fatiar_por_artigos(texto_bruto, titulo, url)
            
            # D. Converte para Document
            for item in chunks_dict:
                doc = Document(
                    text=item['conteudo'],
                    metadata=item['metadata'],
                    excluded_llm_metadata_keys=['url_geral', 'url_direta', 'tipo'],
                    excluded_embed_metadata_keys=['url_geral', 'url_direta']
                )
                lista_final_documentos.append(doc)
            
            # Feedback visual no terminal
            print(f"✅ Processado: {titulo} ({len(chunks_dict)} trechos)")
                
        except Exception as e:
            print(f"🔥 Erro ao processar {url}: {e}")

    # Só salva se tiver documentos novos acumulados
    if lista_final_documentos:
        run_ingestion(lista_final_documentos)
    else:
        print("🏁 Nada novo para salvar.")