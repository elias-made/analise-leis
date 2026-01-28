from llama_index.core import Document, Settings

import LLM
import utils
import ingestion

def main():
    # ==========================================================================
    # PASSO 1: CONFIGURAÇÃO GLOBAL
    # ==========================================================================
    # Define o modelo de Embedding que será usado para vetorizar
    # O LlamaIndex usará isso automaticamente dentro do ingestion.py
    Settings.embed_model = LLM.embed_model
    
    # ==========================================================================
    # PASSO 2: DEFINIR FONTES
    # ==========================================================================
    urls_para_ler = [
        "http://www.planalto.gov.br/ccivil_03/leis/lcp/lcp123.htm",
        "https://www.planalto.gov.br/ccivil_03/decreto-lei/del5452.htm"
    ]
    
    lista_final_documentos = []

    print(f"🚀 Iniciando processamento de {len(urls_para_ler)} leis...")

    # ==========================================================================
    # PASSO 3: LOOP DE PROCESSAMENTO (Extração -> Fatiamento -> Conversão)
    # ==========================================================================
    

    # ==========================================================================
    # PASSO 4: INGESTÃO (Salvar no Qdrant)
    # ==========================================================================
    print(f"\n💾 Iniciando gravação de {len(lista_final_documentos)} vetores no Qdrant...")
    
    # Chama a função do arquivo ingestion.py
    ingestion.run_ingestion(lista_final_documentos)
    
    print("\n✅ Processo Finalizado! Os dados estão indexados.")

if __name__ == "__main__":
    main()