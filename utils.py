import re
from typing import List
import requests
from bs4 import BeautifulSoup
import unicodedata
from llama_index.core.node_parser import SentenceSplitter
import fitz
import logging

import Prompts

# ==============================================================================
# 1. MÓDULO DE EXTRAÇÃO (Mantido igual)
# ==============================================================================
def extract_html(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.encoding = 'latin-1' 
        soup = BeautifulSoup(response.text, 'html.parser')

        for tag in soup.find_all(['strike', 's', 'del', 'script', 'style', 'head', 'footer']):
            tag.decompose()
        for a in soup.find_all('a'):
            if 'Vide' in a.get_text() or 'Redação dada' in a.get_text() or 'Vigência' in a.get_text():
                a.decompose()

        titulo_lei = "Lei Federal"
        p_titulo = soup.find('p', attrs={'align': re.compile(r'center', re.IGNORECASE)})
        if p_titulo:
            titulo_lei = p_titulo.get_text(separator=' ', strip=True)
            titulo_lei = re.sub(r'\s+', ' ', titulo_lei)

        texto = soup.get_text(separator='\n')
        texto = unicodedata.normalize("NFKD", texto)
        texto = re.sub(r'\n{3,}', '\n\n', texto)
        texto = re.sub(r'(?<!\n)\s*(Art[\.\s]\s*\d+)', r'\n\1', texto, flags=re.IGNORECASE)

        return titulo_lei, texto.strip()
    except Exception as e:
        print(f"Erro ao ler {url}: {e}")
        return "Erro", ""

# ==============================================================================
# 2. MÓDULO DE FATIAMENTO HÍBRIDO (Regex + LlamaIndex)
# ==============================================================================
def fatiar_por_artigos(texto_completo, titulo, url):
    """
    1. Usa REGEX para isolar Artigos (garante contexto jurídico).
    2. Usa LLAMAINDEX se o artigo for maior que o chunk_size (garante tokens).
    """
    
    # Configura o Splitter do LlamaIndex
    # chunk_size=1024 tokens é um bom tamanho para leis (pega contexto amplo)
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

    padrao_divisao = r'(?=\nArt[\.\s]\s*\d+)'
    pedacos = re.split(padrao_divisao, texto_completo, flags=re.IGNORECASE)
    
    chunks_processados = []
    
    # --- Preâmbulo ---
    if pedacos:
        preambulo = pedacos[0].strip()
        if preambulo:
            chunks_processados.append({
                "conteudo": preambulo,
                "metadata": {
                    "source": titulo,
                    "url_geral": url,
                    "tipo": "Preambulo",
                    "numero_artigo": "0"
                }
            })
    
    # --- Artigos ---
    for chunk in pedacos[1:]:
        chunk = chunk.strip()
        if not chunk: continue
        
        # Identifica número do artigo para Metadados e Link
        match_num = re.search(r'Art[\.\s]\s*(\d+)', chunk, re.IGNORECASE)
        num_art = match_num.group(1) if match_num else "N/A"
        
        # O split_text só vai quebrar SE o texto for maior que o chunk_size (1024 tokens)
        sub_textos = splitter.split_text(chunk)
        
        for i, sub_texto in enumerate(sub_textos):
            texto_final = sub_texto
            
            if i > 0:
                texto_final = f"[Continuação do Art. {num_art} da {titulo}] ... {sub_texto}"
            
            chunks_processados.append({
                "conteudo": texto_final,
                "metadata": {
                    "source": titulo,
                    "url_geral": url,
                    "tipo": "Artigo",
                    "numero_artigo": num_art,
                    "parte": i + 1 
                }
            })
            
    return chunks_processados

def preparar_historico_estruturado(chat_history: List[str]) -> List[dict]:
    """Transforma a lista de strings ["User: X", "AI: Y"] em [{"role": "user", "content": "X"}, ...]"""
    historico_formatado = []
    for msg in chat_history:
        if msg.startswith("User: "):
            historico_formatado.append({
                "role": "user", 
                "content": msg.replace("User: ", "").strip()
            })
        elif msg.startswith("AI: "):
            historico_formatado.append({
                "role": "assistant", 
                "content": msg.replace("AI: ", "").strip()
            })
    return historico_formatado

def montar_prompt_documento(texto: str) -> str:
    if texto and len(texto) > 10:
        return Prompts.SHARED_TEXT_DOCUMENT.format(texto_documento=texto)
    return ""
    
def preparar_resumo_router(texto: str) -> str:
    if not texto or len(texto.strip()) < 5:
        return "Nenhum documento anexado nesta interação."
    
    return f"--- INÍCIO DO DOCUMENTO ANEXADO ---\n{texto[:2000]}\n--- FIM DO TRECHO ---"

def corrigir_formatacao_markdown(texto: str) -> str:
    """
    Limpa formatação, remove blocos de código indesejados e aplica negrito inteligente.
    """
    if not texto: return ""

    # --- FASE 0: DESEMPACOTAMENTO (REMOVE A CAIXA PRETA/CINZA) ---
    # Remove tags de código markdown que a IA coloca por vício
    texto = texto.replace("```markdown", "")
    texto = texto.replace("```", "")

    # --- FASE 1: CORREÇÃO DE MOEDAS QUEBRADAS ---
    
    # Caso 1: "R 110.000,00**" (Asterisco no fim, falta no início e falta $)
    # Regex: Procura R ou R$ + espaço + numero + ** no fim
    texto = re.sub(r'(?<!\*)\bR\$?\s?([\d\.,]+)\*\*', r'**R$ \1**', texto)
    
    # Caso 2: "**R 110.000,00**" (Tem asteriscos, mas falta o $)
    texto = re.sub(r'\*\*R\s([\d\.,]+)\*\*', r'**R$ \1**', texto)

    # --- FASE 2: APLICAÇÃO AUTOMÁTICA EM VALORES "PELADOS" ---
    
    # 1. MOEDAS (R$ 1.000,00) sem negrito
    # Regex: "R$" seguido de número, que NÃO tenha asterisco antes nem depois
    padrao_moeda = r'(?<!\*)(R\$\s?[\d\.,]+)(?!\*)'
    texto = re.sub(padrao_moeda, r'**\1**', texto)

    # 2. PORCENTAGENS (10%) sem negrito
    padrao_porc = r'(?<!\*)(\b\d+[\.,]?\d*\s?%)(?!\*)'
    texto = re.sub(padrao_porc, r'**\1**', texto)

    # 3. LEIS E ARTIGOS (Lei 123, Art. 5º)
    texto = re.sub(r'(?<!\*)(Lei\sn?º?\s?[\d\./-]+)(?!\*)', r'**\1**', texto, flags=re.IGNORECASE)
    texto = re.sub(r'(?<!\*)(Art\.?|Artigo)\s(\d+[\wº°]*)(?!\*)', r'**\1 \2**', texto, flags=re.IGNORECASE)

    # --- FASE 3: ACABAMENTO PARA O STREAMLIT ---
    
    # Remove espaços duplos criados pelas substituições
    texto = re.sub(r'  +', ' ', texto)
    
    # O PULO DO GATO: Escapa o cifrão ($) para não quebrar o LaTeX do Streamlit
    # Transforma $ em \$ (mas evita duplicar se já tiver)
    texto = texto.replace("\\$", "$") # Normaliza para estado bruto
    texto = texto.replace("$", "\\$") # Escapa novamente
    
    return texto.strip()

def ler_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Recebe o arquivo em bytes, extrai o texto e retorna string.
    Trata erros e trunca se for muito grande.
    """
    if not pdf_bytes:
        return ""

    try:
        texto_extraido = ""
        # Abre o PDF direto da memória
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for pagina in doc:
                texto_extraido += pagina.get_text() + "\n"
        
        # Limite de segurança (ex: 100k caracteres)
        limite_chars = 100000 
        if len(texto_extraido) > limite_chars:
            texto_extraido = texto_extraido[:limite_chars] + "\n...[CONTEÚDO TRUNCADO]..."
            logging.warning(f"PDF truncado em {limite_chars} caracteres.")
        
        return texto_extraido

    except Exception as e:
        logging.error(f"Erro ao ler PDF (Helper): {e}")
        return "ERRO: Arquivo corrompido ou ilegível."