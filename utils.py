import re
import requests
from bs4 import BeautifulSoup
import unicodedata
from llama_index.core.node_parser import SentenceSplitter

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