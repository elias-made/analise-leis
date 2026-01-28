import re
import requests
from bs4 import BeautifulSoup
import unicodedata

# ==============================================================================
# 1. MÓDULO DE EXTRAÇÃO (BeautifulSoup)
# ==============================================================================
def extract_html(url):
    """
    Baixa o HTML, força quebras de linha e limpa o texto.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.encoding = 'latin-1' 
        
        soup = BeautifulSoup(response.text, 'html.parser')

        # Limpeza de Tags
        for tag in soup.find_all(['strike', 's', 'del', 'script', 'style', 'head', 'footer']):
            tag.decompose()
        
        for a in soup.find_all('a'):
            if 'Vide' in a.get_text() or 'Redação dada' in a.get_text() or 'Vigência' in a.get_text():
                a.decompose()

        # Título
        titulo_lei = "Lei Federal"
        p_titulo = soup.find('p', attrs={'align': re.compile(r'center', re.IGNORECASE)})
        if p_titulo:
            titulo_lei = p_titulo.get_text(separator=' ', strip=True)
            titulo_lei = re.sub(r'\s+', ' ', titulo_lei)

        # Extração do texto respeitando parágrafos
        texto = soup.get_text(separator='\n')

        # Normalização
        texto = unicodedata.normalize("NFKD", texto)
        texto = re.sub(r'\n{3,}', '\n\n', texto)
        # Garante "Art." no início da linha
        texto = re.sub(r'(?<!\n)\s*(Art[\.\s]\s*\d+)', r'\n\1', texto, flags=re.IGNORECASE)

        return titulo_lei, texto.strip()

    except Exception as e:
        print(f"Erro ao ler {url}: {e}")
        return "Erro", ""

# ==============================================================================
# 2. MÓDULO DE FATIAMENTO INTELIGENTE
# ==============================================================================
def criar_sub_chunks(texto, tamanho_max=1500, overlap=200):
    """
    Quebra um texto grande em pedaços menores com sobreposição.
    """
    if len(texto) <= tamanho_max:
        return [texto]
    
    pedacos = []
    inicio = 0
    while inicio < len(texto):
        fim = inicio + tamanho_max
        
        # Tenta não cortar no meio da palavra
        if fim < len(texto):
            proximo_espaco = texto.find(' ', fim)
            if proximo_espaco != -1 and proximo_espaco - fim < 100:
                fim = proximo_espaco
        
        chunk = texto[inicio:fim]
        pedacos.append(chunk)
        
        # Avança menos que o fim para criar o overlap (contexto compartilhado)
        inicio = fim - overlap
        
    return pedacos

def fatiar_por_artigos(texto_completo, titulo, url):
    """
    Fatia por Artigo e depois sub-fatia se o artigo for gigante.
    """
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
                    "url_direta": url,
                    "tipo": "Preambulo",
                    "numero_artigo": "0"
                }
            })
    
    # --- Artigos ---
    for chunk in pedacos[1:]:
        chunk = chunk.strip()
        if not chunk: continue
        
        # Identifica número do artigo
        match_num = re.search(r'Art[\.\s]\s*(\d+)', chunk, re.IGNORECASE)
        num_art = match_num.group(1) if match_num else "N/A"
        link_ancora = f"{url}#art{num_art}" if url and num_art != "N/A" else url
        
        # --- AQUI ESTÁ A CORREÇÃO: SUB-CHUNKING ---
        # Se o artigo for gigante (ex: Art 3 da LC123), quebra ele em partes
        sub_textos = criar_sub_chunks(chunk, tamanho_max=2000, overlap=300)
        
        for i, sub_texto in enumerate(sub_textos):
            # Adiciona um cabeçalho para a IA saber do que se trata se for uma "Parte 2"
            texto_final = sub_texto
            if i > 0:
                texto_final = f"[Continuação do Art. {num_art} da {titulo}] ... {sub_texto}"
            
            chunks_processados.append({
                "conteudo": texto_final,
                "metadata": {
                    "source": titulo,
                    "url_geral": url,
                    "url_direta": link_ancora,
                    "tipo": "Artigo",
                    "numero_artigo": num_art,
                    "parte": i + 1 # Metadado extra para controle
                }
            })
            
    return chunks_processados