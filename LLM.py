# LLM.py
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding

# ==============================================================================
# CLAUDE SONNET 4.5
# ==============================================================================
llm_sonnet = Bedrock(
    # ID do Perfil (Com 'us.')
    model='us.anthropic.claude-sonnet-4-5-20250929-v1:0', 
    
    # O PULO DO GATO: Use 'us-east-1' aqui.
    # É obrigatório para perfis que começam com "us."
    region_name='us-east-1', 
    
    temperature=0,
    context_size=200000
)

# ==============================================================================
# CLAUDE HAIKU 4.5
# ==============================================================================
llm_haiku = Bedrock(
    model='us.anthropic.claude-haiku-4-5-20251001-v1:0',
    # Mesma coisa aqui
    region_name='us-east-1',
    temperature=0,
    context_size=200000
)

# ==============================================================================
# EMBEDDINGS (O Titan geralmente aceita qualquer um, mas mantenha padrão)
# ==============================================================================
embed_model = BedrockEmbedding(
    model='amazon.titan-embed-text-v2:0',
    region_name='us-east-1' 
)