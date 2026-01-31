import os

from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding
from pydantic_ai.models.bedrock import BedrockConverseModel

# ==============================================================================
# 1. MODELO "CÉREBRO" (Para Agentes / PydanticAI)
# ==============================================================================
sonnet_bedrock_model = BedrockConverseModel(
    'us.anthropic.claude-sonnet-4-5-20250929-v1:0'
)

# ==============================================================================
# 2. MODELO "LEITOR" (Para RAG / LlamaIndex / App.py)
# ==============================================================================
llm_sonnet = Bedrock(
    model='us.anthropic.claude-3-5-haiku-20241022-v1:0',
    temperature=0,
    context_size=200000
)

# ==============================================================================
# 3. EMBEDDINGS
# ==============================================================================
embed_model = BedrockEmbedding(
    model='amazon.titan-embed-text-v2:0',
)