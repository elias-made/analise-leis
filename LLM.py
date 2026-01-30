# LLM.py
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding

# ==============================================================================
# CLAUDE SONNET 4.5
# ==============================================================================
llm_sonnet = Bedrock(
    model='us.anthropic.claude-sonnet-4-5-20250929-v1:0', 
    region_name='us-east-1', 
    temperature=0,
    context_size=200000
)

# ==============================================================================
# EMBEDDINGS
# ==============================================================================
embed_model = BedrockEmbedding(
    model='amazon.titan-embed-text-v2:0',
    region_name='us-east-1' 
)