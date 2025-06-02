# a2a_wrapper/main.py
import inspect
import uvicorn
from fastapi import FastAPI, Request as FastAPIRequest, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
from datetime import datetime, timezone
from typing import Union, Dict, Any, List, Optional
import logging
import os
import sys
from dotenv import load_dotenv
from types import SimpleNamespace
import httpx  # For OpenAI RT session endpoint
from pydantic import BaseModel

# For LLM calls within pillar/reflector functions
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest

# --- Add project root (Aura2) to sys.path ---
A2A_WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_FOR_A2A = os.path.dirname(A2A_WRAPPER_DIR)
if PROJECT_ROOT_FOR_A2A not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_FOR_A2A)

# --- Load .env file ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This is Aura2
dotenv_path = os.path.join(PROJECT_ROOT, '.env')  # Expect .env in Aura2 directory

# --- Logging Setup for a2a_wrapper/main.py ---
# This should be the primary logging config for the application.
logging.basicConfig(level=os.getenv("LOG_LEVEL_A2A", "INFO").upper(),
                    format='%(asctime)s - A2A_MAIN - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if os.path.exists(dotenv_path):
    logger.info(f"A2A Wrapper: Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    logger.warning(f"A2A Wrapper: .env file not found at {dotenv_path}. Relying on environment variables.")

# --- Module Imports from orchestrator_adk_agent ---
try:
    from orchestrator_adk_agent import (
        adk_runner,  # The Runner instance
        orchestrator_adk_agent_aura,  # The LlmAgent instance
        ADK_APP_NAME,  # ADK App Name string
        memory_blossom_instance,  # Shared MemoryBlossom
        reflector_add_memory,  # The specific function for reflector
        AGENT_MODEL_STRING,  # Model string for helper_llm
        adk_session_service  # Shared InMemorySessionService
    )

    logger.info("Successfully imported components from orchestrator_adk_agent.")
except ImportError as e:
    logger.critical(f"Failed to import from orchestrator_adk_agent.py: {e}. Ensure it's corrected. Exiting.",
                    exc_info=True)
    sys.exit(1)  # Exit if core components can't be loaded

from memory_system.memory_blossom import MemoryBlossom
from memory_system.memory_models import Memory as MemoryModel

from a2a_wrapper.models import (
    A2APart, A2AMessage, A2ATaskSendParams, A2AArtifact,
    A2ATaskStatus, A2ATaskResult, A2AJsonRpcRequest, A2AJsonRpcResponse,
    AgentCard, AgentCardSkill, AgentCardProvider, AgentCardAuthentication, AgentCardCapabilities
)

from google.genai.types import Content as ADKContent, Part as ADKPart
from google.adk.sessions import Session as ADKSession

# --- Configuration & FastAPI App ---
A2A_WRAPPER_HOST = os.getenv("A2A_WRAPPER_HOST", "0.0.0.0")
A2A_WRAPPER_PORT = int(os.getenv("A2A_WRAPPER_PORT", "8098"))  # Client targets this port
A2A_WRAPPER_BASE_URL = os.getenv("A2A_WRAPPER_BASE_URL", f"http://localhost:{A2A_WRAPPER_PORT}")

app = FastAPI(
    title="Aura Agent A2A & Voice API Wrapper",
    description="Exposes Aura (NCF) ADK agent via A2A and provides voice client API endpoints."
)

# --- CORS Configuration ---
origins = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
logger.info(f"CORS allowed origins: {origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Agent Card ---
AGENT_CARD_DATA = AgentCard(
    name="Aura2",
    description="A conversational AI agent, Aura, with advanced memory capabilities using "
                "Narrative Context Framing (NCF). Aura aims to build a deep, "
                "contextual understanding over long interactions.",
    url=A2A_WRAPPER_BASE_URL,  # This should be the public URL if ngrok or similar is used
    version="1.2.1-unified",  # Updated version
    provider=AgentCardProvider(organization="LocalDev", url=os.environ.get("OR_SITE_URL", "http://example.com")),
    capabilities=AgentCardCapabilities(streaming=False, pushNotifications=False),
    authentication=AgentCardAuthentication(schemes=[]),
    skills=[
        AgentCardSkill(
            id="narrative_conversation",
            name="Narrative Conversation with Aura",
            description="Engage in a deep, contextual conversation. Aura uses its "
                        "MemoryBlossom system and Narrative Context Framing to understand "
                        "and build upon previous interactions.",
            tags=["chat", "conversation", "memory", "ncf", "context", "aura", "multi-agent-concept"],
            examples=[
                "Let's continue our discussion about emergent narrative structures.",
                "Based on what we talked about regarding liminal spaces, what do you think about this new idea?",
                "Store this feeling: 'The breakthrough in my research felt incredibly liberating!'",
                "How does our previous conversation on AI ethics relate to this new scenario?"
            ],
            parameters={
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "The textual input from the user for the conversation."
                    },
                    "a2a_task_id_override": {
                        "type": "string",
                        "description": "Optional: Override the A2A task ID for session mapping.",
                        "nullable": True
                    }
                },
                "required": ["user_input"]
            }
        )
    ]
)


@app.get("/.well-known/agent.json", response_model=AgentCard, response_model_exclude_none=True)
async def get_agent_card():
    return AGENT_CARD_DATA


# --- Pydantic Models for Voice Client API ---
class ChatRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    aura_reply: str
    session_id: str
    new_session_created: bool = False
    error: Optional[str] = None


class OpenAIRTSessionResponse(BaseModel):
    ephemeral_key: Optional[str] = None
    error: Optional[str] = None
    openai_session_id: Optional[str] = None
    expires_at: Optional[int] = None


# --- In-memory stores ---
active_aura_chat_sessions: Dict[str, str] = {}  # For /aura/chat endpoint sessions
a2a_task_to_adk_session_map: Dict[str, str] = {}  # For A2A task to ADK session mapping

# --- Helper LLM (for NCF pillars if needed outside ADK agent's main flow) ---
helper_llm = LiteLlm(model=AGENT_MODEL_STRING)


# --- Aura NCF Pillar Functions (Defined in this file) ---
async def get_narrativa_de_fundamento_pilar1(
        current_session_state: Dict[str, Any],
        mb_instance: MemoryBlossom,
        user_id: str
) -> str:
    logger.info(f"[Pilar 1] Generating Narrative Foundation for user {user_id}...")
    if 'foundation_narrative' in current_session_state and current_session_state.get('foundation_narrative_turn_count',
                                                                                     0) < 5:
        current_session_state['foundation_narrative_turn_count'] = current_session_state.get(
            'foundation_narrative_turn_count', 0) + 1
        logger.info(
            f"[Pilar 1] Using cached Narrative Foundation. Turn count: {current_session_state['foundation_narrative_turn_count']}")
        return current_session_state['foundation_narrative']

    relevant_memories_for_foundation: List[MemoryModel] = []
    try:
        explicit_mems = mb_instance.retrieve_memories(
            query="key explicit facts and statements from our past discussions", top_k=2,
            target_memory_types=["Explicit"], apply_criticality=False)
        emotional_mems = mb_instance.retrieve_memories(query="significant emotional moments or sentiments expressed",
                                                       top_k=1, target_memory_types=["Emotional"],
                                                       apply_criticality=False)
        relevant_memories_for_foundation.extend(explicit_mems)
        relevant_memories_for_foundation.extend(emotional_mems)
        seen_ids = set()
        unique_memories = []
        for mem in relevant_memories_for_foundation:
            if mem.id not in seen_ids:
                unique_memories.append(mem)
                seen_ids.add(mem.id)
        relevant_memories_for_foundation = unique_memories
    except Exception as e:
        logger.error(f"[Pilar 1] Error retrieving memories for foundation: {e}", exc_info=True)
        return "Estamos construindo nossa jornada de entendimento mútuo."

    if not relevant_memories_for_foundation:
        narrative = "Nossa jornada de aprendizado e descoberta está apenas começando. Estou ansiosa para explorar vários tópicos interessantes com você."
    else:
        memory_contents = [f"- ({mem.memory_type}): {mem.content}" for mem in relevant_memories_for_foundation]
        memories_str = "\n".join(memory_contents)
        synthesis_prompt = f"""
        Você é um sintetizador de narrativas. Com base nas seguintes memórias chave de interações passadas, crie uma breve narrativa de fundamento (1-2 frases concisas) que capture a essência da nossa jornada de entendimento e os principais temas discutidos. Esta narrativa servirá como pano de fundo para nossa conversa atual.

        Memórias Chave:
        {memories_str}

        Narrativa de Fundamento Sintetizada:
        """
        try:
            logger.info(
                f"[Pilar 1] Calling LLM to synthesize Narrative Foundation from {len(relevant_memories_for_foundation)} memories.")
            request_messages = [ADKContent(parts=[ADKPart(text=synthesis_prompt)])]
            # +++ Use SimpleNamespace for config +++
            # This creates an object for `req.config` with a `tools` attribute that is an empty list.
            minimal_config = SimpleNamespace(tools=[])
            llm_req = LlmRequest(contents=request_messages, config=minimal_config)

            final_text_response = ""
            async for llm_response_event in helper_llm.generate_content_async(
                    llm_req):
                if llm_response_event and llm_response_event.content and \
                        llm_response_event.content.parts and llm_response_event.content.parts[0].text:
                    final_text_response += llm_response_event.content.parts[0].text

            narrative = final_text_response.strip() if final_text_response else "Continuamos a construir nossa compreensão mútua com base em nossas interações anteriores."
        except Exception as e:
            logger.error(f"[Pilar 1] LLM error synthesizing Narrative Foundation: {e}", exc_info=True)
            narrative = "Refletindo sobre nossas conversas anteriores para guiar nosso diálogo atual."

    current_session_state['foundation_narrative'] = narrative
    current_session_state['foundation_narrative_turn_count'] = 1
    logger.info(f"[Pilar 1] Generated new Narrative Foundation: '{narrative[:100]}...'")
    return narrative


async def get_rag_info_pilar2(
        user_utterance: str,
        mb_instance: MemoryBlossom,
        current_session_state: Dict[str, Any]
) -> List[Dict[str, Any]]:
    logger.info(f"[Pilar 2] Retrieving RAG info for utterance: '{user_utterance[:50]}...'")
    try:
        conversation_context = current_session_state.get('conversation_history', [])[-5:]
        recalled_memories_for_rag = mb_instance.retrieve_memories(
            query=user_utterance, top_k=3, conversation_context=conversation_context
        )
        rag_results = [mem.to_dict() for mem in recalled_memories_for_rag]
        logger.info(f"[Pilar 2] Retrieved {len(rag_results)} memories for RAG.")
        return rag_results
    except Exception as e:
        logger.error(f"[Pilar 2] Error in get_rag_info: {e}", exc_info=True)
        return [{"content": f"Erro ao buscar informações RAG específicas: {str(e)}", "memory_type": "Error"}]


def format_chat_history_pilar3(chat_history_list: List[Dict[str, str]], max_turns: int = 15) -> str:
    if not chat_history_list: return "Nenhum histórico de conversa recente disponível."
    recent_history = chat_history_list[-max_turns:]
    formatted_history = [f"{'Usuário' if entry.get('role') == 'user' else 'Aura'}: {entry.get('content')}" for entry in
                         recent_history]
    return "\n".join(
        formatted_history) if formatted_history else "Nenhum histórico de conversa recente disponível para formatar."


def montar_prompt_aura_ncf(
        persona_agente: str, persona_detalhada: str, narrativa_fundamento: str,
        informacoes_rag_list: List[Dict[str, Any]], chat_history_recente_str: str, user_reply: str
) -> str:
    logger.info("[PromptBuilder] Assembling NCF prompt...")
    formatted_rag = ""
    if informacoes_rag_list:
        rag_items_str = [
            f"  - ({item_dict.get('memory_type', 'Info')}): {item_dict.get('content', 'Conteúdo indisponível')}"
            for item_dict in informacoes_rag_list
        ]
        formatted_rag = "Informações e memórias específicas que podem ser úteis para esta interação (RAG):\n" + "\n".join(
            rag_items_str) if rag_items_str \
            else "Nenhuma informação específica (RAG) foi recuperada para esta consulta."
    else:
        formatted_rag = "Nenhuma informação específica (RAG) foi recuperada para esta consulta."

    task_instruction = """## Sua Tarefa:
Responda ao usuário de forma natural, coerente e útil, levando em consideração TODA a narrativa de contexto e o histórico fornecido.
- Incorpore ativamente elementos da "Narrativa de Fundamento" para mostrar continuidade e entendimento profundo.
- Utilize as "Informações RAG" para embasar respostas específicas ou fornecer detalhes relevantes.
- Mantenha a persona definida.
- Se identificar uma aparente contradição entre a "Narrativa de Fundamento", as "Informações RAG" ou o "Histórico Recente", tente abordá-la com humildade epistêmica:
    - Priorize a informação mais recente ou específica, se aplicável.
    - Considere se é uma evolução do entendimento ou um novo aspecto.
    - Se necessário, você pode mencionar sutilmente a aparente diferença ou pedir clarificação ao usuário de forma implícita através da sua resposta. Não afirme categoricamente que há uma contradição, mas navegue a informação com nuance.
- Evite redundância. Se o histórico recente já cobre um ponto, não o repita extensivamente a menos que seja para reforçar uma conexão crucial com a nova informação.
"""
    prompt = f"""<SYSTEM_PERSONA_START>
Você é {persona_agente}.
{persona_detalhada}
<SYSTEM_PERSONA_END>

<NARRATIVE_FOUNDATION_START>
## Nosso Entendimento e Jornada Até Agora (Narrativa de Fundamento):
{narrativa_fundamento if narrativa_fundamento else "Ainda não construímos uma narrativa de fundamento detalhada para nossa interação."}
<NARRATIVE_FOUNDATION_END>

<SPECIFIC_CONTEXT_RAG_START>
## Informações Relevantes para a Conversa Atual (RAG):
{formatted_rag}
<SPECIFIC_CONTEXT_RAG_END>

<RECENT_HISTORY_START>
## Histórico Recente da Nossa Conversa:
{chat_history_recente_str if chat_history_recente_str else "Não há histórico recente disponível para esta conversa."}
<RECENT_HISTORY_END>

<CURRENT_SITUATION_START>
## Situação Atual:
Você está conversando com o usuário. O usuário acabou de dizer:

Usuário: "{user_reply}"

{task_instruction}
<CURRENT_SITUATION_END>

Aura:"""
    logger.info(f"[PromptBuilder] NCF Prompt assembled. Length: {len(prompt)}")
    return prompt


async def aura_reflector_analisar_interacao(
        user_utterance: str, prompt_ncf_usado: str, resposta_de_aura: str,
        mb_instance: MemoryBlossom, user_id: str
):
    logger.info(f"[Reflector] Analisando interação para user {user_id}...")
    logger.debug(
        f"[Reflector] Interaction Log for {user_id}:\nUser Utterance: {user_utterance}\nNCF Prompt (first 500): {prompt_ncf_usado[:500]}...\nAura's Resp: {resposta_de_aura}")

    reflector_prompt = f"""
    Você é um analista de conversas de IA chamado "Reflector". Sua tarefa é analisar a seguinte interação entre um usuário e a IA Aura para identificar se alguma informação crucial deve ser armazenada na memória de longo prazo de Aura (MemoryBlossom).
    Contexto da IA Aura: Aura usa uma Narrativa de Fundamento (resumo de longo prazo), RAG (informações específicas para a query) e Histórico Recente para responder.
    O prompt completo que Aura recebeu já continha muito desse contexto.
    Agora, avalie a *nova* informação trocada (pergunta do usuário e resposta de Aura) e decida se algo dessa *nova troca* merece ser uma memória distinta.
    Critérios para decidir armazenar uma memória:
    1.  **Fatos explícitos importantes declarados pelo usuário ou pela Aura** que provavelmente serão relevantes no futuro (ex: preferências do usuário, decisões chave, novas informações factuais significativas que Aura aprendeu ou ensinou).
    2.  **Momentos emocionais significativos** expressos pelo usuário ou refletidos por Aura que indicam um ponto importante na interação.
    3.  **Insights ou conclusões chave** alcançados durante a conversa.
    4.  **Correções importantes feitas pelo usuário e aceitas por Aura.**
    5.  **Tarefas ou objetivos de longo prazo** mencionados.
    NÃO armazene:
    - Conversa trivial, saudações, despedidas (a menos que contenham emoção significativa).
    - Informação que já está claramente coberta pela Narrativa de Fundamento ou RAG que foi fornecida a Aura (a menos que a interação atual adicione um novo significado ou conexão a ela).
    - Perguntas do usuário, a menos que a pergunta em si revele uma nova intenção de longo prazo ou um fato sobre o usuário.
    Interação para Análise:
    Usuário disse: "{user_utterance}"
    Aura respondeu: "{resposta_de_aura}"
    Com base na sua análise, se você acha que uma ou mais memórias devem ser criadas, forneça a resposta no seguinte formato JSON. Se múltiplas memórias, forneça uma lista de objetos JSON. Se nada deve ser armazenado, retorne um JSON vazio `{{}}` ou uma lista vazia `[]`.
    Formato JSON para cada memória a ser criada:
    {{
      "content": "O conteúdo textual da memória a ser armazenada. Seja conciso mas completo.",
      "memory_type": "Escolha um de: Explicit, Emotional, Procedural, Flashbulb, Liminal, Generative",
      "emotion_score": 0.0-1.0 (se memory_type for Emotional, senão pode ser 0.0),
      "initial_salience": 0.0-1.0 (quão importante parece ser esta memória? 0.5 é neutro, 0.8 é importante),
      "metadata": {{ "source": "aura_reflector_analysis", "user_id": "{user_id}", "related_interaction_turn": "current" }}
    }}
    Sua decisão (JSON):
    """
    try:
        logger.info(f"[Reflector] Chamando LLM para decisão de armazenamento de memória.")
        request_messages = [ADKContent(parts=[ADKPart(text=reflector_prompt)])]
        # +++ Use SimpleNamespace for config +++
        minimal_config = SimpleNamespace(tools=[])
        llm_req = LlmRequest(contents=request_messages, config=minimal_config)

        final_text_response = ""
        async for llm_response_event in helper_llm.generate_content_async(llm_req):
            if llm_response_event and llm_response_event.content and \
                    llm_response_event.content.parts and llm_response_event.content.parts[0].text:
                final_text_response += llm_response_event.content.parts[0].text

        if not final_text_response:
            logger.info("[Reflector] Nenhuma decisão de armazenamento de memória retornada pelo LLM.")
            return

        decision_json_str = final_text_response.strip()
        logger.info(f"[Reflector] Decisão de armazenamento (JSON string): {decision_json_str}")

        if '```json' in decision_json_str:
            decision_json_str = decision_json_str.split('```json')[1].split('```')[0].strip()
        elif not (decision_json_str.startswith('{') and decision_json_str.endswith('}')) and \
                not (decision_json_str.startswith('[') and decision_json_str.endswith(']')):
            match_obj, match_list = None, None
            try:
                obj_start = decision_json_str.index('{');
                obj_end = decision_json_str.rindex('}') + 1
                match_obj = decision_json_str[obj_start:obj_end]
            except ValueError:
                pass
            try:
                list_start = decision_json_str.index('[');
                list_end = decision_json_str.rindex(']') + 1
                match_list = decision_json_str[list_start:list_end]
            except ValueError:
                pass
            if match_obj and (not match_list or len(match_obj) > len(match_list)):
                decision_json_str = match_obj
            elif match_list:
                decision_json_str = match_list
            else:
                logger.warning(f"[Reflector] LLM response not valid JSON after cleaning: {decision_json_str}")
                return

        memories_to_add = []
        try:
            parsed_decision = json.loads(decision_json_str)
            if isinstance(parsed_decision, dict) and "content" in parsed_decision and "memory_type" in parsed_decision:
                memories_to_add.append(parsed_decision)
            elif isinstance(parsed_decision, list):
                memories_to_add = [item for item in parsed_decision if
                                   isinstance(item, dict) and "content" in item and "memory_type" in item]
            elif parsed_decision:  # Non-empty but not valid memory structure
                logger.info(f"[Reflector] Parsed decision is not a valid memory structure: {parsed_decision}")

        except json.JSONDecodeError as e:
            logger.error(f"[Reflector] JSONDecodeError for Reflector decision: {e}. String: {decision_json_str}")
            return

        for mem_data in memories_to_add:
            logger.info(
                f"[Reflector] Adding memory: Type='{mem_data['memory_type']}', Content='{mem_data['content'][:50]}...'")
            reflector_add_memory(
                content=mem_data["content"], memory_type=mem_data["memory_type"],
                emotion_score=float(mem_data.get("emotion_score", 0.0)),
                initial_salience=float(mem_data.get("initial_salience", 0.5)),
                metadata=mem_data.get("metadata", {"source": "aura_reflector_analysis", "user_id": user_id})
            )
    except Exception as e:
        logger.error(f"[Reflector] Error during interaction analysis: {e}", exc_info=True)


# --- Helper: ADK Conversation Turn for Voice Client's /aura/chat ---
async def run_adk_conversation_for_voice_client(user_id: str, session_id: str, user_utterance_raw: str) -> str:
    logger.info(f"VoiceClient Chat: User='{user_id}', Session='{session_id}', Utterance='{user_utterance_raw[:30]}...'")

    current_adk_session: Optional[ADKSession] = adk_session_service.get_session(  # SYNC
        app_name=ADK_APP_NAME, user_id=user_id, session_id=session_id
    )
    if not current_adk_session:
        logger.info(f"  VoiceClient Chat: Creating new ADK session: {session_id} for user: {user_id}")
        current_adk_session = adk_session_service.create_session(  # SYNC
            app_name=ADK_APP_NAME, user_id=user_id, session_id=session_id,
            state={'conversation_history': [], 'foundation_narrative_turn_count': 0, 'foundation_narrative': None}
        )
    if not isinstance(current_adk_session.state, dict):
        current_adk_session.state = {'conversation_history': [], 'foundation_narrative_turn_count': 0}

    current_session_state = current_adk_session.state
    current_session_state.setdefault('conversation_history', []).append({"role": "user", "content": user_utterance_raw})

    narrativa_fundamento = await get_narrativa_de_fundamento_pilar1(current_session_state, memory_blossom_instance,
                                                                    user_id)
    rag_info_list = await get_rag_info_pilar2(user_utterance_raw, memory_blossom_instance, current_session_state)
    chat_history_for_prompt_str = format_chat_history_pilar3(current_session_state['conversation_history'])

    final_ncf_prompt_str = montar_prompt_aura_ncf(
        "Aura (Voice)", AGENT_CARD_DATA.skills[0].description if AGENT_CARD_DATA.skills else "Voice assistant persona.",
        narrativa_fundamento, rag_info_list, chat_history_for_prompt_str, user_utterance_raw
    )
    adk_input_content = ADKContent(role="user", parts=[ADKPart(text=final_ncf_prompt_str)])
    adk_agent_final_text_response = None
    adk_session_service.update_session(current_adk_session)  # SYNC

    async for event in adk_runner.run_async(user_id=user_id, session_id=session_id, new_message=adk_input_content):
        if event.author == orchestrator_adk_agent_aura.name:
            if event.is_final_response() and event.content and event.content.parts and event.content.parts[0].text:
                adk_agent_final_text_response = event.content.parts[0].text.strip()
                break

    adk_agent_final_text_response = adk_agent_final_text_response or "(Aura did not provide a textual response)"
    current_session_state['conversation_history'].append(
        {"role": "assistant", "content": adk_agent_final_text_response})
    adk_session_service.update_session(current_adk_session)  # SYNC

    # Optionally run reflector for voice client interactions too
    await aura_reflector_analisar_interacao(user_utterance_raw, final_ncf_prompt_str, adk_agent_final_text_response,
                                            memory_blossom_instance, user_id)
    return adk_agent_final_text_response


# --- API Endpoints ---
@app.get("/health", status_code=200)
async def health_check_endpoint():
    logger.info("Health check endpoint was called successfully.")
    return {"status": "Aura A2A & Voice API is healthy and running"}


@app.post("/openai/rt/session", response_model=OpenAIRTSessionResponse)
async def get_openai_realtime_session_for_client_endpoint():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key or len(openai_api_key) < 20:  # Basic check
        logger.error(
            "CRITICAL: OPENAI_API_KEY is not configured correctly or is too short in .env for Realtime session.")
        raise HTTPException(status_code=500,
                            detail="OpenAI API Key for Realtime session not configured correctly on server.")

    openai_session_url = "https://api.openai.com/v1/realtime/sessions"
    payload = {"model": "gpt-4o-mini-realtime-preview", "voice": "verse"}  # Ensure model and voice are correct
    headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
    logger.debug(f"Requesting OpenAI RT session from {openai_session_url} with payload: {payload}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(openai_session_url, json=payload, headers=headers)

        logger.debug(
            f"OpenAI RT session response status: {response.status_code}, Response text: {response.text[:500]}...")  # Log part of response text
        if response.status_code != 200:
            detail_msg = response.text
            try:
                detail_msg = response.json().get("error", {}).get("message", response.text)
            except json.JSONDecodeError:
                pass
            logger.error(f"OpenAI API error ({response.status_code}) for Realtime session: {detail_msg}")
            raise HTTPException(status_code=response.status_code, detail=f"OpenAI API Error: {detail_msg}")

        data = response.json()
        client_secret_data = data.get("client_secret")
        if client_secret_data and isinstance(client_secret_data, dict) and client_secret_data.get("value"):
            logger.info(f"Successfully obtained OpenAI Realtime ephemeral key. Session ID: {data.get('id')}")
            return OpenAIRTSessionResponse(ephemeral_key=client_secret_data["value"], openai_session_id=data.get("id"),
                                           expires_at=data.get("expires_at"))
        else:
            logger.error(f"OpenAI RT session response missing client_secret.value. Full Response: {data}")
            raise HTTPException(status_code=500,
                                detail="Invalid response structure from OpenAI RT session API (missing client_secret.value).")
    except httpx.RequestError as e:
        logger.error(f"HTTPX RequestError when trying to contact OpenAI for Realtime session: {str(e)}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Network error when contacting OpenAI: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in /openai/rt/session: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error creating OpenAI RT session: {str(e)}")


@app.post("/aura/chat", response_model=ChatResponse)
async def handle_chat_with_aura_via_voice_client_endpoint(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message
    provided_session_id = request.session_id
    new_session_created_flag = False

    logger.info(f"API /aura/chat: User='{user_id}', Session='{provided_session_id}', Msg='{user_message[:30]}...'")
    adk_session_id_to_use: Optional[str] = provided_session_id
    if not adk_session_id_to_use:
        if user_id in active_aura_chat_sessions:
            adk_session_id_to_use = active_aura_chat_sessions[user_id]
        else:
            adk_session_id_to_use = f"voice_chat_session_{user_id}_{str(uuid.uuid4())[:6]}"
            active_aura_chat_sessions[user_id] = adk_session_id_to_use
            new_session_created_flag = True
            logger.info(f"  Generated new /aura/chat session_id: {adk_session_id_to_use}")
    try:
        aura_response_text = await run_adk_conversation_for_voice_client(
            user_id=user_id, session_id=adk_session_id_to_use, user_utterance_raw=user_message  # type: ignore
        )
        return ChatResponse(
            aura_reply=aura_response_text, session_id=adk_session_id_to_use,
            new_session_created=new_session_created_flag  # type: ignore
        )
    except Exception as e:
        logger.error(f"Error in /aura/chat handling: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error in /aura/chat: {str(e)}")


@app.post("/", response_model=A2AJsonRpcResponse, response_model_exclude_none=True)
async def handle_a2a_rpc(rpc_request: A2AJsonRpcRequest, http_request: FastAPIRequest):
    client_host = http_request.client.host if http_request.client else "unknown"
    logger.info(f"A2A RPC: Method={rpc_request.method}, ID={rpc_request.id} from {client_host}")

    if rpc_request.method == "tasks/send":
        if rpc_request.params is None:
            logger.error("A2A RPC tasks/send: Missing 'params' in request.")
            return A2AJsonRpcResponse(id=rpc_request.id,
                                      error={"code": -32602, "message": "Invalid params: 'params' field is missing."})

        try:
            task_params = rpc_request.params
            logger.info(
                f"A2A RPC: Processing tasks/send for A2A Task ID: {task_params.id}, SessionID: {task_params.sessionId}")

            user_utterance_raw = ""
            if task_params.message and task_params.message.parts:
                first_part = task_params.message.parts[0]
                if first_part.type == "data" and first_part.data and "user_input" in first_part.data:
                    user_utterance_raw = first_part.data["user_input"]
                elif first_part.type == "text" and first_part.text:
                    user_utterance_raw = first_part.text

            if not user_utterance_raw:
                logger.error("A2A RPC tasks/send: 'user_input' missing in message parts.")
                return A2AJsonRpcResponse(id=rpc_request.id, error={"code": -32602,
                                                                    "message": "Invalid params: 'user_input' is missing in message parts."})

            logger.info(f"A2A RPC: Extracted user_utterance_raw: '{user_utterance_raw[:70]}...'")

            adk_session_map_key = task_params.sessionId or task_params.id
            # Check for a2a_task_id_override
            if hasattr(task_params.message.parts[0], 'data') and \
                    task_params.message.parts[0].data and \
                    task_params.message.parts[0].data.get("a2a_task_id_override"):
                override_key = task_params.message.parts[0].data["a2a_task_id_override"]
                if override_key:  # Ensure override is not empty
                    adk_session_map_key = override_key
                    logger.info(f"A2A RPC: Using overridden A2A Task ID for ADK session mapping: {adk_session_map_key}")

            adk_user_id = f"a2a_user_for_{adk_session_map_key}"
            adk_session_id_from_map = a2a_task_to_adk_session_map.get(adk_session_map_key)

            current_adk_session: Optional[ADKSession] = None
            actual_adk_session_id_to_use = adk_session_id_from_map  # Start with this, might change if new session is created

            if adk_session_id_from_map:
                logger.info(
                    f"A2A RPC: Attempting to retrieve ADK session: ID='{adk_session_id_from_map}' for User='{adk_user_id}'")
                current_adk_session = await adk_session_service.get_session(
                    app_name=ADK_APP_NAME, user_id=adk_user_id, session_id=adk_session_id_from_map
                )
                if current_adk_session:
                    logger.info(f"A2A RPC: ADK session '{adk_session_id_from_map}' retrieved.")
                else:
                    logger.warning(
                        f"A2A RPC: ADK session '{adk_session_id_from_map}' not found for user '{adk_user_id}'. Will create a new one.")

            if not current_adk_session:
                new_generated_session_id = f"adk_a2a_sess_{adk_session_map_key}_{str(uuid.uuid4())[:8]}"  # Longer UUID part
                a2a_task_to_adk_session_map[adk_session_map_key] = new_generated_session_id
                actual_adk_session_id_to_use = new_generated_session_id  # This is the ID to use from now on

                logger.info(
                    f"A2A RPC: Creating new ADK session: ID='{actual_adk_session_id_to_use}' for User='{adk_user_id}'")
                current_adk_session = await adk_session_service.create_session(
                    app_name=ADK_APP_NAME, user_id=adk_user_id, session_id=actual_adk_session_id_to_use,
                    state={'conversation_history': [], 'foundation_narrative_turn_count': 0,
                           'foundation_narrative': None}
                )
                logger.info(f"A2A RPC: New ADK session '{actual_adk_session_id_to_use}' created.")

            if current_adk_session is None:  # Should ideally not happen if create_session is robust
                logger.error("A2A RPC: CRITICAL - current_adk_session is None after get/create attempts.")
                return A2AJsonRpcResponse(id=rpc_request.id, error={"code": -32000,
                                                                    "message": "Internal Server Error: Failed to obtain ADK session."})

            # Defensive check and initialization of session state
            if not hasattr(current_adk_session, 'state') or not isinstance(current_adk_session.state, dict):
                logger.warning(
                    f"A2A RPC: ADK session state for '{actual_adk_session_id_to_use}' is invalid or missing. Reinitializing state. Type was: {type(current_adk_session.state)}")
                current_adk_session.state = {'conversation_history': [], 'foundation_narrative_turn_count': 0,
                                             'foundation_narrative': None}

            current_adk_session.state.setdefault('conversation_history', [])
            current_adk_session.state.setdefault('foundation_narrative_turn_count', 0)
            # 'foundation_narrative' is typically set/updated by get_narrativa_de_fundamento_pilar1

            # Add current user utterance to history
            current_adk_session.state['conversation_history'].append({"role": "user", "content": user_utterance_raw})

            logger.info(
                f"--- A2A NCF Prompt Construction for User: {adk_user_id}, Session: {actual_adk_session_id_to_use} ---")
            # Call NCF pillar functions (defined in this file)
            narrativa_fundamento = await get_narrativa_de_fundamento_pilar1(current_adk_session.state,
                                                                            memory_blossom_instance, adk_user_id)
            rag_info_list = await get_rag_info_pilar2(user_utterance_raw, memory_blossom_instance,
                                                      current_adk_session.state)
            chat_history_for_prompt_str = format_chat_history_pilar3(current_adk_session.state['conversation_history'])

            aura_persona_agente_a2a = AGENT_CARD_DATA.name
            aura_persona_detalhada_a2a = AGENT_CARD_DATA.description  # Or a more specific persona string for A2A

            final_ncf_prompt_str = montar_prompt_aura_ncf(
                aura_persona_agente_a2a, aura_persona_detalhada_a2a, narrativa_fundamento,
                rag_info_list, chat_history_for_prompt_str, user_utterance_raw
            )
            logger.debug(f"A2A RPC: NCF Prompt (first 200 chars): {final_ncf_prompt_str[:200]}...")

            adk_input_content = ADKContent(role="user", parts=[ADKPart(text=final_ncf_prompt_str)])

            # No explicit adk_session_service.update_session() needed here for InMemorySessionService
            # as modifications to current_adk_session.state are direct.

            adk_agent_final_text_response = None
            logger.info(
                f"A2A RPC: Running ADK agent for session '{actual_adk_session_id_to_use}' (User: {adk_user_id})")
            async for event in adk_runner.run_async(user_id=adk_user_id, session_id=actual_adk_session_id_to_use,
                                                    new_message=adk_input_content):
                if event.author == orchestrator_adk_agent_aura.name:  # Compare with imported agent's name
                    if event.get_function_calls():
                        fc = event.get_function_calls()[0]
                        logger.info(f"    A2A ADK FunctionCall by {event.author}: {fc.name}({json.dumps(fc.args)})")
                    if event.get_function_responses():
                        fr = event.get_function_responses()[0]
                        logger.info(
                            f"    A2A ADK FunctionResponse to {event.author}: {fr.name} -> {str(fr.response)[:100]}...")
                    if event.is_final_response():
                        if event.content and event.content.parts and event.content.parts[0].text:
                            adk_agent_final_text_response = event.content.parts[0].text.strip()
                            logger.info(f"  A2A Aura ADK Final Response: '{adk_agent_final_text_response[:100]}...'")
                        else:
                            logger.warning("  A2A Aura ADK Final Response event, but no text content found.")
                        break  # Exit loop on final response from agent

            adk_agent_final_text_response = adk_agent_final_text_response or "(Aura A2A: No textual response was generated for this interaction.)"
            current_adk_session.state['conversation_history'].append(
                {"role": "assistant", "content": adk_agent_final_text_response})

            # No explicit adk_session_service.update_session() needed here either.
            # The state of current_adk_session is already updated in memory.

            # Run reflector analysis
            await aura_reflector_analisar_interacao(
                user_utterance_raw, final_ncf_prompt_str, adk_agent_final_text_response,
                memory_blossom_instance, adk_user_id  # memory_blossom_instance imported
            )

            # Prepare A2A response
            a2a_response_artifact = A2AArtifact(parts=[A2APart(type="text", text=adk_agent_final_text_response)])
            a2a_task_status = A2ATaskStatus(state="completed")  # Assuming synchronous completion for now
            a2a_task_result = A2ATaskResult(
                id=task_params.id,
                sessionId=task_params.sessionId,  # Echo back the sessionId from params
                status=a2a_task_status,
                artifacts=[a2a_response_artifact]
            )

            logger.info(f"A2A RPC: Sending A2A response for Task ID {task_params.id}")
            return A2AJsonRpcResponse(id=rpc_request.id, result=a2a_task_result)

        except ValueError as ve:
            logger.error(f"A2A RPC tasks/send: Value Error encountered: {ve}", exc_info=True)
            return A2AJsonRpcResponse(id=rpc_request.id, error={"code": -32602, "message": f"Invalid parameters: {ve}"})
        except HTTPException as he:  # Catch FastAPI's HTTPExceptions if any are raised internally
            logger.error(f"A2A RPC tasks/send: HTTPException: {he.detail}", exc_info=True)
            return A2AJsonRpcResponse(id=rpc_request.id,
                                      error={"code": -32000, "message": f"Internal Server Error: {he.detail}"})
        except Exception as e:
            logger.error(f"A2A RPC tasks/send: Unhandled Internal Error: {e}", exc_info=True)
            return A2AJsonRpcResponse(id=rpc_request.id,
                                      error={"code": -32000, "message": f"Internal Server Error: {e}"})
    else:  # Method not "tasks/send"
        logger.warning(f"A2A RPC: Method '{rpc_request.method}' not supported.")
        return A2AJsonRpcResponse(id=rpc_request.id,
                                  error={"code": -32601, "message": f"Method not found: {rpc_request.method}"})


if __name__ == "__main__":
    logger.info(f"Starting Aura A2A & Voice API Wrapper Server on {A2A_WRAPPER_HOST}:{A2A_WRAPPER_PORT}")
    # Critical check for OpenAI key
    if not os.getenv("OPENAI_API_KEY") or len(os.getenv("OPENAI_API_KEY", "")) < 20:
        logger.critical(
            "CRITICAL FAILURE: OPENAI_API_KEY is not set or is invalid. Voice client functionality will NOT work.")
        # Consider exiting if voice is essential: sys.exit("OPENAI_API_KEY is required for voice features.")
    else:
        logger.info("OPENAI_API_KEY found and appears to be set.")

    if not os.getenv("OPENROUTER_API_KEY"):
        logger.warning("OPENROUTER_API_KEY not set. ADK Agent LLM calls (via OpenRouter) might fail.")

    # A2A_WRAPPER_DIR is the directory containing this main.py file
    uvicorn.run(
        "main:app",  # Points to `app` in `main.py` (this file)
        host=A2A_WRAPPER_HOST,
        port=A2A_WRAPPER_PORT,
        reload=True,
        app_dir=A2A_WRAPPER_DIR  # Tells uvicorn where to find "main.py"
    )
