import asyncio
import base64
import httpx
import os
import uuid
import graphviz
import graphviz.backend
import json
from typing import Annotated, List, Dict, Any, Union, Optional
from google.cloud import firestore
from datetime import datetime, timezone, timedelta


import dspy
import litellm
import google.generativeai as genai
from dotenv import load_dotenv
from mcp import ErrorData, McpError
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field

# --- Load Environment Variables ---
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
assert GEMINI_API_KEY is not None, "Please set GEMINI_API_KEY in your .env file"

# --- Configure Gemini ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Google Generative AI Configured Successfully.")
except Exception as e:
    print(f"An error occurred during genai configuration: {e}")

# --- Custom Gemini DSPy Language Model ---
class CustomGeminiDspyLM(dspy.LM):
    def __init__(
        self,
        model: str,
        api_key: str,
        safety_settings: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ):
        super().__init__(model)
        self.model = model
        self.api_key = api_key
        self.safety_settings = safety_settings
        self.kwargs = kwargs
        self.provider = "custom_gemini_litellm"

    def _prepare_litellm_messages_from_dspy_inputs(
        self, dspy_input: Union[str, List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        if isinstance(dspy_input, str):
            return [{"role": "user", "content": dspy_input}]
        elif isinstance(dspy_input, list):
            return dspy_input
        else:
            raise TypeError(f"Unsupported dspy_input type: {type(dspy_input)}")

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> List[str]:
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")
        
        dspy_input_content = prompt if prompt is not None else messages
        messages_for_litellm = self._prepare_litellm_messages_from_dspy_inputs(dspy_input_content)
        
        final_call_kwargs = self.kwargs.copy()
        final_call_kwargs.update(kwargs)

        extra_body = {}
        if self.safety_settings:
            extra_body['safety_settings'] = self.safety_settings

        try:
            response_obj = litellm.completion(
                model=self.model,
                messages=messages_for_litellm,
                api_key=self.api_key,
                extra_body=extra_body if extra_body else None,
                **final_call_kwargs, # I have Removed the Ratelimiter
            )
            completions = [choice.message.content for choice in response_obj.choices if choice.message and choice.message.content]
            if not completions:
                return ["[WARN: No valid content in response]"]
            return completions
        except Exception as e:
            return [f"[ERROR: {type(e).__name__} - {e}]"]

# --- Pydantic Models for Mind Map ---
class MindMapNode(BaseModel):
    topic: str
    children: List['MindMapNode'] = Field(default_factory=list)

# --- DSPy Signatures and Modules ---
class ConversationSignature(dspy.Signature):
    """
    You are a Conversation Manager. Your job is to gather context to generate a mind map. Ask clarifying questions,
    but don't elongate the conversation. You can ask at most 3 questions. If the context is sufficient, output 'DONE' 
    in the action_code. if the user needs mindmap immediately generate action_code as 'DONE
    """
    conversation_history = dspy.InputField(desc="The history of the conversation so far.")
    question = dspy.OutputField(desc="Your next question to the user.")
    action_code = dspy.OutputField(desc="Output 'DONE' if context is sufficient, otherwise 'PENDING'.")

class MindMapSignature(dspy.Signature):
    """
    You are a mind map generator. Based on the conversation_history, generate the subtopics for the mind map. 
    The structure should be hierarchical."""
    conversation_history = dspy.InputField(desc="Context about the topic for the mind map.")
    reason: str = dspy.OutputField(desc="Explain the reasoning for the structure and topic names.")
    response: MindMapNode = dspy.OutputField(desc="Generate the mind map structure.")

class MindMapInfoGatherer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.ask = dspy.Predict(ConversationSignature)
    def forward(self, conversation_history):
        return self.ask(conversation_history=conversation_history)

class MindMapStructure(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(MindMapSignature)
    def forward(self, conversation_history):
        return self.generator(conversation_history=conversation_history)

# --- Graphviz Conversion ---
def to_graphviz_bytes(root_node: MindMapNode) -> bytes:
    """
    Creates a highly aesthetic mind map using Graphviz and returns it as PNG bytes.
    """
    dot = graphviz.Digraph('FlowMindMap', comment=root_node.topic, engine='osage')
    
    dot.attr(
        bgcolor="#F8F9FA",
        splines='curved',
        nodesep='0.5',
        ranksep='1.0'
    )

    edge_color = '#5F6F7F'
    
    root_node_style = {
        'style': 'rounded,filled', 'shape': 'box', 'penwidth': '2',
        'fontname': 'Helvetica-Bold', 'fontsize': '20', 'fontcolor': 'white',
        'color': '#065F46',
        'fillcolor': '#10B981:#059669',
    }
    
    level1_node_style = {
        'style': 'rounded,filled', 'shape': 'box', 'penwidth': '2',
        'fontname': 'Helvetica-Bold', 'fontsize': '16', 'fontcolor': '#064E3B',
        'color': '#6EE7B7',
        'fillcolor': '#D1FAE5',
    }
    
    default_node_style = {
        'style': 'rounded,filled', 'shape': 'box', 'penwidth': '1.5',
        'fontname': 'Helvetica', 'fontsize': '14', 'fontcolor': '#1F2937',
        'color': '#D1D5DB',
        'fillcolor': '#F3F4F6',
    }

    node_ids = {}

    def get_node_id(node: MindMapNode, parent_id: Optional[str] = None) -> str:
        # Create a unique key for the node based on its topic and its parent's ID
        node_key = (node.topic, parent_id)
        if node_key not in node_ids:
            node_ids[node_key] = str(uuid.uuid4())
        return node_ids[node_key]

    def add_nodes_and_edges(parent_node: MindMapNode, level: int, p_id: Optional[str] = None):
        parent_id = get_node_id(parent_node, p_id)
        
        if level == 0:
            dot.node(parent_id, parent_node.topic, **root_node_style)
        elif level == 1:
            dot.node(parent_id, parent_node.topic, **level1_node_style)
        else:
            dot.node(parent_id, parent_node.topic, **default_node_style)

        for child_node in parent_node.children:
            child_id = get_node_id(child_node, parent_id)
            add_nodes_and_edges(child_node, level + 1, parent_id)
            dot.edge(
                parent_id, child_id,
                color=edge_color, arrowhead='normal', penwidth='1.5'
            )

    add_nodes_and_edges(root_node, 0)

    filename = f"mindmap_{uuid.uuid4()}"
    output_path = dot.render(filename, format='png', cleanup=True, directory='mindmap_images')
    
    with open(output_path, 'rb') as f:
        image_bytes = f.read()
    
    if os.path.exists(output_path):
        os.remove(output_path)
    
    return image_bytes

# --- In-Memory Storage for Conversations ---
# CONVERSATIONS: Dict[str, List[Dict[str, Any]]] = {}
db = firestore.Client()

def format_history_for_dspy(history_list: List[Dict[str, Any]]) -> str:
    return "\n---\n".join([f"{turn.get('role', 'unknown')}: {turn.get('parts', [''])[0]}" for turn in history_list])

async def generate_mindmap_func(
    puch_user_id: Annotated[str, Field(description="The unique ID for the user, provided by the Puch platform.")],
    user_input: Annotated[str, Field(description="The user's message or topic for the mind map.")],
) -> List[Union[TextContent, ImageContent]]:
    
    # Configure DSPy LM
    DEFAULT_MODEL_NAME = "gemini-2.5-flash"
    LITELLM_MODEL_STRING = f"gemini/{DEFAULT_MODEL_NAME}"
    gemini_lm = CustomGeminiDspyLM(
        model=LITELLM_MODEL_STRING,
        api_key=GEMINI_API_KEY,
        temperature=1,
    )
    dspy.settings.configure(lm=gemini_lm)

    # Get or create conversation history
    # conversation_history = CONVERSATIONS.setdefault(user_id, [])
        # NEW: Logic to retrieve conversation history from Firestore
    # Principle: State must be externalized. We define a reference to a document
    # in a 'mindmap_conversations' collection, using the user's ID as the unique key.
    doc_ref = db.collection('mindmap_conversations').document(puch_user_id)
    doc = doc_ref.get()
    
    conversation_history = []
    if doc.exists:
        # The document exists, so we retrieve its data.
        data = doc.to_dict()
        last_updated = data.get('last_updated')
        
        # Principle: Implement the TTL (Time-To-Live) feature in our application logic.
        # We check if the data is older than our defined policy (4 days).
        if last_updated and (datetime.now(timezone.utc) - last_updated) < timedelta(days=4):
            # The data is not expired, so we load it.
            conversation_history = data.get('history', [])
        else:
            # Data is expired or malformed, we treat it as a new conversation.
            pass
            
    # Add user's new input to history (in memory for this request)
    conversation_history.append({'role': 'user', 'parts': [user_input]})

    
    # Initialize DSPy modules
    mindmap_context = MindMapInfoGatherer()
    mindmap_structure = MindMapStructure()
    
    # Check if we have enough context
    formatted_history = format_history_for_dspy(conversation_history)
    context_prediction = mindmap_context(conversation_history=formatted_history)
    
    if context_prediction.action_code.strip().upper() == 'DONE':
        # Generate the mind map
        mind_map_prediction = mindmap_structure(conversation_history=formatted_history)
        
        # Convert the mind map to an image
        try:
            image_bytes = to_graphviz_bytes(mind_map_prediction.response)
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            # If image generation fails, return an error message
            return [
                TextContent(type="text", text="I couldn't generate the mind map image."),
                TextContent(type="text", text=f"Reason: {e}")
            ]

        # Clear conversation for next time
        doc_ref.delete()
        
        # Return the image and the reasoning in the expected format
        image_payload = {
            "image": {
                "mimeType": "image/png",
                "data": image_base64
            }
        }
        return [
            TextContent(type="text", text=json.dumps(image_payload)),
            TextContent(type="text", text=mind_map_prediction.reason)
        ]
    else:
        # Ask for more information
        assistant_question = context_prediction.question
        conversation_history.append({'role': 'assistant', 'parts': [assistant_question]})
        
        # NEW: Save the updated history back to Firestore for the next turn.
        # We save the history and the current timestamp for the TTL calculation.
        # Using firestore.SERVER_TIMESTAMP ensures we use the reliable clock of the database server.
        data_to_save = {
            'history': conversation_history,
            'last_updated': firestore.SERVER_TIMESTAMP
        }
        doc_ref.set(data_to_save)

        return [TextContent(type="text", text=assistant_question)]
