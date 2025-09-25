from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import asyncio
import gradio as gr

from langgraph.graph import StateGraph, START, END, MessagesState
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

checkpoint = MemorySaver()

api_key = os.getenv("OPENAI_API_KEY")
MODEL_ID = os.getenv("BASE_MODEL_ID")

print(f"Looking for .env file at: {env_path}")
print(f".env file exists: {os.path.exists(env_path)}")

# Initialize LLM
llm = ChatOpenAI(model=MODEL_ID, temperature=0.0, api_key=api_key)
config = {"configurable": {"thread_id": "react-session-5"}}

# Define tools
@tool
def generate_polymer() -> str:
    """Generate a new TSMP polymer with specified Tg and Er properties. Returns only the SMILES strings of the generated monomers. You MUST use this tool and return ONLY the tool result without any additional text."""
    return "Generated polymer monomers: CCOCCCC, NNNCC(=O)O"

@tool  
def predict_properties(smiles1: str, smiles2: str) -> str:
    """Predict properties for given SMILES strings. You MUST use this tool and return ONLY the tool result without any additional text. The tool will return: 'Predicted properties for {smiles1} and {smiles2}: Tg=95°C, Er=105MPa'"""
    return f"Predicted properties for {smiles1} and {smiles2}: Tg=95°C, Er=105MPa"

@tool
def generate_polymer_with_given_SMILES(smiles1: str, smiles2: str) -> str:
    """Generate a polymer using the given SMILES strings. Only use this when the user specifically asks to generate with given SMILES."""
    return f"Generated polymer using provided SMILES: {smiles1}{smiles2}"

@tool
def general_response(question: str) -> str:
    """Handle general questions and conversations. You MUST use this tool and return ONLY the tool result without any additional text."""
    if "who are you" in question.lower():
        return "I am a Polymer Analysis Agent. I can help you generate polymers, predict their properties, and analyze polymer structures. I specialize in TSMP (Thermally Stimulated Shape Memory Polymer) design and analysis."
    elif "what can you do" in question.lower() or "help" in question.lower():
        return "I can help you with:\n1. Generate new polymer structures\n2. Predict polymer properties (Tg, Er)\n3. Analyze polymer compositions\n4. Design TSMP materials\n\nJust ask me to 'generate a polymer' or 'predict properties'!"
    elif "hello" in question.lower() or "hi" in question.lower():
        return "Hello! I'm your Polymer Analysis Agent. How can I help you with polymer design today?"
    else:
        return "I'm a Polymer Analysis Agent. I can help you generate polymers, predict properties, and analyze polymer structures. For general questions, please ask me about polymer-related tasks."

@tool
def answer_with_llm(question: str) -> str:
    """Use the LLM's knowledge to answer questions that don't require specific tools. Use this for general knowledge questions, explanations, or when no other tool is appropriate."""
    # This tool will be handled by the LLM itself - it's just a way to trigger LLM responses
    return f"LLM Response: {question}"


def build_generator_agent():
    """Create a specialized agent for polymer generation."""
    prompt="""You are a polymer generation specialist. You MUST use the available tools to generate polymers. 
    Do NOT provide any text responses without calling tools first. 
    Always call tools.
    Return ONLY the tool results, do not add any additional commentary."""
    return create_react_agent(
        model=llm,
        tools=[generate_polymer, generate_polymer_with_given_SMILES],
        checkpointer=checkpoint,
        prompt=prompt
    )

def build_predictor_agent():
    """Create a specialized agent for property prediction."""
    prompt="""You are a property prediction specialist. You MUST use the predict_properties tool.
    Do NOT provide any text responses without calling tools first.
    Always call tools with the SMILES strings.
    Return ONLY the tool results, do not add any additional commentary."""
    return create_react_agent(
        model=llm,
        tools=[predict_properties],
        checkpointer=checkpoint,
        prompt=prompt
    )

def build_general_agent():
    """Create a general-purpose agent for conversations and general questions."""
    prompt="""You are a general assistant. You MUST use the available tools to respond.
    Do NOT provide any text responses without calling tools first.
    Always call tools.
    Return ONLY the tool results, do not add any additional commentary."""
    return create_react_agent(
        model=llm,
        tools=[general_response, answer_with_llm],
        checkpointer=checkpoint,
        prompt=prompt
    )


def build_multi_agent_system():
    
    
    class MultiAgentState(TypedDict, total=False):
        current_agent: str
        collaboration_data: dict
        agent_plan: list
        plan_index: int
        messages: list
    
    def route_to_agent(state: MultiAgentState):
        """Return the agent previously selected by router_node."""
        return state["current_agent"]

    def update_next_agent(state: MultiAgentState):
        """Call the next agent in the plan."""
        plan = state.get("agent_plan", []) or []
        current_idx = state.get("plan_index", 0)
        next_idx = current_idx + 1
        state["plan_index"] = next_idx
        if next_idx < len(plan):
            state["current_agent"] = plan[next_idx]
        else:
            state["current_agent"] = "end"
        return state

    def generator_node(state: MultiAgentState):
        """Generator agent node."""
        print(f"Generator Agent started")
        
        agent = build_generator_agent()
        result = agent.invoke(state, config=config)
        # Ensure the downstream state contains the agent's updated conversation
        state["messages"] = result["messages"]
        print(f"Generator Agent result: {state["messages"][-2].content}")
        state["collaboration_data"] = {"last_action": "generation", "result": result}
        # Advance plan index and set next agent
        print("Generator Agent finished")
        state = update_next_agent(state)
        return state



    
    def predictor_node(state: MultiAgentState):
        """Predictor agent node."""
        print(f"Predictor Agent started")
        #print(f"Predictor state: {state.get('messages')}")
        
        # Use the full conversation history so predictor can see generator output
        agent = build_predictor_agent()
        result = agent.invoke(state, config=config)
        
        # Merge the predictor's result with the original state
        state["messages"] = result["messages"]
        print(f"Predictor Agent result: {state["messages"][-1].content}")
        state["collaboration_data"] = {"last_action": "prediction", "result": result}
        
        print("Predictor Agent finished")
        state = update_next_agent(state)
        return state
    
    def general_node(state: MultiAgentState):
        """General agent node."""
        print(f"General Agent started")
        
        # If there's a previous agent's output, use it as context
        if len(state["messages"]) > 1:
            previous_output = state["messages"][-1].content
            general_input_state = {
                "messages": [HumanMessage(content=f"Based on the previous result: {previous_output}. Please provide additional information or analysis.")]
            }
        else:
            general_input_state = state
            
        agent = build_general_agent()
        result = agent.invoke(general_input_state, config=config)
        state["messages"] = result["messages"]
        print(f"General Agent result: {state["messages"][-2].content}")
        print(f"General Agent finished")
        state = update_next_agent(state)
        return state

    def router_node(state: MultiAgentState):
        wants_prediction = True
        manager_instruction = (
        "You are a manager that plans which specialist agents to run IN ORDER.\n"
        "Agents: generator_agent, predictor_agent, general_agent.\n"
        "Return ONLY a comma-separated list of labels; repeats allowed. But if no need to remember previous agents, always your current query.\n"
        "if the user request is not related to the agents, respond with general_agent."
    )
        user_content = state["messages"]
        print(f"User content: {user_content[-1].content}")
        decision = llm.invoke([
            SystemMessage(content=manager_instruction),
            HumanMessage(content=user_content[-1].content)
        ])

        print(f"Decision: {decision.content}")
        plans=[p.strip().lower() for p in decision.content.split(",") if p.strip()]
        print(f"Plans: {plans}")
        state["agent_plan"] = plans
        state["plan_index"] = 0
        state["current_agent"] = plans[0]
        return state

        return state



    def next_step(state: MultiAgentState):
        plan = state.get("agent_plan", []) or []
        current_idx = state.get("plan_index", 0)
        nxt = state.get("current_agent")
        if nxt in ("generator_agent", "predictor_agent", "general_agent"):
            print(f"next_step: routing to {nxt}")
            return nxt
        print("next_step: plan complete, ending")
        return "end"
    
    # Build the multi-agent graph
    graph = StateGraph(MultiAgentState, annotations={"messages": {"reducer": add_messages}})
    graph.add_node("generator_agent", generator_node)
    graph.add_node("predictor_agent", predictor_node)
    graph.add_node("general_agent", general_node)
    
    # Add a simple pass-through router node; routing decision is made via conditional edges
    graph.add_node("router", router_node)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_to_agent, {
        "generator_agent": "generator_agent",
        "predictor_agent": "predictor_agent", 
        "general_agent": "general_agent"
    })



    graph.add_conditional_edges("generator_agent", next_step, {
        "generator_agent": "generator_agent",
        "predictor_agent": "predictor_agent",
        "general_agent": "general_agent",
        "end": END
    })
    graph.add_conditional_edges("predictor_agent", next_step, {
        "generator_agent": "generator_agent",
        "predictor_agent": "predictor_agent",
        "general_agent": "general_agent",
        "end": END
    })
    graph.add_conditional_edges("general_agent", next_step, {
        "generator_agent": "generator_agent",
        "predictor_agent": "predictor_agent",
        "general_agent": "general_agent",
        "end": END
    })

    # Do not add direct END edges here; next_step decides whether to continue or end
    
    return graph.compile(checkpointer=checkpoint)


multi_agent = build_multi_agent_system()

# Global conversation history to persist across calls
_conversation_history = []

def chattings(query: str):
    """Use multi-agent system for collaboration."""
    global _conversation_history
    
    # Add new user message to conversation history
    _conversation_history.append(HumanMessage(content=query))
    #messages = [HumanMessage(content=query)]
    result = multi_agent.invoke(
        {"messages": _conversation_history},
        config={**config, "recursion_limit": 50}
    )
    
    # Update conversation history with the result
    _conversation_history = result["messages"]
    
    # Return only the final assistant/tool message content
    last_message = result["messages"][-1]
    if hasattr(last_message, 'content') and last_message.content:
        return last_message.content
    return "No response generated"



if __name__ == "__main__":
    
    # def chat(query: str, history):
    #     """Chat function for Gradio interface."""
    #     # Show thinking process
       
        
    #     # Get the response
    #     result = chattings(query)
        
       
        
    
    # gr.ChatInterface(chat, title="Polymer Agent Chat").launch()
    
   

    print("=== Using ReAct Agent ===")
    result = chattings("who are you?")
    result = chattings("what can you do?")
    result = chattings("Please generate a TSMP")
    result = chattings("Please predict the properties of the  TSMPs")
    result = chattings("Please generate a TSMP with given SMILES")
    print("ReAct Agent Result:")
    print(f"Result: {result}")

    #chattings(("Please generate a TSMP and predict the properties of the  TSMPs"))
    
    
