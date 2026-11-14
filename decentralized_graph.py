from typing import Literal
from langchain.chat_models import init_chat_model
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage

from agent import get_agent
from states import DecentralizedState, AgentState
from tools import message_agent, order_move, order_attack, order_support, order_hold, recruit_unit, get_recent_events
from prompts import agent_msg_prompt, msg_no_response_prompt

from strAItegize_common.client.tools import get_generic_tools
from strAItegize_common.client.communication.communication_context import CommunicationContext

def phase_node(state: DecentralizedState) -> DecentralizedState:
    # Adding the round message to the agents
    input_prompt = state.get("input_prompt")
    agents_messages = state["agents_messages"]
    
    if input_prompt:
        for agent_id in agents_messages.keys():
            agents_messages[agent_id] = agents_messages[agent_id] + [input_prompt]
            
    # Blocking selected tools
    if state["upkeep_phase"]:
        blocked_tools = ["order_move", "order_attack", "order_support", "order_hold"]
    else:
        blocked_tools = ["recruit_unit"]
        
    # Activating agents at the beginning of the round
    active_agents = list(state["units"]) if not state["upkeep_phase"] else list(state["regions"])
    finished_agents =list(state["units"]) if state["upkeep_phase"] else list(state["regions"])
    
    
    return {
        "agents_messages": agents_messages,
        "blocked_tools": blocked_tools,
        "active_agents": active_agents,
        "waiting_agents": [],
        "finished_agents": finished_agents,
        "unit_orders": {} if not state["upkeep_phase"] else state.get("unit_orders", {}),
        "region_recruitment": {} if state["upkeep_phase"] else state.get("region_recruitment", {})
    }

def agent_manage_node(state: DecentralizedState) -> DecentralizedState:
    agents_messages = state["agents_messages"]
    agent_types = state["agent_types"]
    
    # Removing missing agents
    for agent in list(agents_messages.keys()):
        if agent not in list(state["units"]) + list(state["regions"]):
            del agents_messages[agent]
            del agent_types[agent]
        
    # Adding new unit agents
    for agent in list(state["units"]):
        if agent not in agents_messages.keys():
            agent_types[agent] = "unit"
            agents_messages[agent] = [state["init_msg"]]
                
    # Adding new region agents
    for agent in list(state["regions"]):
        if agent not in agents_messages.keys():
            agent_types[agent] = "region"
            agents_messages[agent] = [state["init_msg"]]

    return {
        "agents_messages": agents_messages,
        "agent_types": agent_types
    }

def dispatcher_node(state: DecentralizedState) -> DecentralizedState:
    active = state["active_agents"]
    waiting = state["waiting_agents"]
    finished = state["finished_agents"]
    
    agents_messages = state["agents_messages"]
    
    # If no agents are active, wake up waiting agents
    if len(active) == 0 and len(waiting) > 0:
        for agent in waiting:
            agents_messages[agent] = agents_messages[agent] + [
                SystemMessage(content=msg_no_response_prompt())
            ]
            active.append(agent)
            
        waiting = []

    # If there are no more active agents, finish turn
    if len(active) == 0:
        return {
            "active_agents": [],
            "waiting_agents": [],
            "finished_agents": finished,
            "next_action": "end"
        }
        
    # Activate next agent
    current_agent = active.pop(0)
    agent_type = state["agent_types"][current_agent]

    return {
        "active_agents": active,
        "waiting_agents": waiting,
        "finished_agents": finished,
        "agents_messages": agents_messages,
        "active_agent": current_agent,
        "agent_id": current_agent,
        "agent_type": agent_type,
        "agent_messages": agents_messages.get(current_agent, []),
        "messages_to_send": [],
        "next_action": agent_type 
    }

def agent_iteration_router(state: DecentralizedState) -> Literal["unit_agent", "region_agent", "end"]:
    action = state["next_action"]
    
    if action == "unit":
        return "unit_agent"
    elif action == "region":
        return "region_agent"
    return "end"

def agent_output_node(state: DecentralizedState) -> DecentralizedState:
    active = state["active_agents"]
    waiting = state["waiting_agents"]
    finished = state["finished_agents"]
    
    agent_id = state["agent_id"]
    agents_messages = state["agents_messages"]

    # Update agent's message history
    agents_messages[agent_id] = state["agent_messages"]
    
    # Pass messages to other agents
    outgoing_msgs = state["messages_to_send"]
    if outgoing_msgs:
        for msg in outgoing_msgs:
            sendee = msg.sendee
            
            agents_messages[sendee] = agents_messages[sendee] + [
                SystemMessage(content=agent_msg_prompt(sender=agent_id, content=msg.content))
            ]
            
            # Sendee must be active to read this message
            if sendee in finished:
                finished.remove(sendee)
            if sendee in waiting:
                waiting.remove(sendee)
            if sendee not in active:
                active.append(sendee)

        # Sender joins waiting list
        if agent_id not in waiting:
            waiting.append(agent_id)
    else:
        # Agent finishes operating
        if agent_id not in finished:
            finished.append(agent_id)
    
    # Cleaning the shared channel
    return {
        "active_agents": active,
        "waiting_agents": waiting,
        "finished_agents": finished,
        "agents_messages": agents_messages,
        "active_agent": None,
        "agent_messages": None,
        "agent_type": None,
        "messages_to_send": None,
        "agent_id": None
    }

def agent_tool_router(state: AgentState) -> Literal["tools", "end"]:
    last_msg = state["agent_messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    
    return "end"

def get_decentralized_mas(
    communication: CommunicationContext,
    unit_agent_model = init_chat_model("openai:gpt-4o"),
    region_agent_model = init_chat_model("openai:gpt-4o")
):
    # Assigning tools
    unit_agent_tools = [message_agent, order_move, order_attack, order_support, order_hold, get_recent_events] + get_generic_tools(communication=communication)
    region_agent_tools = [message_agent, recruit_unit, get_recent_events] + get_generic_tools(communication=communication)
    
    # Building unit agent subgraph
    unit_agent_builder = StateGraph(AgentState)
    
    unit_agent_builder.add_node("unit_agent", get_agent(tools=unit_agent_tools, model=unit_agent_model))
    unit_agent_builder.add_node("tools", ToolNode(unit_agent_tools, messages_key="agent_messages"))
    
    unit_agent_builder.add_edge(START, "unit_agent")
    unit_agent_builder.add_conditional_edges("unit_agent", agent_tool_router, {
        "tools": "tools",
        "end": END        
    })
    unit_agent_builder.add_edge("tools", "unit_agent")
    
    compiled_unit_agent = unit_agent_builder.compile()
    
    # Building region agent subgraph
    region_agent_builder = StateGraph(AgentState)
    
    region_agent_builder.add_node("region_agent", get_agent(tools=region_agent_tools, model=region_agent_model))
    region_agent_builder.add_node("tools", ToolNode(region_agent_tools, messages_key="agent_messages"))
    
    region_agent_builder.add_edge(START, "region_agent")
    region_agent_builder.add_conditional_edges("region_agent", agent_tool_router, {
        "tools": "tools",
        "end": END        
    })
    region_agent_builder.add_edge("tools", "region_agent")
    
    compiled_region_agent = region_agent_builder.compile()
    
    # Building Decentralized Multi-Agent System graph
    mas_builder = StateGraph(DecentralizedState)
    
    mas_builder.add_node("phase_node", phase_node)
    mas_builder.add_node("agent_manage_node", agent_manage_node)
    mas_builder.add_node("dispatcher_node", dispatcher_node)
    mas_builder.add_node("unit_agent", compiled_unit_agent)
    mas_builder.add_node("region_agent", compiled_region_agent)
    mas_builder.add_node("agent_output_node", agent_output_node)
    
    mas_builder.add_edge(START, "agent_manage_node")
    mas_builder.add_edge("agent_manage_node", "phase_node")
    mas_builder.add_edge("phase_node", "dispatcher_node")
    mas_builder.add_conditional_edges("dispatcher_node", agent_iteration_router, {
        "unit_agent": "unit_agent",
        "region_agent": "region_agent",
        "end": END
    })
    mas_builder.add_edge("unit_agent", "agent_output_node")
    mas_builder.add_edge("region_agent", "agent_output_node")
    mas_builder.add_edge("agent_output_node", "dispatcher_node")
    
    checkpointer = MemorySaver()
    return mas_builder.compile(checkpointer=checkpointer)
    