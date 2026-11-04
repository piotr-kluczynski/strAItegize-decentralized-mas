from typing import Literal
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from states import RegionInputState, RegionState, RegionOutputState
from prompts import region_agent_prompt
from tools import AgentMessage, get_region_tools

from strAItegize_common.others.human import human
from strAItegize_common.others.recruitment_choice import RecruitmentChoice
from strAItegize_common.client.communication.communication_context import CommunicationContext

HUMAN_MODE = False

# Dynamic list of tools initialization
region_tools = []

available_region_tools = []
tools_by_name = {}

# Initializing llm
model = init_chat_model(
    model_name="gpt-4o",
)

# Agent nodes
def input_node(state: RegionInputState) -> RegionState:
    global region_tools, available_region_tools, tools_by_name

    # Dynamically reducing tools if invokation happened during planning phase
    available_region_tools = region_tools

    # During planning phase, the region agent does not make any decisions, but responds to unit agents messages
    if not state["upkeep_phase"]:
        available_region_tools = [tool for tool in region_tools if tool.name not in ["RecruitUnit"]]

    tools_by_name = {tool.name: tool for tool in available_region_tools}

    return {
        "units": state["units"],
        "regions": state["regions"],
        "region_id": state["region_id"],
        "player_id": state["player_id"],
        "recent_events": state["recent_events"],
        "region_messages": state["region_messages"],
        "messages_to_send": [],
        "region_recruitment": state["region_recruitment"],
    }

def region(state: RegionState) -> RegionState:
    return {
        "region_messages": [
            model.bind_tools(available_region_tools).invoke(
                [SystemMessage(content=region_agent_prompt(player_id=state["player_id"], region_id=state["region_id"], units=state["units"], region_recruitment=state["region_recruitment"]))] + state["region_messages"],
                config={}
            )
        ]
    }

def human_region(state: RegionState) -> RegionState:
    return {
        "region_messages":
            human(
                [SystemMessage(content=region_agent_prompt(player_id=state["player_id"], region_id=state["region_id"], units=state["units"], region_recruitment=state["region_recruitment"]))] + state["region_messages"],
                agent_name=state["region_id"],
                tools=available_region_tools
            )
    }

def tool_node(state: RegionState) -> RegionState:
    tool_calls = state["region_messages"][-1].tool_calls
    observations = []

    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]

        if tool_call["name"] == "RecruitUnit":
            unit_type = tool_call["args"]["unit_type"]

            new_recruitment_choice = RecruitmentChoice(
                unit_type=unit_type,
                region=state["region_id"]
            )
            state["region_recruitment"][state["region_id"]] = new_recruitment_choice
            observations.append(f"Your recruitment choice was assigned properly: you will attempt to recruit a {unit_type} unit.")
        elif tool_call["name"] == "MessageAgent":
            sendee = tool_call["args"]["sendee"]
            content = tool_call["args"]["content"]

            state["messages_to_send"].append(AgentMessage(
                sender=state["region_id"],
                sendee=sendee,
                content=content
            ))
            observations.append(f"Message sent to agent {sendee}: '{content}'")
        else:
            # Injecting state schema if needed
            args = {**tool_call["args"]}

            # Collecting argument names
            arg_fields = {name for name, _ in tool.get_input_schema().model_fields.items()}

            if "state" in arg_fields:
                args["state"] = state

            observations.append(tool.invoke(args))

    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"region_messages": state["region_messages"] + tool_outputs}

def output_node(state: RegionState) -> RegionOutputState:
    return RegionOutputState(
        region_messages=state["region_messages"],
        messages_to_send=state["messages_to_send"],
        region_recruitment=state["region_recruitment"]
    )

def should_continue(state: RegionState) -> Literal["tool_node", "output_node"]:
    last_message = state["region_messages"][-1]

    if last_message.tool_calls:
            return "tool_node"
    return "output_node"

def get_region_agent(communication: CommunicationContext, human_mode=False):
    global HUMAN_MODE, region_tools

    # Setting up the tool set
    region_tools = get_region_tools(communication=communication)

    # Graph construction
    region_agent_builder = StateGraph(RegionState, input_schema=RegionInputState, output_schema=RegionOutputState)

    region_agent_builder.add_node("input_node", input_node)
    region_agent_builder.add_node("tool_node", tool_node)
    region_agent_builder.add_node("output_node", output_node)

    if human_mode:
        region_agent_builder.add_node("region", human_region)
        HUMAN_MODE = True
    else:
        region_agent_builder.add_node("region", region)

    region_agent_builder.add_edge(START, "input_node")
    region_agent_builder.add_edge("input_node", "region")
    region_agent_builder.add_edge("tool_node", "region")
    region_agent_builder.add_edge("output_node", END)

    region_agent_builder.add_conditional_edges(
        "region",
        should_continue,
        {
            "tool_node": "tool_node",
            "output_node": "output_node"
        }
    )

    return region_agent_builder.compile()