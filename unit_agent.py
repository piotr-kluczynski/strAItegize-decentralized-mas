from typing import Literal
from pydantic import ValidationError
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from states import UnitInputState, UnitState, UnitOutputState
from prompts import unit_agent_prompt
from tools import UnitAgentMessage, get_unit_tools

from strAItegize_common.others.human import human
from strAItegize_common.others.order import Order, RecruitmentChoice

HUMAN_MODE = False

# Dynamic list of tools initialization
unit_tools = []

available_unit_tools = []
tools_by_name = {}

# Initializing llm
model = init_chat_model(
    model_name="gpt-4o",
)

# Agent nodes
def input_node(state: UnitInputState) -> UnitState:
    global unit_tools, available_unit_tools, tools_by_name

    # Dynamically reducing tools if invokation happened as result of the unit recruitment process
    available_unit_tools = unit_tools
    
    if state["upkeep_phase"]:
        available_unit_tools = [tool for tool in unit_tools if tool.name not in ["OrderMovement", "OrderAttack", "OrderSupport", "OrderHold", "get_recent_events"]]
    else: 
        available_unit_tools = [tool for tool in unit_tools if tool.name != "RecruitUnit"]

    tools_by_name = {tool.name: tool for tool in available_unit_tools}

    return {
        "units": state["units"],
        "regions": state["regions"],
        "player_id": state["player_id"],
        "recent_events": state["recent_events"],
        "unit_messages": state["unit_messages"],
        "messages_to_send": [],
        "order_list": state["order_list"],
        "recruitment_choice_list": state["recruitment_choice_list"]
    }

def unit(state: UnitState) -> UnitState:
    return {
        "unit_messages": [
            model.bind_tools(available_unit_tools).invoke(
                [SystemMessage(content=unit_agent_prompt(state["player_id"], state["generals_units"], state["regions"]))] + state["unit_messages"],
                config={""}
            )
        ]
    }

def human_unit(state: UnitState) -> UnitState:
    return {
        "unit_messages":
            human(
                [SystemMessage(content=unit_agent_prompt(state["player_id"], state["unit_id"], {unit: state["order_list"].get(unit) for unit in state["units"]}, state["regions"]))] + state["unit_messages"],
                agent_name=state["unit_id"],
                tools=available_unit_tools
            )
    }

def tool_node(state: UnitState) -> UnitState:
    tool_calls = state["unit_messages"][-1].tool_calls
    observations = []

    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]

        if tool.name == "OrderMovement":
            move_q = tool_call["args"]["move_q"]
            move_r = tool_call["args"]["move_r"]
            move_s = tool_call["args"]["move_s"]

            new_order = Order(
                order_type="move",
                unit_id=state["unit_id"],
                move_q=move_q,
                move_r=move_r,
                move_s=move_s
            )
            state["unit_orders"][state["unit_id"]] = new_order
        elif tool.name == "OrderAttack":
            target_id = tool_call["args"]["target_id"]
            
            new_order = Order(
                order_type="attack",
                unit_id=state["unit_id"],
                target_id=target_id
            )
            state["unit_orders"][state["unit_id"]] = new_order
        elif tool.name == "OrderSupport":
            target_id = tool_call["args"]["target_id"]
            
            new_order = Order(
                order_type="support",
                unit_id=state["unit_id"],
                target_id=target_id
            )
            state["unit_orders"][state["unit_id"]] = new_order
        elif tool.name == "OrderHold":
            new_order = Order(
                order_type="hold",
                unit_id=state["unit_id"]
            )
            state["unit_orders"][state["unit_id"]] = new_order
        elif tool_call["name"] == "RecruitUnit":
            recruitment_choice = tool_call["args"]["recruitment_choice"]

            # Casting recruitment choice to the appropriate pydantic model
            try:
                recruitment_choice = RecruitmentChoice.model_validate(recruitment_choice)
                state["recruitment_choice"].append(recruitment_choice)
                observations.append(f"Unit recruitment choice: {recruitment_choice.region}(region): {recruitment_choice.unit_type}(unit type) was successfuly added to the list.")
            except ValidationError as e:
                observations.append(f"Invalid input data provided for tool: {tool_call["name"]}. \r\nDetails: {e.errors()}")
        elif tool_call["name"] == "MessageAgent":
            sendee = tool_call["args"]["sendee"]
            content = tool_call["args"]["content"]

            state["messages_to_send"].append(UnitAgentMessage(
                sender=state["unit_id"],
                sendee=sendee,
                content=content
            ))
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

    return {"unit_messages": state["unit_messages"] + tool_outputs}

def output_node(state: UnitState) -> UnitOutputState:
    return UnitOutputState(
        unit_messages=state["unit_messages"],
        messages_to_send=state["messages_to_send"],
        order_list=list(state["unit_orders"].values()),
        recruitment_choice_list=state["recruitment_choice_list"]
    )

def should_continue(state: UnitState) -> Literal["tool_node", "output_node"]:
    unit_messages = state["unit_messages"]
    last_message = unit_messages[-1]

    tool_calls = last_message.tool_calls

    if last_message.tool_calls:
        if "MessageAgent" not in [tool_calls['name'] for tool_call in tool_calls]:
            return "tool_node"
    return "output_node"

def get_unit_agent(communication, human_mode=False):
    global HUMAN_MODE, unit_tools

    # Setting up the tool set
    unit_tools = get_unit_tools(communication)

    # Graph construction
    unit_agent_builder = StateGraph(UnitState, input_schema=UnitInputState)

    unit_agent_builder.add_node("input_node", input_node)
    unit_agent_builder.add_node("tool_node", tool_node)
    unit_agent_builder.add_node("output_node", output_node)

    if human_mode:
        unit_agent_builder.add_node("unit", human_unit)
        HUMAN_MODE = True
    else:
        unit_agent_builder.add_node("unit", unit)

    unit_agent_builder.add_edge(START, "input_node")
    unit_agent_builder.add_edge("input_node", "unit")
    unit_agent_builder.add_edge("tool_node", "unit")
    unit_agent_builder.add_edge("output_node", END)

    unit_agent_builder.add_conditional_edges(
        "unit",
        should_continue,
        {
            "tool_node": "tool_node",
            "output_node": "output_node"
        }
    )

    return unit_agent_builder.compile()