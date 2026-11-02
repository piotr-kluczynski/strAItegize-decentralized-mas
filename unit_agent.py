from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

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
    pass

def unit(state: UnitState) -> UnitState:
    pass

def human_unit(state: UnitState) -> UnitState:
    return {
        "unit_messages":
            human(
                [SystemMessage(content=unit_agent_prompt(state["player_id"], state["generals_units"], state["regions"]))] + state["leader_messages"],
                agent_name="leader",
                tools=available_unit_tools
            )
    }

def tool_node(state: UnitState) -> UnitState:
    pass

def output_node(state: UnitState) -> UnitOutputState:
    pass

def tool_condition