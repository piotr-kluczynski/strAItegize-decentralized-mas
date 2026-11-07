from langchain_core.runnables import RunnableConfig
from langchain.messages import SystemMessage

from states import AgentState, DecentralizedState
from prompts import unit_agent_prompt, region_agent_prompt

from strAItegize_common.others.human import human

def get_agent(tools: list, model):
    def agent(state: AgentState, config: RunnableConfig) -> DecentralizedState:
        configurable = config.get("configurable", {})
        is_human = configurable.get("human_mode", False)
        
        active_tools = [tool for tool in tools if tool.name not in state["blocked_tools"]]
        
        model_with_tools = model.bind_tools(active_tools)
        
        input_prompt = None
        if state["agent_type"] == "unit":
            input_prompt = SystemMessage(
                content=unit_agent_prompt(
                    player_id=state["player_id"],
                    unit_id=state["agent_id"],
                    unit_orders=state["unit_orders"],
                    regions=state["regions"]
                )
            )
        elif state["agent_type"] == "region":
            input_prompt = SystemMessage(
                content=region_agent_prompt(
                    player_id=state["player_id"],
                    region_id=state["agent_id"],
                    units=state["units"],
                    region_recruitment=state["region_recruitment"]
                )
            )
        message_list = [input_prompt] + state["agent_messages"]
        
        if is_human:
            return {
                "agent_messages": human(
                    message_list,
                    agent_name=state["agent_id"],
                    tools=active_tools
                )
            }
        else:
            return {
                "agent_messages": [
                    model_with_tools.invoke(message_list)
                ]
            }
    
    return agent