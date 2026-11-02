from dotenv import load_dotenv
load_dotenv()

import argparse

from strAItegize_common.client.game_controller import GameController
from decentralized_mas import DecentralizedMAS

# Setting up app console line parameters
parser = argparse.ArgumentParser(prog="StrAItegize - Centralized Multi-Agent System")

parser.add_argument("-p", "--port", help="Client port shared by the simulation program to connect to.", type=int, required=True)
parser.add_argument("-d", "--debug", help="If True, during agent invocation, the user takes its place using console line, otherwise all agent calls are directed to the OpenAI LLM.", type=bool, default=True, required=False)

args = parser.parse_args()

if __name__ == '__main__':
    print("Welcome to the Centralized Multi-Agent System Architecture!")
    game_controler = GameController(simulation_port=args.port, multiagent_system=DecentralizedMAS, human_mode=args.debug)

    game_controler.initialize_simulation()

    game_controler.start_simulation()