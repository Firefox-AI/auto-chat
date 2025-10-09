# auto-chat
Repository for creating agent-agent conversations. 

There are two ways to run this repository. The first way, is to have the handler agent (GPT-5) come up with the scenario. This will include generating a user profile, a set of user interests and a goal that the handler agent is trying to accomplish. 

### Example usage -- generated scenario
``` bash
uv run python start_chat.py --model together.ai:Qwen/Qwen3-235B-A22B-Instruct-2507-tput\
                            --eval_model_id openai:gpt-5\
                            --max_turns 5
```

If you wish to create a custom scenario, you can add a `.yaml` file to the `conversations` directory. Sample scenarios are provided. The easiest thing to do would be to use one of these as a template and update accordingly.

### Example usage -- custom scenario
``` bash
uv run python start_chat.py --model together.ai:Qwen/Qwen3-235B-A22B-Instruct-2507-tput\
                            --eval_model_id openai:gpt-5\
                            --conversation_file conversations/san_diego_trip.yaml\
                            --max_turns 5

