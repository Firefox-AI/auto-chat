# auto-chat
Repository for creating agent-agent conversations. 

There are two ways to run this repository. The first way, is to have the handler agent (GPT-5) come up with the scenario. This will include generating a user profile, a set of user interests and a goal that the handler agent is trying to accomplish. 

The default behavior is for the agent to continue to the conversation until its goal has been accomplished. We also provide a flag `--max_turns` to control how long the conversation can continue (in the case that the model is unable to accomplish the agent's goal). 

Another useful flag is `max_tool_calls` which controls the number of consecutive tool calls that can be made without the agent responding. The default is 5, but that might be too low for some scenarios that require looking at the user's open tabs and reading the pages.

### Setup ###
This package is mostly python based, but to replicate the Firefox browser, we use `readability.js` for webpage parsing. In addition, we use `playwright` as a headless browser to navigation. To setup your environment, run the following commands:

1. install node dependencies (`readability.js`)
``` bash
npm install
```

2. install python dependencies
``` bash
uv sync
```

3. install playwrite browsers
``` bash
uv run playwright install
```

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
                            --conversation_file conversations/haunted_dinner_menu.yaml\
                            --max_turns 5

