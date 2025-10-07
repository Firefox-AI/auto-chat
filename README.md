# auto-chat
Repository for creating agent-agent conversations

### Example usage
``` bash
uv run python start_chat.py --model together.ai:Qwen/Qwen3-235B-A22B-Instruct-2507-tput\
                            --eval_model_id openai:gpt-5\
                            --conversation_file conversations/san_diego_trip.yaml\
                            --max_turns 5
