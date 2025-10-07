import pandas as pd
from datasets import load_dataset
import os
import subprocess
import yaml
import json

## model providers
import openai
from openai import OpenAI
import together
from together import Together

import requests
import tempfile
import fitz 
from ast import literal_eval

import prompts as p

import asyncio
from playwright.async_api import async_playwright
from tqdm.asyncio import tqdm_asyncio
from pydantic import BaseModel, Field
from typing import List
from absl import app, flags
from absl import logging as absl_logging

# absl_logging.set_verbosity(absl_logging.ERROR)

FLAGS = flags.FLAGS
flags.DEFINE_string("model", "together.ai:Qwen/Qwen3-Next-80B-A3B-Thinking", "Name of model to evaluate")
flags.DEFINE_string("eval_model_id", "gpt-5", "Name of judge model (OpenAI assumed)")
flags.DEFINE_string("conversation_file", "conversations/san_diego_trip.yaml", "path to conversation file")
flags.DEFINE_string("output_dir", "data", "Location into which output data will be saved")
flags.DEFINE_integer("max_turns", 5, "Maximum number of turns this conversation is allowed to have")
flags.DEFINE_integer("max_tool_calls", 5, "Maximum number of tool calls an agent is allowed to make before getting cut off")

client_oa = OpenAI()    # auth is at os.environ["OPENAI_API_KEY"]
client_tg = Together()  # auth is at os.environ["TOGETHER_API_KEY"]
client_groq = OpenAI(
    api_key=os.environ['GROQ_API_KEY'],
    base_url="https://api.groq.com/openai/v1",
)


## tools
def open_tab(url):
    return "SUCCESS!"


def close_tab(url):
    return "SUCCESS!"


async def get_page_contents(url, title=""):
    try:
        html = await get_html(url)
        d = extract_readable_article(html=html)

        return d['textContent']
    except Exception as e:
        return f"Page Error: could not retrienve page contents. (details: {e})"


def default(*args, **kwargs):
    return "ERROR in function call"


def get_tools():
    with open("tools.yaml", "r") as f:
        tools = yaml.safe_load(f)
    return tools


def get_convo_data(convo_file):
    with open(convo_file, "r") as f:
        convo_data = yaml.safe_load(f)
    return convo_data


def get_response(messages, provider, model_id, tools, tool_choice="auto"):

    match provider:
        case "together.ai":
            response = client_tg.chat.completions.create(
                model=model_id,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
        case "openai":
            response = client_oa.chat.completions.create(
                model=model_id,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice
            )
        case "groq":
            response = client_groq.chat.completions.create(
                model=model_id,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice
            )
        case "vertex":
            response = client_vertex.chat.completions.create(
                model=model_id,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice
            )
    
    return response


def get_pdf(r):
    output = []
    with fitz.open(stream=r.content, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            output.append(page.get_text())

    return "\n".join(output)

def extract_readable_article(html: str = "", url: str = "https://example.com", timeout=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MyScraperBot/1.0; +http://example.com/bot)"
    }

    text = ""
    if not html:
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
        except Exception as e:
            print("EXCEPT: ", e)
            return {}
        if r.status_code == 200:
            if r.text.startswith('%PDF'):
                text = get_pdf(r)
        
        html = r.text
        print(html)

    with tempfile.NamedTemporaryFile(suffix=".html", mode='w+', delete=False) as f:
        f.write(html)
        f.flush()
        result = subprocess.run(
            ["node", "extract_readability.js", url, f.name],
            capture_output=True, text=True
        )
    if result.returncode != 0:
        raise RuntimeError("Readability script failed: " + result.stderr)
        return
    seen_result = json.loads(result.stdout)
    if text:
        seen_result['textContent'] = text
    return seen_result


def make_system_prompt(tabs, prefs):
    return p.CHAT_AGENT_SYSTEM_PROMPT_P1 + tabs + p.CHAT_AGENT_SYSTEM_PROMPT_P2 + prefs + p.CHAT_AGENT_SYSTEM_PROMPT_P3


async def get_html(url):
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        page = await browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        await page.goto(url, timeout=60000)
        content = await page.content()
        # print(content)
        await browser.close()
    return content


def get_agent_response(messages, profile, task, tabs):
    m = [
        {"role": "system",
         "content": p.AGENT_HARNESS_PROMPT.format(profile=profile, task=task, tabs=tabs, messages=messages)}
    ]

    response = get_response(m, 'openai', 'gpt-5', tools=[], tool_choice=None)

    resp = response.choices[0].message.content

    if resp is None:
        print("ERROR in agent: ", response)
    return resp


async def invoke_tool_call(tc, tool_registry):
    # input: toolcall object from response
    
    # tc.function.name
    # tc.function.arguments
    func = tc.function.name
    args = literal_eval(tc.function.arguments)
    if func == "get_page_contents":
        return await get_page_contents(**args)
    return tool_registry.get(func, "default")(**args)


async def do_chat(profile, task_description, tabs, system_prompt, first_user_prompt, tool_registry, provider, model_id, tools, max_turns, max_tool_calls):
    messages = [
        {
            "content": system_prompt,
            "role": "system"
        },
        {
            "content": first_user_prompt,
            "role": "user"
        },
    ]

    print(messages)

    tokens = []
    ctr = 0
    tool_count = 0

    while True:    
        response = get_response(messages, provider, model_id, tools)
        tokens.append(response.usage.model_dump())
        print(response.usage)
        
        print(response.choices[0].message.content, response.choices[0].message.tool_calls, end="\n\n")
        messages.append(response.choices[0].message.model_dump())
        
        if (response.choices[0].message.tool_calls is not None) and len(response.choices[0].message.tool_calls):
            for tool_call in response.choices[0].message.tool_calls:
                tool_count += 1
                fcn_id = tool_call.id
                
                resp = await invoke_tool_call(tool_call, tool_registry)
                # resp = input("Tool response: ")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": fcn_id,
                    "content": json.dumps(resp),
                    "name": tool_call.function.name
                })

                if tool_count >=max_tool_calls:
                    print("Maximum tool call limit reached.")
                    messages.append({"role": "system", "content": "Maximum tool call limit reach. Ending chat"})
                    return messages, tokens

        else:
            # resp = input("Response: ")
            tool_count = 0
            resp = get_agent_response(messages, profile=profile, task=task_description, tabs=tabs)
            ctr += 1
            print("AGENT RESPONSE: ", resp)
            print("\n")

            # this allows the user to "open a tab" and then reference it to the model
            # eg see update3 below -- after this the user can say, "I opened the menus for luigis and beres park pies
            if resp.startswith("SYS:"):
                messages.append(
                    {"role": "system",
                    "content": resp[4:]
                    }
                )
                resp = input("Response: ")
                print("\n")
            
            if resp == 'q' or ctr > max_turns:
                break
            messages.append(
                {"role": "user",
                "content": resp
                }
            )
    
    return messages, tokens


async def _main(_):
    print("Calling _main")
    provider, model_id = FLAGS.model.split(":")
    model_id_simple = model_id.split("/")[-1]
    print(" | ".join([provider, model_id, model_id_simple]))

    tools = get_tools()
    conversation_data = get_convo_data(FLAGS.conversation_file)

    TABS = json.dumps(conversation_data['open_tabs'], indent=3)
    PREFS = json.dumps(conversation_data['user_preferences'], indent=3)

    system_prompt = make_system_prompt(TABS, PREFS)
    print(system_prompt)

    def get_tabs():
        return TABS

    def search_history(search_term, *args, **kwargs):
        return "No history found"

    def get_preferences(query=None, *args, **kwargs):
        return PREFS

    TOOL_REGISTRY = {
        "open_tab": open_tab,
        "close_tab": close_tab,
        "get_page_contents": get_page_contents,
        "search_history": search_history,
        "get_tabs": get_tabs,
        "get_preferences": get_preferences,
        "default": default
    }

    messages, tokens = await do_chat(
        profile=conversation_data["user_profile"], 
        task_description=conversation_data["task_description"], 
        tabs=TABS, 
        system_prompt=system_prompt, 
        first_user_prompt=conversation_data["first_user_prompt"], 
        tool_registry=TOOL_REGISTRY,
        provider=provider,
        model_id=model_id,
        tools=tools,
        max_turns=FLAGS.max_turns,
        max_tool_calls=FLAGS.max_tool_calls
        )
    
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    convo_name, _ = os.path.splitext(os.path.basename(FLAGS.conversation_file))

    with open(os.path.join(FLAGS.output_dir, f"{convo_name}.json"), 'w') as f:
        json.dump(messages, f)
    with open(os.path.join(FLAGS.output_dir, f"{convo_name}_tokens.json"), 'w') as f:
        json.dump(messages, f)


def main(_):
    print("Starting")
    asyncio.run(_main(None))

if __name__ == "__main__":
    app.run(main)
