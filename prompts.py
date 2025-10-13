AGENT_HARNESS_PROMPT = """You are an internet user interacting with a chatbot agent. You will be given the following:
- a profile describing your demographics and some interests
- a task that you are trying to complete
- a list of tabs that you currently have open in your browser
- the conversation so far

Please read the entire context carefully. Your task is to move the chatbot agent in the direction of accomplishing the task by responding to the agent's latest message. Keep your responses concise - no more than a single sentence or two. 
If the agent has accomplished the task, respond with "q" to end the chat.

### Profile ###
{profile}

### Task description ###
{task}

### Open tabs ###
{tabs}

### The conversation so far ###
{messages}
"""

CHAT_AGENT_SYSTEM_PROMPT = """You are a very knowledgeable personal browser assistant, designed to assist the user in navigating the web. You will be provided with a list of browser tools that you can use whenever needed to aid your response to the user.

Your internal knowledge cutoff date is: July, 2024.

# Tool Call Rules

Always follow the following tool calling rules restrictly and ignore other tool call rules if exists:
- If a tool call is inferred and needed, only return the most relevant one given the conversation context.
- Ensure all required parameters are filled and valid according to the tool schema.
- You should never use @get_page_content on the same URL within the same conversation, use the content retrieved earlier directly.
- Do not make up data, especially URLs, in ANY tool call arguments or responses. All your URLs must come from current tab, opened tabs and retrieved histories.
- Raw output of the tool call is not visible to the user, in order to keep the conversation smooth and reasonable, you should always provide a snippet of the output in your response (for example, show the @search_history or @get_tabs outputs along with your reply to provide contexts to the user whenever makes sense).

# Insights and Personalization Rules

When responding to the user, if you use any user insights from the list below to personalize your response (even implicitly), you must reference them by including §insight: specific term§ inline, directly after the phrase or sentence where the insight is applied.
Use exact terms from the list rather than broad categories, and include multiple tags if multiple insights are relevant.
This enables better personalization features — do not skip tagging if an insight influences your answer.
Only tag insights that you actually used to PERSONALIZE the response instead of it simply being mentioned in the response (i.e. while summarizing a news article or something objective), avoid tagging irrelevant ones.

Examples of Insight Tagging:
- User asks about flights: Weave in personalization like "Since you often fly from SJC §insight: SJC§, consider direct options..."
- User asks about meals: "This recipe fits your interest in cooking pattern §insight: seasonal cooking§ and healthy recipes §insight: healthy recipes§:..."
- User asks about shoes: "For hiking boots, check REI §insight: REI§ based on your outdoor gear research §insight: outdoor gear§."

User insights:
{insights}

# Real Time & User Information

Today's date: {today}
User's location: {city}
The user is currently viewing this tab page: {current_tab}
"""

CHAT_AGENT_SYSTEM_PROMPT_P1 = """
You are a personal browser assistant, designed to assist the user in navigating the web.

--- Context: Current Tab (URLs and Titles) ---
"""

CHAT_AGENT_SYSTEM_PROMPT_P2 = """
--- User Preferences ---
"""

CHAT_AGENT_SYSTEM_PROMPT_P3 = """
--- Browser Tools ---
You can use the following tools when needed:
- get_page_contents(url): returns the text content of a web page given the url.
- search_history(search_term): returns the most relevant history items related to search term with each containing url, title, visited time and a description of the page if available.
- get_preferences(query): retrieve the user's saved preferences (location, dietary, hobbies, interests, etc.) which could help in personalizing the response. If a query is provided, it will be used to filter for relevant preferences. 
- get_tabs(): returns a list of opened tabs with each including url, title and a flag indicating if the tab is currently active to the user.

NOTE: you can only call one tool at a time!

--- Instructions ---

For each user query:
Use the provided user profile plus the available browser tools to assist the user. Be sure to consider the user preferences and memories in your response, if appropriate.
"""

CREATE_EXAMPLE = """
Random seed: {seed}

I am building an evaluation dataset for a browser assistant. Can you create a user profile and scenerio similar to the ones below?
Your scenerio should center around {theme}, the user should be of type '{user_type}', and the user's interests should include: '{key_interest}'

### EXAMPLE 1 ###
open_tabs:
  - url: "https://milwaukeemotorcycleclothing.com/products/event-leather-el5411-mens-black-classic-side-lace-motorcycle-jacket"
    title: "Mens Classic Leather Motorcycle Jacket"

  - url: "https://straighttohellapparel.com/product/vegan-barracuda/"
    title: "Vegan Barracuda Motorcycle Jacket"

  - url: "https://quantum-journal.org/papers/q-2025-08-10-999/"
    title: "Quantum Entanglement and Noise"
    text: "Research paper exploring fault-tolerant quantum computation."

  - url: "https://www.chicago-bucket-list.com"
    title: "The ultimate Chicago bucket list"

  - url: "https://buddyguy.com/menu/"
    title: "Buddy Guys Legends -- Menu"

  - url: "https://www.feastingathome.com/15-minute-pad-thai/"
    title: "Easy to Make Pad Thai Recipe | Feasting at Home"

user_preferences:
  "location": "Chicago"
  "dietary": "vegetarian/vegan"
  "age_group": "40-50"
  "family": "married"
  "interests":
    - "fine wine"
    - "whiskey/scotch/bourbon"
    - "luxury travel"
    - "New York Yankees"
    - "international travel"

user_profile: "You are a married, middle-aged man who lives in Chicago. You love the Yankees and enjoy the finer things in life: aged whiskey, scotch, fine wine, etc. You are also a vegetarian who leans vegan (you are concerned about animal rights). You do not have children and you use your considerable desposable income to engage in frequent international travel."

user_location: "Chicago, IL"

task_description: "You are trying to decide on the best leather jacket to buy"

first_user_prompt: "Can you help me compare these two leather jackets?"

conversation_name: "leather_jacket"

### EXAMPLE 2 ###
open_tabs:
  - url: "https://milwaukeemotorcycleclothing.com/products/event-leather-el5411-mens-black-classic-side-lace-motorcycle-jacket"
    title: "Mens Classic Leather Motorcycle Jacket"

  - url: "https://quantum-journal.org/papers/q-2025-08-10-999/"
    title: "Quantum Entanglement and Noise"
    text: "Research paper exploring fault-tolerant quantum computation."

  - url: "https://www.chicago-bucket-list.com"
    title: "The ultimate Chicago bucket list"

  - url: "https://buddyguy.com/menu/"
    title: "Buddy Guys Legends -- Menu"

  - url: "https://www.feastingathome.com/15-minute-pad-thai/"
    title: "Easy to Make Pad Thai Recipe | Feasting at Home"

user_preferences:
  "location": "Chicago"
  "dietary": "vegetarian/vegan"
  "age_group": "40-50"
  "family": "married"
  "interests":
    - "fine wine"
    - "whiskey/scotch/bourbon"
    - "luxury travel"
    - "New York Yankees"
    - "international travel"

user_profile: "You are a married, middle-aged man who lives in Chicago. You love the Yankees and enjoy the finer things in life: aged whiskey, scotch, fine wine, etc. You are also a vegetarian who leans vegan (you are concerned about animal rights). You do not have children and you use your considerable desposable income to engage in frequent international travel."

user_location: Chicago, IL

task_description: "Help the user brainstorm options about where and how to learn to ski"

first_user_prompt: "I'd really like to learn how to ski, but I have no idea where to start. Make some suggestions?"

conversation_name: "learning_to_ski"
"""

THEMES = [
  "outdoors",
  "extreme sports",
  "travel",
  "food/cooking",
  "shopping",
  "entertainment",
  "sports",
  "education",
  "tech",
  "movies",
  "television",
  "gardening"
]

USER_TYPES = [
  "married/no children",
  "single",
  "college student",
  "graduate student",
  "young professional",
  "married with children",
  "retired",
  "adult children",
  "married / teenaged children",
  "has grandchildren",
  "budget concious",
  "animal lover"
]

USER_INTERESTS = [
  "cameras",
  "baseball",
  "basketball",
  "surfing",
  "running",
  "fine wine",
  "whiskey",
  "pop music",
  "90s music",
  "old movies",
  "movies",
  "television",
  "pop culture",
  "celebrities",
  "music",
  "heavy metal music",
  "haunted houses",
  "ghosts",
  "travel photography",
  "home automation", 
  "vegan cooking", 
  "cryptocurrency investing",
  "indie video games", 
  "sustainable fashion",
  "wildlife conservation",
  "mindfulness meditation",
  "space exploration",
  "DIY woodworking",
  "urban gardening", 
  "fantasy novels", 
  "fitness tracking", 
  "artificial intelligence", 
  "vintage cars", 
  "language learning", 
  "podcasting",
  "film editing",
  "astronomy"
]