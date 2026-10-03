import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# Load environment variables (.env) for OPENAI_API_KEY and SERPER_API_KEY
load_dotenv()

# Initialize the search tool
search = SerperDevTool()

# 1. Define the Researcher Agent (Has the search tool)
researcher = Agent(
    role="Senior Tech News Researcher",
    goal="Discover the top 3 tech news stories of yesterday and summarize their core facts accurately.",
    backstory=(
        "You are an expert investigative tech journalist with an eye for breaking news, "
        "AI developments, and emerging tech breakthroughs. You find verified, timely information."
    ),
    tools=[search],
    verbose=True
)

# 2. Define the Writer Agent (No search tools, relies entirely on the Researcher's context)
writer = Agent(
    role="Tech Newsletter Editor",
    goal="Synthesize raw research notes into a sharp, engaging 5-line newsletter.",
    backstory=(
        "You are a concise newsletter writer known for high-signal, zero-fluff editorial style. "
        "You never need to research on your own; you excel at turning notes into polished copy."
    ),
    verbose=True
)

# 3. Define the Finding Task
find = Task(
    description=(
        "Search the web for yesterday's top 3 major technology news stories. "
        "Identify the main headline, key details, and significance of each story."
    ),
    expected_output=(
        "A structured summary containing the 3 stories, each with a brief 1-2 sentence "
        "overview and the source/context."
    ),
    agent=researcher
)

# 4. Define the Writing Task (Takes the 'find' task as context)
write = Task(
    description=(
        "Using only the provided research, compose a concise yesterday's tech newsletter. "
        "The format must be exactly 5 lines total: 1 catchy headline line, followed by "
        "3 single-line bullet points (one for each story), and 1 single closing takeaway line."
    ),
    expected_output=(
        "Exactly 5 lines of text with no extra blank lines, headers, or intro/outro text. "
        "Line 1: Headline. Lines 2-4: Bullet summaries. Line 5: Closing takeaway."
    ),
    agent=writer,
    context=[find] # This explicitly hands the Researcher's output to the Writer
)

# 5. Assemble the Crew and Run sequentially
crew = Crew(
    agents=[researcher, writer],
    tasks=[find, write],
    process=Process.sequential,
    verbose=True
)

print("\n--- Starting Newsletter Crew ---")
result = crew.kickoff()

print("\n--- Final Newsletter Output ---")
print(result)