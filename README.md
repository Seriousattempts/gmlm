# GameMaker Language Methods
### *Based on the GameMaker manual, "accessed" on ‎Wednesday, ‎August ‎28, ‎2024, ‏‎10:56:38 PM*

These are my ideas based on a thought I had that one days I can afford to make the list of games and romhacks I've written down. Then I watched my first WAN Show https://youtu.be/Luz82RG5PqA. I've spent a few years thinking, still writing down ideas while slowly getting better at accessing models for project creations. Then around this time last year, I started converting the manual towards the idea of fine-tuning a model https://www.reddit.com/r/LocalLLaMA/. Watching and reading how people layer upon their LLMs with their data. But after using sites like Perplexity and Phind, I realize that if you tell the right bot to use specific data, it will actually use that instead of it's internal data more concurrently. As LLMs get better, why should I fine-tune for an old model that will be outdated by the time you read this sentence in expected features? Vision models are getting better, vocal, diffusion, etc. Just have those models use a more processable feed of data. Back to school

### What do you remember from school?
You're in school, taking notes. You use those notes to learn the subject, take a test to prove you know the subject.
- Convert your notes on the subject to a known parsing format https://github.com/Seriousattempts/gmlm/blob/main/Notes/Maths_And_Numbers_references.xlsx

Decide what to include. For example, would you copy "DEPRECIATED" functions? How long do you intend to know the material you wrote down? Just for the semester? Career?
You're not perfect, neither is the material. Spell check, proof read, etc. https://github.com/Seriousattempts/gmlm/blob/main/Notes/Spelling%20error%20checker.py
- Turn to accessible data of your choosing. I choose .jsonl, .npy, .txt
  1. https://github.com/Seriousattempts/gmlm/blob/main/Notes/Excel%20Data%20Conversion.py
  2. https://github.com/Seriousattempts/gmlm/blob/main/Notes/Excel%20Folder%20Data%20Conversion.py

Think of possible shortcomings when accessing data. You're creating the format to be digestible to recreate. Don't forget to organize that data. But now that you got this, what can you do with this? Well to me it depends on your current state of using GameMaker.
- From my short perspective if you're a big proponent of using GML, interested in automating task, then you're just a few workflows away from seeing some crazy stuff with the right idea.

## Basics asking an LLM for assistance
- 2023 sample example: https://www.youtube.com/watch?v=g-azEjhzTyk
- 2023 manual paired with bots on POE https://github.com/Seriousattempts/gmlm/blob/main/POE/poe_usecase.md

Now instead of current RAG techniques, and because there's so many files that come with a manual, lets use the .jsonl, .npy, .txt. Surface level, it seems easier to parse than thousands of files
- RapidAPI: https://rapidapi.com/user/Seriousattempts
- numpy: https://github.com/Seriousattempts/gmlm/tree/main/numpy

Now what can we do with it? [But I decided to test software to think of possible workflow integrations for a week](https://www.youtube.com/watch?v=lX_JabC6I5o). Here's some workflow assumptions paired with half baked ideas
- Discord | https://github.com/Seriousattempts/gmlm/blob/main/Discord/discord_usercase.md
- Lovable | https://github.com/Seriousattempts/gmlm/blob/main/LOVABLE/lovable_usecase.md
- Model Context Protocol | https://github.com/Seriousattempts/gmlm/blob/main/MCP/mcp_usecase.md
- n8n | https://github.com/Seriousattempts/gmlm/tree/main/n8n

How about something more daring if you're the type to let your computer be accessed by LLMs through your device (Local or API, you're clearly a LMSoaker)
## DIAMBRA https://github.com/diambra
- Diambra Reinforcement Learning Agents https://github.com/diambra/agents
- POE https://poe.com/DIAMBRAArena

Can your test your game with an LLM bot?
Can your game be played with an LLM bot (why would you?)

Think of multiplayer games that's unplayable in some circumstance
- No Wifi access, so you're playing locally
- No one to play with
  1. Could you make your own bot to play with you? Or you could be a more civil coder and just make your own in game algorithm from scratch (It's called VS CPU).

*Note Diambra has their own* crypto *economy*



