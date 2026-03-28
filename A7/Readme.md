# A7: MCP-Server, AI Agent, and External Tool Integration

**Course:** AT82.05 Artificial Intelligence: Natural Language Understanding (NLU)  
**Assignment:** A7 — MCP-Server, AI Agent, and External Tool Integration

---

##  Overview

This assignment builds an integrated AI Agent ecosystem using the **Model Context Protocol (MCP)**. The agent is capable of managing real-world schedules and communicating via Telegram, demonstrating practical Natural Language Understanding (NLU).

The system is deployed locally using **Docker (n8n)** and exposed to the internet via **ngrok**.

---

##  Tools & Technologies

| Tool | Purpose |
|------|---------|
| [n8n](https://n8n.io/) | Workflow automation platform |
| [Docker](https://www.docker.com/) | Run n8n locally |
| [ngrok](https://ngrok.com/) | Expose local n8n to the internet |
| [Groq](https://console.groq.com/) | Free LLM API (llama-3.3-70b-versatile) |
| [Telegram Bot API](https://core.telegram.org/bots/api) | Messaging interface |
| [Google Calendar API](https://developers.google.com/calendar) | Schedule management |
| MCP (Model Context Protocol) | Tool integration protocol |

---

##  Setup Instructions

### Prerequisites
- Docker installed
- ngrok account and binary
- Groq API key (free at https://console.groq.com)
- Telegram account + BotFather bot token
- Google account with Calendar API enabled

### Step 1 — Start n8n with Docker
```bash
docker run -it --rm --name n8n -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  -e WEBHOOK_URL=https://YOUR-NGROK-URL.ngrok-free.dev \
  docker.io/n8nio/n8n
```

### Step 2 — Start ngrok tunnel
```bash
ngrok.exe http 5678
```
Copy the Forwarding URL (e.g. `https://xxxx.ngrok-free.dev`) and use it as your `WEBHOOK_URL` above.

### Step 3 — Access n8n
Open your browser and go to:
```
http://localhost:5678
```

---

## Task 1: MCP Infrastructure & Server Setup

### MCP Server Workflow
- Created an n8n workflow acting as an **MCP Server**
- Implemented an **MCP Server Trigger** with 3 internal tools:
  - **Calculator** — performs mathematical calculations
  - **Date & Time** — retrieves current date and time
  - **Code Tool** — formats and processes text
- Published the workflow to generate a Production URL

### AI Agent Workflow
- Created a separate **AI Agent** workflow with:
  - **Chat Trigger** — receives user messages
  - **Groq Chat Model** — LLM using `llama-3.3-70b-versatile`
  - **Simple Memory** — maintains conversation context
  - **MCP Client** — connects to the MCP Server Production URL
- Verified the agent can use MCP tools via the n8n chat interface

---

##  Task 2: Telegram & Google Calendar Integration

### Telegram Agent Workflow
- Created a **Telegram Agent** workflow with:
  - **Telegram Trigger** — receives messages from Telegram bot
  - **AI Agent** — processes messages using Groq LLM
  - **Simple Memory** — maintains context per chat session
  - **MCP Client** — access to MCP tools
  - **Google Calendar (Create)** — creates calendar events
  - **Google Calendar (Get Many)** — reads calendar events
  - **Telegram Send Message** — replies back to user

### Project Scheduling Test
The agent was commanded via Telegram to create a 4-phase project schedule:

| Phase | Event | Date |
|-------|-------|------|
| 1st | Literature Review | April 6, 2026 |
| 2nd | Project Proposal | April 7, 2026 |
| 3rd | Update Progress | April 8, 2026 |
| 4th | Final Presentation | April 9, 2026 |

All 4 events were successfully created in Google Calendar via a single Telegram command.

---

##  Configuration Notes

- **Memory:** Used `Simple Memory` instead of `Window Buffer Memory` as n8n's newer version consolidates memory options
- **Session ID:** Set to `{{ $('Telegram Trigger').item.json.message.chat.id }}` for per-user context
- **Webhook URL:** Must use ngrok HTTPS URL (not localhost) for Telegram webhooks to work
- **Google OAuth:** Created via Google Cloud Console with ngrok callback URL

---

##  Screenshots

See the `/screenshots` folder (or report PDF) for:
1. ngrok tunnel running
2. MCP Server workflow
3. AI Agent workflow with n8n chat verification
4. Telegram Agent workflow
5. Telegram conversation — project schedule command
6. Google Calendar showing all 4 events

---

##  Known Limitations

- ngrok free tier generates a new URL on each restart — Docker must be restarted with the new `WEBHOOK_URL`
- Groq free tier has token-per-minute limits — wait ~1 minute if rate limited
- Timezone offset between n8n and Google Calendar may cause event times to appear shifted

---

## References

- [n8n Local Setup](https://github.com/chaklam-silpasuwanchai/Python-fo-Natural-Language-Processing/tree/main/Code/11%20-%20Agentic%20AI/local-n8n)
- [Groq Console](https://console.groq.com/)
- [Typhoon AI](https://playground.opentyphoon.ai/)
- [n8n MCP Documentation](https://docs.n8n.io/)