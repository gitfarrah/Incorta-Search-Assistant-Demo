# Workspace AI Search Assistant

A Streamlit application that searches Slack and Confluence, then uses Google Gemini to generate an intelligent answer with citations.

## Features

- Search Slack messages (user token, not bot) and Confluence pages/blogs
- Parallel retrieval and unified context formatting
- Gemini synthesis with inline citations
- Expanders to view raw sources
- Input validation, basic error handling, and logging

## Tech Stack

- Frontend: Streamlit
- APIs: `slack_sdk`, `atlassian-python-api`, `google-generativeai`
- Python: 3.9+

## Prerequisites

- Python 3.9+
- Slack user token (`xoxp-`) with scopes:
  - `channels:history`, `channels:read`, `search:read`, `users:read`, `groups:history`, `im:history`
- Atlassian Confluence Cloud account and API token
- Google AI Studio API key (Gemini)

## Setup

1. Clone this repository and enter the directory.
2. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and populate values:

```bash
cp .env.example .env
```

Fill in:

- `SLACK_USER_TOKEN=xoxp-...`
- `CONFLUENCE_URL=https://your-domain.atlassian.net`
- `CONFLUENCE_EMAIL=you@example.com`
- `CONFLUENCE_API_TOKEN=...`
- `GEMINI_API_KEY=...`

## Running Locally

```bash
streamlit run app.py
```

Open the provided local URL in your browser.

## Usage

- Enter a question in the text area.
- Optionally set filters in the sidebar (date hints, Slack channel hint, Confluence space hint).
- Click Search. The app will:
  - Search Slack and Confluence in parallel
  - Merge results into a context
  - Ask Gemini for a concise answer with citations
  - Show the answer and expandable raw sources

## Deployment (Streamlit Cloud)

1. Push this project to a Git repository.
2. In Streamlit Cloud, create a new app pointing to your repo.
3. Configure Secrets (same keys as `.env`):

```toml
SLACK_USER_TOKEN="xoxp-..."
CONFLUENCE_URL="https://your-domain.atlassian.net"
CONFLUENCE_EMAIL="you@example.com"
CONFLUENCE_API_TOKEN="..."
GEMINI_API_KEY="..."
```

4. Deploy. The app should launch automatically.

## Slack App Notes

- Use a user token (`xoxp-`), not a bot token (`xoxb-`).
- Ensure the token’s workspace membership matches expected visibility.
- Required scopes: `channels:history`, `channels:read`, `search:read`, `users:read`, `groups:history`, `im:history`.

## Confluence Notes

- Generate an API token from your Atlassian account security settings.
- `CONFLUENCE_URL` must match your Cloud base URL, e.g., `https://company.atlassian.net`.

## Gemini Notes

- Create an API key from Google AI Studio.
- The app prefers `gemini-1.5-pro` and falls back to `gemini-pro`.

## Troubleshooting

- Missing env vars: The app shows a warning banner listing them.
- Authentication errors: Validate tokens and permissions; check Streamlit logs.
- No results: Try simpler queries or remove filters; ensure the user token can access channels/spaces.
- Rate limiting: The app has basic retries for Gemini; Slack/Confluence errors are logged.

## File Structure

- `app.py`: Streamlit app entrypoint
- `slack_search.py`: Slack search via `search.messages`
- `confluence_search.py`: Confluence CQL search
- `gemini_handler.py`: Prompting Gemini and returning answer
- `requirements.txt`: Dependencies
- `.env.example`: Environment variable template
- `README.md`: This file

## Security

- Never commit `.env` (ignored by `.gitignore`).
- Do not hardcode credentials. Use environment variables or Streamlit Secrets.

## License

Internal demo; add a license as needed for your org.
