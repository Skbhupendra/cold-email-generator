# Cold Email Generator

A lightweight Streamlit app that turns a job posting URL into a tailored cold email draft using AI and portfolio matching.

## Overview

This project helps automate the first step of job outreach by:

- reading a public job posting URL
- extracting the role, skills, and description
- identifying relevant portfolio examples based on technology stack
- generating a concise email draft for the hiring team
- letting the user review and edit the draft before sending

## What this app does

- Parses the job description from a webpage
- Extracts contact emails mentioned in the posting
- Matches relevant portfolio entries from a CSV dataset
- Uses a Groq LLM to create a draft cold email
- Keeps the company field optional, so it is hidden from the email when left blank

## Tech stack

- Python
- Streamlit
- LangChain
- Groq API
- ChromaDB
- Pandas
- BeautifulSoup / WebBaseLoader

## Repository structure

```text
cold-email-generator/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── docs/
│   └── screenshots/
│           └──generated-email.png
├── data/
│   └── portfolios.csv
├── notebooks/
│   └── cold-email-generator.ipynb
```

> The local virtual environment and generated runtime folders should not be committed to GitHub.

## Requirements

- Python 3.10+
- A Groq API key
- Internet access to fetch job-posting pages

## Setup

1. Clone the repository.
2. Create a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a local environment file:

```bash
copy .env.example .env
```

On macOS/Linux:

```bash
cp .env.example .env
```

5. Add your Groq API key in `.env`:

```env
GROQ_API_KEY=your_api_key_here
```

## Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How to use it

1. Paste a job posting URL.
2. Add your name.
3. Optionally add a company name.
4. Click “Generate email”.
5. Review the draft and edit it if needed.
6. Download the output as a text file.

## Important notes

- The company field is optional.
- If the field is empty, it is not included in the generated draft.
- The project uses a local portfolio dataset for matching relevant projects to the role.

## License

This project is intended for personal or internal use unless a different license is specified.

## Future ideas

- Improve portfolio matching quality
- Add more job source support
- Add tests for prompt and email generation
- Add export to email clients or templates
- Improve UI polish and error handling
