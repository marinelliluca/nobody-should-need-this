# Using the Job Hunt apps

This is a guide for using the two job-hunting apps in your browser. You don't need to write any code or understand what's happening behind the scenes. You just type one short command to start each app, and everything after that happens on a normal-looking web page.

There are two apps:

- **Find Jobs** — searches lots of job boards at once and collects the results into one file.
- **Match Jobs** — takes that file plus your CV and ranks the jobs by how well they fit you.

Find Jobs gives you a results `*.parquet` file, **SAVE IT!** Match Jobs reads that file.

---
## Opening an app

You start each app with one short command, then use it in your browser. You do this once per session.

> **First time ever?** The tool needs Python installed and a `.env` file holding the access keys (a token for searching jobs, plus the AI settings). If that hasn't been done yet, the apps will show an orange warning box and won't work. See [Before your first run](#before-your-first-run) below.

To open an app:

1. Open a **Terminal** (Mac and Linux: the app called "Terminal"; Windows: "Command Prompt" or "PowerShell").
2. Go to the tool's folder: type `cd ` (with a space after it), drag the tool's folder into the window, and press **Enter**.
3. Copy-paste **one** of these lines and press **Enter**:
   - Find Jobs: `python -m interface.scraper_app`
   - Match Jobs: `python -m interface.rag_app`
4. It will print a web address like `http://localhost:7860`. Open that in your browser (Chrome, Firefox, Safari — whatever you use). That's the app.
5. When you're done, click back on the Terminal and press **Ctrl + C** to close the app.

You run one app at a time, which fits the normal flow anyway: open Find Jobs, get your results file, close it, then open Match Jobs.

If you ever see an **orange warning box** at the top mentioning "environment variables," the access keys in the `.env` file aren't set. See [Before your first run](#before-your-first-run) for how to fix it.

---

## Before your first run

You only do this part once, ever. If the apps already open and don't show an orange warning box, skip straight to the apps below.

If the terminal steps above feel unfamiliar, two short, trustworthy guides cover the background:

- **What a terminal is and the handful of commands you'll use** (like `cd`): Real Python — [The Terminal: First Steps and Useful Commands](https://realpython.com/terminal-commands/).
- **Installing Python and running things from the terminal**: the official [Python For Beginners](https://www.python.org/about/gettingstarted/) page on python.org, with downloads at [python.org/downloads](https://www.python.org/downloads/) (tick **"Add Python to PATH"** in the installer).

The tool itself needs three things in place:

1. **Python installed** — check by opening a Terminal and typing `python --version` (or `python3 --version`). A version number means you're set; an error means install it using the links above.
2. **The tool's components installed** — once, from inside the tool's folder, run `pip install -r requirements.txt`. (See Real Python — [How to Run Your Python Scripts](https://realpython.com/run-python-scripts/) if `pip` or the `-m` commands are new to you.)
3. **A `.env` file with access keys** — the tool reads a small text file named `.env` for the job-search token and AI settings. There's a template named `.env.example` in the folder: copy it to `.env` and fill in the values. Without it, the apps show the orange warning box and won't run.

Once those three are done, every future session is just the [Opening an app](#opening-an-app) steps.

---

## App 1 — Find Jobs

This app runs a list of searches and gathers everything it finds into one results file.

1. **Look at the list of searches.** The form starts with a ready-made list of job searches (things like "data scientist", "machine learning engineer"). You can:
   - **Edit** any line to change what it searches for.
   - **Remove** lines you don't want.
   - **Add** new lines for other roles you're curious about.
2. **Set the basics** — the location (e.g. Berlin), the country, and how many results you want per search. The defaults are usually fine.
3. **Click Scrape** (the run button). Each search runs one at a time. You'll see progress text appear as it works. If one search fails, the others still keep going, so you'll still get results.
4. **Download your results** when it finishes. You'll get a couple of files to save — most importantly a **`.parquet`** file. **Keep this file somewhere you can find it.** This is what the Match Jobs app needs.

> **Tip:** more searches and a higher result count mean a longer wait. If you just want to try it out, trim the list down to two or three searches first.

---

## App 2 — Match Jobs

This app reads the results file from App 1, looks at your CV and ranks every job by fit. It works in steps, top to bottom on the page.

Before beginning make sure you converted your CV to `.txt` or markdown `.md` ([example CV](example_cv.md)).

1. **Upload your jobs file.** Use the `.parquet` file you downloaded from Find Jobs. The app will confirm it loaded correctly.
2. **Set up your CV profile.** This is a short summary of what you're looking for. You have two options:
   - **Auto-create it from your CV** — upload your CV file and the app figures out your roles, must-haves, and deal-breakers for you.
   - **Upload an existing profile** if you've saved one before ([example profile](example_profile.json)).
3. **Check and edit the profile.** You'll see a few editable lists — things like the kinds of roles that fit you, things a job *must* have, and things that are deal-breakers. **This step matters most.** Add or remove anything so it really reflects what you want. For example, add "remote-friendly" to the must-haves, or "on-call shifts" to the deal-breakers. You can save this profile to reuse later.
4. **Click Run / Match.** The app scores every job. This is the slow part — it's reading each posting carefully — so give it time. Progress text shows what it's doing.
5. **Read the ranking.** When it's done, you get a table of jobs sorted best-fit first, with two scores per job:
   - a **recruiter score** (would a recruiter likely move you forward?), and
   - a **candidate score** (does this match what *you* want?).
6. **Fine-tune without re-running.** You can slide the importance of each score up or down, and set a minimum cutoff to hide weaker matches. The table updates instantly — no waiting. Play with it until the top of the list looks right.
7. **Export** the final list when you're happy, to keep or open in a spreadsheet.

---

## Quick troubleshooting

- **An app won't open / the link is dead.** The app probably isn't running. Go back to the Terminal: if you closed it or pressed Ctrl + C, just run the command again. If the command itself errors out (for example "python is not recognized" or "No module named ..."), the install isn't complete — see [Before your first run](#before-your-first-run).
- **Orange warning box about "environment variables."** The `.env` access keys aren't filled in. See [Before your first run](#before-your-first-run).
- **Match Jobs won't accept my file.** Make sure you're uploading the **`.parquet`** file from Find Jobs, not the spreadsheet (`.csv`) or anything else.
- **"Run" seems frozen.** It's not — scoring reads hundreds of postings one at a time and is genuinely slow. Watch the progress text; it'll tell you when it's done.
- **A search returned nothing.** Try broader wording or a different location, and check the result count isn't set too low.
- **The rankings feel off.** Almost always the fix is the CV profile: go back, correct the role / must-have / deal-breaker lists so they really match what you want, and run again.

That's the whole thing: start an app, search, save the file, match, skim the top of the list.