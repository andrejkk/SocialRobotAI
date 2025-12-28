# Social Robot AI

## General

GIT:
Code: andrejkk/SocialRobotAI
App: https://github.com/TineCrnugelj/ContextAnnotationApp

Buttons showed: event_codes, e_active

App dataset: supabase hosted baza.
https://supabase.com/dashboard/project/vuglhpxfvefcutjaocgd/editor/17486?sort=created_at:desc

Structure of data:
V datoteki uIDs_sIDs.xlsx je evidenca snemanj in ima stolpce
uID Name Date sID Notes
Name in notes sta za človeško evidenco, strukturo direktorijev sestavimo iz uID, Date in sID.
Torej za
66001 A 2025-12-11 S1
bo struktura map
66001/2025-12-11/S1

## data-import.py script

Usage:

1. Create .env file in root with the following structure:<br>
   `VITE_SUPABASE_PROJECT_ID`=""<br>
   `VITE_SUPABASE_PUBLISHABLE_KEY`=""<br>
   `VITE_SUPABASE_URL`=""<br>

2. Install libraries (requirements.txt), using pip or a virtual environment:

   1. python3 -m venv venv
   2. source venv/bin/activate
   3. (pip install --upgrade pip)
   4. pip install -r requirements.txt

3. Running the script: the signal-fetcher.py script requires 4 arguments:
   - recording_id: Recording DB id
   - uID: User id, for example: 66001
   - date: Date, for example: 2025-12-11
   - sID: Recording id, for example: S1

Example command: `python3 ./data-import.py 0565b6af-c324-47da-b684-458970d6e48c 66001 2025-12-11 S2`

This is required so the script knows where to save the generated files (plots)
