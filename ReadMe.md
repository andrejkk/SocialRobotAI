# Social Robot AI

## signal-fetcher.py

Usage:

1. Create .env file in root with the following structure:
   `VITE_SUPABASE_PROJECT_ID`=""<br>
   `VITE_SUPABASE_PUBLISHABLE_KEY`=""<br>
   `VITE_SUPABASE_URL`=""

2. Install libraries, using pip or a virtual environment

3. Running the script: the signal-fetcher.py script requires 4 arguments:
   - recording_id: Recording DB id
   - uID: User id, for example: 66001
   - date: Date, for example: 2025-12-11
   - sID: Recording id, for example: S1

Example command: `python3 ./signal-fetcher.py 0565b6af-c324-47da-b684-458970d6e48c 66001 2025-12-11 S2`

This is required so the script knows where to save the generated files (plots)
