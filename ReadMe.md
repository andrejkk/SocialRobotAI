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

## Video (.webm) to .wav

To convert video to audio format (for example .wav) for signal analysis:

1. Move to directory containing the video
2. Run `ffmpeg -i ./16ef52e7-8ae6-474c-8911-a6aec7bafe58.webm  output.wav` - replace parameters with your file names

## baseline.py

Pipeline:

- Parse labeled activities: Read ground-truth activity files with timestamps and labels.

- Align with sensor events: Map each labeled activity to the corresponding raw sensor activations.

- Segment into time windows: Divide the continuous sensor stream into fixed-length windows to capture local activity.

- Extract features per window: Compute statistics such as total sensor activations, room-level counts, sensor diversity, and time-of-day indicators.

- Combine features with labels: Build a windowed dataset where each row represents one time window and its associated activity.

- Produce structured dataset: The resulting dataset is ready for baseline algorithms or machine-learning models for activity recognition.

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

## Video (.webm) to .wav

To convert video to audio format (for example .wav) for signal analysis:

1. Move to directory containing the video
2. Run `ffmpeg -i ./Data/66001/2025-12-11/S1/16ef52e7-8ae6-474c-8911-a6aec7bafe58.webm  ./Data/66001/2025-12-11/S1/output.wav` - replace parameters with your file names

## Fixing broken video

When downloading the video from <b>Annotation app</b> the video's progress bar does not work and it does display duration (I am still looking for a fix). For now we can run the following command:

`ffmpeg -i broken.webm -c copy fixed.webm`

This rewrites the WebM container and rebuilds cues. Video should now be fixed.

## baseline.py

Pipeline:

- Parse labeled activities: Read ground-truth activity files with timestamps and labels.

- Align with sensor events: Map each labeled activity to the corresponding raw sensor activations.

- Segment into time windows: Divide the continuous sensor stream into fixed-length windows to capture local activity.

- Extract features per window: Compute statistics such as total sensor activations, room-level counts, sensor diversity, and time-of-day indicators.

- Combine features with labels: Build a windowed dataset where each row represents one time window and its associated activity.

- Produce structured dataset: The resulting dataset is ready for baseline algorithms or machine-learning models for activity recognition.
