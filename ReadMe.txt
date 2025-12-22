ReadMe of robot data analysis

GIT:
Code: andrejkk/SocialRobotAI
App: https://github.com/TineCrnugelj/ContextAnnotationApp

Buttons showed: event_codes, e_active

App dataset: supabase hosted baza.
https://supabase.com/dashboard/project/vuglhpxfvefcutjaocgd/editor/17486?sort=created_at:desc



Structure of data:
V datoteki uIDs_sIDs.xlsx je evidenca snemanj in ima stolpce
uID	Name	Date	sID	Notes
Name in notes sta za človeško evidenco, strukturo direktorijev sestavimo iz uID, Date in sID. 
Torej za 
66001	A	2025-12-11	S1	
bo struktura map
66001/2025-12-11/S1
