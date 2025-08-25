We will make a Discord bot that gives quizzes using the information we can get from using rapidapiIntegration.ts. This will be converted to python. It will allow group participation in a server to where it will display the highest score and the user with the highest score at the end of each game. This game will be called Melee Match GMQ 08-28-24

1 user must first be allowed to login with their RAPIDAPI_KEY & RAPIDAPI_HOST. After submitting their information, it will connect immediately by doing a GET /v1/health request and set up the match. Discord server owners can limit how many times they can do a tournament quiz per day, week or month.

Multiple choice only, it's based on matching the 'text_fields' of a random 'id' from a {filename} users will vote for before starting. You must load a file before searching it's /v1/schema and then constructing quizzes with /v1/records.

At random it can ask a quarter of the users to vote on not using a quarter of the {filenames}, but the file used for that game will be selected at random to start. Note that if a {filename} doesn't have 'text_fields', you ask users to vote for another {filename}. If the {filename} doesn't have more than 1 parameter within text_fields, it asks users to vote for another {filename}.

It will tournament style to where it will eliminate users after getting two straight answers wrong. New users cannot join an active tournament. After every quarter of participants are eliminated, remaining users can vote on delaying the round for an increment amount of minutes. Base starts at 3. More minutes will be available after each increasing quarter elimination, maxed at 10.

In the final round where there's less than a fifth of the total participants are remaining, the final questions will be based on another {filename}. The game continues until there's one participant is remaining.

Discord server admin can customize the game to where if certain role of users are participating, each round can do a different POST /v1/load{file_name}, or change the answers to call for a different paramter that's not 'text_field'. They can do multiplie answer choice with 'id' being the answer, but it pulls 3-5 different 'id' from 3 - 5 different POST /v1/load/{file_name}. For example, the question is using "Examples", and then shows different "id" that are randomize in order from using /POST /v1/load{file_name}. One of the 'id' must match the 'id' of the parameter used, so in this case;"Examples".


A message will be sent to the server for users in that channel to select the reaction role to join the upcoming game. It will countdown before starting, and the channel administration can set the time before the game begins.

Then It will allow the user to select options that will use fastapi endpoint commands available to obtain the 'text_fields" information to continue further.

The available endpoints:
- POST /v1/load/{file_name} https://github.com/Seriousattempts/gmlm/blob/main/RapidAPI_docs/Filename.txt
- POST /v1/search https://github.com/Seriousattempts/gmlm/blob/main/RapidAPI_docs/Search.txt
- GET /v1/schema https://github.com/Seriousattempts/gmlm/blob/main/RapidAPI_docs/Schema.txt
- GET /v1/records https://github.com/Seriousattempts/gmlm/blob/main/RapidAPI_docs/Records.txt
- GET /v1/record_id https://github.com/Seriousattempts/gmlm/blob/main/RapidAPI_docs/Record_id.txt
- GET /v1/health https://github.com/Seriousattempts/gmlm/blob/main/RapidAPI_docs/Health.txt

Here's an example:

``` {
  "id",
  "sheet",
  "text": {
    "text_fields": [
      "Argument",
      "Data Type",
      "Definition",
      "Description",
      "Examples",
      "Examples Explanation",
      "Function",
      "Returns",
      "Syntax",
      "Type",
      "Variable Name"
    ]
  }
}
```


So it will display one of the 'text_fields' and then display between 3 to 5 different answers to select (assign different reaction roles to represent a selected answer). For example, the question is using "Examples Explanation", and then shows different "Returns" that are randomize in order from using GET /v1/records. One of the 'Returns' must match the 'id' of the parameter used, so in this case; "Examples Explanation". Users will have a time limit of 12 seconds before they're unable to change reaction roles for that question. Every new elimination also reduces the time limit by a quarter, until there's 4 seconds remaining, and keeps it at that speed.

Note, if no users get eliminated for 4 rounds straight, it will reduce the time by 2, every 2 rounds until there's 4 seconds remaining. When at 4 seconds if there's 2 straight rounds of no elimination, it will add to the seconds remaining by the power of 2 every round until no users get eliminated for 4 rounds straight. Then it repeats the loop, but adds a second to every time change in that loop.

Save the last tournament to a database file (.xlsx, .csv). It must have the following:
- Number of participants
- Discord username
- Questions asked
- corresponding answers given with question
- Highlight correct answer

This is it's required metadata:
- Discord server
- Discord Channel
- Roles participated
- Time tournament started
- Time of each elimination started
- Time of final person standing (winner)


The available filenames that users can select are:
- Animation_Curves_text.jsonl
- Asset_and_Tag_text.jsonl
- Asynchronous_Functions_text.jsonl
- Audio_text.jsonl
- Buffer_text.jsonl
- Cameras_And_Display_text.jsonl
- Code-Specific_Format_text.jsonl
- Constant_Description_Data_text.jsonl
- Constant_Description_Value_Data_text.jsonl
- Data_Structures_text.jsonl
- Debugging_text.jsonl
- Drawing_text.jsonl
- Extension_text.jsonl
- File_handling_text.jsonl
- Font_text.jsonl
- Full_Example_Data_text.jsonl
- Game_Input_text.jsonl
- Garbage_collection_text.jsonl
- General_Game_Control_text.jsonl
- GPU_State_DS_Map_Key_Data_text.jsonl
- GXC_text.jsonl
- HTML5_Valid_Targets_Description_Data_text.jsonl
- Instances_text.jsonl
- Joint_Constant_Data_text.jsonl
- Language_Features_text.jsonl
- Layer_text.jsonl
- Live_Wallpaper_text.jsonl
- Maths_And_Numbers_text.jsonl
- Movement_And_Collisions_text.jsonl
- Multi-task_Format_text.jsonl
- Networking_text.jsonl
- Object_text.jsonl
- OS_and_Compiler_text.jsonl
- Path_text.jsonl
- Physics_text.jsonl
- Rooms_text.jsonl
- Script_text.jsonl
- Sequences_text.jsonl
- Shader_text.jsonl
- Sprites_text.jsonl
- Static_Struct_text.jsonl
- Strings_text.jsonl
- StructPointer_Data_text.jsonl
- Struct_Forbidden_Variables_text.jsonl
- Struct_Variable_Type_Data_text.jsonl
- Terms_and_Definitions_text.jsonl
- Text_Alignment_Constant_Data_text.jsonl
- Tilset_text.jsonl
- Timeline_text.jsonl
- Time_Sources_text.jsonl
- Variables_text.jsonl
- Variable_Functions_text.jsonl
- Variable_Name_Data_Type_Description_Data_text.jsonl
- Web_and_HTML5_text.jsonl 


# DISCORD GAMES

What can you devs do now?

- How can those participated take every non answer for that tournament and build a game using only those functions or examples?
- How can those participated take every correct answer for that tournaments and build a game without using those functions or examples

Development Server?

- Assign roles based on respective focus
1. Menu Methods
2. Battle System
3. Database score keeping
4. Ergonomic Inputs
5. Color correctiona
etc

Ideas?
https://discord.com/developers/docs/monetization/overview


1. Card Discord Game
Create a card game on discord
https://www.freepublicapis.com/deck-of-cards
https://boardgamegeek.com/thread/3414659/what-games-can-be-created-and-played-on-discord
- DECKED OUT MAZE (n8n)

2. YuGiOh Discord Game 
- https://ygoprodeck.com/premium/
- https://poe.com/YGOPRODeckAPIv7 | https://github.com/Seriousattempts/gmlm/blob/main/Discord/YGOProDeck_cardset.py


## You vs LLM
Use 40 - 60 cards to describe a fantasy story
YGOProDeck_cardset.py
Build with GameMaker Roles
- Build with restraints of last Melee Tournament results
- Build it right into discord, he's an example https://top.gg/bot/812323565012647997





