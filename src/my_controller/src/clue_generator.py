
import requests

from openai import OpenAI

def generateClues():
    '''
    '''

    URL = "https://phas.ubc.ca/~miti/ENPH353/ENPH353Keys.txt"

    response = requests.get(URL)
    API_KEY,_ = response.text.split(',')

    client = OpenAI(
            api_key=API_KEY
        )

    prompt = f"""You will generate clues that describe a potential funny crime 
                for your game in random order. 
                The clues must have less than 13 characters. 
                Use themes from planet Earth.
                Display the clues in the following order:
                    NUMBER OF VICTIMS
                    WHO ARE THE VICTIMS
                    WHAT IS THE CRIME
                    WHEN WAS THE CRIME COMMITTED
                    WHERE WAS THE CRIME COMMITTED
                    WHY WAS THE CRIME COMMITED
                    WHAT WEAPON WAS THE CRIME COMMITED WITH
                    WHO WAS THE CRIMINAL
                """

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"""You are a game creator making a 
             fun game similar to the boardgame Clue. You want the game to be
             different than the game of Clue so you make changes to it. Your
             game's victim has a funny name related to events from today. 
             This name can be common noun so don't use only proper nouns. 
             The criminals are also different than the original game of Clue 
             and they also make one smile. The location is not limited to 
             human scale, it can be anywhere in the universe from galaxies 
             to atomic nuclei. And the weapon needs to be a fun one too."""},
            {"role": "user", "content": prompt}
        ])

    story = completion.choices[0].message.content
    print(story)

generateClues()
exit(0)
