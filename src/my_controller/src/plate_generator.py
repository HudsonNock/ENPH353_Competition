#!/usr/bin/env python3

import cv2
import csv
import numpy as np
import os
import pyqrcode
import random
import string
from skimage.util import random_noise


from random import randint
from PIL import Image, ImageFont, ImageDraw

# TODO: increase the number of entries that are here by a lot 
import requests

from openai import OpenAI

import requests
from openai import OpenAI
import re


entries = {'SIZE': ["100", "10 GOOGLES", "314", "A PAIR", "BAKER DOZEN",
                    "COUNTLESS", "DOZEN", "FEW", "FIVE", "HALF DOZEN",
                    "LEGIONS", "MANY", "QUINTUPLETS", "RAYO10", "SINGLE",
                    "THREE", "TRIPLETS", "TWO", "UNCOUNTABLE", "ZEPTILLION"],
           'VICTIM': ["ALIENS", "ANTS", "BACTERIA", "BED BUGS", "BUNNIES",
                      "CITIZENS", "DINOSAURS", "FRODOS", "JEDIS", "KANGAROO",
                      "KOALAS", "PANDAS", "PARROTS", "PHYSICISTS", "QUOKKAS",
                      "ROBOTS", "RABBITS", "TOURISTS", "ZOMBIES"],
           'CRIME': ["ACCELERATE", "BITE", "CURSE", "DECELERATE", "DEFRAUD",
                     "DESTROY", "HEADBUT", "IRRADIATE", "LIE TO", "POKE",
                     "PUNCH", "PUSH", "SCARE", "STEAL", "STRIKE", "SWEAR",
                     "TELEPORT", "THINKING", "TICKLE", "TRANSMOGRIFY",
                     "TRESPASS"],
           'TIME': ["2023", "AUTUMN", "DAWN", "D DAY", "DUSK", "EONS AGO",
                    "JURASIC", "MIDNIGHT", "NOON", "Q DAY", "SPRING",
                    "SUMMER", "TOMORROW", "TWILIGHT", "WINTER", "YESTERDAY"],
           'PLACE': ["AMAZON", "ARCTIC", "BASEMENT", "BEACH", "BENU", "CAVE",
                     "CLASS", "EVEREST", "EXIT 8", "FIELD", "FOREST",
                     "HOSPITAL", "HOTEL", "JUNGLE", "MADAGASCAR", "MALL",
                     "MARS", "MINE", "MOON", "SEWERS", "SWITZERLAND",
                     "THE HOOD", "UNDERGROUND", "VILLAGE"],
           'MOTIVE': ["ACCIDENT", "BOREDOM", "CURIOSITY", "FAME", "FEAR",
                      "FOOLISHNESS", "GLAMOUR", "GLUTTONY", "GREED", "HATE",
                      "HASTE", "IGNORANCE", "IMPULSE", "LOVE", "LOATHING",
                      "PASSION", "PRIDE", "RAGE", "REVENGE", "REVOLT",
                      "SELF DEFENSE", "THRILL", "ZEALOUSNESS"],
           'WEAPON': ["ANTIMATTER", "BALOON", "CHEESE", "ELECTRON", "FIRE",
                      "FLASHLIGHT", "HIGH VOLTAGE", "HOLY GRENADE", "ICYCLE",
                      "KRYPTONITE", "NEUTRINOS", "PENCIL", "PLASMA",
                      "POLONIUM", "POSITRON", "POTATO GUN", "ROCKET", "ROPE",
                      "SHURIKEN", "SPONGE", "STICK", "TAMAGOCHI", "WATER",
                      "WRENCH"],
           'BANDIT': ["BARBIE", "BATMAN", "CAESAR", "CAO CAO", "EINSTEIN",
                      "GODZILA", "GOKU", "HANNIBAL", "L", "LENIN", "LUCIFER",
                      "LUIGI", "PIKACHU", "SATOSHI", "SHREK", "SAURON",
                      "THANOS", "TEMUJIN", "THE DEVIL", "ZELOS"]
           }
def generateClue():
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

    return completion.choices[0].message.content

# Call generateClue 10,000 times and update entries dictionary
for _ in range(100):
    clue = generateClue()
    # Split the generated clue into parts and update entries
    parts = clue.split('\n')
    if (len(parts) >= 8):
    # weird sometimes this isn't generating enough parts which is why is why im getting an index out of range error 
      entries['SIZE'].append(re.sub(r'^[\d-]+\.', '', parts[0]).strip().replace(" ", ""))
      entries['VICTIM'].append(re.sub(r'^[\d-]+\.', '', parts[1]).strip().replace(" ", ""))
      entries['CRIME'].append(re.sub(r'^[\d-]+\.', '', parts[2]).strip().replace(" ", ""))
      entries['TIME'].append(re.sub(r'^[\d-]+\.', '', parts[3]).strip().replace(" ", ""))
      entries['PLACE'].append(re.sub(r'^[\d-]+\.', '', parts[4]).strip().replace(" ", ""))
      entries['MOTIVE'].append(re.sub(r'^[\d-]+\.', '', parts[5]).strip().replace(" ", ""))
      entries['WEAPON'].append(re.sub(r'^[\d-]+\.', '', parts[6]).strip().replace(" ", ""))
      entries['BANDIT'].append(re.sub(r'^[\d-]+\.', '', parts[7]).strip().replace(" ", ""))
    for key in entries:
        if len(entries[key][-1]) >= 13:
            entries[key][-1] = entries[key][-1][:12]
print("Entries updated successfully!")
# Print entries in a readable format
imageCount = 0
imageCount2 = 0
for _ in range(100):
  for key, values in entries.items():
    entries[key] = [re.sub(r'^[^:]+:', '', value).strip().replace(":", "").replace("-", "") for value in values]

      #print("--------------------")


  # Print a few examples to verify
  # print(entries['SIZE'][-3:])  # Print the last 3 added SIZE entries
  # print(entries['VICTIM'][-3:])  # Print the last 3 added VICTIM entries


  print("len of entries " + str(len(entries)))
  # Find the path to this script
  SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
  TEXTURE_PATH = '../media/materials/textures/'

  banner_canvas = cv2.imread(SCRIPT_PATH+'clue_banner.png')
  PLATE_HEIGHT = 600
  PLATE_WIDTH = banner_canvas.shape[1]
  IMG_DEPTH = 3

  # write plates to plates.csv
  with open(SCRIPT_PATH + "plates.csv", 'w') as plates_file:
      csvwriter = csv.writer(plates_file)

      i = 0
      for key in entries:
          # pick a random criminal
          j = random.randint(0, len(entries[key])-1)
          random_value = entries[key][j]

          #if len(random_value) < 11:
              #random_value = random.choice(string.ascii_uppercase) + " " + random_value

          entry = key + "," + random_value
          #print(entry)
          csvwriter.writerow([key, random_value])

          # Generate plate
     
          # To use monospaced font for the license plate we need to use the PIL
          # package.
          # Convert into a PIL image (this is so we can use the monospaced fonts)
          blank_plate_pil = Image.fromarray(banner_canvas)
          # Get a drawing context
          draw = ImageDraw.Draw(blank_plate_pil)
          font_size = 90
          monospace = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 
                                         font_size)
          font_color = (255,0,0)
          draw.text((250, 30), key, font_color, font=monospace)
          draw.text((30, 250), random_value, font_color, font=monospace)
          # Convert back to OpenCV image and save
          populated_banner = np.array(blank_plate_pil)


          # Apply Gaussian blur
          blurred_image = cv2.GaussianBlur(populated_banner, (9, 9), 0)
          blurred_image = cv2.GaussianBlur(blurred_image, (9, 9), 0)
          blurred_image = cv2.GaussianBlur(blurred_image, (9, 9), 0)
          blurred_image = cv2.GaussianBlur(blurred_image, (9, 9), 0)
          gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
          _, black_white_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
          black_white_image = cv2.bitwise_not(black_white_image)
          # blurred_image = cv2.GaussianBlur(blurred_image, (9, 9), 0)
          # blurred_image = cv2.GaussianBlur(blurred_image, (9, 9), 0)
          # blurred_image = cv2.GaussianBlur(blurred_image, (9, 9), 0)
          # blurred_image = cv2.GaussianBlur(blurred_image, (9, 9), 0)
          # blurred_image = cv2.GaussianBlur(blurred_image, (9, 9), 0)
          kernel_size = (5, 5)

          # Create the Gaussian kernel
          gaussian_kernel = cv2.getGaussianKernel(kernel_size[0], sigma=0)

          # Apply the 2D convolution with the Gaussian kernel
          img = cv2.filter2D(black_white_image, -1, gaussian_kernel)
          # Convert the blurred image to grayscale
          imageCount = 0

          # Apply adaptive thresholding to convert the grayscale image to black and white
          
          height, width = img.shape[:2]

          clueType = img[0:int(height/2), int(width/3)+15:width]
          clueValue = img[int(height/2):height, 0:width]
          dirValue = "/home/fizzer/ros_ws/src/live_data/processed_real_data/split_clueT_clueV/Value/"
          dirType = "/home/fizzer/ros_ws/src/live_data/processed_real_data/split_clueT_clueV/Type/"


          filename = os.path.join(dirType, f"image_{imageCount}.jpg")
          cv2.imwrite(filename, clueType)
          imageCount+=1
          #print(f"image saved as {filename}")
          filename = os.path.join(dirValue, f"image_{imageCount2}.jpg")
          cv2.imwrite(filename, clueValue)
          imageCount2+=1


          img = clueValue

          # Set up the detector with default parameters.
          _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

          # Find contours
          area_max = 0
          arrayCoords = []
          newOrderedCoords = []
          xCoords = []
          contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)
            # order letters
            if h < 100 and h > 10:
              arrayCoords.append([x,y,w,h])
          # Sort bounding boxes by x-coordinate
          arrayCoords.sort(key=lambda coord: coord[0])
          # Extract and save letters

          for i, coord in enumerate(arrayCoords):
            x, y, w, h = coord
            letter_image = img[y:y+h, x:x+w]
            output_directory = "/home/fizzer/ros_ws/src/live_data/processed_real_data/clue_value_letters/"
            filename = os.path.join(output_directory, f"image_{imageCount}.jpg")
            cv2.imwrite(filename, letter_image)
            imageCount+=1

   
            # maybe add a black padding to the images a bit? 

          img = clueType

          # is detecting the image and finding those contours 

          # Set up the detector with default parameters.
          _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

          # Find contours
          area_max = 0
          arrayCoords = []
          newOrderedCoords = []
          xCoords = []
          contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)
            # order letters
            if h < 100 and h > 10:
              arrayCoords.append([x,y,w,h])
          # Sort bounding boxes by x-coordinate
          arrayCoords.sort(key=lambda coord: coord[0])
          # Extract and save letters

          for i, coord in enumerate(arrayCoords):
            x, y, w, h = coord
            letter_image = img[y:y+h, x:x+w]
            output_directory = "/home/fizzer/ros_ws/src/live_data/processed_real_data/clue_value_letters/"
            filename = os.path.join(output_directory, f"image_{imageCount2}.jpg")
            cv2.imwrite(filename, letter_image)
            imageCount2+=1

          # maybe add a 
