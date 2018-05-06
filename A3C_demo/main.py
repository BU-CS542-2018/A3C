
#--------------------------------------------------------------------------------------------------------------------------------
# CS 542 Machine Learning Project, Winter 2018, Boston University
# Authors: Alex Burkatovskiy, Siqi Zhang
# Description: This is used for demo.
#--------------------------------------------------------------------------------------------------------------------------------
import gym
import universe  # register the universe environments
import threading
import cv2
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, ELU, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
# global parameters
env_id = 'flashgames.NeonRace-v0'
train = False

user_history = []
user_time_step = 0
ai_history = []
ai_time_step = 0
try:
    # Python2
    import Tkinter as tk
except ImportError:
    # Python3
    import tkinter as tk

train = False;
quit = False
history = []
ACTIONS = [[("KeyEvent", "ArrowUp", True),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", True),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", True),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", True),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", True)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", True),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", True),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", True)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", True),("KeyEvent", "ArrowLeft", True),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", True),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", True)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", False)]]

AI_ACTIONS = [[("KeyEvent", "ArrowUp", True),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", True),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", True),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", True),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", True)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", True),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", True),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", True)]]


#===========================================================================================================
def convertHistoryToKey():
    key = ""
    global history
    if "Up" in history:
        key = key + "Up"
    elif "Down" in history:
        key = key + "Down"
        
    if "Left" in history:
        key = key + "Left"
    elif "Right" in history:
        key = key + "Right"

    return key

#===========================================================================================================
def getActionFromKeyPress():
    keyPressed = convertHistoryToKey()

    global ACTIONS
    if keyPressed == "Up":
        return ACTIONS[0]
    elif keyPressed == "Left":
        return ACTIONS[4]
    elif keyPressed == "Right":
        return ACTIONS[5]
    elif keyPressed == "Down":
        return ACTIONS[3]
    elif keyPressed == "UpLeft":
        return ACTIONS[1]
    elif keyPressed == "UpRight":
        return ACTIONS[2]
    elif keyPressed == "DownLeft":
        return ACTIONS[6]
    elif keyPressed == "DownRight":
        return ACTIONS[7] 
    
    #Do nothing
    return ACTIONS[8]

#===========================================================================================================
def runCarGame():
    global user_time_step
    env = gym.make('flashgames.NeonRace-v0')
    env.configure(fps=30, remotes=1)  # automatically creates a local docker container

    observation_n = env.reset()
    while True:
        global quit
        if quit:
            break
        a = getActionFromKeyPress()
        action_n = [a for ob in observation_n]  # your agent here
        observation_n, reward_n, done_n, info = env.step(action_n)
        #only record the game frame when it's ready
	#and we only record for 30s(or 1800 images)
        if(observation_n[-1] != None):
            img_input = observation_n[0]['vision'][85:565, 20:660]
            user_history.append(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
            user_time_step += 1
            if(user_time_step > 300):
                break
        env.render()

#===========================================================================================================
def keyup(event):
    kP = str(event.keysym)
    if kP == 'Escape':
        global quit
        quit = True
        root.destroy()
    
    global history
    if kP in history :
        history.pop(history.index(kP))


def keydown(event):
    kP = str(event.keysym)
    if kP == 'Escape':
        global quit
        quit = True
        root.destroy()
    
    if kP == "Up" or kP == "Down" or kP == "Left" or kP == "Right":
        global history
        if not kP in history :
            history.append(kP)



#Start up the car game 
task = runCarGame
t = threading.Thread(target=task)
t.start()

#Start up a GUI that can conveniently get keyboard input
root = tk.Tk()
root.bind("<KeyPress>", keydown)
root.bind("<KeyRelease>", keyup)

info = ("Welcome to the Racing Game!\n\n"
        "Use arrow keys to control car\n"
        "Exit with Esc key\n"
        "\nNOTE: This window must be in focus to play\n"
        )
l = tk.Label(root, font=("Helvetica", 20), text=info)
l.pack()

root.mainloop()
#Wait for the game thread to also quit when pressing Esc
t.join();

#===========================================================================================================
#load model
model = load_model('my_model.h5')
#start the agent
env = gym.make(env_id)

env.configure(fps = 60.0, remotes=1) 
observation_n = env.reset()
while True:
            
    #only process when environment is ready
    #current frame
    if(observation_n[-1] != None):
        img_input = observation_n[0]['vision'][85:565, 20:660]
        ai_history.append(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
        ai_time_step += 1
        #crop
        # downsize and we finally have the input image to the network
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_input = cv2.pyrDown(img_input)
        img_input = cv2.pyrDown(img_input)
		
        #image dimension
        input_row = img_input.shape[0]
        input_col = img_input.shape[1]
        #reshape the input 
        img_input = img_input.reshape(1, 1, input_row * input_col)
        #normalization
        img_input = img_input.astype('float32')
        img_input /= 255
	#make the prediction using trained model
        action = np.argmax(model.predict(img_input))
        action_n = [AI_ACTIONS[action] for ob in observation_n] 
        observation_n, reward_n, done_n, info = env.step(action_n)
        #display
        env.render()
    else:
        action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
        observation_n, reward_n, done_n, info = env.step(action_n)
    if (ai_time_step > 300):
        break;
    #display
    env.render()
#video writor
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1280,480))
#cencatenate the 2 image sets
for i in range(0, len(user_history)):
    user_img = user_history[i]
    ai_img = ai_history[i]
    out_video_frame = np.concatenate((user_img, ai_img), axis=1)
    out.write(out_video_frame)

out.release()


 
