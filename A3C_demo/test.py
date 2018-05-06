import gym
import universe
import cv2
import numpy as np
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, ELU, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
# global parameters
env_id = 'flashgames.NeonRace-v0'
train = False
load = False
#create env
env = gym.make(env_id)

env.configure(fps = 30.0, remotes=1) 
observation_n = env.reset()
#initializa networksF
model = Sequential()
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
if(load):
	del model	
	model = load_model('my_model.h5')
else:
	state_input = Input(shape=(1,160*120))
	h1 = Dense(24, activation='relu')(state_input)
	h2 = Dense(48, activation='relu')(h1)
	h3 = Dense(24, activation='relu')(h2)
	output = Dense(6, activation='relu')(h3)
	model = Model(input=state_input, output=output)
	# 8. Compile model
	model.compile(loss='mse',
		      optimizer='adam',
		      metrics=['accuracy'])
	test = np.random.rand(1,1,160*120)
	result = model.predict(test)
	print(result.shape)
	model.save('my_model.h5')

while True:
	#only process when environment is ready
	#if(observation_n[-1] != None):
	#current frame
	if(observation_n[-1] != None):
		#this is the current frame
		vision = observation_n[0]['vision']
		#crop
		img_input = vision[85:565, 20:660]  #480x640
		img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)	
		# downsize
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
		#used for initialization
		action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
		observation_n, reward_n, done_n, info = env.step(action_n)
		env.render()
out.release();

