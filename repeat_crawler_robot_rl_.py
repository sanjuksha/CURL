from gpiozero import DistanceSensor
import numpy as np
import random
import pandas as pd
import RPi.GPIO as GPIO
import time
import csv

#  Pins Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
q = GPIO.PWM(4, 50)
p = GPIO.PWM(18, 50)
ultrasonic = DistanceSensor(echo=20, trigger=21)
p.start(10)
q.start(10)


print("Started the motors")

# Declaring Variables
global currentHandPosition
currentHandPosition = 0
global currentArmPosition
currentArmPosition = 0
global oldDistance
oldDistance = ultrasonic.distance
time.sleep(1)
global currentDistance
currentDistance=oldDistance
global lastAction
lastAction = 0


print("Variables Declared")

class QLearner(object):
	

    # Functions
    def __init__(self,
                 num_states=4,
                 num_actions=4,
                 alpha=0.8,  # Learning Rate aplha  
                 gamma=0.5,  # discount rate gamma  
                 random_action_rate=0.2,  # Epsilon    
                 random_action_decay_rate=0.99):  # Decrease epsilon with time at random_action_decay_rate
        

		print ("Initializing......")
		self.num_states = num_states
		self.num_actions = num_actions
		self.alpha = alpha
		self.gamma = gamma
		self.state = 0
		self.action = 0
		self.random_action_rate = random_action_rate
		self.random_action_decay_rate = random_action_decay_rate
		
		temp = pd.read_csv('qvalues.csv', sep = ";", header=None)
		#print(temp)
		self.qtable = np.asarray(temp)
		#self.qtable = np.random.uniform(low=0, high=1, size=(num_states, num_actions))
        
		
        
		
        
    def calculateDistance(self):

        dist = ultrasonic.distance
        time.sleep(2)
        return dist

		
            
		
    def getCurrentState(self):
		print("Getting Current State...")
		state = str(int(currentHandPosition)) + str(int(currentArmPosition))
		if state == "10": # Arm Down Hand Left
			state = 2
		elif state == "11": # Arm Down Hand Right
			state = 3
		elif state == "01": # Arm Up Hand Right
			state = 1
		elif state == "00": #Arm Up Hand Left
			state = 0
		print("Current State: %.1f" %state)
		return int(state) 
		
    def set_initial_state(self, state):
		print("I entered the Initial State Function ")
		self.state = state
		self.action = self.qtable[state].argsort()[-1]
		return self.action
   
        
    def moveArmUp(self):
		p.ChangeDutyCycle(10)
		
		global currentArmPosition
		currentArmPosition = 0
		global lastAction
		lastAction=	0
			
    def moveArmDown(self):
		p.ChangeDutyCycle(8)	
		
		global currentArmPosition
		currentArmPosition = 1
		global lastAction
		lastAction=	1	
		
    def moveHandLeft(self):
		q.ChangeDutyCycle(8)
		
		global currentHandPosition
		currentHandPosition = 0
		global lastAction
		lastAction=	2
		
    def moveHandRight(self):
		q.ChangeDutyCycle(3)	
		
		global currentHandPosition
		currentHandPosition = 1
		global lastAction
		lastAction=	3
				


    def move(self, state_prime, reward):
		"""
		@summary: Moves to the given state with given reward and returns action
		@param state_prime: The new state
		@param reward: The reward
		@returns: The selected action
		"""
		alpha = self.alpha
		gamma = self.gamma
		state = self.state
		action = self.action
		qtable = self.qtable

		"""In the follwing line we choose epsilon(random_action_rate) and check if it is
		   between o and 1. atually it is the (1-epsilon) term or probabily term as we call it.
		   But most of the time it would be false as the epsilon is smaller. Hence, as the
		   epsilon decreases the randomness decreases. """
		
        
		
		random_value = np.random.uniform(0,1)
		choose_random_action = (1 - self.random_action_rate) <= random_value
		print("Random value is %.1f" %random_value)

		if choose_random_action:  # if the above value is between 0 and 1
			"""action_prime or the action to be taken in the state
			   should be between 0 and less than the total number of actions. """
			
			action_prime = random.randint(0, self.num_actions - 1)
			print("Inside Random Action {0}".format(action_prime))
			

		else:
			"""else use the action which gives the maximum value"""
			action_prime = self.qtable[state_prime].argsort()[-1]
			
			print("Inside Definite Action {0}".format(action_prime))
		"""Now multiply the epsilon(random_action_rate) with the random_action_decay_rate"""


		self.random_action_rate *= self.random_action_decay_rate
		print("Random Action Rate is %.2f " %self.random_action_rate)
		""" hence, after finding a random value to initialize Q we now find the
				converging value of Q^(s,a) to Q(s,a). Following is the Q value convergence formula."""

		qtable[state, action] = (1 - alpha) * qtable[state, action] + alpha * (reward + gamma * qtable[state_prime, action_prime])

		self.state = state_prime  # now move to the next state
		self.action = action_prime  # take the action_prime calculated above
		#np.savetxt("qvalues.csv", qtable, delimiter=";", fmt="%.2f")
		np.savetxt("qvalues.csv", qtable, delimiter=";", fmt="%.2f")
		print(qtable)
              
		return self.action	
		
    def calculateReward(self):
		currentDistance = self.calculateDistance()
		global oldDistance
		print("Current Distance inside CalculateReward is %.2f" %currentDistance)
		print("Old Distance inside CalculateReward is %.2f" %oldDistance)
		
		if currentDistance < (oldDistance - 0.04):
			
			reward = 1
			print ("Reward %.1f" % reward)
		elif currentDistance > (oldDistance + 0.04):
			reward = -2
			print ("Reward %.1f" % reward)
		else:
			reward = -2
			print("No Reward")
		
		
		oldDistance = currentDistance
		
		return reward
		
	

    
    def step(self, action):
		global lastAction
		if lastAction== 0:
		    if action == 1:
			    self.moveArmDown()
		    elif action == 2:
			    self.moveHandLeft()
		    elif action == 3:
			    self.moveHandRight()
		elif lastAction== 1:
			if action == 0:
			   self.moveArmUp()
			elif action == 2:
			    self.moveHandLeft()
			elif action == 3:
			    self.moveHandRight()
		elif lastAction== 2:
			if action == 0:
				self.moveArmUp()
			elif action == 1:
			    self.moveArmDown()
			elif action == 3:
			    self.moveHandRight() 
		elif lastAction== 3:
			if action == 0:
			    self.moveArmUp()
			elif action == 1:
			    self.moveArmDown()
			elif action == 2:
			    self.moveHandLeft()
		reward = self.calculateReward()
		return reward
			
    
		
		
# Main Program

def crawler_robot_with_qlearning():
	goal_average_steps = 15  # average of the timesteps to reach the goal, or basically THE GOAL
	max_number_of_steps = 20 # number of steps after which the episode id terminated
	number_of_iterations_to_average = 10  # number of values used to calculate average
	#number_of_features = 1
	last_time_steps = np.ndarray(0)
	print("Starting to Learn")
	learner = QLearner(num_states=4,
					   num_actions=4,
					   alpha=0.5,
					   gamma=0.5,
					   random_action_rate=0.5,
					   random_action_decay_rate=0.99)

	for episode in xrange(10):
		# observation = env.reset()
		
		
		print ("{0} Episode Started".format(episode))
		
		state = learner.getCurrentState()
		"""Now using the set_initial _state function it would return the highest value action  to be taken"""
		action = learner.set_initial_state(state)

		for step in xrange(max_number_of_steps - 1):
			print("Inside step {0}".format(step))
			reward = learner.step(action)
			print ("Action is {0}".format(action))
			print ("Calculating Reward....")

			
			state_prime = learner.getCurrentState()


			action = learner.move(state_prime, reward)  # move to the next state using either random action or argsort
			global currentDistance
			currentDistance=ultrasonic.distance
			if currentDistance <= 0.1:
				
				print ("Reached the Wall")
				print (last_time_steps.mean())
				print episode
				last_time_steps = np.append(last_time_steps, [int(step + 1)])  # creates an array of last time steps
				
				if len(last_time_steps) > number_of_iterations_to_average:
					last_time_steps = np.delete(last_time_steps, 0)  # Deletes the extra time steps if the iterations are more than 100
				break
				GPIO.cleanup()


		if last_time_steps.mean() > goal_average_steps:
			print "Goal reached!"
			print "Episodes before solve: ", episode + 1

			break

print ("230")			

random.seed(0)
print("234")
crawler_robot_with_qlearning()





