# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
from util import nearestPoint
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DefensiveAgent', second = 'OffensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

from distanceCalculator import Distancer

 
class BlockerAgent(CaptureAgent):
  
  def registerInitialState(self, gameState):
   
    CaptureAgent.registerInitialState(self, gameState)
    self.onGuard = False
    
  def chooseAction(self, gameState):
    
    actions = gameState.getLegalActions(self.index)
    default = 'Stop'
    capsules = CaptureAgent.getCapsulesYouAreDefending(self, gameState)
    
    if(capsules):
      capsuleToGoTo = capsules[0]
      myPos = gameState.getAgentPosition(self.index)
      
      if(myPos != capsuleToGoTo and not self.onGuard):
        for action in actions: 
          newState = self.getSuccessor(gameState, action)
          newDis = CaptureAgent.getMazeDistance(self, newState.getAgentState(self.index).getPosition(),capsuleToGoTo);
          currentDis = CaptureAgent.getMazeDistance(self, myPos,capsuleToGoTo);
          if newDis < currentDis:
              return action
      else:
        if not self.onGuard:
          default = actions[0]
          self.onGuard = True
 
    return default
    
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

class DefensiveAgent(CaptureAgent):
  
  # Initialisation function
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    
    # Boolean representing if the agent currently is powered up
    self.powerUp = False
    # Keeps track of how much time is left on the power up
    self.powerUpTimer = 0
    self.foodLeft = len(self.getFood(gameState).asList())
    self.inferenceModules = []
    
    # Create an ExactInference mod for each opponent
    for index in self.getOpponents(gameState):
      self.inferenceModules.append(ExactInference(gameState, self.index, index))
    
    self.firstMove = True
    
    # Initialise opponent beliefs
    self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
    
    # Changes the weighting on score based on what team the agents on,
    # by default the score is the red teams score
    if self.red:
      self.scoreWeight = 500
    else:
      self.scoreWeight = -500
      
  # Choose the best action from the current gamestate based on features 
  def chooseAction(self, gameState):
    
    # Update the beliefs for the opponents
    for index, inf in enumerate(self.inferenceModules):
      if not self.firstMove: inf.elapseTime(gameState)
      self.firstMove = False
      inf.observe(gameState)
      self.ghostBeliefs[index] = inf.getBeliefDistribution()
    
    # Displays the beliefs on the game board, provides zero actual functionality
    self.displayDistributionsOverPositions(self.ghostBeliefs)
    
    # Build up scores for each action based on features
    actionScores = util.Counter()
    for action in gameState.getLegalActions(self.index):
      newState = gameState.generateSuccessor(self.index, action)
      actionScores[self.getActionScore(newState, action)] = action
    
    #print(actionScores)
    # Choose the action with the best score
    bestAction = actionScores[max(actionScores)]
    
    # If the action leads to eating a power up, set the powerUp boolean and start the timer
    if gameState.generateSuccessor(self.index,bestAction).getAgentPosition(self.index) in self.getCapsules(gameState):
      self.powerUp = True
      self.powerUpTimer = 80
      #print("POWER UP!")
    
    # Keeps track of how much food is left
    if gameState.generateSuccessor(self.index,bestAction).getAgentPosition(self.index) in self.getFood(gameState).asList():
      self.foodLeft -= 1
    # If the agent is currently powered up, decrement the timer
    if self.powerUp:
      self.powerUpTimer -= 1
    # When the timer reaches zero, reset the boolean value 
    if self.powerUpTimer == 0:
      self.powerUp = False
    
    return bestAction
    
  def getActionScore(self, gameState, action):
    features = self.getFeatures(gameState, action)
    # Get the dot product of the weight and feature vectors
    score = sum([self.getWeights()[i]*features[i] for i in features])
    #print(self.getWeights())
    #print(features)
    return score
    
  def getFeatures(self, gameState, action):
    features =  {
      # The farther away the capsule is, the greater the negative value
      'nearestPowerUp': 1.0 if len(self.getCapsules(gameState))==0 else -min(self.getMazeDistance(gameState.getAgentPosition(self.index),p) for p in self.getCapsules(gameState)),
      # If the inferred distance is farther than half the width of the grid, ignore it, otherwise reward it for being closer
      'inferedGhost': 0 if (self.getInferedGhostDistance(gameState) > len(gameState.getWalls()[0])/2) else self.getInferedGhostDistance(gameState),
      # This will either be zero (farther than 5 spaces away) or the distance (less than five)
      'nearGhost': -self.getNearGhostDistance(gameState, action),
      # Discourages stopping
      'stop': 1 if action == Directions.STOP else 0,
      # Discourages going to the offensive side
      'offensiveSide': self.getSide(gameState)
    }
    return features
    
  def getWeights(self):
    return {
      'nearestPowerUp': 2.0,
      'inferedGhost': 1.0,
      'nearGhost': 100000000.0,
      'stop': -100,
      'offensiveSide': -1000
    } 
  
  # Returns a 1 if on the offensive side, 0 if own side
  def getSide(self, gameState):
    midpoint = len(gameState.getWalls()[0])/2
    myPos = gameState.getAgentPosition(self.index)
    if (self.red):
      return int (myPos[0] > midpoint)
    else:
      return int (myPos[0] < midpoint)
  
  # Returns 0 if no ghosts can be seen (they are farther than 5 spaces away from either agents)
  def getNearGhostDistance(self, gameState,action):
    # Computes distance to invaders we can see
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    nearest = 0
    if len(invaders) > 0:
      dists = [self.getMazeDistance(gameState.getAgentPosition(self.index), a.getPosition()) for a in invaders]
      nearest = min(dists)
    return nearest
  
  # Returns the distance to the closest ghost based on the inference modules
  def getInferedGhostDistance(self, gameState):
    probPositions = []
    myPosition = gameState.getAgentPosition(self.index)
    for inf in self.inferenceModules:
      probPositions.append(inf.getBeliefDistribution().argMax())
    distances = map(lambda x: self.getMazeDistance(x, myPosition), probPositions)
    return min(distances)
    
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

class OffensiveAgent(CaptureAgent):
  
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.powerUp = False
    self.powerUpTimer = 0
    self.foodLeft = len(self.getFood(gameState).asList())
    self.inferenceModules = []
    #create an ExactInference mod for each opponent
    for index in self.getOpponents(gameState):
      self.inferenceModules.append(ExactInference(gameState, self.index, index))
    
    self.firstMove = True
    self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
    if self.red:
      self.scoreWeight = 500
    else:
      self.scoreWeight = -500
      
    
  def chooseAction(self, gameState):
    """
    for index, inf in enumerate(self.inferenceModules):
      if not self.firstMove: inf.elapseTime(gameState)
      self.firstMove = False
      inf.observe(gameState)
      self.ghostBeliefs[index] = inf.getBeliefDistribution()
    self.displayDistributionsOverPositions(self.ghostBeliefs)
    """
    #build up scores for each action based on features
    actionScores = util.Counter()
    for action in gameState.getLegalActions(self.index):
      newState = gameState.generateSuccessor(self.index, action)
      actionScores[self.getActionScore(newState, action)] = action
    
    #choose the action with the best score
    bestAction = actionScores[max(actionScores)]
    
    #if the action leads to eating a power up, set the powerUp boolean
    if gameState.generateSuccessor(self.index,bestAction).getAgentPosition(self.index) in self.getCapsules(gameState):
      self.powerUp = True
      self.powerUpTimer = 80
      #print("POWER UP!")
      
    if gameState.generateSuccessor(self.index,bestAction).getAgentPosition(self.index) in self.getFood(gameState).asList():
      self.foodLeft -= 1
    
    if self.powerUp:
      self.powerUpTimer -= 1
      
    if self.powerUpTimer == 0:
      self.powerUp = False
    
    
    return bestAction
    
  def getActionScore(self, gameState, action):
    features = self.getFeatures(gameState, action)
    score = sum([self.getWeights()[i]*features[i] for i in features])
    return score
    
  def getFeatures(self, gameState, action):
    features =  {
      'nearestFood':1.0/min(self.getMazeDistance(gameState.getAgentPosition(self.index),p) for p in self.getFood(gameState).asList()),
      'nearestPowerUp': 1.0 if len(self.getCapsules(gameState))==0 else 1.0/min(self.getMazeDistance(gameState.getAgentPosition(self.index),p) for p in self.getCapsules(gameState)),
      'nearestGhost': 0.0, #(-(10*self.powerUpTimer) if self.powerUp else 1.0)*self.getNearestGhostDistance(gameState),
      'score': gameState.getScore(),
      'powerUp': 1.0 if gameState.getAgentPosition(self.index) in self.getCapsules(gameState) else 0,
      'foodEaten': 1.0 if len(self.getFood(gameState).asList()) < self.foodLeft else 0,
    }
    return features
    
  def getWeights(self):
    return {
      'nearestFood':200.0,
      'nearestPowerUp': 50.0,
      'nearestGhost': -1.0,
      'score': self.scoreWeight,
      'powerUp': 100000000000.0,
      'foodEaten': 50.0
    }  
    
  def getNearestGhostDistance(self, gameState):
    probPositions = []
    myPosition = gameState.getAgentPosition(self.index)
    for inf in self.inferenceModules:
      probPositions.append(inf.getBeliefDistribution().argMax())
    distances = map(lambda x: self.getMazeDistance(x, myPosition), probPositions)
    return min(distances)
    
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
 
from game import Actions
 
class ExactInference:
  
  def __init__(self, gameState, myIndex, enemyIndex):
    "Begin with a uniform distribution over ghost positions."
    self.beliefs = util.Counter()
    self.allLegalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.initBeliefs();
    self.enemyIndex = enemyIndex
    self.myIndex = myIndex
  
  def observe(self, gameState):

    # Get current noisy pos and our agent position
    noisyDistance = gameState.getAgentDistances()[self.enemyIndex]
    exactPos = gameState.getAgentPosition(self.enemyIndex)
    myPosition = gameState.getAgentPosition(self.myIndex)


    # Create new beliefs variable
    newBeliefs = util.Counter()

    # Check if within 5 spaces
    if exactPos:
      newBeliefs[exactPos] = 1.0
    else:
      # Iterate through every legal position
      for p in self.beliefs:
        # Get the true distance from agent pos -> this pos
        trueDistance = util.manhattanDistance(p, myPosition)
        # Find the probability that this is a noisy position
        prob = gameState.getDistanceProb(trueDistance, noisyDistance) * self.beliefs[p]
        # Only add position to the new beliefs array if it is noisy
        #if(prob != 0): 
        newBeliefs[p] =  prob

    # Save updated belief locations to instance variable
    newBeliefs.normalize()
    self.beliefs = newBeliefs

  def elapseTime(self, gameState):
    newBeliefs = util.Counter()

    # Updates all the possible ghost legal positions for the elapsed state
    possiblePositions = self.getAllPossibleNextPositions(gameState);
    
    # Iterates over every position in the current belief
    for oldPos in self.beliefs:

      # Iterates over each possible move for this position
      if(len(possiblePositions) > 0):
        for legalMovePos in possiblePositions[oldPos]:

          # Add (1/num_moves * old_belief) to the new belief for this legal move
          newBeliefs[legalMovePos] += (1.0/len(possiblePositions[oldPos])) * self.beliefs[oldPos] #p(t+1, t) = p(t+1 | t) * p(t)
      
    newBeliefs.normalize()   
    self.beliefs = newBeliefs      
  
  # Returns dictionary of (x,y) : [possible moves for x,y] for every self.belief
  def getAllPossibleNextPositions(self, gameState):
    possiblePositions = {}
    for pos in self.beliefs:
      nextPos = Actions.getLegalNeighbors(pos, gameState.getWalls())#[(x+1,y),(x-1,y),(x,y+1),(x,y-1)]

      possiblePositions[pos] = nextPos#filter(lambda x: not gameState.hasWall(x[0],x[1]),nextPos))
    return possiblePositions

  def getPossibleNextPositions(self, gameState, pos):
    return Actions.getLegalNeighbors(pos, gameState.getWalls())

  def getBeliefDistribution(self):
    return self.beliefs

  def initBeliefs(self):
    # Reset beliefs to 1.0 for all legal positions on map
    for p in self.allLegalPositions: self.beliefs[p] = 1.0
    self.beliefs.normalize();
