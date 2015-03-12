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
               first = 'BlockerAgent', second = 'OffensiveAgent'):
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

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on). 
    
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    ''' 
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py. 
    '''
    CaptureAgent.registerInitialState(self, gameState)

    ''' 
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    ''' 
    You should change this in your own agent.
    '''

    return random.choice(actions)

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
      print("POWER UP!")
      
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
      'nearestFood':2.0/min(self.getMazeDistance(gameState.getAgentPosition(self.index),p) for p in self.getFood(gameState).asList()),
      'nearestPowerUp': 1.0 if len(self.getCapsules(gameState))==0 else 1.0/min(self.getMazeDistance(gameState.getAgentPosition(self.index),p) for p in self.getCapsules(gameState)),
      'nearestGhost': 0.0, #(-(10*self.powerUpTimer) if self.powerUp else 1.0)*self.getNearestGhostDistance(gameState),
      'score': gameState.getScore(),
      'powerUp': 1.0 if gameState.getAgentPosition(self.index) in self.getCapsules(gameState) else 0,
      'foodEaten': 1.0 if len(self.getFood(gameState).asList()) < self.foodLeft else 0,
    }
    return features
    
  def getWeights(self):
    return {
      'nearestFood':100.0,
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
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    for p in self.legalPositions: self.beliefs[p] = 1.0
    self.beliefs.normalize()
    self.possiblePositions = self.getPossibleNextPositions(gameState)
    self.prob = 1.0/len(self.possiblePositions)
    self.enemyIndex = enemyIndex
    self.myIndex = myIndex
  
  def observe(self, gameState):
    noisyDistance = gameState.getAgentDistances()[self.enemyIndex]
    myPosition = gameState.getAgentPosition(self.myIndex)
    print(myPosition, self.myIndex)
    
    newBeliefs = util.Counter()
    for p in self.beliefs:
      trueDistance = util.manhattanDistance(p, myPosition)
      newBeliefs[p] = gameState.getDistanceProb(trueDistance, noisyDistance) * self.beliefs[p]
    newBeliefs.normalize()
    self.beliefs = newBeliefs
    
    
    
  def elapseTime(self, gameState):
    newBeliefs = util.Counter()
    for oldPos in self.beliefs:
      #just assume all the possible positions have the same probability
      for newPos in self.possiblePositions:
        newBeliefs[newPos] += self.prob * self.beliefs[oldPos] #p(t+1, t) = p(t+1 | t) * p(t)
    newBeliefs.normalize()   
    self.beliefs = newBeliefs      
  
  def getPossibleNextPositions(self, gameState):
    possiblePositions = []
    for pos in self.beliefs:
      nextPos = Actions.getLegalNeighbors(pos, gameState.getWalls())#[(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
      possiblePositions.extend(nextPos)#filter(lambda x: not gameState.hasWall(x[0],x[1]),nextPos))
    return possiblePositions

  def getBeliefDistribution(self):
    return self.beliefs