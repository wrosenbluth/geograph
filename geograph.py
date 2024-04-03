import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import math
import heapq

# a rasterGraph object is a node-weighted graph defined by a nodesDict attribute:
# a coordinate-keyed dictionary storing a tuple of neighbors and node weight
# to initialize a rasterGraph object, pass in a friction map as a numpy array,
# and the geotransform of that array

class RasterGraph():
  def __init__(self, array, geotransform):
    self.array = array
    self.geotransform = geotransform
    self.xscaleLong = geotransform[1]
    self.yscaleLat = geotransform[5]
    self.yscaleMeters = self.yscaleLat * 111.32*1000 # convert from long to meters
    self.xscalesMeters = self.getPixelXScales()
    self.nodesDict = self.getAllNodes() #the friction map in graph form

  def getPixelXScale(self, node):
    """get the x scale of a pixel. X scale (meters per pixel) is dependent on
    lattitude and so cannot be set globally during initialization.
    Args: node, tuple
    Returns: Xscale in meters"""
    # handle negative indexing
    if node[0]<0:
      y = node[0] + np.shape(self.array)[0]
    else:
      y = node[0]

    # calculate width of longitude line, and then convert scale to meters
    topLat = self.geotransform[3]
    pixlat = topLat + y*self.yscaleLat
    xscaleMeters = abs(40075*math.cos(math.radians(pixlat))/360*1000*self.xscaleLong) # assuming spherical earth with 40075 km circumference
    return xscaleMeters

  # takes ~ 15 seconds for a 1.8 million pixel array
  # surprisingly long!
  def getPixelXScales(self):
    """get X scale for all pixels"""
    pixelXScales = {}
    for i in range(0, np.shape(self.array)[0]):
      pixelXScales[i] = self.getPixelXScale((i,0))
    return pixelXScales

  def getNeighbors(self, node, width=1):
    """takes a node ((y, x) coordinates), and a width and returns all the
    neighboring node coordinates within the width buffer"""
    neighbors = []
    (ysize, xsize) = np.shape(self.array)
    x = node[1]
    y = node[0]

    yRange = range(max(0, y-width), min(ysize, y+width+1))
    xRange = range(max(0, x-width), min(xsize, x+width+1))

    neighbors = [(yIter, xIter) for yIter in yRange for xIter in xRange]
    neighbors.remove(node)
    return tuple(neighbors)

  def getAllNodes(self):
    """builds a dictionary of nodes, where the key is the coordinate in the array
    and the value is a tuple containing 1) all neighbors and 2) the friction"""
    (ysize, xsize) = np.shape(self.array)
    nodesDict = {}
    for yIter in range(0, ysize):
      for xIter in range(0, xsize):
        nodesDict[(yIter, xIter)] = (self.getNeighbors((yIter, xIter)), self.array[yIter, xIter])
    return nodesDict

  # helper function to get the path length of a path
  def getPathLength(self, path):
    pathLength = 0
    for i in range(0, len(path)):
      currentNode = path[i]
      nextNode = path[i-1]
      if nextNode[0] == currentNode[0]: # if y's are equal the jump length should be x scale
        jumpLength = abs(self.xscalesMeters[currentNode[0]])
      elif nextNode[1] == currentNode[1]: # if x's are equal the jump length should be y scale
        jumpLength = abs(self.yscaleMeters)
      else: # otherwise we compute the pythagorean distance
        jumpLength = math.sqrt(self.yscaleMeters**2 + self.xscalesMeters[currentNode[0]]**2)
      pathLength += jumpLength
    # convert meters to miles
    pathLength = pathLength*0.000621371
    return pathLength

  def buildCumulativeCost(self, startNodes, endNode=None):
    """the first part of Dijkstra's algorithm. Takes a list of start nodes and
     returns a coordinate-indexed dictionary of cumulative costs. NB that this
     function does not return a numpy array of cumulative costs; call
     cumulativeCostToArray() method if array format is desired"""

    # initializations
    nodes = self.nodesDict
    path = []
    # empty set to store already visited nodes to prevent revisiting
    visited = set()
    # initialize Cumulative Cost map
    cumulativeCost = {}
    for key in nodes:
      cumulativeCost[key] = math.inf
    # initialize heap of envelope nodes -- nodes that have been touched but not explored
    envelopeNodesHeap = []
    for startNode in startNodes:
      cumulativeCost[startNode] = 0
      heapq.heappush(envelopeNodesHeap, (cumulativeCost[startNode], startNode))
    # shortestPathTree stores the best edge to traverse to reach each node
    shortestPathTree = {}

    ##################################################
    # consider adding:
    # if endNode != None
    # have a while loop with different exit conditions
    ##################################################

    # iterate the algorithm as long any nodes are in the envelope heap
    while envelopeNodesHeap:
      currentCost, currentNode = heapq.heappop(envelopeNodesHeap)

      # check to make sure we dont explore nodes more than once.
      # An explored node is already on the optimal path,
      # and its neighbors have already been evaluated on this path,
      # so we can just pop and discard it if we run into it again
      if currentNode in visited:
        continue

      visited.add(currentNode)

      # for all neighbors, check distance
      #nodes[currentNode][0] contains all neighbors
      neighborCoords = nodes[currentNode][0]

      # for each neighbor, find and potentially update cumulative cost
      # if we update cumulative cost to a node, push it to the heap
      for neighborCoord in neighborCoords:
        # first check if neighberCoord is in visited to reduce computation time
        if neighborCoord in visited:
          continue

        # if y jump, use y distance
        if neighborCoord[0] == currentNode[0]:
          jumpCost = abs(self.yscaleMeters)*nodes[currentNode][1]

        # if x jump, use x distance
        elif neighborCoord[1] == currentNode[1]:
          jumpCost = abs(self.xscalesMeters[currentNode[0]])*nodes[currentNode][1]

        # else it is a lateral jump, use pythagorean theorem
        else:
          jumpCost = math.sqrt(self.yscaleMeters**2 + self.xscalesMeters[currentNode[0]]**2)*nodes[currentNode][1]

        neighborCost = currentCost + jumpCost

        # if we find new shortest path, update distance and push to the heap
        # also update the shortest path tree
        if neighborCost < cumulativeCost[neighborCoord]:
          cumulativeCost[neighborCoord] = neighborCost
          shortestPathTree[neighborCoord] = currentNode
          heapq.heappush(envelopeNodesHeap, (neighborCost, neighborCoord))

    return cumulativeCost, shortestPathTree

  def tracebackPath(self, shortestPathTree, endNode=None):
    """The second half of dijkstra's algorithm. Takes a shortestPathTree
    (a dictionary containing node coordinate keys with parent node as value),
    and end node from which to start backward tracing. Returns a tuple containing:
    (the shortest path from the end node to the starting region, the length of
    the path in miles, the shortestPathTree)"""
    path = []
    currentNode = endNode
    while currentNode is not None:
      path.append(currentNode)
      currentNode = shortestPathTree.get(currentNode)
    pathInfo = (path, self.getPathLength(path), shortestPathTree)

    return pathInfo

  def dijkstrasAlgorithm(self, startNodes, endNode=None):
    """Implementation of Dijkstra's algorithm. Takes start nodes as a tuple, and
    an end node. Returns an array of cumulative costs, and a pathInfo tuple
    generated by tracebackPath. If endNode is None, the CC map will be generated
    but pathInfo will contain ([], 0, {})."""
    cumulativeCost, shortestPathTree = self.buildCumulativeCost(startNodes, endNode)
    pathInfo = self.tracebackPath(shortestPathTree, endNode)
    return (self.cumulativeCostToArray(cumulativeCost), pathInfo)

  def cumulativeCostToArray(self, cumulativeCost):
    """helper function to convert cumulative cost from dictionary to array"""
    arrayDims = np.shape(self.array)
    CCarray = np.zeros(arrayDims)
    for i in range(0, arrayDims[0]):
      for j in range(0, arrayDims[1]):
        CCarray[i,j] = cumulativeCost[i,j]
    return CCarray

  def pathToArray(self, path, width=1):
    """helper function to convert path list to array"""
    arrayDims = np.shape(self.array)
    pathArray = np.zeros(arrayDims)
    if width==1:
      for coord in path:
        pathArray[coord]=1
    else:
      for coord in path:
        pathArray[coord]=1
        for neighbor in self.getNeighbors(coord, width):
          pathArray[neighbor]=1
    return pathArray


# I/O functions

def readFrictionTiff(path):
  """import numpy array and geotransform from tiff with gdal"""
  frictionTif = gdal.Open(r'%s'%path)
  friction = frictionTif.GetRasterBand(1).ReadAsArray()
  frictionGeotransform = frictionTif.GetGeoTransform()
  return friction, frictionGeotransform

def saveRasterAsGTiff(raster, filename, geotransform):
  """save a raster (numpy) as a geotiff"""
  [rows, cols] = raster.shape
  driver = gdal.GetDriverByName("GTiff")
  outdata = driver.Create(filename, cols, rows, 1, gdal.GDT_UInt16)
  outdata.SetGeoTransform(geotransform)##sets same geotransform as input
  outdata.SetProjection(raster.GetProjection())##sets same projection as input
  outdata.GetRasterBand(1).WriteArray(raster)
  outdata.GetRasterBand(1).SetNoDataValue(0) # zero values set to transparent
  outdata.FlushCache() ##saves to disk!!
  outdata = None
  band=None
  ds=None

