#!/usr/bin/env python3
# put your routing program here!

"""
Abstract: 
1.I create two object Highway and City for store the info,
2.city_hash and highway_hash is used to store the graph info which is created by graph function

3.buildcity and buildroad function is used to store the info into turpe, the info(city_gps and road_segms)  is gained from city_gps and road_segment file
4.Graph function is used to create graph according to the info gain from tuple created by buildcity and buildroad function(city_gps and road_segms)

5.BFS, DFS, uniform, A*, IDS is search function. And Distance function is heuristic funciton used in uniform and a*.

6.last part is input and testing part.

Explaination about heuristic: 
my heuristic come from: "https://en.wikipedia.org/wiki/Great-circle_distance" for calculate the great circle distance. we can give the weights to the heuristic value for computing. We also can opotimize the heuristic function based on how many turns what we had been used , which is for the requiring sepcificlly cost function.

"""
import math
import sys
import time
from collections import defaultdict

#define Highway and City

class Highway(object):

	def __init__(self,start,end,dist,speedlimit,name):
		self.start = start
		self.end = end
		self.dist = float(dist)
		self.speedlimit = float(speedlimit)
		self.name = name
		
class City(object):
	def __init__(self,name,latitude,longitude):
		self.name = name
		self.latitude = latitude
		self.longitude = longitude
		
	def __hash__(self):
		return hash(self.name)

	def __eq__(self, other):
		return self.name == other.name
		

		
#global variable for used input:
city_hash ={} #used for store city info and city object info
highway_hash = defaultdict(set)  #used for store the info for highway object

#system input
average_speed = 0


#build city and road format
def buildcity(line):
	name, latitude, longitude = line.split()
	return (str(name),float(latitude),float(longitude))
	
def buildroad(line):
	temp = line.split()
	if(len(temp) == 5):
		start_city, end_city,dist,speed_limit,highway = temp[0],temp[1],temp[2],temp[3],temp[4]
	else:
		start_city, end_city, dist, speed_limit, highway = temp[0], temp[1], temp[2], float(45) , temp[3]
	return (str(start_city),str(end_city),float(dist),float(speed_limit),str(highway))
	
#distance from source to dest for A* and uniform
#the husristic equation get from wiki 
def distance(s,d):
	lat_source,long_source, lat_dest, long_dest = map(math.radians,[city_hash[s].latitude, city_hash[s].longitude, city_hash[d].latitude, city_hash[d].longitude])
	
	long_diff = long_source - long_dest
	lat_diff = lat_source - lat_dest
	
	result = math.sin(lat_diff / 2) ** 2 + math.cos(lat_source) * math.cos(lat_dest) * math.sin(long_diff / 2) ** 2
	return 6371 * 2 * math.asin(math.sqrt(result))

#build in the graph:

def Graph(city_gps,road_segm):

	#allCities,allRoads = set(),set()    #Make sets of both
	city_info = set()
	road_info = set()	
	global highway_hash, city_hash
	global average_speed
	
	temp = 0
#	count = 0

	for highways in road_segm:
		if(len(highways)==5) and float(highways[3]):
			
			road_info.add(Highway(highways[0],highways[1],highways[2],highways[3],highways[4]))
			#get the sum of total sppeed and store city info(start end)
			  
			city_info.add(highways[0])  
			city_info.add(highways[1])  
			temp += float(highways[3]) 

	
	
	average_speed = temp / len(road_info)
	


	#find all unique cites if there is no exists in the txt file
	for cities in city_info:    
		if cities not in (city[0] for city in city_gps):
			city_gps.append([cities,None,None])   #unkonw lat and long setting up with None value
##			
#			
	#build city map
	for r in city_gps:
		if r[1]:
			l = float(r[1])
		else: l = 0.0
		
		if r[2]:
			ls = float(r[2])
		else: ls = 0.0
		
#		temp1 = City(r[0], float(r[1]) if r[1] else 0.0,float(r[2]) if r[2] else 0.0)
		temp1 = City(r[0], l,ls)
		city_hash[r[0]] = temp1 #store the value city to the hash table
		
	for k in city_hash:
#		for obj in road_info:
#			if obj == k or obj ==k:
#				continue
#			else: obj = None 
#		highway_hash[k] = filer(None,obj)
		
		highway_hash[k] = filter(None,{obj if obj.start == k or obj.end==k else None for obj in road_info})
#	print("\nbuild the graph")
	
	return
	
	
#BFS
	
def BFS(s,d):
#	print("BFS starting")
	fringe = [(0, 0, [s])]
	visited = []

	while(fringe):
		successor = fringe.pop(0)
		route = successor[2]
		current = route[len(route)-1]
		if(current not in visited):
			if(current == d):
				return successor

			for e in highway_hash[current]:
				new_route = list(route)
				if e.end == current:
					next = e.start
				else:
					if e.start == current:
						next = e.end
					else: next == None

				new_route.append(next)
				#temp result for recording element
				result = []
				
				for x in fringe:
					result.append(x[0])
				if(next not in result):
					temp1 = (successor[0] + e.dist, successor[1] + e.speedlimit,new_route)
					fringe.append(temp1)
			visited.append(current)
#	print("error, route no found")
	return None
		
#####DFS
def DFS(s,d):
#	print("DFS start")
	fringe = [(0,0,[s])]
	
	visited = []

	while (fringe):
		successor = fringe.pop()
		route = successor[2]  
		current = route[len(route)-1]
		
		if (current not in visited):
			if (current == d):  
				return successor

			for e in highway_hash[current]:
		
				new_route = list(route)	
				
			#determination of postion
				if e.end == current:
					next = e.start
				else:
					if e.start == current:
						next = e.end
					else: next == None
					
				new_route.append(next)
				result = []
				
				for x in fringe:
					result.append(x[0])
				if(next not in result):
					temp1 = (successor[0]+e.dist, successor[1]+e.speedlimit, new_route)
					fringe.append(temp1)
			visited.append(current)
#	print("route not found! Please try again")
	return None	

	
#A*


def Astar(s,d,func):
#	print("A* starting")
	fringe = [(0, 0, [s], distance(s, d), 0, 0)]  # Tuple of Distance, speedlimit, route, Haversine, Turns, Time
	visited = set()

	while (fringe):

		
		if (str(func) == "distance"):
			fringe.sort(key=lambda tup: tup[3])
		elif (str(func) == "segment"):
			fringe.sort(key=lambda tup: tup[4])
		elif (str(func) == "time"):
			fringe.sort(key=lambda tup: tup[5])
			
		successor = fringe.pop(0)
		route = successor[2]
		current = route[len(route)-1]
		if (current not in visited):
			if (current == d):
				
				return successor
				
			for e in highway_hash[current]:
				new_route = list(route)
				
				if e.end == current:
					next = e.start
				else:
					if e.start == current:
						next = e.end
					else: next == None
#				next = e.start if e.end == current else e.end if e.start == current  else None
				
				new_route.append(next)
				
				#heuristic
				heuristic = distance(next,d)

				# according to cost function choose,to get the different successor
				if (str(func) == "distance"):  
					result = []
					
					for x in fringe:
						result.append(x[0])
					if(next not in result):
						temp1 = (successor[0] + e.dist, successor[1] + e.speedlimit, new_route,successor[3] + distance(next, d) + heuristic, successor[4] + 1,successor[5] + e.dist / e.speedlimit)
						fringe.append(temp1)
						
				if (str(func) == "segment"):    
					result = []
					
					for x in fringe:
						result.append(x[0])
					if(next not in result):
						temp1 = (successor[0] + e.dist, successor[1] + e.speedlimit, new_route,successor[3] + distance(next, d), successor[4] + 1 + heuristic,successor[5] + e.dist / e.speedlimit)
						fringe.app(temp1)
						

				
				if (str(func) == "time"):  
					result = []
					
					for x in fringe:
						result.append(x[0])
					if(next not in result):
						temp1 = (successor[0] + e.dist, successor[1] + e.speedlimit, new_route,successor[3] + distance(next, d), successor[4] + 1,successor[5] + e.dist / e.speedlimit + heuristic)
						fringe.append(temp1)

				
				visited.add(current)

#	print("error, try again")
	return None
	

#uniform search	
def Uniform(s,d,func):
#	print("uniform starting")
	
	fringe = [(0,0,[s],distance(s,d),0,0)]   
	visited = []

	while(fringe):
		
		if(str(func)=="distance"):
			fringe.sort(key=lambda t: t[0])
			
		elif(str(func)=="segment"):
			fringe.sort(key=lambda t: t[4])
			
		elif (str(func) == "time"):
			fringe.sort(key=lambda t: t[5])
			

		successor = fringe.pop(0)
		route = successor[2]
		current = route[len(route)-1]
		if(current not in visited):
			if(current == d):
			
				return successor
			for e in highway_hash[current]:
				new_route = list(route)
				

				if e.end == current:
					next = e.start
				else:
					if e.start == current:
						next = e.end
					else: next == None
					
				new_route = list(route)
				new_route.append(next)
				
				result = []
				
				for x in fringe:
					result.append(x[0])
				if(next not in result):
					temp1 = (successor[0] + e.dist, successor[1]+e.speedlimit, new_route,successor[3] + distance(next, d),successor[4] + 1,successor[5] + e.dist / e.speedlimit)

					fringe.append(temp1)
			visited.append(current)
#	print("error,try again")
	return None

	
#IDS
def IDS(s,d):
#	print("IDS starting")
	fringe = [(0, 0, [s])]
	visited = []

	while(fringe):
		successor = fringe.pop(0)
		route = successor[2]
		current = route[len(route)-1]
		if(current not in visited):
			if(current == d):
				return successor

			for e in highway_hash[current]:
				new_route = list(route)
				if e.end == current:
					next = e.start
				else:
					if e.start == current:
						next = e.end
					else: next == None

				new_route.append(next)
				#temp result for recording element
				result = []
				
				for x in fringe:
					result.append(x[0])
				if(next not in result):
					temp1 = (successor[0] + e.dist, successor[1] + e.speedlimit,new_route)
					fringe.append(temp1)
			visited.append(current)
#	print("error, route no found")
	return None
	
	

#print("read files")

city_gps = [buildcity(line) for line in open('city-gps.txt', 'r')]
road_segm = [buildroad(line) for line in open('road-segments.txt', 'r')]

#print("Files read, graph generation")

source,dest, algo, func = sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]

Graph(city_gps,road_segm)

#function call



start_time = time.time()

if(str(algo).lower() == "bfs"):
#	final = BFS(source,dest,func)
	final = BFS(source,dest)
	distance, total = final[0], final[0]/average_speed
	print("bfs",distance," ",total," "," ".join(final[2]))
	
elif(str(algo).lower() == "dfs"):
#	final = DFS(source,dest,func)
	final = DFS(source, dest)
	distance, total = final[0], final[0] / average_speed
	print("no",distance, " ", total, " ", " ".join(final[2]))
	
elif(str(algo).lower() == "a*"):
	final = Astar(source,dest,func)      
	distance, total = final[0], final[0] / average_speed
	print("yes",distance, " ", total, " ", " ".join(final[2]))
	
elif(str(algo).lower() == "uniform"):  
	final = Uniform(source,dest,func)
	distance, total = final[0], final[0] / average_speed
	print("yes",distance, " ", total, " ", " ".join(final[2]))
	
elif(str(algo).lower() == "ids"):
	final = IDS(source, dest)
	distance, total = final[0], final[0] / average_speed
	print("yes",distance, " ", total, " ", " ".join(final[2]))
	
	


end_time = time.time()

#print("total time: ", end_time - start_time)
