# 1. Binary Search
def binary_search(arr, target):
    # Initialize the lower and upper bounds of the search range
    low, high = 0, len(arr) - 1
    
    # Continue searching while the lower bound is less than or equal to the upper bound
    while low <= high:
        # Calculate the middle index
        mid = (low + high) // 2
        
        # If the middle element is the target, return its index
        if arr[mid] == target:
            return mid
        # If the middle element is less than the target, search the right half
        elif arr[mid] < target:
            low = mid + 1
        # If the middle element is greater than the target, search the left half
        else:
            high = mid - 1
    
    # If the target is not found, return -1
    return -1

# 2. Bubble Sort
def bubble_sort(arr):
    # Get the length of the array
    n = len(arr)
    
    # Iterate through the array n times
    for i in range(n):
        # In each iteration, compare adjacent elements up to n-i-1
        for j in range(0, n - i - 1):
            # If the element found is greater than the next element, swap them
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    
    # Return the sorted array
    return arr

# 3. Quick Sort
def quick_sort(arr):
    # Base case: if the array has 1 or fewer elements, it's already sorted
    if len(arr) <= 1:
        return arr
    
    # Choose the middle element as the pivot
    pivot = arr[len(arr) // 2]
    
    # Partition the array into left (elements smaller than pivot)
    left = [x for x in arr if x < pivot]
    
    # Middle (elements equal to pivot)
    middle = [x for x in arr if x == pivot]
    
    # Right (elements larger than pivot)
    right = [x for x in arr if x > pivot]
    
    # Recursively sort left and right partitions, then combine with middle
    return quick_sort(left) + middle + quick_sort(right)

# 4. Depth-First Search (DFS)
def dfs(graph, start, visited=None):
    # Initialize visited set if not provided
    if visited is None:
        visited = set()
    
    # Mark the current node as visited
    visited.add(start)
    
    # Print the current node (can be replaced with any processing)
    print(start)
    
    # Recursively visit all unvisited neighbors
    for next in graph[start] - visited:
        dfs(graph, next, visited)
    
    # Return the set of visited nodes
    return visited

# 5. Breadth-First Search (BFS)
from collections import deque

def bfs(graph, start):
    # Initialize a set to keep track of visited nodes
    visited = set()
    
    # Initialize a queue with the starting node
    queue = deque([start])
    
    # Continue while there are nodes to process
    while queue:
        # Remove and get the next vertex from the queue
        vertex = queue.popleft()
        
        # If this vertex hasn't been visited yet
        if vertex not in visited:
            # Mark it as visited
            visited.add(vertex)
            
            # Process the vertex (here, we're just printing it)
            print(vertex)
            
            # Add all unvisited neighbors to the queue
            queue.extend(graph[vertex] - visited)
    
    # Return the set of visited nodes
    return visited

# 6. Merge Sort
def merge_sort(arr):
    # Base case: if the array has 1 or fewer elements, it's already sorted
    if len(arr) <= 1:
        return arr
    
    # Find the middle point to divide the array into two halves
    mid = len(arr) // 2
    
    # Recursively sort the left half
    left = merge_sort(arr[:mid])
    
    # Recursively sort the right half
    right = merge_sort(arr[mid:])
    
    # Merge the sorted halves
    return merge(left, right)

def merge(left, right):
    # Initialize an empty result list and index pointers for left and right arrays
    result = []
    i, j = 0, 0
    
    # Compare elements from left and right and add the smaller one to the result
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add any remaining elements from the left array
    result.extend(left[i:])
    
    # Add any remaining elements from the right array
    result.extend(right[j:])
    
    # Return the merged result
    return result

# 7. Dijkstra's Algorithm
import heapq

def dijkstra(graph, start):
    # Initialize distances to all nodes as infinity
    distances = {node: float('infinity') for node in graph}
    
    # Distance to the start node is 0
    distances[start] = 0
    
    # Initialize the priority queue with the start node
    pq = [(0, start)]
    
    # Continue while there are nodes to process
    while pq:
        # Get the node with the smallest distance
        current_distance, current_node = heapq.heappop(pq)
        
        # If we've found a longer path, skip this node
        if current_distance > distances[current_node]:
            continue
        
        # Check all neighbors of the current node
        for neighbor, weight in graph[current_node].items():
            # Calculate the distance to the neighbor through the current node
            distance = current_distance + weight
            
            # If this path is shorter than any previous path
            if distance < distances[neighbor]:
                # Update the distance
                distances[neighbor] = distance
                # Add the neighbor to the priority queue
                heapq.heappush(pq, (distance, neighbor))
    
    # Return the shortest distances from the start node to all other nodes
    return distances

# 8. Fibonacci Sequence
def fibonacci(n):
    # Base cases: fib(0) = 0, fib(1) = 1
    if n <= 1:
        return n
    
    # Recursive case: fib(n) = fib(n-1) + fib(n-2)
    return fibonacci(n-1) + fibonacci(n-2)

# 9. Sieve of Eratosthenes
def sieve_of_eratosthenes(n):
    # Initialize all numbers as potential primes
    primes = [True] * (n + 1)
    
    # 0 and 1 are not prime numbers
    primes[0] = primes[1] = False
    
    # Check all numbers up to the square root of n
    for i in range(2, int(n**0.5) + 1):
        # If i is prime
        if primes[i]:
            # Mark all multiples of i as not prime
            for j in range(i*i, n+1, i):
                primes[j] = False
    
    # Return all numbers that are still marked as prime
    return [i for i in range(n+1) if primes[i]]

# 10. Knapsack Problem (Dynamic Programming)
def knapsack(values, weights, capacity):
    # Get the number of items
    n = len(values)
    
    # Initialize the DP table with zeros
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Build the DP table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            # If the current item can fit in the knapsack
            if weights[i-1] <= w:
                # Choose the maximum of including or excluding the current item
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                # If the item doesn't fit, exclude it
                dp[i][w] = dp[i-1][w]
    
    # Return the maximum value that can be achieved
    return dp[n][capacity]

# 11. Linear Search
def linear_search(arr, target):
    # Iterate through each element in the array
    for i in range(len(arr)):
        # If the current element matches the target, return its index
        if arr[i] == target:
            return i
    
    # If the target is not found, return -1
    return -1

# 12. Selection Sort
def selection_sort(arr):
    # Iterate through the array
    for i in range(len(arr)):
        # Find the minimum element in the unsorted part of the array
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # Swap the found minimum element with the first element of the unsorted part
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    
    return arr

# 13. Insertion Sort
def insertion_sort(arr):
    # Iterate through the array starting from the second element
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        
        # Move elements that are greater than key to one position ahead of their current position
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        
        # Insert the key in its correct position
        arr[j + 1] = key
    
    return arr

# 14. Heap Sort
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    # Check if left child exists and is greater than root
    if left < n and arr[largest] < arr[left]:
        largest = left

    # Check if right child exists and is greater than largest so far
    if right < n and arr[largest] < arr[right]:
        largest = right

    # Change root if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)

    # Build a max-heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements from the heap one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

    return arr

# 15. Counting Sort
def counting_sort(arr):
    # Find the maximum element in the array
    max_element = max(arr)
    
    # Create a count array to store the count of each unique object
    count = [0] * (max_element + 1)
    
    # Store the count of each element
    for i in arr:
        count[i] += 1
    
    # Change count[i] so that count[i] now contains actual position of this element in output array
    for i in range(1, len(count)):
        count[i] += count[i-1]
    
    # Build the output array
    output = [0] * len(arr)
    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i]] - 1] = arr[i]
        count[arr[i]] -= 1
    
    return output

# 16. Radix Sort
def counting_sort_for_radix(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    for i in range(n):
        arr[i] = output[i]

def radix_sort(arr):
    max_num = max(arr)
    exp = 1
    while max_num // exp > 0:
        counting_sort_for_radix(arr, exp)
        exp *= 10
    return arr

# 17. Topological Sort
from collections import defaultdict

def topological_sort(graph):
    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(node)

    visited = set()
    stack = []
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return stack[::-1]

# 18. Floyd-Warshall Algorithm
def floyd_warshall(graph):
    V = len(graph)
    dist = [row[:] for row in graph]

    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist

# 19. Kruskal's Algorithm
class DisjointSet:
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        elif self.rank[xroot] > self.rank[yroot]:
            self.parent[yroot] = xroot
        else:
            self.parent[yroot] = xroot
            self.rank[xroot] += 1

def kruskal(graph):
    edges = [(w, u, v) for u, edges in graph.items() for v, w in edges.items()]
    edges.sort()
    vertices = list(graph.keys())
    ds = DisjointSet(vertices)
    mst = []

    for w, u, v in edges:
        if ds.find(u) != ds.find(v):
            ds.union(u, v)
            mst.append((u, v, w))
    
    return mst
