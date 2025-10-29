Linear Data Structures
Arrays and Lists
Array (Static)

When: You know the size beforehand and need fast random access
Why: O(1) access time, cache-friendly, minimal memory overhead
Examples: Storing fixed configuration data, lookup tables, small collections

```python
# Python implementation of a static array using the array module
import array

# Creating a static array of integers
static_array = array.array('i', [1, 2, 3, 4, 5])  # 'i' for signed integer

# Accessing elements (O(1))
first_element = static_array[0]  # Returns 1

# Modifying elements (O(1))
static_array[2] = 10  # Changes the third element to 10

# Getting array size
array_size = len(static_array)  # Returns 5

# Iterating through the array
for element in static_array:
    print(element)

# Note: Python's built-in list is actually a dynamic array, but the array module
# provides a more memory-efficient static array implementation for primitive types.
```

Dynamic Array (ArrayList, Vector)

When: Size changes but insertions/deletions are mostly at the end
Why: O(1) amortized append, O(1) random access, better than linked lists for most cases
Examples: General-purpose collections, building lists incrementally

```python
# Python implementation of a dynamic array
class DynamicArray:
    def __init__(self, capacity=1):
        """Initialize with a small capacity"""
        self.capacity = capacity
        self.size = 0
        self.array = [None] * self.capacity
    
    def _resize(self, new_capacity):
        """Resize the internal array to new_capacity"""
        new_array = [None] * new_capacity
        for i in range(self.size):
            new_array[i] = self.array[i]
        self.array = new_array
        self.capacity = new_capacity
    
    def append(self, element):
        """Add element to the end, O(1) amortized"""
        if self.size == self.capacity:
            # Double the capacity when full
            self._resize(2 * self.capacity)
        
        self.array[self.size] = element
        self.size += 1
    
    def insert(self, index, element):
        """Insert element at index, O(n) worst case"""
        if index < 0 or index > self.size:
            raise IndexError("Index out of bounds")
        
        if self.size == self.capacity:
            self._resize(2 * self.capacity)
        
        # Shift elements to the right
        for i in range(self.size, index, -1):
            self.array[i] = self.array[i-1]
        
        self.array[index] = element
        self.size += 1
    
    def remove(self, index):
        """Remove element at index, O(n) worst case"""
        if index < 0 or index >= self.size:
            raise IndexError("Index out of bounds")
        
        # Shift elements to the left
        for i in range(index, self.size - 1):
            self.array[i] = self.array[i+1]
        
        self.size -= 1
        
        # Shrink if too much unused space
        if self.size <= self.capacity // 4 and self.capacity > 1:
            self._resize(self.capacity // 2)
    
    def get(self, index):
        """Get element at index, O(1)"""
        if index < 0 or index >= self.size:
            raise IndexError("Index out of bounds")
        return self.array[index]
    
    def set(self, index, element):
        """Set element at index, O(1)"""
        if index < 0 or index >= self.size:
            raise IndexError("Index out of bounds")
        self.array[index] = element
    
    def __len__(self):
        """Return the number of elements"""
        return self.size
    
    def __str__(self):
        """String representation of the array"""
        return str([self.array[i] for i in range(self.size)])

# Example usage
arr = DynamicArray()
arr.append(1)
arr.append(2)
arr.append(3)
arr.insert(1, 4)  # [1, 4, 2, 3]
print(arr)  # Output: [1, 4, 2, 3]
arr.remove(2)  # [1, 4, 3]
print(arr)  # Output: [1, 4, 3]
print(arr.get(0))  # Output: 1

# Note: Python's built-in list is already a dynamic array implementation
# This example demonstrates how it works internally
```

Singly Linked List

When: Frequent insertions/deletions at the beginning, unknown size, memory fragmentation concerns
Why: O(1) insertion/deletion at head, no reallocation needed, can grow indefinitely
Examples: Implementing stacks, simple queues, undo functionality

```python
# Python implementation of a singly linked list
class Node:
    def __init__(self, data=None):
        """Initialize a node with data and next pointer"""
        self.data = data
        self.next = None

class SinglyLinkedList:
    def __init__(self):
        """Initialize an empty linked list"""
        self.head = None
        self.size = 0
    
    def is_empty(self):
        """Check if the list is empty"""
        return self.head is None
    
    def append(self, data):
        """Add a new node at the end of the list, O(n)"""
        new_node = Node(data)
        
        if self.is_empty():
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        
        self.size += 1
    
    def prepend(self, data):
        """Add a new node at the beginning of the list, O(1)"""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def delete(self, data):
        """Delete the first occurrence of data, O(n)"""
        if self.is_empty():
            return False
        
        # Special case: head node
        if self.head.data == data:
            self.head = self.head.next
            self.size -= 1
            return True
        
        # Find the node before the one to delete
        current = self.head
        while current.next and current.next.data != data:
            current = current.next
        
        if current.next:
            current.next = current.next.next
            self.size -= 1
            return True
        
        return False  # Data not found
    
    def find(self, data):
        """Find a node with the given data, O(n)"""
        current = self.head
        while current:
            if current.data == data:
                return current
            current = current.next
        return None
    
    def get_at(self, index):
        """Get the node at the given index, O(n)"""
        if index < 0 or index >= self.size:
            return None
        
        current = self.head
        for _ in range(index):
            current = current.next
        
        return current
    
    def insert_at(self, index, data):
        """Insert data at the specified index, O(n)"""
        if index < 0 or index > self.size:
            return False
        
        if index == 0:
            self.prepend(data)
            return True
        
        new_node = Node(data)
        current = self.head
        for _ in range(index - 1):
            current = current.next
        
        new_node.next = current.next
        current.next = new_node
        self.size += 1
        return True
    
    def __len__(self):
        """Return the size of the list"""
        return self.size
    
    def __str__(self):
        """String representation of the list"""
        if self.is_empty():
            return "[]"
        
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        
        return "[" + ", ".join(elements) + "]"

# Example usage
ll = SinglyLinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
ll.prepend(0)  # [0, 1, 2, 3]
print(ll)  # Output: [0, 1, 2, 3]

ll.delete(2)  # [0, 1, 3]
print(ll)  # Output: [0, 1, 3]

ll.insert_at(2, 5)  # [0, 1, 5, 3]
print(ll)  # Output: [0, 1, 5, 3]

node = ll.find(5)
print(node.data if node else "Not found")  # Output: 5
```

Doubly Linked List

When: Need bidirectional traversal or deletions from middle with node reference
Why: O(1) deletion with node pointer, backward traversal, easier manipulation
Examples: LRU cache implementation, browser history, music playlists

```python
# Python implementation of a doubly linked list
class Node:
    def __init__(self, data=None):
        """Initialize a node with data, prev and next pointers"""
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        """Initialize an empty doubly linked list"""
        self.head = None
        self.tail = None
        self.size = 0
    
    def is_empty(self):
        """Check if list is empty"""
        return self.head is None
    
    def append(self, data):
        """Add a new node at the end of the list, O(1)"""
        new_node = Node(data)
        
        if self.is_empty():
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        
        self.size += 1
    
    def prepend(self, data):
        """Add a new node at the beginning of the list, O(1)"""
        new_node = Node(data)
        
        if self.is_empty():
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        
        self.size += 1
    
    def delete(self, data):
        """Delete the first occurrence of data, O(n)"""
        if self.is_empty():
            return False
        
        current = self.head
        
        while current and current.data != data:
            current = current.next
        
        if current is None:
            return False  # Data not found
        
        # Update head if needed
        if current.prev is None:
            self.head = current.next
        else:
            current.prev.next = current.next
        
        # Update tail if needed
        if current.next is None:
            self.tail = current.prev
        else:
            current.next.prev = current.prev
        
        self.size -= 1
        return True
    
    def delete_node(self, node):
        """Delete a specific node, O(1) if node reference is available"""
        if node is None:
            return False
        
        # Update head if needed
        if node.prev is None:
            self.head = node.next
        else:
            node.prev.next = node.next
        
        # Update tail if needed
        if node.next is None:
            self.tail = node.prev
        else:
            node.next.prev = node.prev
        
        self.size -= 1
        return True
    
    def find(self, data):
        """Find a node with the given data, O(n)"""
        current = self.head
        while current:
            if current.data == data:
                return current
            current = current.next
        return None
    
    def find_from_tail(self, data):
        """Find a node starting from the tail, O(n)"""
        current = self.tail
        while current:
            if current.data == data:
                return current
            current = current.prev
        return None
    
    def get_at(self, index):
        """Get the node at the given index, O(n)"""
        if index < 0 or index >= self.size:
            return None
        
        # Choose the closer end to start from
        if index < self.size // 2:
            current = self.head
            for _ in range(index):
                current = current.next
        else:
            current = self.tail
            for _ in range(self.size - 1, index, -1):
                current = current.prev
        
        return current
    
    def insert_at(self, index, data):
        """Insert data at the specified index, O(n)"""
        if index < 0 or index > self.size:
            return False
        
        if index == 0:
            self.prepend(data)
            return True
        
        if index == self.size:
            self.append(data)
            return True
        
        # Find the node at the insertion position
        current = self.get_at(index)
        if current is None:
            return False
        
        new_node = Node(data)
        new_node.prev = current.prev
        new_node.next = current
        current.prev.next = new_node
        current.prev = new_node
        
        self.size += 1
        return True
    
    def insert_after(self, node, data):
        """Insert data after a specific node, O(1)"""
        if node is None:
            return False
        
        new_node = Node(data)
        new_node.prev = node
        new_node.next = node.next
        
        if node.next is None:
            self.tail = new_node
        else:
            node.next.prev = new_node
        
        node.next = new_node
        self.size += 1
        return True
    
    def insert_before(self, node, data):
        """Insert data before a specific node, O(1)"""
        if node is None:
            return False
        
        new_node = Node(data)
        new_node.prev = node.prev
        new_node.next = node
        
        if node.prev is None:
            self.head = new_node
        else:
            node.prev.next = new_node
        
        node.prev = new_node
        self.size += 1
        return True
    
    def __len__(self):
        """Return the size of the list"""
        return self.size
    
    def __str__(self):
        """String representation of the list"""
        if self.is_empty():
            return "[]"
        
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        
        return "[" + ", ".join(elements) + "]"

# Example usage
dll = DoublyLinkedList()
dll.append(1)
dll.append(2)
dll.append(3)
dll.prepend(0)  # [0, 1, 2, 3]
print(dll)  # Output: [0, 1, 2, 3]

dll.delete(2)  # [0, 1, 3]
print(dll)  # Output: [0, 1, 3]

dll.insert_at(2, 5)  # [0, 1, 5, 3]
print(dll)  # Output: [0, 1, 5, 3]

node = dll.find(5)
if node:
    dll.insert_after(node, 7)  # [0, 1, 5, 7, 3]
    print(dll)  # Output: [0, 1, 5, 7, 3]

# Demonstrate backward traversal
print("Backward traversal:")
current = dll.tail
while current:
    print(current.data, end=" ")
    current = current.prev
print()  # Output: 3 7 5 1 0
```

Circular Linked List

When: Need to cycle through elements repeatedly
Why: No null end, constant-time connection of last to first
Examples: Round-robin scheduling, multiplayer turn management, circular buffers

```python
# Python implementation of a circular linked list
class Node:
    def __init__(self, data=None):
        """Initialize a node with data and next pointer"""
        self.data = data
        self.next = None

class CircularLinkedList:
    def __init__(self):
        """Initialize an empty circular linked list"""
        self.head = None
        self.size = 0
    
    def is_empty(self):
        """Check if the list is empty"""
        return self.head is None
    
    def append(self, data):
        """Add a new node at the end of the list, O(1)"""
        new_node = Node(data)
        
        if self.is_empty():
            new_node.next = new_node  # Points to itself
            self.head = new_node
        else:
            # Find the last node (node before head)
            last = self.head
            while last.next != self.head:
                last = last.next
            
            last.next = new_node
            new_node.next = self.head
        
        self.size += 1
    
    def prepend(self, data):
        """Add a new node at the beginning of the list, O(1)"""
        new_node = Node(data)
        
        if self.is_empty():
            new_node.next = new_node  # Points to itself
            self.head = new_node
        else:
            # Find the last node (node before head)
            last = self.head
            while last.next != self.head:
                last = last.next
            
            last.next = new_node
            new_node.next = self.head
            self.head = new_node
        
        self.size += 1
    
    def delete(self, data):
        """Delete the first occurrence of data, O(n)"""
        if self.is_empty():
            return False
        
        current = self.head
        prev = None
        
        # Find the node to delete
        while True:
            if current.data == data:
                break
            prev = current
            current = current.next
            
            # If we've looped back to the head, data not found
            if current == self.head:
                return False
        
        # If we're deleting the only node
        if self.size == 1:
            self.head = None
        # If we're deleting the head
        elif current == self.head:
            # Find the last node
            last = self.head
            while last.next != self.head:
                last = last.next
            last.next = current.next
            self.head = current.next
        else:
            prev.next = current.next
        
        self.size -= 1
        return True
    
    def find(self, data):
        """Find a node with the given data, O(n)"""
        if self.is_empty():
            return None
        
        current = self.head
        while True:
            if current.data == data:
                return current
            current = current.next
            if current == self.head:
                return None
    
    def get_at(self, index):
        """Get the node at the given index, O(n)"""
        if self.is_empty() or index < 0 or index >= self.size:
            return None
        
        current = self.head
        for _ in range(index):
            current = current.next
        
        return current
    
    def insert_at(self, index, data):
        """Insert data at the specified index, O(n)"""
        if index < 0 or index > self.size:
            return False
        
        if index == 0:
            self.prepend(data)
            return True
        
        if index == self.size:
            self.append(data)
            return True
        
        # Find the node at the insertion position
        prev_node = self.get_at(index - 1)
        if prev_node is None:
            return False
        
        new_node = Node(data)
        new_node.next = prev_node.next
        prev_node.next = new_node
        
        self.size += 1
        return True
    
    def rotate(self, steps=1):
        """Rotate the list by the given number of steps, O(n)"""
        if self.is_empty() or steps % self.size == 0:
            return
        
        steps = steps % self.size
        for _ in range(steps):
            self.head = self.head.next
    
    def __len__(self):
        """Return the size of the list"""
        return self.size
    
    def __str__(self):
        """String representation of the list"""
        if self.is_empty():
            return "[]"
        
        elements = []
        current = self.head
        for _ in range(self.size):
            elements.append(str(current.data))
            current = current.next
        
        return "[" + ", ".join(elements) + "]"
    
    def __iter__(self):
        """Make the list iterable"""
        if self.is_empty():
            return iter([])
        
        elements = []
        current = self.head
        for _ in range(self.size):
            elements.append(current.data)
            current = current.next
        
        return iter(elements)

# Example usage
cll = CircularLinkedList()
cll.append(1)
cll.append(2)
cll.append(3)
cll.prepend(0)  # [0, 1, 2, 3]
print(cll)  # Output: [0, 1, 2, 3]

cll.delete(2)  # [0, 1, 3]
print(cll)  # Output: [0, 1, 3]

cll.insert_at(2, 5)  # [0, 1, 5, 3]
print(cll)  # Output: [0, 1, 5, 3]

# Rotate the list
cll.rotate(2)  # [5, 3, 0, 1]
print(cll)  # Output: [5, 3, 0, 1]

# Iterate through the list
print("Iteration:")
for item in cll:
    print(item, end=" ")
print()  # Output: 5 3 0 1
```

Skip List

When: Need sorted data with faster search than linked lists but simpler than trees
Why: O(log n) search/insert/delete on average, simpler to implement than balanced trees
Examples: Redis sorted sets, concurrent data structures, in-memory databases

```python
# Python implementation of a Skip List
import random

class SkipNode:
    def __init__(self, data=None, level=0):
        """Initialize a node with data and next pointers for each level"""
        self.data = data
        self.next = [None] * (level + 1)

class SkipList:
    def __init__(self, max_level=16):
        """Initialize an empty skip list with a maximum level"""
        self.max_level = max_level
        self.level = 0  # Current highest level in use
        self.head = SkipNode(None, max_level)  # Head node with max level
        self.p = 0.5  # Probability factor for level generation
    
    def _random_level(self):
        """Generate a random level for a new node"""
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level
    
    def insert(self, data):
        """Insert data into the skip list, O(log n) average"""
        update = [None] * (self.max_level + 1)
        current = self.head
        
        # Find the insertion point for each level
        for i in range(self.level, -1, -1):
            while current.next[i] and current.next[i].data < data:
                current = current.next[i]
            update[i] = current
        
        current = current.next[0]
        
        # If data already exists, update it
        if current and current.data == data:
            current.data = data
            return
        
        # Create a new node with a random level
        new_level = self._random_level()
        
        # Update the skip list's level if needed
        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.head
            self.level = new_level
        
        # Create the new node
        new_node = SkipNode(data, new_level)
        
        # Update pointers at each level
        for i in range(new_level + 1):
            new_node.next[i] = update[i].next[i]
            update[i].next[i] = new_node
    
    def search(self, data):
        """Search for data in the skip list, O(log n) average"""
        current = self.head
        
        # Start from the highest level and work down
        for i in range(self.level, -1, -1):
            while current.next[i] and current.next[i].data < data:
                current = current.next[i]
        
        current = current.next[0]
        
        # Check if we found the data
        if current and current.data == data:
            return current
        return None
    
    def delete(self, data):
        """Delete data from the skip list, O(log n) average"""
        update = [None] * (self.max_level + 1)
        current = self.head
        
        # Find the node to delete
        for i in range(self.level, -1, -1):
            while current.next[i] and current.next[i].data < data:
                current = current.next[i]
            update[i] = current
        
        current = current.next[0]
        
        # If data not found, return False
        if not current or current.data != data:
            return False
        
        # Update pointers at each level
        for i in range(self.level + 1):
            if update[i].next[i] != current:
                break
            update[i].next[i] = current.next[i]
        
        # Update the skip list's level if needed
        while self.level > 0 and self.head.next[self.level] is None:
            self.level -= 1
        
        return True
    
    def __str__(self):
        """String representation of the skip list"""
        elements = []
        current = self.head.next[0]
        while current:
            elements.append(str(current.data))
            current = current.next[0]
        return "[" + ", ".join(elements) + "]"
    
    def display_levels(self):
        """Display the skip list structure by levels"""
        for level in range(self.level, -1, -1):
            nodes = []
            current = self.head.next[level]
            while current:
                nodes.append(str(current.data))
                current = current.next[level]
            print(f"Level {level}: {' -> '.join(nodes) if nodes else 'Empty'}")

# Example usage
sl = SkipList()

# Insert elements
sl.insert(3)
sl.insert(6)
sl.insert(7)
sl.insert(9)
sl.insert(12)
sl.insert(19)
sl.insert(17)
sl.insert(26)
sl.insert(21)
sl.insert(25)

print("Skip List:", sl)  # Output: [3, 6, 7, 9, 12, 17, 19, 21, 25, 26]

# Search for elements
print("Search for 12:", sl.search(12).data if sl.search(12) else "Not found")  # Output: 12
print("Search for 15:", sl.search(15).data if sl.search(15) else "Not found")  # Output: Not found

# Delete elements
sl.delete(12)
sl.delete(19)
print("After deletions:", sl)  # Output: [3, 6, 7, 9, 17, 21, 25, 26]

# Display the internal structure
print("\nInternal structure by levels:")
sl.display_levels()
```

Unrolled Linked List

When: Want linked list flexibility with better cache performance
Why: Stores multiple elements per node, reduces pointer overhead, better cache locality
Examples: Text editors, large sequence storage

```python
# Python implementation of an Unrolled Linked List
class UnrolledNode:
    def __init__(self, capacity=16):
        """Initialize a node with a fixed capacity"""
        self.capacity = capacity
        self.elements = []  # List to store elements
        self.next = None
    
    def is_full(self):
        """Check if the node is full"""
        return len(self.elements) >= self.capacity
    
    def is_empty(self):
        """Check if the node is empty"""
        return len(self.elements) == 0
    
    def add_element(self, element):
        """Add an element to this node"""
        if not self.is_full():
            self.elements.append(element)
            return True
        return False
    
    def remove_element(self, index):
        """Remove an element at a given index"""
        if 0 <= index < len(self.elements):
            return self.elements.pop(index)
        return None
    
    def insert_element(self, index, element):
        """Insert an element at a given index"""
        if 0 <= index <= len(self.elements) and not self.is_full():
            self.elements.insert(index, element)
            return True
        return False

class UnrolledLinkedList:
    def __init__(self, node_capacity=16):
        """Initialize an empty unrolled linked list"""
        self.node_capacity = node_capacity
        self.head = None
        self.size = 0
    
    def is_empty(self):
        """Check if the list is empty"""
        return self.head is None
    
    def append(self, element):
        """Add an element to the end of the list, O(1) amortized"""
        if self.is_empty():
            self.head = UnrolledNode(self.node_capacity)
            self.head.add_element(element)
        else:
            # Find the last node
            current = self.head
            while current.next:
                current = current.next
            
            # If the last node is full, create a new node
            if current.is_full():
                new_node = UnrolledNode(self.node_capacity)
                current.next = new_node
                new_node.add_element(element)
            else:
                current.add_element(element)
        
        self.size += 1
    
    def prepend(self, element):
        """Add an element to the beginning of the list, O(1)"""
        if self.is_empty():
            self.head = UnrolledNode(self.node_capacity)
            self.head.add_element(element)
        else:
            # If the head node is full, create a new node
            if self.head.is_full():
                new_head = UnrolledNode(self.node_capacity)
                new_head.add_element(element)
                new_head.next = self.head
                self.head = new_head
            else:
                self.head.insert_element(0, element)
        
        self.size += 1
    
    def get_at(self, index):
        """Get the element at a given index, O(n)"""
        if index < 0 or index >= self.size:
            return None
        
        current = self.head
        elements_seen = 0
        
        while current:
            if index < elements_seen + len(current.elements):
                # The element is in this node
                return current.elements[index - elements_seen]
            
            elements_seen += len(current.elements)
            current = current.next
        
        return None
    
    def insert_at(self, index, element):
        """Insert an element at a given index, O(n)"""
        if index < 0 or index > self.size:
            return False
        
        # Special case: inserting at the beginning
        if index == 0:
            self.prepend(element)
            return True
        
        # Special case: inserting at the end
        if index == self.size:
            self.append(element)
            return True
        
        # Find the node where the element should be inserted
        current = self.head
        elements_seen = 0
        
        while current:
            if index < elements_seen + len(current.elements):
                # The element should be inserted in this node
                insert_pos = index - elements_seen
                
                if current.is_full():
                    # Need to split the node
                    new_node = UnrolledNode(self.node_capacity)
                    
                    # Move half of the elements to the new node
                    split_pos = len(current.elements) // 2
                    new_node.elements = current.elements[split_pos:]
                    current.elements = current.elements[:split_pos]
                    
                    # Insert the new node after current
                    new_node.next = current.next
                    current.next = new_node
                    
                    # Determine where to insert the element
                    if insert_pos <= split_pos:
                        current.insert_element(insert_pos, element)
                    else:
                        new_node.insert_element(insert_pos - split_pos, element)
                else:
                    # Node has space, insert directly
                    current.insert_element(insert_pos, element)
                
                self.size += 1
                return True
            
            elements_seen += len(current.elements)
            current = current.next
        
        return False
    
    def delete_at(self, index):
        """Delete the element at a given index, O(n)"""
        if index < 0 or index >= self.size:
            return False
        
        # Special case: deleting the first element
        if index == 0 and self.head:
            if len(self.head.elements) > 1:
                self.head.remove_element(0)
            else:
                self.head = self.head.next
            
            self.size -= 1
            return True
        
        # Find the node containing the element
        prev = None
        current = self.head
        elements_seen = 0
        
        while current:
            if index < elements_seen + len(current.elements):
                # The element is in this node
                delete_pos = index - elements_seen
                current.remove_element(delete_pos)
                
                # If the node is now empty, remove it
                if current.is_empty() and prev:
                    prev.next = current.next
                
                self.size -= 1
                return True
            
            prev = current
            elements_seen += len(current.elements)
            current = current.next
        
        return False
    
    def __len__(self):
        """Return the size of the list"""
        return self.size
    
    def __str__(self):
        """String representation of the list"""
        if self.is_empty():
            return "[]"
        
        elements = []
        current = self.head
        
        while current:
            elements.extend(str(e) for e in current.elements)
            current = current.next
        
        return "[" + ", ".join(elements) + "]"
    
    def debug_print(self):
        """Print the internal structure for debugging"""
        if self.is_empty():
            print("Empty list")
            return
        
        node_index = 0
        current = self.head
        
        while current:
            print(f"Node {node_index}: {current.elements} (capacity: {current.capacity})")
            current = current.next
            node_index += 1

# Example usage
ull = UnrolledLinkedList(node_capacity=4)

# Add elements
for i in range(1, 11):
    ull.append(i)

print("Unrolled Linked List:", ull)  # Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("Size:", len(ull))  # Output: 10

# Show internal structure
print("\nInternal structure:")
ull.debug_print()

# Get elements
print("\nGet element at index 5:", ull.get_at(5))  # Output: 6
print("Get element at index 9:", ull.get_at(9))  # Output: 10

# Insert elements
ull.insert_at(5, 99)  # Insert 99 at index 5
print("\nAfter inserting 99 at index 5:", ull)  # Output: [1, 2, 3, 4, 5, 99, 6, 7, 8, 9, 10]

# Delete elements
ull.delete_at(5)  # Delete element at index 5 (99)
print("After deleting element at index 5:", ull)  # Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Add more elements to trigger node splits
for i in range(11, 20):
    ull.append(i)

print("\nAfter adding more elements:", ull)
print("Size:", len(ull))

# Show updated internal structure
print("\nUpdated internal structure:")
ull.debug_print()
```

Self-organizing List

When: Access patterns favor recently/frequently accessed items
Why: Adapts to usage patterns, moves frequently accessed items to front
Examples: Cache implementations, compression algorithms

```python
# Python implementation of a Self-organizing List
class Node:
    def __init__(self, data=None):
        """Initialize a node with data and next pointer"""
        self.data = data
        self.next = None
        self.access_count = 0  # Track access frequency for MFU strategy

class SelfOrganizingList:
    def __init__(self, strategy="move-to-front"):
        """Initialize an empty self-organizing list
        
        Strategies:
        - "move-to-front": Move accessed element to front (MTF)
        - "transpose": Swap accessed element with previous element
        - "count": Maintain access count, reorder by frequency (MFU)
        """
        self.head = None
        self.size = 0
        self.strategy = strategy
    
    def is_empty(self):
        """Check if the list is empty"""
        return self.head is None
    
    def append(self, data):
        """Add a new node at the end of the list, O(n)"""
        new_node = Node(data)
        
        if self.is_empty():
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        
        self.size += 1
    
    def find(self, data):
        """Find data and reorganize based on strategy, O(n)"""
        if self.is_empty():
            return None
        
        current = self.head
        prev = None
        
        while current and current.data != data:
            prev = current
            current = current.next
        
        if current is None:
            return None  # Data not found
        
        # Increment access count for MFU strategy
        current.access_count += 1
        
        # Reorganize based on strategy
        if self.strategy == "move-to-front":
            self._move_to_front(current, prev)
        elif self.strategy == "transpose":
            self._transpose(current, prev)
        elif self.strategy == "count":
            self._reorder_by_count()
        
        return current
    
    def _move_to_front(self, node, prev):
        """Move the accessed node to the front (MTF strategy)"""
        if prev is None:  # Already at front
            return
        
        prev.next = node.next
        node.next = self.head
        self.head = node
    
    def _transpose(self, node, prev):
        """Swap the accessed node with its previous one (Transpose strategy)"""
        if prev is None:  # Already at front
            return
        
        # Swap node with its previous node
        prev_next = prev.next
        node_next = node.next
        
        if prev_next == node:  # Node is directly after prev
            prev.next = node_next
            node.next = prev
            
            # Update head if needed
            if self.head == prev:
                self.head = node
        else:
            # Find the node before prev
            prev_prev = self.head
            while prev_prev and prev_prev.next != prev:
                prev_prev = prev_prev.next
            
            if prev_prev:
                prev_prev.next = node
            else:
                self.head = node
            
            prev.next = node_next
            node.next = prev
    
    def _reorder_by_count(self):
        """Reorder the list by access count (MFU strategy)"""
        # Create a list of nodes sorted by access count
        nodes = []
        current = self.head
        while current:
            nodes.append(current)
            current = current.next
        
        # Sort nodes by access count (descending)
        nodes.sort(key=lambda x: x.access_count, reverse=True)
        
        # Rebuild the list
        if nodes:
            self.head = nodes[0]
            for i in range(len(nodes) - 1):
                nodes[i].next = nodes[i + 1]
            nodes[-1].next = None
    
    def delete(self, data):
        """Delete the first occurrence of data, O(n)"""
        if self.is_empty():
            return False
        
        current = self.head
        prev = None
        
        while current and current.data != data:
            prev = current
            current = current.next
        
        if current is None:
            return False  # Data not found
        
        if prev is None:
            # Deleting the head
            self.head = current.next
        else:
            prev.next = current.next
        
        self.size -= 1
        return True
    
    def __len__(self):
        """Return the size of the list"""
        return self.size
    
    def __str__(self):
        """String representation of the list"""
        if self.is_empty():
            return "[]"
        
        elements = []
        current = self.head
        while current:
            elements.append(f"{current.data}({current.access_count})")
            current = current.next
        
        return "[" + ", ".join(elements) + "]"
    
    def debug_print(self):
        """Print the list with access counts"""
        print(f"Self-organizing List (strategy: {self.strategy}):")
        print(f"List: {self}")
        
        if self.strategy == "count":
            print("\nAccess counts:")
            current = self.head
            while current:
                print(f"  {current.data}: {current.access_count}")
                current = current.next

# Example usage
# Move-to-Front strategy
mtf_list = SelfOrganizingList(strategy="move-to-front")
for i in range(1, 6):
    mtf_list.append(i)

print("Initial MTF list:", mtf_list)
print("Size:", len(mtf_list))

# Access elements to trigger reorganization
print("\nAccessing element 3:")
mtf_list.find(3)
print("After accessing 3:", mtf_list)

print("\nAccessing element 5:")
mtf_list.find(5)
print("After accessing 5:", mtf_list)

# Transpose strategy
transpose_list = SelfOrganizingList(strategy="transpose")
for i in range(1, 6):
    transpose_list.append(i)

print("\n\nInitial Transpose list:", transpose_list)
print("Size:", len(transpose_list))

# Access elements to trigger reorganization
print("\nAccessing element 3:")
transpose_list.find(3)
print("After accessing 3:", transpose_list)

print("\nAccessing element 3 again:")
transpose_list.find(3)
print("After accessing 3 again:", transpose_list)

# Count (MFU) strategy
mfu_list = SelfOrganizingList(strategy="count")
for i in range(1, 6):
    mfu_list.append(i)

print("\n\nInitial MFU list:", mfu_list)
print("Size:", len(mfu_list))

# Access elements multiple times to build access counts
for _ in range(3):
    mfu_list.find(3)
for _ in range(2):
    mfu_list.find(5)
for _ in range(4):
    mfu_list.find(2)

print("\nAfter building access counts:")
mfu_list.debug_print()
```

Stacks and Queues
Stack

When: Need LIFO (Last In First Out) behavior
Why: O(1) push/pop operations, natural for recursive problems
Examples: Function call stack, expression evaluation, backtracking algorithms, undo/redo

```python
# Python implementation of a Stack
class Stack:
    def __init__(self):
        """Initialize an empty stack"""
        self.items = []
    
    def is_empty(self):
        """Check if the stack is empty"""
        return len(self.items) == 0
    
    def push(self, item):
        """Add an item to the top of the stack, O(1)"""
        self.items.append(item)
    
    def pop(self):
        """Remove and return the top item, O(1)"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self.items.pop()
    
    def peek(self):
        """Return the top item without removing it, O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.items[-1]
    
    def size(self):
        """Return the number of items in the stack"""
        return len(self.items)
    
    def __str__(self):
        """String representation of the stack"""
        return str(self.items)

# Example usage
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)

print("Stack:", stack)  # Output: [1, 2, 3]
print("Top item:", stack.peek())  # Output: 3
print("Pop:", stack.pop())  # Output: 3
print("Stack after pop:", stack)  # Output: [1, 2]
print("Size:", stack.size())  # Output: 2

# Example: Expression evaluation
def evaluate_postfix(expression):
    """Evaluate a postfix expression using a stack"""
    stack = Stack()
    operators = set(['+', '-', '*', '/', '^'])
    
    for token in expression.split():
        if token.isdigit():
            stack.push(int(token))
        elif token in operators:
            if stack.size() < 2:
                raise ValueError("Invalid expression")
            
            right = stack.pop()
            left = stack.pop()
            
            if token == '+':
                stack.push(left + right)
            elif token == '-':
                stack.push(left - right)
            elif token == '*':
                stack.push(left * right)
            elif token == '/':
                stack.push(left / right)
            elif token == '^':
                stack.push(left ** right)
        else:
            raise ValueError(f"Invalid token: {token}")
    
    if stack.size() != 1:
        raise ValueError("Invalid expression")
    
    return stack.pop()

# Evaluate a postfix expression
expression = "3 4 2 * 1 5 - / +"
result = evaluate_postfix(expression)
print(f"\nExpression: {expression}")
print(f"Result: {result}")  # Output: 3.6666666666666665

# Example: Backtracking algorithm (maze solving)
def solve_maze(maze, start, end):
    """Solve a maze using backtracking with a stack"""
    rows, cols = len(maze), len(maze[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    stack = Stack()
    
    # Push the starting position
    stack.push((start[0], start[1], []))  # (row, col, path)
    visited[start[0]][start[1]] = True
    
    while not stack.is_empty():
        row, col, path = stack.pop()
        path = path + [(row, col)]
        
        # Check if we've reached the end
        if (row, col) == end:
            return path
        
        # Explore all possible moves (up, down, left, right)
        moves = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        for r, c in moves:
            if (0 <= r < rows and 0 <= c < cols and
                not visited[r][c] and maze[r][c] == 0):
                visited[r][c] = True
                stack.push((r, c, path))
    
    return None  # No path found

# Example maze (0 = path, 1 = wall)
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
end = (4, 4)

path = solve_maze(maze, start, end)
if path:
    print(f"\nMaze path from {start} to {end}:")
    for pos in path:
        print(pos)
else:
    print(f"\nNo path found from {start} to {end}")
```

Queue

When: Need FIFO (First In First Out) behavior
Why: O(1) enqueue/dequeue, fair ordering
Examples: Task scheduling, BFS traversal, print queue, request handling

```python
# Python implementation of a Queue
class Queue:
    def __init__(self):
        """Initialize an empty queue"""
        self.items = []
        self.front = 0  # Index of the front element
        self.rear = 0   # Index where next element will be inserted
    
    def is_empty(self):
        """Check if the queue is empty"""
        return self.front == self.rear
    
    def enqueue(self, item):
        """Add an item to the rear of the queue, O(1) amortized"""
        self.items.append(item)
        self.rear += 1
    
    def dequeue(self):
        """Remove and return the front item, O(1) amortized"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        
        item = self.items[self.front]
        self.front += 1
        
        # If we've removed a lot of elements, reclaim space
        if self.front > len(self.items) // 2:
            self.items = self.items[self.front:]
            self.rear = len(self.items)
            self.front = 0
        
        return item
    
    def peek(self):
        """Return the front item without removing it, O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self.items[self.front]
    
    def size(self):
        """Return the number of items in the queue"""
        return self.rear - self.front
    
    def __str__(self):
        """String representation of the queue"""
        return str(self.items[self.front:self.rear])

# Example usage
queue = Queue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)

print("Queue:", queue)  # Output: [1, 2, 3]
print("Front item:", queue.peek())  # Output: 1
print("Dequeue:", queue.dequeue())  # Output: 1
print("Queue after dequeue:", queue)  # Output: [2, 3]
print("Size:", queue.size())  # Output: 2

# Example: BFS traversal of a graph
class Graph:
    def __init__(self, vertices):
        """Initialize a graph with the given number of vertices"""
        self.V = vertices
        self.adj = [[] for _ in range(vertices)]
    
    def add_edge(self, u, v):
        """Add an edge between vertices u and v"""
        self.adj[u].append(v)
        self.adj[v].append(u)  # For undirected graph
    
    def bfs(self, start):
        """Breadth-First Search starting from vertex start"""
        visited = [False] * self.V
        result = []
        queue = Queue()
        
        # Mark the starting vertex as visited and enqueue it
        visited[start] = True
        queue.enqueue(start)
        
        while not queue.is_empty():
            # Dequeue a vertex and add it to the result
            u = queue.dequeue()
            result.append(u)
            
            # Visit all adjacent vertices
            for v in self.adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.enqueue(v)
        
        return result

# Create a graph and perform BFS
g = Graph(6)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.add_edge(1, 4)
g.add_edge(2, 4)
g.add_edge(3, 5)

print("\nBFS traversal starting from vertex 0:")
bfs_result = g.bfs(0)
print(bfs_result)  # Output: [0, 1, 2, 3, 4, 5]

# Example: Task scheduling
class TaskScheduler:
    def __init__(self):
        """Initialize a task scheduler"""
        self.queue = Queue()
    
    def add_task(self, task):
        """Add a task to the scheduler"""
        self.queue.enqueue(task)
        print(f"Task '{task}' added to the queue")
    
    def process_tasks(self):
        """Process all tasks in the queue"""
        print("\nProcessing tasks:")
        while not self.queue.is_empty():
            task = self.queue.dequeue()
            print(f"Processing task: {task}")
        print("All tasks completed!")

# Create a task scheduler and add tasks
scheduler = TaskScheduler()
scheduler.add_task("Initialize system")
scheduler.add_task("Load configuration")
scheduler.add_task("Start services")
scheduler.add_task("Run health check")

# Process all tasks
scheduler.process_tasks()
```

Deque (Double-ended Queue)

When: Need to add/remove from both ends efficiently
Why: O(1) operations at both ends, combines stack and queue functionality
Examples: Sliding window problems, palindrome checking, work stealing in thread pools

```python
# Python implementation of a Deque (Double-ended Queue)
class Deque:
    def __init__(self):
        """Initialize an empty deque"""
        self.items = []
    
    def is_empty(self):
        """Check if deque is empty"""
        return len(self.items) == 0
    
    def append(self, item):
        """Add an item to right end, O(1) amortized"""
        self.items.append(item)
    
    def appendleft(self, item):
        """Add an item to left end, O(n)"""
        self.items.insert(0, item)
    
    def pop(self):
        """Remove and return rightmost item, O(1) amortized"""
        if self.is_empty():
            raise IndexError("pop from empty deque")
        return self.items.pop()
    
    def popleft(self):
        """Remove and return leftmost item, O(n)"""
        if self.is_empty():
            raise IndexError("popleft from empty deque")
        return self.items.pop(0)
    
    def peek(self):
        """Return rightmost item without removing it, O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty deque")
        return self.items[-1]
    
    def peekleft(self):
        """Return leftmost item without removing it, O(1)"""
        if self.is_empty():
            raise IndexError("peekleft from empty deque")
        return self.items[0]
    
    def size(self):
        """Return the number of items in the deque"""
        return len(self.items)
    
    def __str__(self):
        """String representation of the deque"""
        return str(self.items)

# Example usage
deque = Deque()
deque.append(1)
deque.append(2)
deque.appendleft(0)

print("Deque:", deque)  # Output: [0, 1, 2]
print("Rightmost item:", deque.peek())  # Output: 2
print("Leftmost item:", deque.peekleft())  # Output: 0
print("Pop right:", deque.pop())  # Output: 2
print("Pop left:", deque.popleft())  # Output: 0
print("Deque after pops:", deque)  # Output: [1]

# Example: Palindrome checking
def is_palindrome(s):
    """Check if a string is a palindrome using a deque"""
    deque = Deque()
    
    # Add all characters to deque (case-insensitive, ignoring non-alphanumeric)
    for char in s.lower():
        if char.isalnum():
            deque.append(char)
    
    # Compare characters from both ends
    while deque.size() > 1:
        left = deque.popleft()
        right = deque.pop()
        if left != right:
            return False
    
    return True

# Test palindrome checking
test_strings = [
    "racecar",
    "A man, a plan, a canal: Panama",
    "hello",
    "Was it a car or a cat I saw?"
]

print("\nPalindrome checking:")
for s in test_strings:
    result = is_palindrome(s)
    print(f"'{s}': {'Palindrome' if result else 'Not a palindrome'}")

# Example: Sliding window maximum
def sliding_window_max(nums, k):
    """Find maximum in each sliding window of size k"""
    if not nums or k <= 0:
        return []
    
    result = []
    deque = Deque()  # Store indices of elements
    
    for i, num in enumerate(nums):
        # Remove indices that are out of the current window
        while not deque.is_empty() and deque.peekleft() <= i - k:
            deque.popleft()
        
        # Remove indices whose corresponding values are less than the current value
        while not deque.is_empty() and nums[deque.peek()] < num:
            deque.pop()
        
        # Add current index
        deque.append(i)
        
        # Add maximum of current window to result
        if i >= k - 1:
            result.append(nums[deque.peekleft()])
    
    return result

# Test sliding window maximum
print("\nSliding window maximum:")
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
result = sliding_window_max(nums, k)
print(f"Array: {nums}")
print(f"Window size: {k}")
print(f"Maximum in each window: {result}")  # Output: [3, 3, 5, 5, 6, 7]

# Example: Work-stealing queue for thread pools
class WorkStealingQueue:
    def __init__(self):
        """Initialize a work-stealing queue"""
        self.deque = Deque()
    
    def push(self, task):
        """Add a task to the local end (right)"""
        self.deque.append(task)
    
    def pop(self):
        """Remove a task from the local end (right)"""
        return self.deque.pop()
    
    def steal(self):
        """Steal a task from the remote end (left)"""
        return self.deque.popleft()
    
    def is_empty(self):
        """Check if the queue is empty"""
        return self.deque.is_empty()
    
    def size(self):
        """Return the number of tasks in the queue"""
        return self.deque.size()

# Simulate work-stealing between threads
print("\nWork-stealing simulation:")
local_queue = WorkStealingQueue()
for i in range(1, 6):
    local_queue.push(f"Task-{i}")

print(f"Local queue size: {local_queue.size()}")

# Local worker processes tasks
while local_queue.size() > 2:
    task = local_queue.pop()
    print(f"Local worker processed: {task}")

print(f"Local queue size after processing: {local_queue.size()}")

# Remote worker steals tasks
while not local_queue.is_empty():
    task = local_queue.steal()
    print(f"Remote worker stole: {task}")

print(f"Local queue size after stealing: {local_queue.size()}")
```

Priority Queue

When: Need to always access the minimum/maximum element
Why: O(log n) insertion, O(1) access to min/max, O(log n) deletion
Examples: Dijkstra's algorithm, task scheduling with priorities, event simulation, Huffman coding

```python
# Python implementation of a Priority Queue using a min-heap
import heapq

class PriorityQueue:
    def __init__(self):
        """Initialize an empty priority queue"""
        self.heap = []
    
    def is_empty(self):
        """Check if the priority queue is empty"""
        return len(self.heap) == 0
    
    def insert(self, item, priority):
        """Insert an item with the given priority, O(log n)"""
        heapq.heappush(self.heap, (priority, item))
    
    def extract_min(self):
        """Remove and return the item with the minimum priority, O(log n)"""
        if self.is_empty():
            raise IndexError("extract_min from empty priority queue")
        return heapq.heappop(self.heap)[1]
    
    def peek(self):
        """Return the item with the minimum priority without removing it, O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty priority queue")
        return self.heap[0][1]
    
    def size(self):
        """Return the number of items in the priority queue"""
        return len(self.heap)
    
    def __str__(self):
        """String representation of the priority queue"""
        return str([(p, i) for p, i in self.heap])

# Example usage
pq = PriorityQueue()
pq.insert("Task 1", 3)
pq.insert("Task 2", 1)
pq.insert("Task 3", 2)

print("Priority Queue:", pq)  # Output: [(1, 'Task 2'), (3, 'Task 1'), (2, 'Task 3')]
print("Minimum priority item:", pq.peek())  # Output: Task 2
print("Extract minimum:", pq.extract_min())  # Output: Task 2
print("Priority Queue after extraction:", pq)  # Output: [(2, 'Task 3'), (3, 'Task 1')]
print("Size:", pq.size())  # Output: 2

# Example: Task scheduling with priorities
class TaskScheduler:
    def __init__(self):
        """Initialize a task scheduler with a priority queue"""
        self.pq = PriorityQueue()
    
    def add_task(self, task, priority):
        """Add a task with the given priority"""
        self.pq.insert(task, priority)
        print(f"Added task '{task}' with priority {priority}")
    
    def process_tasks(self):
        """Process all tasks in priority order"""
        print("\nProcessing tasks in priority order:")
        while not self.pq.is_empty():
            task = self.pq.extract_min()
            print(f"Processing task: {task}")
        print("All tasks completed!")

# Create a task scheduler and add tasks
scheduler = TaskScheduler()
scheduler.add_task("Critical bug fix", 1)
scheduler.add_task("Feature development", 3)
scheduler.add_task("Documentation update", 4)
scheduler.add_task("Code review", 2)

# Process all tasks
scheduler.process_tasks()

# Example: Dijkstra's algorithm using priority queue
def dijkstra(graph, start):
    """Find shortest paths from start to all vertices using Dijkstra's algorithm"""
    # Initialize distances and priority queue
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    pq = PriorityQueue()
    pq.insert(start, 0)
    
    # Track visited vertices
    visited = set()
    
    while not pq.is_empty():
        # Get vertex with minimum distance
        current = pq.extract_min()
        
        # Skip if already visited
        if current in visited:
            continue
        
        # Mark as visited
        visited.add(current)
        
        # Update distances to neighbors
        for neighbor, weight in graph[current].items():
            if neighbor not in visited:
                distance = distances[current] + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    pq.insert(neighbor, distance)
    
    return distances

# Example graph (adjacency list representation)
graph = {
    'A': {'B': 5, 'C': 1},
    'B': {'A': 5, 'C': 2, 'D': 1},
    'C': {'A': 1, 'B': 2, 'D': 4, 'E': 8},
    'D': {'B': 1, 'C': 4, 'E': 3, 'F': 6},
    'E': {'C': 8, 'D': 3},
    'F': {'D': 6}
}

start_vertex = 'A'
distances = dijkstra(graph, start_vertex)

print(f"\nShortest distances from {start_vertex}:")
for vertex, distance in distances.items():
    print(f"{vertex}: {distance}")

# Example: Huffman coding using priority queue
class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        """Initialize a Huffman tree node"""
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        """Comparison operator for priority queue"""
        return self.freq < other.freq

def huffman_encoding(frequencies):
    """Build Huffman tree and return codes for each character"""
    # Create initial nodes for each character
    pq = PriorityQueue()
    for char, freq in frequencies.items():
        pq.insert(HuffmanNode(char, freq), freq)
    
    # Build the Huffman tree
    while pq.size() > 1:
        # Extract two nodes with minimum frequencies
        left = pq.extract_min()
        right = pq.extract_min()
        
        # Create a new internal node
        combined_freq = left.freq + right.freq
        combined_node = HuffmanNode(None, combined_freq, left, right)
        pq.insert(combined_node, combined_freq)
    
    # Generate codes from the tree
    root = pq.extract_min()
    codes = {}
    
    def generate_codes(node, code=""):
        if node is None:
            return
        
        # If it's a leaf node, store the code
        if node.char is not None:
            codes[node.char] = code
            return
        
        # Recursively generate codes for left and right children
        generate_codes(node.left, code + "0")
        generate_codes(node.right, code + "1")
    
    generate_codes(root)
    return codes

# Example frequencies for characters
char_frequencies = {
    'A': 5,
    'B': 9,
    'C': 12,
    'D': 13,
    'E': 16,
    'F': 45
}

# Generate Huffman codes
huffman_codes = huffman_encoding(char_frequencies)

print("\nHuffman codes:")
for char, code in sorted(huffman_codes.items()):
    print(f"{char}: {code}")
```

Circular Queue

When: Fixed-size buffer with wraparound needed
Why: Efficient use of fixed memory, no shifting required
Examples: Buffering data streams, producer-consumer problems, ring buffers in hardware

```python
# Python implementation of a Circular Queue (Ring Buffer)
class CircularQueue:
    def __init__(self, capacity):
        """Initialize a circular queue with given capacity"""
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = 0  # Index of front element
        self.rear = 0   # Index where next element will be inserted
        self.size = 0   # Current number of elements
    
    def is_empty(self):
        """Check if circular queue is empty"""
        return self.size == 0
    
    def is_full(self):
        """Check if circular queue is full"""
        return self.size == self.capacity
    
    def enqueue(self, item):
        """Add an item to rear of queue, O(1)"""
        if self.is_full():
            raise OverflowError("Queue is full")
        
        self.queue[self.rear] = item
        self.rear = (self.rear + 1) % self.capacity
        self.size += 1
    
    def dequeue(self):
        """Remove and return front item, O(1)"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        
        item = self.queue[self.front]
        self.queue[self.front] = None  # Optional: Clear reference
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        
        return item
    
    def peek(self):
        """Return front item without removing it, O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self.queue[self.front]
    
    def size(self):
        """Return number of items in queue"""
        return self.size
    
    def __str__(self):
        """String representation of circular queue"""
        if self.is_empty():
            return "[]"
        
        elements = []
        for i in range(self.size):
            index = (self.front + i) % self.capacity
            elements.append(str(self.queue[index]))
        
        return "[" + ", ".join(elements) + "]"

# Example usage
cq = CircularQueue(5)

print("Initial circular queue:", cq)  # Output: []
print("Is empty:", cq.is_empty())  # Output: True

# Enqueue elements
cq.enqueue(1)
cq.enqueue(2)
cq.enqueue(3)

print("\nAfter enqueuing 1, 2, 3:")
print("Circular queue:", cq)  # Output: [1, 2, 3]
print("Front item:", cq.peek())  # Output: 1
print("Size:", cq.size())  # Output: 3

# Dequeue elements
print("\nDequeued:", cq.dequeue())  # Output: 1
print("Circular queue after dequeue:", cq)  # Output: [2, 3]

# Fill queue
cq.enqueue(4)
cq.enqueue(5)
cq.enqueue(6)  # This will wrap around

print("\nAfter filling queue:")
print("Circular queue:", cq)  # Output: [2, 3, 4, 5, 6]
print("Is full:", cq.is_full())  # Output: True

# Try to enqueue when full
try:
    cq.enqueue(7)
except OverflowError as e:
    print("\nError when enqueuing to full queue:", e)

# Example: Producer-Consumer pattern
import time
import random
import threading

class ProducerConsumer:
    def __init__(self, buffer_size):
        """Initialize producer-consumer with a circular buffer"""
        self.buffer = CircularQueue(buffer_size)
        self.produced_count = 0
        self.consumed_count = 0
        self.running = True
    
    def producer(self, items_to_produce):
        """Producer function that adds items to buffer"""
        for i in range(items_to_produce):
            if not self.running:
                break
            
            # Wait if buffer is full
            while self.buffer.is_full() and self.running:
                time.sleep(0.1)
            
            if self.running:
                item = f"Item-{self.produced_count + 1}"
                self.buffer.enqueue(item)
                self.produced_count += 1
                print(f"Produced: {item}, Buffer: {self.buffer}")
                time.sleep(random.uniform(0.1, 0.5))
    
    def consumer(self, items_to_consume):
        """Consumer function that removes items from buffer"""
        for i in range(items_to_consume):
            if not self.running:
                break
            
            # Wait if buffer is empty
            while self.buffer.is_empty() and self.running:
                time.sleep(0.1)
            
            if self.running:
                item = self.buffer.dequeue()
                self.consumed_count += 1
                print(f"Consumed: {item}, Buffer: {self.buffer}")
                time.sleep(random.uniform(0.2, 0.6))
    
    def stop(self):
        """Stop the producer-consumer simulation"""
        self.running = False

# Simulate producer-consumer
print("\nProducer-Consumer Simulation:")
pc = ProducerConsumer(3)

# Create producer and consumer threads
producer_thread = threading.Thread(target=pc.producer, args=(10,))
consumer_thread = threading.Thread(target=pc.consumer, args=(8,))

# Start threads
producer_thread.start()
consumer_thread.start()

# Let them run for a bit
time.sleep(2)

# Stop the simulation
pc.stop()

# Wait for threads to complete
producer_thread.join()
consumer_thread.join()

print(f"\nSimulation complete. Produced: {pc.produced_count}, Consumed: {pc.consumed_count}")

# Example: Sliding window with circular queue
class SlidingWindow:
    def __init__(self, window_size):
        """Initialize a sliding window with given size"""
        self.window = CircularQueue(window_size)
        self.sum = 0
    
    def add(self, value):
        """Add a new value to sliding window"""
        # If window is full, remove the oldest value from sum
        if self.window.is_full():
            oldest = self.window.dequeue()
            self.sum -= oldest
        
        # Add the new value
        self.window.enqueue(value)
        self.sum += value
    
    def average(self):
        """Calculate average of values in window"""
        if self.window.is_empty():
            return 0
        return self.sum / self.window.size()
    
    def __str__(self):
        """String representation of sliding window"""
        return str(self.window)

# Example usage of sliding window
print("\nSliding Window Example:")
sw = SlidingWindow(3)

values = [5, 10, 15, 20, 25]
for val in values:
    sw.add(val)
    print(f"Added {val}, Window: {sw}, Average: {sw.average():.2f}")
```

Tree Data Structures
Binary Trees
Binary Tree

When: Hierarchical data without specific ordering requirements
Why: Simple structure, foundation for other tree types
Examples: Expression trees, decision trees, organizational hierarchies

```python
# Python implementation of a Binary Tree
class Node:
    def __init__(self, data=None):
        """Initialize a node with data and left/right child pointers"""
        self.data = data
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        """Initialize an empty binary tree"""
        self.root = None
    
    def is_empty(self):
        """Check if the tree is empty"""
        return self.root is None
    
    def insert(self, data):
        """Insert data as a new node in the first available position (level-order)"""
        new_node = Node(data)
        
        if self.is_empty():
            self.root = new_node
            return
        
        # Use a queue for level-order traversal to find the first available position
        queue = [self.root]
        
        while queue:
            current = queue.pop(0)
            
            # Check left child
            if current.left is None:
                current.left = new_node
                return
            else:
                queue.append(current.left)
            
            # Check right child
            if current.right is None:
                current.right = new_node
                return
            else:
                queue.append(current.right)
    
    def preorder_traversal(self, node=None, result=None):
        """Root -> Left -> Right traversal"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            result.append(node.data)
            self.preorder_traversal(node.left, result)
            self.preorder_traversal(node.right, result)
        
        return result
    
    def inorder_traversal(self, node=None, result=None):
        """Left -> Root -> Right traversal"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            self.inorder_traversal(node.left, result)
            result.append(node.data)
            self.inorder_traversal(node.right, result)
        
        return result
    
    def postorder_traversal(self, node=None, result=None):
        """Left -> Right -> Root traversal"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            self.postorder_traversal(node.left, result)
            self.postorder_traversal(node.right, result)
            result.append(node.data)
        
        return result
    
    def levelorder_traversal(self):
        """Breadth-first traversal"""
        if self.is_empty():
            return []
        
        result = []
        queue = [self.root]
        
        while queue:
            current = queue.pop(0)
            result.append(current.data)
            
            if current.left:
                queue.append(current.left)
            
            if current.right:
                queue.append(current.right)
        
        return result
    
    def height(self, node=None):
        """Calculate the height of the tree"""
        if node is None:
            node = self.root
        
        if node is None:
            return -1  # Height of empty tree is -1
        
        left_height = self.height(node.left)
        right_height = self.height(node.right)
        
        return max(left_height, right_height) + 1
    
    def size(self, node=None):
        """Count the number of nodes in the tree"""
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        return 1 + self.size(node.left) + self.size(node.right)
    
    def __str__(self):
        """String representation of the tree (level-order)"""
        return str(self.levelorder_traversal())

# Example usage
bt = BinaryTree()

# Insert elements to form a binary tree
bt.insert(1)
bt.insert(2)
bt.insert(3)
bt.insert(4)
bt.insert(5)
bt.insert(6)
bt.insert(7)

print("Binary Tree:", bt)  # Output: [1, 2, 3, 4, 5, 6, 7]
print("Height:", bt.height())  # Output: 2
print("Size:", bt.size())  # Output: 7

# Tree traversals
print("\nPre-order traversal:", bt.preorder_traversal())  # Output: [1, 2, 4, 5, 3, 6, 7]
print("In-order traversal:", bt.inorder_traversal())    # Output: [4, 2, 5, 1, 6, 3, 7]
print("Post-order traversal:", bt.postorder_traversal()) # Output: [4, 5, 2, 6, 7, 3, 1]
print("Level-order traversal:", bt.levelorder_traversal()) # Output: [1, 2, 3, 4, 5, 6, 7]

# Example: Expression tree
class ExpressionTree(BinaryTree):
    def __init__(self):
        """Initialize an empty expression tree"""
        super().__init__()
    
    def build_from_postfix(self, expression):
        """Build an expression tree from a postfix expression"""
        stack = []
        operators = set(['+', '-', '*', '/', '^'])
        
        for token in expression.split():
            if token not in operators:
                # Operand, create a node and push to stack
                node = Node(token)
                stack.append(node)
            else:
                # Operator, create a node with top two elements as children
                node = Node(token)
                
                # Pop the top two elements (right operand first)
                right = stack.pop()
                left = stack.pop()
                
                node.right = right
                node.left = left
                
                stack.append(node)
        
        if stack:
            self.root = stack.pop()
    
    def evaluate(self, node=None):
        """Evaluate the expression tree"""
        if node is None:
            node = self.root
        
        if node is None:
            return None
        
        # If it's a leaf node (operand), return its value
        if node.left is None and node.right is None:
            try:
                return float(node.data)
            except ValueError:
                return node.data  # Return as string if not a number
        
        # Evaluate left and right subtrees
        left_val = self.evaluate(node.left)
        right_val = self.evaluate(node.right)
        
        # Apply the operator
        if node.data == '+':
            return left_val + right_val
        elif node.data == '-':
            return left_val - right_val
        elif node.data == '*':
            return left_val * right_val
        elif node.data == '/':
            return left_val / right_val
        elif node.data == '^':
            return left_val ** right_val
        else:
            raise ValueError(f"Unknown operator: {node.data}")

# Create and evaluate an expression tree
expr_tree = ExpressionTree()
postfix_expr = "3 4 2 * 1 5 - / +"
expr_tree.build_from_postfix(postfix_expr)

print("\nExpression Tree:")
print("Postfix expression:", postfix_expr)
print("In-order traversal:", expr_tree.inorder_traversal())
print("Evaluation result:", expr_tree.evaluate())  # Output: 3.6666666666666665

# Example: Decision tree for a simple game
class DecisionTree:
    def __init__(self, question=None):
        """Initialize a decision tree node"""
        self.question = question
        self.yes_branch = None
        self.no_branch = None
    
    def add_yes_branch(self, node):
        """Add a branch for 'yes' answer"""
        self.yes_branch = node
    
    def add_no_branch(self, node):
        """Add a branch for 'no' answer"""
        self.no_branch = node
    
    def query(self):
        """Traverse the decision tree based on user input"""
        if self.yes_branch is None and self.no_branch is None:
            # This is a leaf node (answer)
            print(self.question)
            return
        
        # This is a decision node
        answer = input(f"{self.question} (yes/no): ").lower()
        
        if answer == 'yes':
            if self.yes_branch:
                self.yes_branch.query()
            else:
                print("No further information for 'yes' path.")
        elif answer == 'no':
            if self.no_branch:
                self.no_branch.query()
            else:
                print("No further information for 'no' path.")
        else:
            print("Please answer 'yes' or 'no'.")
            self.query()

# Build a simple decision tree
root = DecisionTree("Is the animal a mammal?")
mammal = DecisionTree("Does it live in water?")
land_mammal = DecisionTree("Does it have stripes?")
water_mammal = DecisionTree("Is it a dolphin?")

root.add_yes_branch(mammal)
root.add_no_branch(DecisionTree("Is it a bird?"))

mammal.add_yes_branch(water_mammal)
mammal.add_no_branch(land_mammal)

land_mammal.add_yes_branch(DecisionTree("It's a zebra!"))
land_mammal.add_no_branch(DecisionTree("It's a lion!"))

water_mammal.add_yes_branch(DecisionTree("It's a dolphin!"))
water_mammal.add_no_branch(DecisionTree("It's a whale!"))

print("\nDecision Tree Example:")
print("(This is an interactive example - in a real program, it would ask for input)")
print("Tree structure:")
print("Root: Is the animal a mammal?")
print("  Yes: Does it live in water?")
print("    Yes: Is it a dolphin?")
print("      Yes: It's a dolphin!")
print("      No: It's a whale!")
print("    No: Does it have stripes?")
print("      Yes: It's a zebra!")
print("      No: It's a lion!")
print("  No: Is it a bird?")
```

Binary Search Tree (BST)

When: Need sorted data with decent search/insert/delete (without balance guarantee)
Why: O(log n) average case, simple implementation, in-order traversal gives sorted sequence
Examples: Simple symbol tables, when data is roughly random, educational purposes

```python
# Python implementation of a Binary Search Tree (BST)
class Node:
    def __init__(self, data=None):
        """Initialize a node with data and left/right child pointers"""
        self.data = data
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        """Initialize an empty BST"""
        self.root = None
    
    def is_empty(self):
        """Check if BST is empty"""
        return self.root is None
    
    def insert(self, data):
        """Insert data into BST"""
        if self.is_empty():
            self.root = Node(data)
            return
        
        self._insert_recursive(self.root, data)
    
    def _insert_recursive(self, node, data):
        """Helper method to insert data recursively"""
        if data < node.data:
            # Insert in left subtree
            if node.left is None:
                node.left = Node(data)
            else:
                self._insert_recursive(node.left, data)
        elif data > node.data:
            # Insert in right subtree
            if node.right is None:
                node.right = Node(data)
            else:
                self._insert_recursive(node.right, data)
        # If data == node.data, we can either ignore or update
        # Here we're ignoring duplicates
    
    def search(self, data):
        """Search for data in BST"""
        return self._search_recursive(self.root, data)
    
    def _search_recursive(self, node, data):
        """Helper method to search recursively"""
        if node is None:
            return False
        
        if data == node.data:
            return True
        elif data < node.data:
            return self._search_recursive(node.left, data)
        else:  # data > node.data
            return self._search_recursive(node.right, data)
    
    def delete(self, data):
        """Delete data from BST"""
        self.root = self._delete_recursive(self.root, data)
    
    def _delete_recursive(self, node, data):
        """Helper method to delete recursively"""
        if node is None:
            return None
        
        if data < node.data:
            node.left = self._delete_recursive(node.left, data)
        elif data > node.data:
            node.right = self._delete_recursive(node.right, data)
        else:  # data == node.data, this is the node to delete
            # Case 1: Node with no children
            if node.left is None and node.right is None:
                return None
            
            # Case 2: Node with one child
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            
            # Case 3: Node with two children
            # Find inorder successor (smallest in right subtree)
            successor = self._find_min(node.right)
            node.data = successor.data
            node.right = self._delete_recursive(node.right, successor.data)
        
        return node
    
    def _find_min(self, node):
        """Find node with minimum value in subtree"""
        while node.left is not None:
            node = node.left
        return node
    
    def inorder_traversal(self, node=None, result=None):
        """Left -> Root -> Right traversal (gives sorted order)"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            self.inorder_traversal(node.left, result)
            result.append(node.data)
            self.inorder_traversal(node.right, result)
        
        return result
    
    def preorder_traversal(self, node=None, result=None):
        """Root -> Left -> Right traversal"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            result.append(node.data)
            self.preorder_traversal(node.left, result)
            self.preorder_traversal(node.right, result)
        
        return result
    
    def postorder_traversal(self, node=None, result=None):
        """Left -> Right -> Root traversal"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            self.postorder_traversal(node.left, result)
            self.postorder_traversal(node.right, result)
            result.append(node.data)
        
        return result
    
    def height(self, node=None):
        """Calculate height of BST"""
        if node is None:
            node = self.root
        
        if node is None:
            return -1  # Height of empty tree is -1
        
        left_height = self.height(node.left)
        right_height = self.height(node.right)
        
        return max(left_height, right_height) + 1
    
    def size(self, node=None):
        """Count number of nodes in BST"""
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        return 1 + self.size(node.left) + self.size(node.right)
    
    def is_bst(self, node=None, min_val=float('-inf'), max_val=float('inf')):
        """Check if tree is a valid BST"""
        if node is None:
            node = self.root
        
        if node is None:
            return True
        
        # Check if current node violates BST property
        if node.data <= min_val or node.data >= max_val:
            return False
        
        # Recursively check left and right subtrees
        return (self.is_bst(node.left, min_val, node.data) and
                self.is_bst(node.right, node.data, max_val))
    
    def __str__(self):
        """String representation of BST (inorder)"""
        return str(self.inorder_traversal())

# Example usage
bst = BinarySearchTree()

# Insert elements
bst.insert(50)
bst.insert(30)
bst.insert(70)
bst.insert(20)
bst.insert(40)
bst.insert(60)
bst.insert(80)

print("BST (inorder):", bst)  # Output: [20, 30, 40, 50, 60, 70, 80]
print("Height:", bst.height())  # Output: 2
print("Size:", bst.size())  # Output: 7
print("Is valid BST:", bst.is_bst())  # Output: True

# Tree traversals
print("\nPre-order traversal:", bst.preorder_traversal())  # Output: [50, 30, 20, 40, 70, 60, 80]
print("In-order traversal:", bst.inorder_traversal())    # Output: [20, 30, 40, 50, 60, 70, 80]
print("Post-order traversal:", bst.postorder_traversal()) # Output: [20, 40, 30, 60, 80, 70, 50]

# Search operations
print("\nSearch operations:")
print("Search 40:", bst.search(40))  # Output: True
print("Search 90:", bst.search(90))  # Output: False

# Delete operations
print("\nDelete operations:")
print("Original BST:", bst.inorder_traversal())

# Delete leaf node
bst.delete(20)
print("After deleting 20 (leaf):", bst.inorder_traversal())

# Delete node with one child
bst.delete(30)
print("After deleting 30 (one child):", bst.inorder_traversal())

# Delete node with two children
bst.delete(50)
print("After deleting 50 (two children):", bst.inorder_traversal())

# Example: BST as a simple symbol table
class SymbolTable:
    def __init__(self):
        """Initialize an empty symbol table using BST"""
        self.bst = BinarySearchTree()
    
    def put(self, key, value):
        """Add or update a key-value pair"""
        if self.bst.is_empty():
            self.bst.root = Node((key, value))
        else:
            self._put_recursive(self.bst.root, key, value)
    
    def _put_recursive(self, node, key, value):
        """Helper method to put recursively"""
        if key < node.data[0]:
            if node.left is None:
                node.left = Node((key, value))
            else:
                self._put_recursive(node.left, key, value)
        elif key > node.data[0]:
            if node.right is None:
                node.right = Node((key, value))
            else:
                self._put_recursive(node.right, key, value)
        else:  # key == node.data[0], update value
            node.data = (key, value)
    
    def get(self, key):
        """Get value for a given key"""
        return self._get_recursive(self.bst.root, key)
    
    def _get_recursive(self, node, key):
        """Helper method to get recursively"""
        if node is None:
            return None
        
        if key == node.data[0]:
            return node.data[1]
        elif key < node.data[0]:
            return self._get_recursive(node.left, key)
        else:  # key > node.data[0]
            return self._get_recursive(node.right, key)
    
    def keys(self):
        """Get all keys in sorted order"""
        result = []
        self._inorder_keys(self.bst.root, result)
        return result
    
    def _inorder_keys(self, node, result):
        """Helper method to get keys in inorder"""
        if node:
            self._inorder_keys(node.left, result)
            result.append(node.data[0])
            self._inorder_keys(node.right, result)

# Create and use a symbol table
print("\nSymbol Table Example:")
st = SymbolTable()

st.put("apple", 1)
st.put("banana", 2)
st.put("cherry", 3)
st.put("date", 4)
st.put("elderberry", 5)

print("All keys:", st.keys())
print("Value of 'cherry':", st.get("cherry"))
print("Value of 'grape':", st.get("grape"))

st.put("cherry", 10)  # Update existing key
print("Updated value of 'cherry':", st.get("cherry"))
```

AVL Tree

When: Need guaranteed balanced BST with more frequent searches than modifications
Why: Strict balancing (height difference ≤ 1), best search performance O(log n) guaranteed
Examples: Databases with read-heavy workloads, in-memory databases, language compilers

```python
# Python implementation of an AVL Tree
class Node:
    def __init__(self, data=None):
        """Initialize a node with data, height, and left/right child pointers"""
        self.data = data
        self.left = None
        self.right = None
        self.height = 1  # Height of node initially 1 (leaf node)

class AVLTree:
    def __init__(self):
        """Initialize an empty AVL tree"""
        self.root = None
    
    def is_empty(self):
        """Check if AVL tree is empty"""
        return self.root is None
    
    def get_height(self, node):
        """Get height of a node"""
        if node is None:
            return 0
        return node.height
    
    def get_balance(self, node):
        """Get balance factor of a node"""
        if node is None:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)
    
    def update_height(self, node):
        """Update height of a node based on its children's heights"""
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))
    
    def rotate_right(self, y):
        """Right rotate the subtree rooted with y"""
        x = y.left
        T2 = x.right
        
        # Perform rotation
        x.right = y
        y.left = T2
        
        # Update heights
        self.update_height(y)
        self.update_height(x)
        
        # Return new root
        return x
    
    def rotate_left(self, x):
        """Left rotate the subtree rooted with x"""
        y = x.right
        T2 = y.left
        
        # Perform rotation
        y.left = x
        x.right = T2
        
        # Update heights
        self.update_height(x)
        self.update_height(y)
        
        # Return new root
        return y
    
    def insert(self, data):
        """Insert data into the AVL tree"""
        self.root = self._insert_recursive(self.root, data)
    
    def _insert_recursive(self, node, data):
        """Helper method to insert data recursively and rebalance"""
        # Standard BST insertion
        if node is None:
            return Node(data)
        
        if data < node.data:
            node.left = self._insert_recursive(node.left, data)
        elif data > node.data:
            node.right = self._insert_recursive(node.right, data)
        else:  # Duplicate keys not allowed
            return node
        
        # Update height of this ancestor node
        self.update_height(node)
        
        # Get the balance factor to check if unbalanced
        balance = self.get_balance(node)
        
        # If unbalanced, there are 4 cases
        
        # Left Left Case
        if balance > 1 and data < node.left.data:
            return self.rotate_right(node)
        
        # Right Right Case
        if balance < -1 and data > node.right.data:
            return self.rotate_left(node)
        
        # Left Right Case
        if balance > 1 and data > node.left.data:
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)
        
        # Right Left Case
        if balance < -1 and data < node.right.data:
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)
        
        # Return unchanged node pointer
        return node
    
    def search(self, data):
        """Search for data in the AVL tree"""
        return self._search_recursive(self.root, data)
    
    def _search_recursive(self, node, data):
        """Helper method to search recursively"""
        if node is None:
            return False
        
        if data == node.data:
            return True
        elif data < node.data:
            return self._search_recursive(node.left, data)
        else:  # data > node.data
            return self._search_recursive(node.right, data)
    
    def delete(self, data):
        """Delete data from the AVL tree"""
        self.root = self._delete_recursive(self.root, data)
    
    def _delete_recursive(self, node, data):
        """Helper method to delete recursively and rebalance"""
        # Standard BST deletion
        if node is None:
            return node
        
        if data < node.data:
            node.left = self._delete_recursive(node.left, data)
        elif data > node.data:
            node.right = self._delete_recursive(node.right, data)
        else:  # This is the node to be deleted
            # Node with only one child or no child
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            
            # Node with two children
            # Get the inorder successor (smallest in right subtree)
            temp = self._get_min_value_node(node.right)
            
            # Copy the inorder successor's data to this node
            node.data = temp.data
            
            # Delete the inorder successor
            node.right = self._delete_recursive(node.right, temp.data)
        
        # If the tree had only one node, return it
        if node is None:
            return node
        
        # Update height of the current node
        self.update_height(node)
        
        # Get balance factor
        balance = self.get_balance(node)
        
        # If unbalanced, there are 4 cases
        
        # Left Left Case
        if balance > 1 and self.get_balance(node.left) >= 0:
            return self.rotate_right(node)
        
        # Left Right Case
        if balance > 1 and self.get_balance(node.left) < 0:
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)
        
        # Right Right Case
        if balance < -1 and self.get_balance(node.right) <= 0:
            return self.rotate_left(node)
        
        # Right Left Case
        if balance < -1 and self.get_balance(node.right) > 0:
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)
        
        return node
    
    def _get_min_value_node(self, node):
        """Get node with minimum value in subtree"""
        current = node
        while current.left is not None:
            current = current.left
        return current
    
    def inorder_traversal(self, node=None, result=None):
        """Left -> Root -> Right traversal (gives sorted order)"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            self.inorder_traversal(node.left, result)
            result.append(node.data)
            self.inorder_traversal(node.right, result)
        
        return result
    
    def preorder_traversal(self, node=None, result=None):
        """Root -> Left -> Right traversal"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            result.append(node.data)
            self.preorder_traversal(node.left, result)
            self.preorder_traversal(node.right, result)
        
        return result
    
    def postorder_traversal(self, node=None, result=None):
        """Left -> Right -> Root traversal"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            self.postorder_traversal(node.left, result)
            self.postorder_traversal(node.right, result)
            result.append(node.data)
        
        return result
    
    def height(self, node=None):
        """Calculate height of AVL tree"""
        if node is None:
            node = self.root
        
        if node is None:
            return 0  # Height of empty tree is 0
        
        return node.height
    
    def size(self, node=None):
        """Count number of nodes in AVL tree"""
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        return 1 + self.size(node.left) + self.size(node.right)
    
    def is_avl(self, node=None):
        """Check if tree is a valid AVL tree"""
        if node is None:
            node = self.root
        
        if node is None:
            return True
        
        # Check if current node is balanced
        balance = abs(self.get_balance(node))
        if balance > 1:
            return False
        
        # Recursively check left and right subtrees
        return self.is_avl(node.left) and self.is_avl(node.right)
    
    def __str__(self):
        """String representation of AVL tree (inorder)"""
        return str(self.inorder_traversal())

# Example usage
avl = AVLTree()

# Insert elements
avl.insert(10)
avl.insert(20)
avl.insert(30)
avl.insert(40)
avl.insert(50)
avl.insert(25)

print("AVL Tree (inorder):", avl)  # Output: [10, 20, 25, 30, 40, 50]
print("Height:", avl.height())  # Output: 2 (balanced)
print("Size:", avl.size())  # Output: 6
print("Is valid AVL:", avl.is_avl())  # Output: True

# Tree traversals
print("\nPre-order traversal:", avl.preorder_traversal())  # Output: [30, 20, 10, 25, 40, 50]
print("In-order traversal:", avl.inorder_traversal())    # Output: [10, 20, 25, 30, 40, 50]
print("Post-order traversal:", avl.postorder_traversal()) # Output: [10, 25, 20, 50, 40, 30]

# Search operations
print("\nSearch operations:")
print("Search 25:", avl.search(25))  # Output: True
print("Search 35:", avl.search(35))  # Output: False

# Delete operations
print("\nDelete operations:")
print("Original AVL:", avl.inorder_traversal())

# Delete leaf node
avl.delete(10)
print("After deleting 10 (leaf):", avl.inorder_traversal())

# Delete node with one child
avl.delete(20)
print("After deleting 20 (one child):", avl.inorder_traversal())

# Delete node with two children
avl.delete(30)
print("After deleting 30 (two children):", avl.inorder_traversal())

print("Height after deletions:", avl.height())  # Output: 2 (still balanced)
print("Is valid AVL after deletions:", avl.is_avl())  # Output: True

# Example: AVL tree for a simple database index
class DatabaseIndex:
    def __init__(self):
        """Initialize a database index using AVL tree"""
        self.avl = AVLTree()
    
    def add_record(self, key, value):
        """Add a record to the index"""
        # In a real implementation, we'd store (key, record_id) pairs
        # and maintain a separate structure for the actual records
        self.avl.insert(key)
        print(f"Added record with key {key}")
    
    def find_record(self, key):
        """Find a record by key"""
        if self.avl.search(key):
            print(f"Found record with key {key}")
            return True
        else:
            print(f"Record with key {key} not found")
            return False
    
    def remove_record(self, key):
        """Remove a record from the index"""
        if self.avl.search(key):
            self.avl.delete(key)
            print(f"Removed record with key {key}")
        else:
            print(f"Cannot remove: Record with key {key} not found")
    
    def list_all_records(self):
        """List all records in sorted order"""
        keys = self.avl.inorder_traversal()
        print(f"All records (sorted by key): {keys}")
        return keys

# Create and use a database index
print("\nDatabase Index Example:")
db_index = DatabaseIndex()

db_index.add_record(1001)
db_index.add_record(1005)
db_index.add_record(1003)
db_index.add_record(1002)
db_index.add_record(1004)

db_index.list_all_records()

db_index.find_record(1003)
db_index.find_record(1007)

db_index.remove_record(1002)
db_index.remove_record(1007)  # This record doesn't exist

db_index.list_all_records()
```

Red-Black Tree

When: Need balanced BST with frequent insertions/deletions
Why: Less strict balancing than AVL (faster insertions), O(log n) guaranteed, fewer rotations
Examples: Java TreeMap/TreeSet, C++ std::map/set, Linux kernel scheduling, process management

```python
# Python implementation of a Red-Black Tree
class Color:
    """Enumeration for node colors"""
    RED = 0
    BLACK = 1

class Node:
    def __init__(self, data=None):
        """Initialize a node with data, color, and left/right/parent pointers"""
        self.data = data
        self.left = None
        self.right = None
        self.parent = None
        self.color = Color.RED  # New nodes are red by default

class RedBlackTree:
    def __init__(self):
        """Initialize an empty Red-Black tree with a NIL node"""
        # Create a single NIL node that will be used for all leaf references
        self.NIL = Node(None)
        self.NIL.color = Color.BLACK
        self.root = self.NIL
    
    def is_empty(self):
        """Check if Red-Black tree is empty"""
        return self.root == self.NIL
    
    def insert(self, data):
        """Insert data into the Red-Black tree"""
        # Create a new node
        node = Node(data)
        node.left = self.NIL
        node.right = self.NIL
        
        # Standard BST insertion
        parent = None
        current = self.root
        
        while current != self.NIL:
            parent = current
            if node.data < current.data:
                current = current.left
            else:
                current = current.right
        
        # Set parent
        node.parent = parent
        
        if parent is None:  # Tree was empty
            self.root = node
        elif node.data < parent.data:
            parent.left = node
        else:
            parent.right = node
        
        # Fix Red-Black properties
        self._insert_fixup(node)
    
    def _insert_fixup(self, node):
        """Fix Red-Black properties after insertion"""
        while node.parent.color == Color.RED:
            if node.parent == node.parent.parent.left:
                # Node's parent is a left child
                uncle = node.parent.parent.right
                
                if uncle.color == Color.RED:
                    # Case 1: Uncle is RED
                    node.parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        # Case 2: Node is right child
                        node = node.parent
                        self._left_rotate(node)
                    
                    # Case 3: Node is left child
                    node.parent.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    self._right_rotate(node.parent.parent)
            else:
                # Node's parent is a right child (mirror of above)
                uncle = node.parent.parent.left
                
                if uncle.color == Color.RED:
                    # Mirror Case 1
                    node.parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        # Mirror Case 2
                        node = node.parent
                        self._right_rotate(node)
                    
                    # Mirror Case 3
                    node.parent.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    self._left_rotate(node.parent.parent)
        
        # Ensure root is black
        self.root.color = Color.BLACK
    
    def delete(self, data):
        """Delete data from the Red-Black tree"""
        node = self._search(self.root, data)
        if node == self.NIL:
            return False  # Key not found
        
        y = node
        y_original_color = y.color
        
        if node.left == self.NIL:
            x = node.right
            self._transplant(node, node.right)
        elif node.right == self.NIL:
            x = node.left
            self._transplant(node, node.left)
        else:
            # Node has two children
            y = self._minimum(node.right)
            y_original_color = y.color
            x = y.right
            
            if y.parent == node:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = node.right
                y.right.parent = y
            
            self._transplant(node, y)
            y.left = node.left
            y.left.parent = y
            y.color = node.color
        
        if y_original_color == Color.BLACK:
            self._delete_fixup(x)
        
        return True
    
    def _transplant(self, u, v):
        """Replace subtree rooted at u with subtree rooted at v"""
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent
    
    def _delete_fixup(self, x):
        """Fix Red-Black properties after deletion"""
        while x != self.root and x.color == Color.BLACK:
            if x == x.parent.left:
                w = x.parent.right  # Sibling
                
                if w.color == Color.RED:
                    # Case 1: Sibling is RED
                    w.color = Color.BLACK
                    x.parent.color = Color.RED
                    self._left_rotate(x.parent)
                    w = x.parent.right
                
                if w.left.color == Color.BLACK and w.right.color == Color.BLACK:
                    # Case 2: Both of sibling's children are BLACK
                    w.color = Color.RED
                    x = x.parent
                else:
                    if w.right.color == Color.BLACK:
                        # Case 3: Sibling's right child is BLACK
                        w.left.color = Color.BLACK
                        w.color = Color.RED
                        self._right_rotate(w)
                        w = x.parent.right
                    
                    # Case 4: Sibling's right child is RED
                    w.color = x.parent.color
                    x.parent.color = Color.BLACK
                    w.right.color = Color.BLACK
                    self._left_rotate(x.parent)
                    x = self.root
            else:
                # Mirror of above cases
                w = x.parent.left
                
                if w.color == Color.RED:
                    # Mirror Case 1
                    w.color = Color.BLACK
                    x.parent.color = Color.RED
                    self._right_rotate(x.parent)
                    w = x.parent.left
                
                if w.right.color == Color.BLACK and w.left.color == Color.BLACK:
                    # Mirror Case 2
                    w.color = Color.RED
                    x = x.parent
                else:
                    if w.left.color == Color.BLACK:
                        # Mirror Case 3
                        w.right.color = Color.BLACK
                        w.color = Color.RED
                        self._left_rotate(w)
                        w = x.parent.left
                    
                    # Mirror Case 4
                    w.color = x.parent.color
                    x.parent.color = Color.BLACK
                    w.left.color = Color.BLACK
                    self._right_rotate(x.parent)
                    x = self.root
        
        x.color = Color.BLACK
    
    def _minimum(self, node):
        """Find minimum node in subtree"""
        while node.left != self.NIL:
            node = node.left
        return node
    
    def search(self, data):
        """Search for data in the Red-Black tree"""
        return self._search(self.root, data) != self.NIL
    
    def _search(self, node, data):
        """Helper method to search recursively"""
        while node != self.NIL:
            if data == node.data:
                return node
            elif data < node.data:
                node = node.left
            else:
                node = node.right
        return node  # Will be NIL if not found
    
    def _left_rotate(self, x):
        """Left rotate the subtree rooted at x"""
        y = x.right
        x.right = y.left
        y.left.parent = x
        
        if x.right != self.NIL:
            x.right.parent = x
        
        y.parent = x.parent
        
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        
        x.parent = y
    
    def _right_rotate(self, x):
        """Right rotate the subtree rooted at x"""
        y = x.left
        x.left = y.right
        y.right.parent = x
        
        if x.left != self.NIL:
            x.left.parent = x
        
        y.parent = x.parent
        
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        
        x.parent = y
    
    def inorder_traversal(self, node=None, result=None):
        """Left -> Root -> Right traversal (gives sorted order)"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node != self.NIL:
            self.inorder_traversal(node.left, result)
            result.append(node.data)
            self.inorder_traversal(node.right, result)
        
        return result
    
    def preorder_traversal(self, node=None, result=None):
        """Root -> Left -> Right traversal"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node != self.NIL:
            result.append(node.data)
            self.preorder_traversal(node.left, result)
            self.preorder_traversal(node.right, result)
        
        return result
    
    def postorder_traversal(self, node=None, result=None):
        """Left -> Right -> Root traversal"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node != self.NIL:
            self.postorder_traversal(node.left, result)
            self.postorder_traversal(node.right, result)
            result.append(node.data)
        
        return result
    
    def height(self, node=None):
        """Calculate height of Red-Black tree"""
        if node is None:
            node = self.root
        
        return self._height_recursive(node)
    
    def _height_recursive(self, node):
        """Helper method to calculate height recursively"""
        if node == self.NIL:
            return 0
        
        left_height = self._height_recursive(node.left)
        right_height = self._height_recursive(node.right)
        
        return max(left_height, right_height) + 1
    
    def size(self, node=None):
        """Count number of nodes in Red-Black tree"""
        if node is None:
            node = self.root
        
        return self._size_recursive(node)
    
    def _size_recursive(self, node):
        """Helper method to count nodes recursively"""
        if node == self.NIL:
            return 0
        
        return 1 + self._size_recursive(node.left) + self._size_recursive(node.right)
    
    def is_red_black(self, node=None):
        """Check if tree satisfies Red-Black properties"""
        if node is None:
            node = self.root
        
        # Property 1: Every node is either red or black (always true)
        # Property 2: Root is black
        if node.color != Color.BLACK:
            return False
        
        # Property 3: All leaves (NIL) are black (always true by construction)
        
        # Property 4: Red nodes have black children
        if not self._check_red_children(node):
            return False
        
        # Property 5: All paths from root to leaves have same number of black nodes
        black_height = self._black_height(node)
        return black_height != -1  # -1 indicates different black heights
    
    def _check_red_children(self, node):
        """Check if all red nodes have black children"""
        if node == self.NIL:
            return True
        
        if node.color == Color.RED:
            if node.left.color != Color.BLACK or node.right.color != Color.BLACK:
                return False
        
        return (self._check_red_children(node.left) and
                self._check_red_children(node.right))
    
    def _black_height(self, node):
        """Check if all paths have same black height, return -1 if not"""
        if node == self.NIL:
            return 0
        
        left_black_height = self._black_height(node.left)
        if left_black_height == -1:
            return -1
        
        right_black_height = self._black_height(node.right)
        if right_black_height == -1:
            return -1
        
        if left_black_height != right_black_height:
            return -1
        
        # Add 1 if current node is black
        return left_black_height + (1 if node.color == Color.BLACK else 0)
    
    def __str__(self):
        """String representation of Red-Black tree (inorder)"""
        return str(self.inorder_traversal())

# Example usage
rbt = RedBlackTree()

# Insert elements
rbt.insert(10)
rbt.insert(20)
rbt.insert(30)
rbt.insert(15)
rbt.insert(25)
rbt.insert(5)

print("Red-Black Tree (inorder):", rbt)  # Output: [5, 10, 15, 20, 25, 30]
print("Height:", rbt.height())  # Output: 2
print("Size:", rbt.size())  # Output: 6
print("Is valid Red-Black:", rbt.is_red_black())  # Output: True

# Tree traversals
print("\nPre-order traversal:", rbt.preorder_traversal())  # Output: [15, 10, 5, 20, 25, 30]
print("In-order traversal:", rbt.inorder_traversal())    # Output: [5, 10, 15, 20, 25, 30]
print("Post-order traversal:", rbt.postorder_traversal()) # Output: [5, 10, 25, 30, 20, 15]

# Search operations
print("\nSearch operations:")
print("Search 15:", rbt.search(15))  # Output: True
print("Search 35:", rbt.search(35))  # Output: False

# Delete operations
print("\nDelete operations:")
print("Original Red-Black:", rbt.inorder_traversal())

rbt.delete(5)   # Delete leaf node
print("After deleting 5 (leaf):", rbt.inorder_traversal())

rbt.delete(15)  # Delete node with one child
print("After deleting 15 (one child):", rbt.inorder_traversal())

rbt.delete(20)  # Delete node with two children
print("After deleting 20 (two children):", rbt.inorder_traversal())

print("Height after deletions:", rbt.height())  # Output: 2
print("Is valid Red-Black after deletions:", rbt.is_red_black())  # Output: True

# Example: Red-Black tree for a simple map/dictionary
class RedBlackMap:
    def __init__(self):
        """Initialize an empty map using Red-Black tree"""
        self.rbt = RedBlackTree()
    
    def put(self, key, value):
        """Add or update a key-value pair"""
        # In a real implementation, we'd store (key, value) pairs
        # For simplicity, we're just storing the key
        self.rbt.insert(key)
        print(f"Added/Updated key {key}")
    
    def contains_key(self, key):
        """Check if key exists in map"""
        return self.rbt.search(key)
    
    def remove_key(self, key):
        """Remove a key from map"""
        if self.rbt.search(key):
            self.rbt.delete(key)
            print(f"Removed key {key}")
        else:
            print(f"Cannot remove: Key {key} not found")
    
    def get_all_keys(self):
        """Get all keys in sorted order"""
        keys = self.rbt.inorder_traversal()
        print(f"All keys (sorted): {keys}")
        return keys

# Create and use a Red-Black map
print("\nRed-Black Map Example:")
rb_map = RedBlackMap()

rb_map.put("apple", 1)
rb_map.put("banana", 2)
rb_map.put("cherry", 3)
rb_map.put("date", 4)
rb_map.put("elderberry", 5)

rb_map.get_all_keys()

print("Contains 'cherry':", rb_map.contains_key("cherry"))
print("Contains 'grape':", rb_map.contains_key("grape"))

rb_map.remove_key("banana")
rb_map.remove_key("grape")  # This key doesn't exist

rb_map.get_all_keys()
```

Splay Tree

When: Access patterns show temporal locality (recently accessed items accessed again)
Why: Self-adjusting, frequently accessed items move to root, O(log n) amortized
Examples: Caches, garbage collection, network routing tables

```python
# Python implementation of a Splay Tree
class Node:
    def __init__(self, data=None):
        """Initialize a node with data and left/right/parent pointers"""
        self.data = data
        self.left = None
        self.right = None
        self.parent = None

class SplayTree:
    def __init__(self):
        """Initialize an empty Splay tree"""
        self.root = None
    
    def is_empty(self):
        """Check if Splay tree is empty"""
        return self.root is None
    
    def splay(self, node):
        """Splay operation to bring node to root"""
        while node.parent is not None:
            parent = node.parent
            grandparent = parent.parent
            
            if grandparent is None:
                # Zig case
                if node == parent.left:
                    self._right_rotate(parent)
                else:
                    self._left_rotate(parent)
            else:
                if parent == grandparent.left:
                    if node == parent.left:
                        # Zig-zig case
                        self._right_rotate(grandparent)
                        self._right_rotate(parent)
                    else:
                        # Zig-zag case
                        self._left_rotate(parent)
                        self._right_rotate(grandparent)
                else:
                    if node == parent.right:
                        # Zig-zig case (mirror)
                        self._left_rotate(grandparent)
                        self._left_rotate(parent)
                    else:
                        # Zig-zag case (mirror)
                        self._right_rotate(parent)
                        self._left_rotate(grandparent)
    
    def _right_rotate(self, y):
        """Right rotate at node y"""
        x = y.left
        y.left = x.right
        if x.right:
            x.right.parent = y
        
        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y == y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x
        
        x.right = y
        y.parent = x
    
    def _left_rotate(self, x):
        """Left rotate at node x"""
        y = x.right
        x.right = y.left
        if y.left:
            y.left.parent = x
        
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        
        y.left = x
        x.parent = y
    
    def insert(self, data):
        """Insert data into Splay tree"""
        node = Node(data)
        
        # Standard BST insertion
        y = None
        x = self.root
        
        while x is not None:
            y = x
            if node.data < x.data:
                x = x.left
            else:
                x = x.right
        
        node.parent = y
        
        if y is None:  # Tree was empty
            self.root = node
        elif node.data < y.data:
            y.left = node
        else:
            y.right = node
        
        # Splay the newly inserted node to root
        self.splay(node)
        self.root = node
    
    def search(self, data):
        """Search for data in Splay tree"""
        node = self._search(self.root, data)
        
        # If found, splay it to root
        if node:
            self.splay(node)
            self.root = node
        
        return node is not None
    
    def _search(self, node, data):
        """Helper method to search recursively"""
        while node is not None:
            if data == node.data:
                return node
            elif data < node.data:
                node = node.left
            else:
                node = node.right
        return node  # Will be None if not found
    
    def delete(self, data):
        """Delete data from Splay tree"""
        if not self.search(data):
            return False  # Key not found
        
        # Now the node to delete is at root
        node = self.root
        
        if node.left is None:
            self.root = node.right
            if self.root:
                self.root.parent = None
        elif node.right is None:
            self.root = node.left
            if self.root:
                self.root.parent = None
        else:
            # Find the inorder predecessor (maximum in left subtree)
            predecessor = self._maximum(node.left)
            
            # If predecessor is not the left child
            if predecessor != node.left:
                # Transplant predecessor with its right child
                self._transplant(predecessor, predecessor.right)
                predecessor.right = node.right
                predecessor.right.parent = predecessor
            
            # Transplant node with predecessor
            self._transplant(node, predecessor)
            predecessor.left = node.left
            predecessor.left.parent = predecessor
        
        return True
    
    def _maximum(self, node):
        """Find maximum node in subtree"""
        while node.right is not None:
            node = node.right
        return node
    
    def _transplant(self, u, v):
        """Replace subtree rooted at u with subtree rooted at v"""
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        
        if v:
            v.parent = u.parent
    
    def join(self, left_tree, right_tree):
        """Join two Splay trees where all elements in left_tree < all elements in right_tree"""
        if left_tree is None:
            return right_tree
        if right_tree is None:
            return left_tree
        
        # Find maximum in left_tree
        max_node = self._maximum(left_tree.root)
        
        # Splay the maximum to root
        self.splay(max_node)
        
        # Attach right_tree as right child of max_node
        max_node.right = right_tree.root
        if right_tree.root:
            right_tree.root.parent = max_node
        
        return left_tree
    
    def split(self, data):
        """Split the tree into two trees: left tree with keys < data and right tree with keys >= data"""
        if not self.search(data):
            return None, None  # Key not found
        
        # After search, the node with data is at root
        left_tree = SplayTree()
        right_tree = SplayTree()
        
        left_tree.root = self.root.left
        if left_tree.root:
            left_tree.root.parent = None
        
        right_tree.root = self.root.right
        if right_tree.root:
            right_tree.root.parent = None
        
        return left_tree, right_tree
    
    def inorder_traversal(self, node=None, result=None):
        """Left -> Root -> Right traversal (gives sorted order)"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            self.inorder_traversal(node.left, result)
            result.append(node.data)
            self.inorder_traversal(node.right, result)
        
        return result
    
    def preorder_traversal(self, node=None, result=None):
        """Root -> Left -> Right traversal"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            result.append(node.data)
            self.preorder_traversal(node.left, result)
            self.preorder_traversal(node.right, result)
        
        return result
    
    def postorder_traversal(self, node=None, result=None):
        """Left -> Right -> Root traversal"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            self.postorder_traversal(node.left, result)
            self.postorder_traversal(node.right, result)
            result.append(node.data)
        
        return result
    
    def height(self, node=None):
        """Calculate height of Splay tree"""
        if node is None:
            node = self.root
        
        return self._height_recursive(node)
    
    def _height_recursive(self, node):
        """Helper method to calculate height recursively"""
        if node is None:
            return -1
        
        left_height = self._height_recursive(node.left)
        right_height = self._height_recursive(node.right)
        
        return max(left_height, right_height) + 1
    
    def size(self, node=None):
        """Count number of nodes in Splay tree"""
        if node is None:
            node = self.root
        
        return self._size_recursive(node)
    
    def _size_recursive(self, node):
        """Helper method to count nodes recursively"""
        if node is None:
            return 0
        
        return 1 + self._size_recursive(node.left) + self._size_recursive(node.right)
    
    def __str__(self):
        """String representation of Splay tree (inorder)"""
        return str(self.inorder_traversal())

# Example usage
st = SplayTree()

# Insert elements
st.insert(10)
st.insert(20)
st.insert(30)
st.insert(15)
st.insert(25)
st.insert(5)

print("Splay Tree (inorder):", st)  # Output: [5, 10, 15, 20, 25, 30]
print("Height:", st.height())  # Output: 2
print("Size:", st.size())  # Output: 6

# Tree traversals
print("\nPre-order traversal:", st.preorder_traversal())  # Output varies based on splay operations
print("In-order traversal:", st.inorder_traversal())    # Output: [5, 10, 15, 20, 25, 30]
print("Post-order traversal:", st.postorder_traversal()) # Output varies based on splay operations

# Search operations (note that search will splay the found node to root)
print("\nSearch operations:")
print("Search 15:", st.search(15))  # Output: True
print("Root after searching 15:", st.root.data if st.root else None)  # Output: 15
print("Search 35:", st.search(35))  # Output: False

# Delete operations
print("\nDelete operations:")
print("Original Splay Tree:", st.inorder_traversal())

st.delete(5)   # Delete leaf node
print("After deleting 5 (leaf):", st.inorder_traversal())
print("Root after deletion:", st.root.data if st.root else None)

st.delete(20)  # Delete node with two children
print("After deleting 20 (two children):", st.inorder_traversal())
print("Root after deletion:", st.root.data if st.root else None)

# Example: Splay tree for a cache
class SplayCache:
    def __init__(self, capacity):
        """Initialize a cache with Splay tree and capacity"""
        self.tree = SplayTree()
        self.capacity = capacity
        self.size = 0
    
    def get(self, key):
        """Get value from cache"""
        if self.tree.search(key):
            print(f"Cache HIT for key {key}")
            return True
        else:
            print(f"Cache MISS for key {key}")
            return False
    
    def put(self, key):
        """Add key to cache"""
        if self.size >= self.capacity:
            # Simple eviction policy: remove a random key
            # In a real implementation, we'd use LRU or another policy
            print(f"Cache FULL, evicting a key to make space for {key}")
            # For simplicity, we're not actually removing anything here
        
        if not self.tree.search(key):
            self.tree.insert(key)
            self.size += 1
            print(f"Added key {key} to cache")
        else:
            # Key already exists, splay it to root (accessed)
            print(f"Key {key} already in cache, moved to root")

# Create and use a Splay cache
print("\nSplay Cache Example:")
cache = SplayCache(capacity=5)

cache.put("apple")
cache.put("banana")
cache.put("cherry")
cache.put("date")
cache.put("elderberry")

cache.get("cherry")  # Should be a hit
cache.get("grape")   # Should be a miss

cache.put("fig")     # Cache is full
cache.get("apple")   # Hit, and apple moves to root
cache.get("banana")  # Hit, and banana moves to root
```

Treap

When: Need randomized balancing that's simpler than deterministic trees
Why: Combines BST and heap properties, probabilistic balance, simpler implementation
Examples: Competitive programming, when simplicity matters more than worst-case guarantees

```python
# Python implementation of a Treap (Tree + Heap)
import random

class TreapNode:
    def __init__(self, key=None, priority=None):
        """Initialize a node with key, priority, and left/right child pointers"""
        self.key = key
        self.priority = priority if priority is not None else random.random()
        self.left = None
        self.right = None

class Treap:
    def __init__(self):
        """Initialize an empty Treap"""
        self.root = None
    
    def is_empty(self):
        """Check if Treap is empty"""
        return self.root is None
    
    def rotate_right(self, y):
        """Right rotate at node y"""
        x = y.left
        y.left = x.right
        x.right = y
        return x
    
    def rotate_left(self, x):
        """Left rotate at node x"""
        y = x.right
        x.right = y.left
        y.left = x
        return y
    
    def insert(self, key):
        """Insert a key into the Treap"""
        self.root = self._insert_recursive(self.root, key)
    
    def _insert_recursive(self, node, key):
        """Helper method to insert recursively and maintain heap property"""
        if node is None:
            return TreapNode(key)
        
        if key < node.key:
            node.left = self._insert_recursive(node.left, key)
            
            # Check if heap property is violated
            if node.left.priority < node.priority:
                node = self.rotate_right(node)
        else:
            node.right = self._insert_recursive(node.right, key)
            
            # Check if heap property is violated
            if node.right.priority < node.priority:
                node = self.rotate_left(node)
        
        return node
    
    def search(self, key):
        """Search for a key in the Treap"""
        return self._search_recursive(self.root, key)
    
    def _search_recursive(self, node, key):
        """Helper method to search recursively"""
        if node is None:
            return False
        
        if key == node.key:
            return True
        elif key < node.key:
            return self._search_recursive(node.left, key)
        else:
            return self._search_recursive(node.right, key)
    
    def delete(self, key):
        """Delete a key from the Treap"""
        self.root = self._delete_recursive(self.root, key)
    
    def _delete_recursive(self, node, key):
        """Helper method to delete recursively"""
        if node is None:
            return None
        
        if key < node.key:
            node.left = self._delete_recursive(node.left, key)
        elif key > node.key:
            node.right = self._delete_recursive(node.right, key)
        else:
            # Node to delete found
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                # Both children exist
                # Rotate the child with higher priority up
                if node.left.priority < node.right.priority:
                    node = self.rotate_right(node)
                    node.right = self._delete_recursive(node.right, key)
                else:
                    node = self.rotate_left(node)
                    node.left = self._delete_recursive(node.left, key)
        
        return node
    
    def inorder_traversal(self, node=None, result=None):
        """Left -> Root -> Right traversal (gives sorted order)"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            self.inorder_traversal(node.left, result)
            result.append(node.key)
            self.inorder_traversal(node.right, result)
        
        return result
    
    def preorder_traversal(self, node=None, result=None):
        """Root -> Left -> Right traversal"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            result.append(node.key)
            self.preorder_traversal(node.left, result)
            self.preorder_traversal(node.right, result)
        
        return result
    
    def height(self, node=None):
        """Calculate height of Treap"""
        if node is None:
            node = self.root
        
        return self._height_recursive(node)
    
    def _height_recursive(self, node):
        """Helper method to calculate height recursively"""
        if node is None:
            return -1
        
        left_height = self._height_recursive(node.left)
        right_height = self._height_recursive(node.right)
        
        return max(left_height, right_height) + 1
    
    def size(self, node=None):
        """Count number of nodes in Treap"""
        if node is None:
            node = self.root
        
        return self._size_recursive(node)
    
    def _size_recursive(self, node):
        """Helper method to count nodes recursively"""
        if node is None:
            return 0
        
        return 1 + self._size_recursive(node.left) + self._size_recursive(node.right)
    
    def is_treap(self, node=None):
        """Check if tree satisfies both BST and heap properties"""
        if node is None:
            node = self.root
        
        return self._is_bst(node, float('-inf'), float('inf')) and self._is_heap(node)
    
    def _is_bst(self, node, min_val, max_val):
        """Check if subtree satisfies BST property"""
        if node is None:
            return True
        
        if node.key <= min_val or node.key >= max_val:
            return False
        
        return (self._is_bst(node.left, min_val, node.key) and
                self._is_bst(node.right, node.key, max_val))
    
    def _is_heap(self, node):
        """Check if subtree satisfies heap property"""
        if node is None:
            return True
        
        if node.left and node.left.priority < node.priority:
            return False
        
        if node.right and node.right.priority < node.priority:
            return False
        
        return (self._is_heap(node.left) and self._is_heap(node.right))
    
    def __str__(self):
        """String representation of Treap (inorder)"""
        return str(self.inorder_traversal())

# Example usage
treap = Treap()

# Insert elements
treap.insert(5)
treap.insert(3)
treap.insert(8)
treap.insert(1)
treap.insert(7)
treap.insert(2)
treap.insert(6)
treap.insert(9)
treap.insert(4)

print("Treap (inorder):", treap)  # Output: [1, 2, 3, 4, 5, 6, 7, 8, 9]
print("Height:", treap.height())  # Output varies, but expected to be ~log(n)
print("Size:", treap.size())  # Output: 9
print("Is valid Treap:", treap.is_treap())  # Output: True

# Tree traversals
print("\nPre-order traversal:", treap.preorder_traversal())  # Output varies based on priorities
print("In-order traversal:", treap.inorder_traversal())    # Output: [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Search operations
print("\nSearch operations:")
print("Search 6:", treap.search(6))  # Output: True
print("Search 10:", treap.search(10))  # Output: False

# Delete operations
print("\nDelete operations:")
print("Original Treap:", treap.inorder_traversal())

treap.delete(1)  # Delete leaf node
print("After deleting 1 (leaf):", treap.inorder_traversal())

treap.delete(5)  # Delete node with two children
print("After deleting 5 (two children):", treap.inorder_traversal())

# Example: Treap for a simple priority queue
class TreapPriorityQueue:
    def __init__(self):
        """Initialize a priority queue using a Treap"""
        self.treap = Treap()
    
    def enqueue(self, item, priority):
        """Add an item with given priority"""
        # In a real implementation, we'd store (priority, item) pairs
        # For simplicity, we're just using the priority as the key
        self.treap.insert(priority)
        print(f"Enqueued item with priority {priority}")
    
    def dequeue(self):
        """Remove and return the item with minimum priority"""
        if self.treap.is_empty():
            print("Queue is empty")
            return None
        
        # Find the minimum element (leftmost in inorder traversal)
        min_priority = self._find_min(self.treap.root)
        self.treap.delete(min_priority)
        print(f"Dequeued item with priority {min_priority}")
        return min_priority
    
    def _find_min(self, node):
        """Find the minimum key in the subtree"""
        if node is None:
            return None
        
        while node.left is not None:
            node = node.left
        
        return node.key
    
    def is_empty(self):
        """Check if the priority queue is empty"""
        return self.treap.is_empty()

# Create and use a Treap-based priority queue
print("\nTreap Priority Queue Example:")
pq = TreapPriorityQueue()

pq.enqueue("Task 1", 5)
pq.enqueue("Task 2", 2)
pq.enqueue("Task 3", 8)
pq.enqueue("Task 4", 1)
pq.enqueue("Task 5", 4)

while not pq.is_empty():
    pq.dequeue()
```

Scapegoat Tree

When: Want balance without per-node metadata
Why: No balance information stored, periodic rebalancing, good amortized performance
Examples: Memory-constrained environments, when pointer overhead matters

```python
# Python implementation of a Scapegoat Tree
import math

class ScapegoatNode:
    def __init__(self, key=None):
        """Initialize a node with key and left/right child pointers"""
        self.key = key
        self.left = None
        self.right = None

class ScapegoatTree:
    def __init__(self, alpha=0.75):
        """Initialize an empty Scapegoat tree with balance factor alpha"""
        self.root = None
        self.alpha = alpha  # Balance factor (typically between 0.5 and 1.0)
        self.max_size = 0  # Maximum size since last rebuild
        self.size = 0  # Current size of the tree
    
    def is_empty(self):
        """Check if Scapegoat tree is empty"""
        return self.root is None
    
    def insert(self, key):
        """Insert a key into the Scapegoat tree"""
        if self.search(key):
            return  # Key already exists
        
        self.root = self._insert_recursive(self.root, key)
        self.size += 1
        self.max_size = max(self.max_size, self.size)
        
        # Check if tree needs rebuilding
        if self.size > 0 and self._size(self.root) > self.alpha * self.max_size:
            self.root = self._rebuild_tree(self.root)
            self.max_size = self.size
    
    def _insert_recursive(self, node, key):
        """Helper method to insert recursively"""
        if node is None:
            return ScapegoatNode(key)
        
        if key < node.key:
            node.left = self._insert_recursive(node.left, key)
        else:
            node.right = self._insert_recursive(node.right, key)
        
        return node
    
    def search(self, key):
        """Search for a key in the Scapegoat tree"""
        return self._search_recursive(self.root, key)
    
    def _search_recursive(self, node, key):
        """Helper method to search recursively"""
        if node is None:
            return False
        
        if key == node.key:
            return True
        elif key < node.key:
            return self._search_recursive(node.left, key)
        else:
            return self._search_recursive(node.right, key)
    
    def delete(self, key):
        """Delete a key from the Scapegoat tree"""
        if not self.search(key):
            return False  # Key not found
        
        self.root = self._delete_recursive(self.root, key)
        self.size -= 1
        
        # Check if tree needs rebuilding
        if self.size > 0 and self._size(self.root) > self.alpha * self.max_size:
            self.root = self._rebuild_tree(self.root)
            self.max_size = self.size
        
        return True
    
    def _delete_recursive(self, node, key):
        """Helper method to delete recursively"""
        if node is None:
            return None
        
        if key < node.key:
            node.left = self._delete_recursive(node.left, key)
        elif key > node.key:
            node.right = self._delete_recursive(node.right, key)
        else:
            # Node to delete found
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                # Node has two children
                # Find inorder successor (smallest in right subtree)
                successor = self._find_min(node.right)
                node.key = successor.key
                node.right = self._delete_recursive(node.right, successor.key)
        
        return node
    
    def _find_min(self, node):
        """Find node with minimum key in subtree"""
        while node.left is not None:
            node = node.left
        return node
    
    def _size(self, node):
        """Calculate size of subtree"""
        if node is None:
            return 0
        return 1 + self._size(node.left) + self._size(node.right)
    
    def _depth(self, node):
        """Calculate depth of node in tree"""
        if node is None:
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))
    
    def _is_balanced(self, node, depth):
        """Check if subtree is balanced according to alpha"""
        if node is None:
            return True
        
        size = self._size(node)
        return size <= self.alpha * (2 ** depth)
    
    def _rebuild_tree(self, node):
        """Rebuild a subtree to restore balance"""
        if node is None:
            return None
        
        # Get all keys in inorder traversal
        keys = []
        self._inorder_traversal(node, keys)
        
        # Build balanced tree from sorted keys
        return self._build_balanced(keys, 0, len(keys) - 1)
    
    def _inorder_traversal(self, node, result):
        """Helper method for inorder traversal"""
        if node is None:
            return
        
        self._inorder_traversal(node.left, result)
        result.append(node.key)
        self._inorder_traversal(node.right, result)
    
    def _build_balanced(self, keys, start, end):
        """Build a balanced tree from sorted keys"""
        if start > end:
            return None
        
        mid = (start + end) // 2
        node = ScapegoatNode(keys[mid])
        
        node.left = self._build_balanced(keys, start, mid - 1)
        node.right = self._build_balanced(keys, mid + 1, end)
        
        return node
    
    def inorder_traversal(self, node=None, result=None):
        """Left -> Root -> Right traversal (gives sorted order)"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            self.inorder_traversal(node.left, result)
            result.append(node.key)
            self.inorder_traversal(node.right, result)
        
        return result
    
    def preorder_traversal(self, node=None, result=None):
        """Root -> Left -> Right traversal"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            result.append(node.key)
            self.preorder_traversal(node.left, result)
            self.preorder_traversal(node.right, result)
        
        return result
    
    def postorder_traversal(self, node=None, result=None):
        """Left -> Right -> Root traversal"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            self.postorder_traversal(node.left, result)
            self.postorder_traversal(node.right, result)
            result.append(node.key)
        
        return result
    
    def height(self, node=None):
        """Calculate height of Scapegoat tree"""
        if node is None:
            node = self.root
        
        return self._height_recursive(node)
    
    def _height_recursive(self, node):
        """Helper method to calculate height recursively"""
        if node is None:
            return -1
        
        left_height = self._height_recursive(node.left)
        right_height = self._height_recursive(node.right)
        
        return max(left_height, right_height) + 1
    
    def size(self):
        """Return the size of the tree"""
        return self.size
    
    def __str__(self):
        """String representation of Scapegoat tree (inorder)"""
        return str(self.inorder_traversal())

# Example usage
sgt = ScapegoatTree(alpha=0.75)

# Insert elements
sgt.insert(10)
sgt.insert(20)
sgt.insert(30)
sgt.insert(15)
sgt.insert(25)
sgt.insert(5)
sgt.insert(35)
sgt.insert(1)
sgt.insert(40)
sgt.insert(45)

print("Scapegoat Tree (inorder):", sgt)  # Output: [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
print("Height:", sgt.height())  # Output varies, but expected to be ~log(n)
print("Size:", sgt.size())  # Output: 10

# Tree traversals
print("\nPre-order traversal:", sgt.preorder_traversal())  # Output varies based on rebuilds
print("In-order traversal:", sgt.inorder_traversal())    # Output: [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
print("Post-order traversal:", sgt.postorder_traversal()) # Output varies based on rebuilds

# Search operations
print("\nSearch operations:")
print("Search 15:", sgt.search(15))  # Output: True
print("Search 35:", sgt.search(35))  # Output: True
print("Search 50:", sgt.search(50))  # Output: False

# Delete operations
print("\nDelete operations:")
print("Original Scapegoat Tree:", sgt.inorder_traversal())
sgt.delete(20)
print("After deleting 20:", sgt.inorder_traversal())
sgt.delete(5)
print("After deleting 5:", sgt.inorder_traversal())
sgt.delete(40)
print("After deleting 40:", sgt.inorder_traversal())
print("Final size:", sgt.size())
```

Weight-balanced Tree

When: Need order statistics and balanced BST operations
Why: Maintains subtree size information, efficient for rank/select operations
Examples: Order-maintaining data structures, competitive programming

```python
# Python implementation of a Weight-balanced Tree
import random

class WeightNode:
    def __init__(self, key=None):
        """Initialize a node with key, weight, and left/right child pointers"""
        self.key = key
        self.left = None
        self.right = None
        self.weight = 1  # Weight is the size of the subtree

class WeightBalancedTree:
    def __init__(self, delta=0.7, gamma=0.2):
        """Initialize an empty weight-balanced tree with balance factors"""
        self.root = None
        self.delta = delta  # Balance factor for rotations (typically 0.5-0.8)
        self.gamma = gamma  # Balance factor for rotations (typically 0.1-0.4)
    
    def is_empty(self):
        """Check if weight-balanced tree is empty"""
        return self.root is None
    
    def get_weight(self, node):
        """Get weight of a node (size of subtree)"""
        if node is None:
            return 0
        return node.weight
    
    def update_weight(self, node):
        """Update weight of a node based on its children"""
        if node is not None:
            node.weight = 1 + self.get_weight(node.left) + self.get_weight(node.right)
    
    def rotate_right(self, y):
        """Right rotate at node y"""
        x = y.left
        T2 = x.right
        
        # Perform rotation
        x.right = y
        y.left = T2
        
        # Update weights
        self.update_weight(y)
        self.update_weight(x)
        
        return x
    
    def rotate_left(self, x):
        """Left rotate at node x"""
        y = x.right
        T2 = y.left
        
        # Perform rotation
        y.left = x
        x.right = T2
        
        # Update weights
        self.update_weight(x)
        self.update_weight(y)
        
        return y
    
    def is_balanced(self, node):
        """Check if a node is weight-balanced"""
        if node is None:
            return True
        
        left_weight = self.get_weight(node.left)
        right_weight = self.get_weight(node.right)
        total_weight = left_weight + right_weight + 1
        
        # Check delta condition
        if left_weight > self.delta * total_weight or right_weight > self.delta * total_weight:
            return False
        
        # Check gamma condition for children
        if node.left and self.get_weight(node.left.left) < self.gamma * left_weight:
            return False
        if node.left and self.get_weight(node.left.right) < self.gamma * left_weight:
            return False
        if node.right and self.get_weight(node.right.left) < self.gamma * right_weight:
            return False
        if node.right and self.get_weight(node.right.right) < self.gamma * right_weight:
            return False
        
        return True
    
    def rebalance(self, node):
        """Rebalance a node if necessary"""
        if node is None:
            return node
        
        left_weight = self.get_weight(node.left)
        right_weight = self.get_weight(node.right)
        total_weight = left_weight + right_weight + 1
        
        # Check if left subtree is too heavy
        if left_weight > self.delta * total_weight:
            if self.get_weight(node.left.left) >= self.gamma * left_weight:
                # Single rotation
                node = self.rotate_right(node)
            else:
                # Double rotation
                node.left = self.rotate_left(node.left)
                node = self.rotate_right(node)
        
        # Check if right subtree is too heavy
        elif right_weight > self.delta * total_weight:
            if self.get_weight(node.right.right) >= self.gamma * right_weight:
                # Single rotation
                node = self.rotate_left(node)
            else:
                # Double rotation
                node.right = self.rotate_right(node.right)
                node = self.rotate_left(node)
        
        return node
    
    def insert(self, key):
        """Insert a key into the weight-balanced tree"""
        self.root = self._insert_recursive(self.root, key)
    
    def _insert_recursive(self, node, key):
        """Helper method to insert recursively and rebalance"""
        if node is None:
            return WeightNode(key)
        
        if key < node.key:
            node.left = self._insert_recursive(node.left, key)
        elif key > node.key:
            node.right = self._insert_recursive(node.right, key)
        else:
            # Key already exists
            return node
        
        # Update weight
        self.update_weight(node)
        
        # Rebalance if necessary
        return self.rebalance(node)
    
    def search(self, key):
        """Search for a key in the weight-balanced tree"""
        return self._search_recursive(self.root, key)
    
    def _search_recursive(self, node, key):
        """Helper method to search recursively"""
        if node is None:
            return False
        
        if key == node.key:
            return True
        elif key < node.key:
            return self._search_recursive(node.left, key)
        else:
            return self._search_recursive(node.right, key)
    
    def delete(self, key):
        """Delete a key from the weight-balanced tree"""
        self.root = self._delete_recursive(self.root, key)
    
    def _delete_recursive(self, node, key):
        """Helper method to delete recursively and rebalance"""
        if node is None:
            return None
        
        if key < node.key:
            node.left = self._delete_recursive(node.left, key)
        elif key > node.key:
            node.right = self._delete_recursive(node.right, key)
        else:
            # Node to delete found
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                # Node has two children
                # Find inorder successor (smallest in right subtree)
                successor = self._find_min(node.right)
                node.key = successor.key
                node.right = self._delete_recursive(node.right, successor.key)
        
        # Update weight
        self.update_weight(node)
        
        # Rebalance if necessary
        return self.rebalance(node)
    
    def _find_min(self, node):
        """Find node with minimum key in subtree"""
        while node.left is not None:
            node = node.left
        return node
    
    def find_kth_smallest(self, k):
        """Find the k-th smallest element (1-indexed)"""
        if k <= 0 or k > self.get_weight(self.root):
            return None
        
        return self._find_kth_recursive(self.root, k)
    
    def _find_kth_recursive(self, node, k):
        """Helper method to find k-th smallest recursively"""
        if node is None:
            return None
        
        left_weight = self.get_weight(node.left)
        
        if k <= left_weight:
            return self._find_kth_recursive(node.left, k)
        elif k == left_weight + 1:
            return node.key
        else:
            return self._find_kth_recursive(node.right, k - left_weight - 1)
    
    def rank(self, key):
        """Find the rank of a key (number of elements less than key)"""
        return self._rank_recursive(self.root, key)
    
    def _rank_recursive(self, node, key):
        """Helper method to find rank recursively"""
        if node is None:
            return 0
        
        if key < node.key:
            return self._rank_recursive(node.left, key)
        elif key == node.key:
            return self.get_weight(node.left)
        else:
            return 1 + self.get_weight(node.left) + self._rank_recursive(node.right, key)
    
    def inorder_traversal(self, node=None, result=None):
        """Left -> Root -> Right traversal (gives sorted order)"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            self.inorder_traversal(node.left, result)
            result.append(node.key)
            self.inorder_traversal(node.right, result)
        
        return result
    
    def preorder_traversal(self, node=None, result=None):
        """Root -> Left -> Right traversal"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            result.append(node.key)
            self.preorder_traversal(node.left, result)
            self.preorder_traversal(node.right, result)
        
        return result
    
    def postorder_traversal(self, node=None, result=None):
        """Left -> Right -> Root traversal"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            self.postorder_traversal(node.left, result)
            self.postorder_traversal(node.right, result)
            result.append(node.key)
        
        return result
    
    def height(self, node=None):
        """Calculate height of weight-balanced tree"""
        if node is None:
            node = self.root
        
        return self._height_recursive(node)
    
    def _height_recursive(self, node):
        """Helper method to calculate height recursively"""
        if node is None:
            return -1
        
        left_height = self._height_recursive(node.left)
        right_height = self._height_recursive(node.right)
        
        return max(left_height, right_height) + 1
    
    def size(self):
        """Return the size of the tree"""
        return self.get_weight(self.root)
    
    def __str__(self):
        """String representation of weight-balanced tree (inorder)"""
        return str(self.inorder_traversal())

# Example usage
wbt = WeightBalancedTree(delta=0.7, gamma=0.2)

# Insert elements
wbt.insert(10)
wbt.insert(20)
wbt.insert(30)
wbt.insert(15)
wbt.insert(25)
wbt.insert(5)
wbt.insert(35)
wbt.insert(1)
wbt.insert(40)
wbt.insert(45)

print("Weight-balanced Tree (inorder):", wbt)  # Output: [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
print("Height:", wbt.height())  # Output varies, but expected to be ~log(n)
print("Size:", wbt.size())  # Output: 10

# Tree traversals
print("\nPre-order traversal:", wbt.preorder_traversal())  # Output varies based on rebalancing
print("In-order traversal:", wbt.inorder_traversal())    # Output: [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
print("Post-order traversal:", wbt.postorder_traversal()) # Output varies based on rebalancing

# Search operations
print("\nSearch operations:")
print("Search 15:", wbt.search(15))  # Output: True
print("Search 35:", wbt.search(35))  # Output: True
print("Search 50:", wbt.search(50))  # Output: False

# Order statistics operations
print("\nOrder statistics operations:")
for k in range(1, 11):
    kth = wbt.find_kth_smallest(k)
    print(f"{k}-th smallest: {kth}")

print("\nRank operations:")
for key in [5, 15, 25, 35]:
    rank = wbt.rank(key)
    print(f"Rank of {key}: {rank}")

# Delete operations
print("\nDelete operations:")
print("Original Weight-balanced Tree:", wbt.inorder_traversal())
wbt.delete(20)
print("After deleting 20:", wbt.inorder_traversal())
wbt.delete(5)
print("After deleting 5:", wbt.inorder_traversal())
wbt.delete(40)
print("After deleting 40:", wbt.inorder_traversal())
print("Final size:", wbt.size())

# Example: Order statistics tree for a leaderboard
class Leaderboard:
    def __init__(self):
        """Initialize a leaderboard using weight-balanced tree"""
        self.tree = WeightBalancedTree()
        self.players = {}  # Map player name to score
    
    def add_player(self, name, score):
        """Add a player to the leaderboard"""
        if name in self.players:
            # Update existing player
            old_score = self.players[name]
            self.tree.delete(old_score)
        
        self.players[name] = score
        self.tree.insert(score)
        print(f"Added/updated player {name} with score {score}")
    
    def remove_player(self, name):
        """Remove a player from the leaderboard"""
        if name in self.players:
            score = self.players[name]
            self.tree.delete(score)
            del self.players[name]
            print(f"Removed player {name} with score {score}")
        else:
            print(f"Player {name} not found")
    
    def get_rank(self, name):
        """Get the rank of a player (1-indexed)"""
        if name in self.players:
            score = self.players[name]
            rank = self.tree.size() - self.tree.rank(score)
            print(f"Player {name} is ranked #{rank}")
            return rank
        else:
            print(f"Player {name} not found")
            return -1
    
    def get_top_players(self, n):
        """Get the top n players"""
        if n <= 0:
            return []
        
        size = self.tree.size()
        top_players = []
        
        for i in range(1, min(n, size) + 1):
            # Get the i-th largest element (size - i + 1)-th smallest
            score = self.tree.find_kth_smallest(size - i + 1)
            # Find player with this score
            for name, player_score in self.players.items():
                if player_score == score:
                    top_players.append((name, score))
                    break
        
        return top_players

# Create and use a leaderboard
print("\nLeaderboard Example:")
leaderboard = Leaderboard()

leaderboard.add_player("Alice", 850)
leaderboard.add_player("Bob", 920)
leaderboard.add_player("Charlie", 780)
leaderboard.add_player("Diana", 950)
leaderboard.add_player("Eve", 890)

leaderboard.get_rank("Alice")
leaderboard.get_rank("Bob")
leaderboard.get_rank("Charlie")

print("\nTop 3 players:")
top_players = leaderboard.get_top_players(3)
for i, (name, score) in enumerate(top_players, 1):
    print(f"{i}. {name}: {score}")
```

sgt.delete(5)   # Delete leaf node
print("After deleting 5 (leaf):", sgt.inorder_traversal())

sgt.delete(20)  # Delete node with one child
print("After deleting 20 (one child):", sgt.inorder_traversal())

sgt.delete(10)  # Delete node with two children
print("After deleting 10 (two children):", sgt.inorder_traversal())

# Example: Scapegoat tree for a simple set
class ScapegoatSet:
    def __init__(self, alpha=0.75):
        """Initialize a set using a Scapegoat tree"""
        self.sgt = ScapegoatTree(alpha)
    
    def add(self, element):
        """Add an element to the set"""
        if not self.sgt.search(element):
            self.sgt.insert(element)
            print(f"Added {element} to set")
        else:
            print(f"{element} already in set")
    
    def contains(self, element):
        """Check if element is in the set"""
        return self.sgt.search(element)
    
    def remove(self, element):
        """Remove an element from the set"""
        if self.sgt.search(element):
            self.sgt.delete(element)
            print(f"Removed {element} from set")
        else:
            print(f"{element} not in set")
    
    def get_all_elements(self):
        """Get all elements in sorted order"""
        elements = self.sgt.inorder_traversal()
        print(f"All elements in set: {elements}")
        return elements

# Create and use a Scapegoat set
print("\nScapegoat Set Example:")
sgt_set = ScapegoatSet()

sgt_set.add("apple")
sgt_set.add("banana")
sgt_set.add("cherry")
sgt_set.add("date")
sgt_set.add("elderberry")

sgt_set.add("apple")  # Already in set

print("\nContains 'cherry':", sgt_set.contains("cherry"))
print("Contains 'grape':", sgt_set.contains("grape"))

sgt_set.remove("banana")
sgt_set.remove("grape")  # Not in set

sgt_set.get_all_elements()
```

Weight-balanced Tree

When: Need randomized balancing that's simpler than deterministic trees
Why: Combines BST and heap properties, probabilistic balance, simpler implementation
Examples: Competitive programming, when simplicity matters more than worst-case guarantees

Scapegoat Tree

When: Want balance without per-node metadata
Why: No balance information stored, periodic rebalancing, good amortized performance
Examples: Memory-constrained environments, when pointer overhead matters

Weight-balanced Tree

When: Need to maintain subtree sizes for ranking/selection
Why: Can find k-th smallest element in O(log n)
Examples: Order statistics, ranked data structures

B-Trees
B-Tree

When: Data is on disk/SSD and minimizing disk accesses is critical
Why: High branching factor reduces tree height, minimizes I/O operations
Examples: Database indexes, file systems, large datasets on disk

```python
# Python implementation of a B-Tree
class BTreeNode:
    def __init__(self, degree, is_leaf=False):
        """Initialize a B-Tree node"""
        self.degree = degree  # Minimum degree
        self.keys = []  # List of keys
        self.children = []  # List of child pointers
        self.is_leaf = is_leaf  # True if leaf node
        self.num_keys = 0  # Number of keys currently in node
    
    def is_full(self):
        """Check if node is full"""
        return self.num_keys == 2 * self.degree - 1
    
    def search(self, key):
        """Search for key in this node"""
        # Find the first key greater than or equal to key
        i = 0
        while i < self.num_keys and key > self.keys[i]:
            i += 1
        
        # If key is found in this node
        if i < self.num_keys and self.keys[i] == key:
            return self, i
        
        # If this is a leaf node, key is not present
        if self.is_leaf:
            return None, -1
        
        # Recurse to the appropriate child
        return self.children[i].search(key)
    
    def split_child(self, i):
        """Split the child at index i"""
        degree = self.degree
        child = self.children[i]
        
        # Create a new node to hold (degree-1) keys from child
        new_node = BTreeNode(degree, child.is_leaf)
        new_node.num_keys = degree - 1
        
        # Copy the last (degree-1) keys of child to new_node
        for j in range(degree - 1):
            new_node.keys.append(child.keys[j + degree])
        
        # Copy the last degree children of child to new_node
        if not child.is_leaf:
            for j in range(degree):
                new_node.children.append(child.children[j + degree])
        
        # Reduce the number of keys in child
        child.num_keys = degree - 1
        
        # Insert new child in this node's children list
        self.children.insert(i + 1, new_node)
        
        # Move the middle key of child up to this node
        self.keys.insert(i, child.keys[degree - 1])
        self.num_keys += 1
    
    def insert_non_full(self, key):
        """Insert a key into a non-full node"""
        i = self.num_keys - 1
        
        # If this is a leaf node, find the correct position and insert
        if self.is_leaf:
            # Move all greater keys one position ahead
            self.keys.append(None)  # Make space
            while i >= 0 and key < self.keys[i]:
                self.keys[i + 1] = self.keys[i]
                i -= 1
            
            # Insert the new key
            self.keys[i + 1] = key
            self.num_keys += 1
        else:
            # Find the child to receive the new key
            while i >= 0 and key < self.keys[i]:
                i -= 1
            
            # If the child is full, split it first
            if self.children[i + 1].is_full():
                self.split_child(i + 1)
                
                # After split, the middle key moves up, determine which child to use
                if key > self.keys[i]:
                    i += 1
            
            # Recursively insert into the appropriate child
            self.children[i + 1].insert_non_full(key)
    
    def __str__(self):
        """String representation of the node"""
        return f"Keys: {self.keys[:self.num_keys]}, Leaf: {self.is_leaf}"

class BTree:
    def __init__(self, degree=3):
        """Initialize an empty B-Tree with minimum degree"""
        self.root = None
        self.degree = degree  # Minimum degree
    
    def search(self, key):
        """Search for a key in the B-Tree"""
        if self.root is None:
            return None
        
        node, _ = self.root.search(key)
        return node is not None
    
    def insert(self, key):
        """Insert a key into the B-Tree"""
        if self.root is None:
            # If tree is empty, create a root node
            self.root = BTreeNode(self.degree, True)
            self.root.keys.append(key)
            self.root.num_keys = 1
        else:
            # If root is full, split it first
            if self.root.is_full():
                # Create a new root
                new_root = BTreeNode(self.degree, False)
                new_root.children.append(self.root)
                
                # Split the old root
                new_root.split_child(0)
                
                # Update the root
                self.root = new_root
                # Insert the key into the appropriate child
                i = 0
                if key > self.root.keys[0]:
                    i = 1
                if key > self.root.keys[1]:
                    i = 2
                
                self.root.children[i].insert_non_full(key)
            else:
                # If root is not full, insert directly
                self.root.insert_non_full(key)
    
    def inorder_traversal(self, node=None, result=None):
        """Inorder traversal of the B-Tree"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            i = 0
            # Traverse the first child and all keys and children in order
            for i in range(node.num_keys):
                if not node.is_leaf:
                    self.inorder_traversal(node.children[i], result)
                result.append(node.keys[i])
            
            # Traverse the last child
            if not node.is_leaf:
                self.inorder_traversal(node.children[i + 1], result)
        
        return result
    
    def print_tree(self, node=None, level=0):
        """Print the B-Tree structure for debugging"""
        if node is None:
            node = self.root
        
        if node:
            print("  " * level, end="")
            print(node.keys[:node.num_keys])
            
            if not node.is_leaf:
                for i in range(node.num_keys + 1):
                    self.print_tree(node.children[i], level + 1)
    
    def height(self, node=None):
        """Calculate the height of the B-Tree"""
        if node is None:
            node = self.root
        
        if node is None or node.is_leaf:
            return 0
        
        max_child_height = 0
        for i in range(node.num_keys + 1):
            child_height = self.height(node.children[i])
            max_child_height = max(max_child_height, child_height)
        
        return max_child_height + 1
    
    def size(self, node=None):
        """Count the number of keys in the B-Tree"""
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        if node.is_leaf:
            return node.num_keys
        
        total = node.num_keys
        for i in range(node.num_keys + 1):
            total += self.size(node.children[i])
        
        return total

# Example usage
bt = BTree(degree=3)  # Minimum degree of 3 (max 5 keys per node)

# Insert elements
keys = [10, 20, 5, 6, 12, 30, 7, 17]
for key in keys:
    bt.insert(key)
    print(f"Inserted {key}")

print("\nB-Tree structure:")
bt.print_tree()

print("\nInorder traversal (sorted keys):")
print(bt.inorder_traversal())

print("\nSearch operations:")
for key in [6, 15, 30]:
    found = bt.search(key)
    print(f"Search {key}: {'Found' if found else 'Not found'}")

print("\nHeight:", bt.height())
print("Size:", bt.size())

# Example: B-Tree for a simple database index
class DatabaseIndex:
    def __init__(self, degree=4):
        """Initialize a database index using a B-Tree"""
        self.btree = BTree(degree)
        self.records = {}  # In a real DB, this would reference actual records
    
    def add_record(self, key, data):
        """Add a record to the index"""
        self.btree.insert(key)
        self.records[key] = data
        print(f"Added record with key {key}")
    
    def find_record(self, key):
        """Find a record by key"""
        if self.btree.search(key):
            print(f"Found record with key {key}: {self.records.get(key)}")
            return self.records.get(key)
        else:
            print(f"Record with key {key} not found")
            return None
    
    def delete_record(self, key):
        """Delete a record from the index"""
        # In a real implementation, we'd need to implement B-Tree deletion
        # For simplicity, we're just removing from our records dictionary
        if key in self.records:
            del self.records[key]
            print(f"Deleted record with key {key}")
        else:
            print(f"Cannot delete: Record with key {key} not found")
    
    def list_all_records(self):
        """List all records in sorted order"""
        keys = self.btree.inorder_traversal()
        print(f"All records (sorted by key):")
        for key in keys:
            if key in self.records:
                print(f"  Key {key}: {self.records[key]}")
        return keys

# Create and use a database index
print("\nDatabase Index Example:")
db_index = DatabaseIndex(degree=3)

db_index.add_record(1001, "Customer: Alice")
db_index.add_record(1005, "Customer: Bob")
db_index.add_record(1003, "Customer: Charlie")
db_index.add_record(1002, "Customer: Diana")
db_index.add_record(1004, "Customer: Eve")

db_index.find_record(1003)
db_index.find_record(1007)

db_index.list_all_records()
```

B+ Tree

When: Need range queries and all data at leaves (most common in databases)
Why: All values at leaves, internal nodes for routing only, better for range scans
Examples: Database indexes (MySQL InnoDB), file system indexes, most DBMS implementations

```python
# Python implementation of a B+ Tree
class BPlusTreeNode:
    def __init__(self, degree, is_leaf=False):
        """Initialize a B+ Tree node"""
        self.degree = degree  # Minimum degree
        self.keys = []  # List of keys
        self.children = []  # List of child pointers (for internal nodes)
        self.values = []  # List of values (for leaf nodes)
        self.is_leaf = is_leaf  # True if leaf node
        self.num_keys = 0  # Number of keys currently in node
        self.next = None  # Pointer to next leaf node (for leaf nodes)
    
    def is_full(self):
        """Check if node is full"""
        return self.num_keys == 2 * self.degree - 1
    
    def search(self, key):
        """Search for key in this node"""
        # Find the first key greater than or equal to key
        i = 0
        while i < self.num_keys and key > self.keys[i]:
            i += 1
        
        # If key is found in this node and it's a leaf
        if self.is_leaf and i < self.num_keys and self.keys[i] == key:
            return self, i
        
        # If this is a leaf node and key is not present
        if self.is_leaf:
            return None, -1
        
        # Recurse to the appropriate child
        return self.children[i].search(key)
    
    def split_child(self, i):
        """Split the child at index i"""
        degree = self.degree
        child = self.children[i]
        
        # Create a new node to hold (degree-1) keys from child
        new_node = BPlusTreeNode(degree, child.is_leaf)
        new_node.num_keys = degree - 1
        
        if child.is_leaf:
            # For leaf nodes, copy the last (degree-1) keys and values
            for j in range(degree - 1):
                new_node.keys.append(child.keys[j + degree])
                new_node.values.append(child.values[j + degree])
            
            # Set up the linked list of leaf nodes
            new_node.next = child.next
            child.next = new_node
        else:
            # For internal nodes, copy the last (degree-1) keys and children
            for j in range(degree - 1):
                new_node.keys.append(child.keys[j + degree])
                new_node.children.append(child.children[j + degree])
            
            # Copy the last child
            new_node.children.append(child.children[2 * degree - 1])
        
        # Reduce the number of keys in child
        child.num_keys = degree - 1
        
        # Insert new child in this node's children list
        self.children.insert(i + 1, new_node)
        
        # Move the middle key of child up to this node
        if child.is_leaf:
            # For leaf nodes, the first key of the new node moves up
            self.keys.insert(i, new_node.keys[0])
        else:
            # For internal nodes, the middle key moves up
            self.keys.insert(i, child.keys[degree - 1])
        
        self.num_keys += 1
    
    def insert_non_full(self, key, value=None):
        """Insert a key into a non-full node"""
        i = self.num_keys - 1
        
        # If this is a leaf node, find the correct position and insert
        if self.is_leaf:
            # Move all greater keys one position ahead
            self.keys.append(None)  # Make space
            if value is not None:
                self.values.append(None)  # Make space for value
            
            while i >= 0 and key < self.keys[i]:
                self.keys[i + 1] = self.keys[i]
                if value is not None:
                    self.values[i + 1] = self.values[i]
                i -= 1
            
            # Insert the new key and value
            self.keys[i + 1] = key
            if value is not None:
                self.values[i + 1] = value
            self.num_keys += 1
        else:
            # Find the child to receive the new key
            while i >= 0 and key < self.keys[i]:
                i -= 1
            
            # If the child is full, split it first
            if self.children[i + 1].is_full():
                self.split_child(i + 1)
                
                # After split, the middle key moves up, determine which child to use
                if key > self.keys[i]:
                    i += 1
            
            # Recursively insert into the appropriate child
            self.children[i + 1].insert_non_full(key, value)
    
    def __str__(self):
        """String representation of the node"""
        if self.is_leaf:
            return f"Leaf - Keys: {self.keys[:self.num_keys]}, Values: {self.values[:self.num_keys]}"
        else:
            return f"Internal - Keys: {self.keys[:self.num_keys]}"

class BPlusTree:
    def __init__(self, degree=3):
        """Initialize an empty B+ Tree with minimum degree"""
        self.root = None
        self.degree = degree  # Minimum degree
    
    def search(self, key):
        """Search for a key in the B+ Tree"""
        if self.root is None:
            return None
        
        node, _ = self.root.search(key)
        if node and node.is_leaf:
            # Find the exact key in the leaf
            for i in range(node.num_keys):
                if node.keys[i] == key:
                    return node.values[i]
        return None
    
    def insert(self, key, value=None):
        """Insert a key-value pair into the B+ Tree"""
        if self.root is None:
            # If tree is empty, create a root node
            self.root = BPlusTreeNode(self.degree, True)
            self.root.keys.append(key)
            if value is not None:
                self.root.values.append(value)
            self.root.num_keys = 1
        else:
            # If root is full, split it first
            if self.root.is_full():
                # Create a new root
                new_root = BPlusTreeNode(self.degree, False)
                new_root.children.append(self.root)
                
                # Split the old root
                new_root.split_child(0)
                
                # Update the root
                self.root = new_root
            
            # Insert the key
            self.root.insert_non_full(key, value)
    
    def range_query(self, start_key, end_key):
        """Perform a range query from start_key to end_key"""
        result = []
        
        # Find the first leaf with a key >= start_key
        if self.root is None:
            return result
        
        # Navigate to the first leaf that might contain start_key
        node = self.root
        while not node.is_leaf:
            i = 0
            while i < node.num_keys and start_key > node.keys[i]:
                i += 1
            node = node.children[i]
        
        # Now node is a leaf, find the first key >= start_key
        i = 0
        while i < node.num_keys and start_key > node.keys[i]:
            i += 1
        
        # Collect all keys in the range by following leaf links
        while node and i < node.num_keys and node.keys[i] <= end_key:
            result.append((node.keys[i], node.values[i]))
            i += 1
            
            # Move to the next leaf if needed
            if i >= node.num_keys:
                node = node.next
                i = 0
        
        return result
    
    def inorder_traversal(self, node=None, result=None):
        """Inorder traversal of the B+ Tree"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            if node.is_leaf:
                # For leaf nodes, collect all key-value pairs
                for i in range(node.num_keys):
                    result.append((node.keys[i], node.values[i]))
                
                # Follow the leaf chain
                current = node.next
                while current:
                    for i in range(current.num_keys):
                        result.append((current.keys[i], current.values[i]))
                    current = current.next
            else:
                # For internal nodes, traverse children in order
                for i in range(node.num_keys + 1):
                    self.inorder_traversal(node.children[i], result)
        
        return result
    
    def print_tree(self, node=None, level=0):
        """Print the B+ Tree structure for debugging"""
        if node is None:
            node = self.root
        
        if node:
            print("  " * level, end="")
            print(node)
            
            if not node.is_leaf:
                for i in range(node.num_keys + 1):
                    self.print_tree(node.children[i], level + 1)
    
    def height(self, node=None):
        """Calculate the height of the B+ Tree"""
        if node is None:
            node = self.root
        
        if node is None or node.is_leaf:
            return 0
        
        max_child_height = 0
        for i in range(node.num_keys + 1):
            child_height = self.height(node.children[i])
            max_child_height = max(max_child_height, child_height)
        
        return max_child_height + 1
    
    def size(self, node=None):
        """Count the number of keys in the B+ Tree"""
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        if node.is_leaf:
            return node.num_keys
        
        total = 0
        for i in range(node.num_keys + 1):
            total += self.size(node.children[i])
        
        return total

# Example usage
bpt = BPlusTree(degree=3)  # Minimum degree of 3

# Insert key-value pairs
data = [
    (10, "Value 10"), (20, "Value 20"), (5, "Value 5"),
    (6, "Value 6"), (12, "Value 12"), (30, "Value 30"),
    (7, "Value 7"), (17, "Value 17"), (8, "Value 8"),
    (25, "Value 25"), (15, "Value 15"), (35, "Value 35")
]

for key, value in data:
    bpt.insert(key, value)
    print(f"Inserted ({key}, {value})")

print("\nB+ Tree structure:")
bpt.print_tree()

print("\nInorder traversal (sorted key-value pairs):")
pairs = bpt.inorder_traversal()
for key, value in pairs:
    print(f"  {key}: {value}")

print("\nSearch operations:")
for key in [6, 15, 30]:
    value = bpt.search(key)
    print(f"Search {key}: {'Found ' + str(value) + "'" if value is not None else 'Not found'}")

print("\nRange query (keys between 10 and 25):")
range_result = bpt.range_query(10, 25)
for key, value in range_result:
    print(f"  {key}: {value}")

print("\nHeight:", bpt.height())
print("Size:", bpt.size())

# Example: B+ Tree for a database index with range queries
class DatabaseIndex:
    def __init__(self, degree=4):
        """Initialize a database index using a B+ Tree"""
        self.bpt = BPlusTree(degree)
        self.records = {}  # In a real DB, this would reference actual records
    
    def add_record(self, key, data):
        """Add a record to the index"""
        self.bpt.insert(key, data)
        self.records[key] = data
        print(f"Added record with key {key}")
    
    def find_record(self, key):
        """Find a record by key"""
        value = self.bpt.search(key)
        if value is not None:
            print(f"Found record with key {key}: {value}")
            return value
        else:
            print(f"Record with key {key} not found")
            return None
    
    def range_query(self, start_key, end_key):
        """Find records in a key range"""
        results = self.bpt.range_query(start_key, end_key)
        print(f"Records with keys between {start_key} and {end_key}:")
        for key, value in results:
            print(f"  {key}: {value}")
        return results
    
    def list_all_records(self):
        """List all records in sorted order"""
        pairs = self.bpt.inorder_traversal()
        print(f"All records (sorted by key):")
        for key, value in pairs:
            print(f"  {key}: {value}")
        return pairs

# Create and use a database index
print("\nDatabase Index Example:")
db_index = DatabaseIndex(degree=3)

db_index.add_record(1001, "Customer: Alice")
db_index.add_record(1005, "Customer: Bob")
db_index.add_record(1003, "Customer: Charlie")
db_index.add_record(1002, "Customer: Diana")
db_index.add_record(1004, "Customer: Eve")
db_index.add_record(1007, "Customer: Frank")
db_index.add_record(1006, "Customer: Grace")

db_index.find_record(1003)
db_index.find_record(1008)

db_index.range_query(1002, 1005)

db_index.list_all_records()
```

B Tree*

When: Want better space utilization than B-Tree
Why: Delays splits, keeps nodes fuller (2/3 instead of 1/2)
Examples: Systems where storage efficiency is critical

```python
# Python implementation of a B* Tree
class BStarTreeNode:
    def __init__(self, degree, is_leaf=False):
        """Initialize a B* Tree node"""
        self.degree = degree  # Minimum degree
        self.keys = []  # List of keys
        self.children = []  # List of child pointers
        self.is_leaf = is_leaf  # True if leaf node
        self.num_keys = 0  # Number of keys currently in node
    
    def is_full(self):
        """Check if node is full"""
        return self.num_keys == 2 * self.degree - 1
    
    def search(self, key):
        """Search for key in this node"""
        # Find first key greater than or equal to key
        i = 0
        while i < self.num_keys and key > self.keys[i]:
            i += 1
        
        # If key is found in this node
        if i < self.num_keys and self.keys[i] == key:
            return self, i
        
        # If this is a leaf node, key is not present
        if self.is_leaf:
            return None, -1
        
        # Recurse to the appropriate child
        return self.children[i].search(key)
    
    def split_child(self, i):
        """Split child at index i with B* Tree redistribution"""
        degree = self.degree
        child = self.children[i]
        
        # Create a new node to hold (degree-1) keys from child
        new_node = BStarTreeNode(degree, child.is_leaf)
        
        # B* Tree tries to redistribute before splitting
        # Try to redistribute with left sibling
        if i > 0:
            left_sibling = self.children[i-1]
            if left_sibling.num_keys < 2 * degree - 1:  # Has space
                # Redistribute from child to left sibling
                keys_to_move = min(degree // 2, child.num_keys)
                for j in range(keys_to_move):
                    left_sibling.keys.append(child.keys[0])
                    left_sibling.num_keys += 1
                    child.keys.pop(0)
                    child.num_keys -= 1
                
                # If not a leaf, also redistribute children
                if not child.is_leaf:
                    for j in range(keys_to_move):
                        left_sibling.children.append(child.children[0])
                        child.children.pop(0)
                
                # Update parent key
                self.keys[i-1] = child.keys[0] if child.num_keys > 0 else self.keys[i-1]
                return
        
        # Try to redistribute with right sibling
        if i < len(self.children) - 1:
            right_sibling = self.children[i+1]
            if right_sibling.num_keys < 2 * degree - 1:  # Has space
                # Redistribute from child to right sibling
                keys_to_move = min(degree // 2, child.num_keys)
                for j in range(keys_to_move):
                    right_sibling.keys.insert(0, child.keys[-1])
                    right_sibling.num_keys += 1
                    child.keys.pop()
                    child.num_keys -= 1
                
                # If not a leaf, also redistribute children
                if not child.is_leaf:
                    for j in range(keys_to_move):
                        right_sibling.children.insert(0, child.children[-1])
                        child.children.pop()
                
                # Update parent key
                self.keys[i] = right_sibling.keys[0] if right_sibling.num_keys > 0 else self.keys[i]
                return
        
        # If redistribution is not possible, perform split
        new_node.num_keys = degree - 1
        
        # Copy last (degree-1) keys of child to new_node
        for j in range(degree - 1):
            new_node.keys.append(child.keys[j + degree])
        
        # Copy last degree children of child to new_node
        if not child.is_leaf:
            for j in range(degree):
                new_node.children.append(child.children[j + degree])
        
        # Reduce the number of keys in child
        child.num_keys = degree - 1
        
        # Insert new child in this node's children list
        self.children.insert(i + 1, new_node)
        
        # Move middle key of child up to this node
        self.keys.insert(i, child.keys[degree - 1])
        self.num_keys += 1
    
    def insert_non_full(self, key, value=None):
        """Insert a key into a non-full node"""
        i = self.num_keys - 1
        
        # If this is a leaf node, find the correct position and insert
        if self.is_leaf:
            # Move all greater keys one position ahead
            self.keys.append(None)  # Make space
            if value is not None:
                self.values.append(None)  # Make space for value
            
            while i >= 0 and key < self.keys[i]:
                self.keys[i + 1] = self.keys[i]
                if value is not None:
                    self.values[i + 1] = self.values[i]
                i -= 1
            
            # Insert the new key
            self.keys[i + 1] = key
            if value is not None:
                self.values[i + 1] = value
            self.num_keys += 1
        else:
            # Find the child to receive the new key
            while i >= 0 and key < self.keys[i]:
                i -= 1
            
            # If the child is full, split it first
            if self.children[i + 1].is_full():
                self.split_child(i + 1)
                
                # After split, the middle key moves up, determine which child to use
                if key > self.keys[i]:
                    i += 1
            
            # Recursively insert into the appropriate child
            self.children[i + 1].insert_non_full(key, value)
    
    def __str__(self):
        """String representation of the node"""
        if self.is_leaf:
            return f"Leaf - Keys: {self.keys[:self.num_keys]}"
        else:
            return f"Internal - Keys: {self.keys[:self.num_keys]}"

class BStarTree:
    def __init__(self, degree=3):
        """Initialize an empty B* Tree with minimum degree"""
        self.root = None
        self.degree = degree  # Minimum degree
    
    def search(self, key):
        """Search for a key in the B* Tree"""
        if self.root is None:
            return None
        
        node, _ = self.root.search(key)
        return node is not None
    
    def insert(self, key, value=None):
        """Insert a key-value pair into the B* Tree"""
        if self.root is None:
            # If the tree is empty, create a root node
            self.root = BStarTreeNode(self.degree, True)
            self.root.keys.append(key)
            if value is not None:
                self.root.values = [value]
            self.root.num_keys = 1
        else:
            # If the root is full, split it first
            if self.root.is_full():
                # Create a new root
                new_root = BStarTreeNode(self.degree, False)
                new_root.children.append(self.root)
                
                # Split the old root
                new_root.split_child(0)
                
                # Update the root
                self.root = new_root
            
            # Insert the key
            self.root.insert_non_full(key, value)
    
    def inorder_traversal(self, node=None, result=None):
        """Inorder traversal of the B* Tree"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            if node.is_leaf:
                # For leaf nodes, collect all key-value pairs
                for i in range(node.num_keys):
                    if hasattr(node, 'values') and i < len(node.values):
                        result.append((node.keys[i], node.values[i]))
                    else:
                        result.append(node.keys[i])
            else:
                # For internal nodes, traverse children in order
                for i in range(node.num_keys + 1):
                    self.inorder_traversal(node.children[i], result)
                
                # Also collect the keys
                for i in range(node.num_keys):
                    result.append(node.keys[i])
        
        return result
    
    def print_tree(self, node=None, level=0):
        """Print the B* Tree structure for debugging"""
        if node is None:
            node = self.root
        
        if node:
            print("  " * level, end="")
            print(node)
            
            if not node.is_leaf:
                for i in range(node.num_keys + 1):
                    self.print_tree(node.children[i], level + 1)
    
    def height(self, node=None):
        """Calculate the height of the B* Tree"""
        if node is None:
            node = self.root
        
        if node is None or node.is_leaf:
            return 0
        
        max_child_height = 0
        for i in range(node.num_keys + 1):
            child_height = self.height(node.children[i])
            max_child_height = max(max_child_height, child_height)
        
        return max_child_height + 1
    
    def size(self, node=None):
        """Count the number of keys in the B* Tree"""
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        if node.is_leaf:
            return node.num_keys
        
        total = node.num_keys
        for i in range(node.num_keys + 1):
            total += self.size(node.children[i])
        
        return total

# Example usage
bst = BStarTree(degree=3)  # Minimum degree of 3

# Insert key-value pairs
data = [
    (10, "Value10"), (20, "Value20"), (5, "Value5"),
    (6, "Value6"), (12, "Value12"), (30, "Value30"),
    (7, "Value7"), (17, "Value17"), (8, "Value8"),
    (25, "Value25"), (15, "Value15"), (35, "Value35")
]

for key, value in data:
    bst.insert(key, value)
    print(f"Inserted ({key}, {value})")

print("\nB* Tree structure:")
bst.print_tree()

print("\nInorder traversal (sorted key-value pairs):")
pairs = bst.inorder_traversal()
for key, value in pairs:
    if isinstance(value, tuple):
        print(f"  {key}: {value[1]}")
    else:
        print(f"  {key}: {value}")

print("\nSearch operations:")
for key in [6, 15, 30]:
    found = bst.search(key)
    print(f"Search {key}: {'Found' if found else 'Not found'}")

print("\nHeight:", bst.height())
print("Size:", bst.size())

# Example: B* Tree for a database index with better space utilization
class DatabaseIndex:
    def __init__(self, degree=4):
        """Initialize a database index using a B* Tree"""
        self.bst = BStarTree(degree)
        self.records = {}  # In a real DB, this would reference actual records
    
    def add_record(self, key, data):
        """Add a record to the index"""
        self.bst.insert(key, data)
        self.records[key] = data
        print(f"Added record with key {key}")
    
    def find_record(self, key):
        """Find a record by key"""
        if self.bst.search(key):
            print(f"Found record with key {key}: {self.records.get(key)}")
            return self.records.get(key)
        else:
            print(f"Record with key {key} not found")
            return None
    
    def list_all_records(self):
        """List all records in sorted order"""
        keys = [item[0] if isinstance(item, tuple) else item for item in self.bst.inorder_traversal()]
        print(f"All records (sorted by key):")
        for key in keys:
            if key in self.records:
                print(f"  Key {key}: {self.records[key]}")
        return keys

# Create and use a database index
print("\nDatabase Index Example:")
db_index = DatabaseIndex(degree=3)

db_index.add_record(1001, "Customer: Alice")
db_index.add_record(1005, "Customer: Bob")
db_index.add_record(1003, "Customer: Charlie")
db_index.add_record(1002, "Customer: Diana")
db_index.add_record(1004, "Customer: Eve")

db_index.find_record(1003)
db_index.find_record(1007)

db_index.list_all_records()
```

2-3 Tree

When: Teaching balanced trees or theoretical analysis
Why: Simpler concept than B-Trees, naturally balanced, foundation for understanding
Examples: Educational purposes, theoretical computer science

```python
# Python implementation of a 2-3 Tree
class TwoThreeNode:
    def __init__(self):
        """Initialize a 2-3 tree node"""
        self.keys = []  # Can hold 1 or 2 keys
        self.children = []  # Can hold 2 or 3 children
        self.num_keys = 0  # Number of keys in the node
    
    def is_leaf(self):
        """Check if the node is a leaf"""
        return len(self.children) == 0
    
    def is_full(self):
        """Check if the node has 2 keys (maximum)"""
        return self.num_keys == 2
    
    def insert_key(self, key):
        """Insert a key into a node that is not full"""
        if self.num_keys == 0:
            self.keys.append(key)
            self.num_keys = 1
        elif self.num_keys == 1:
            if key < self.keys[0]:
                self.keys.insert(0, key)
            else:
                self.keys.append(key)
            self.num_keys = 2
    
    def get_child_index(self, key):
        """Get the index of the child that should contain the key"""
        if self.num_keys == 0:
            return 0
        elif self.num_keys == 1:
            if key < self.keys[0]:
                return 0
            else:
                return 1
        else:  # self.num_keys == 2
            if key < self.keys[0]:
                return 0
            elif key < self.keys[1]:
                return 1
            else:
                return 2
    
    def __str__(self):
        """String representation of the node"""
        if self.is_leaf():
            return f"Leaf({self.keys[:self.num_keys]})"
        else:
            return f"Internal({self.keys[:self.num_keys]})"

class TwoThreeTree:
    def __init__(self):
        """Initialize an empty 2-3 tree"""
        self.root = None
    
    def is_empty(self):
        """Check if the tree is empty"""
        return self.root is None
    
    def search(self, key):
        """Search for a key in the 2-3 tree"""
        if self.is_empty():
            return False
        
        return self._search_recursive(self.root, key)
    
    def _search_recursive(self, node, key):
        """Helper method to search recursively"""
        if node.is_leaf():
            # Check if key is in this leaf node
            return key in node.keys[:node.num_keys]
        
        # Find which child to search
        child_index = node.get_child_index(key)
        return self._search_recursive(node.children[child_index], key)
    
    def insert(self, key):
        """Insert a key into the 2-3 tree"""
        if self.is_empty():
            # Create a new root if tree is empty
            self.root = TwoThreeNode()
            self.root.insert_key(key)
            return
        
        # Insert recursively and handle splits
        self.root = self._insert_recursive(self.root, key)
    
    def _insert_recursive(self, node, key):
        """Helper method to insert recursively and handle splits"""
        if node.is_leaf():
            # Insert into leaf node
            node.insert_key(key)
            
            # Check if node needs to be split
            if node.num_keys > 2:
                return self._split_node(node)
            return node
        
        # Find which child to insert into
        child_index = node.get_child_index(key)
        
        # Recursively insert into the appropriate child
        result = self._insert_recursive(node.children[child_index], key)
        
        # Check if child was split
        if isinstance(result, tuple) and len(result) == 2:
            # Child was split, need to insert the middle key and new child
            new_child, middle_key = result
            
            # Insert middle key into current node
            node.insert_key(middle_key)
            
            # Insert new child at the correct position
            node.children.insert(child_index + 1, new_child)
            
            # Check if current node needs to be split
            if node.num_keys > 2:
                return self._split_node(node)
        
        return node
    
    def _split_node(self, node):
        """Split a 2-3 node"""
        if node.num_keys != 3:
            return node
        
        # Create two new nodes
        left_node = TwoThreeNode()
        right_node = TwoThreeNode()
        
        # Left node gets the smallest key
        left_node.insert_key(node.keys[0])
        
        # Right node gets the largest key
        right_node.insert_key(node.keys[2])
        
        # Middle key moves up
        middle_key = node.keys[1]
        
        # Distribute children
        if not node.is_leaf():
            # Left node gets the first two children
            left_node.children = node.children[:2]
            
            # Right node gets the last two children
            right_node.children = node.children[2:]
        
        # Return the new nodes and the middle key
        return (left_node, middle_key, right_node)
    
    def inorder_traversal(self, node=None, result=None):
        """Inorder traversal of the 2-3 tree"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            if node.is_leaf():
                # Add all keys in the leaf
                for i in range(node.num_keys):
                    result.append(node.keys[i])
            else:
                # Traverse children in order
                for i in range(node.num_keys + 1):
                    self.inorder_traversal(node.children[i], result)
                
                # Add the keys in between
                for i in range(node.num_keys):
                    result.append(node.keys[i])
        
        return result
    
    def preorder_traversal(self, node=None, result=None):
        """Preorder traversal of the 2-3 tree"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            # Add all keys in the node
            for i in range(node.num_keys):
                result.append(node.keys[i])
            
            # Traverse all children
            for i in range(node.num_keys + 1):
                self.preorder_traversal(node.children[i], result)
        
        return result
    
    def height(self, node=None):
        """Calculate the height of the 2-3 tree"""
        if node is None:
            node = self.root
        
        if node is None or node.is_leaf():
            return 0
        
        # Height is 1 + max height of children
        max_child_height = 0
        for i in range(node.num_keys + 1):
            child_height = self.height(node.children[i])
            max_child_height = max(max_child_height, child_height)
        
        return max_child_height + 1
    
    def size(self, node=None):
        """Count the number of keys in the 2-3 tree"""
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        if node.is_leaf():
            return node.num_keys
        
        # Count keys in node and all children
        total = node.num_keys
        for i in range(node.num_keys + 1):
            total += self.size(node.children[i])
        
        return total
    
    def __str__(self):
        """String representation of the 2-3 tree (inorder)"""
        return str(self.inorder_traversal())

# Example usage
tt = TwoThreeTree()

# Insert elements
for key in [30, 20, 10, 40, 50, 60, 70, 25, 35, 45]:
    tt.insert(key)
    print(f"Inserted {key}")

print("\n2-3 Tree (inorder):", tt)  # Sorted keys
print("Height:", tt.height())
print("Size:", tt.size())

# Tree traversals
print("\nPre-order traversal:", tt.preorder_traversal())
print("In-order traversal:", tt.inorder_traversal())

# Search operations
print("\nSearch operations:")
for key in [20, 25, 55]:
    found = tt.search(key)
    print(f"Search {key}: {'Found' if found else 'Not found'}")

# Example: 2-3 Tree for a simple symbol table
class SymbolTable:
    def __init__(self):
        """Initialize a symbol table using a 2-3 tree"""
        self.tree = TwoThreeTree()
        self.values = {}  # Map keys to values
    
    def put(self, key, value):
        """Add or update a key-value pair"""
        if key not in self.values:
            self.tree.insert(key)
        self.values[key] = value
        print(f"Added/Updated key {key} with value {value}")
    
    def get(self, key):
        """Get value for a given key"""
        if self.tree.search(key):
            return self.values.get(key)
        return None
    
    def contains_key(self, key):
        """Check if key exists in symbol table"""
        return self.tree.search(key)
    
    def remove_key(self, key):
        """Remove a key from symbol table"""
        if self.tree.search(key):
            # In a full implementation, we'd need to implement deletion
            # For simplicity, we're just removing from our dictionary
            del self.values[key]
            print(f"Removed key {key}")
        else:
            print(f"Cannot remove: Key {key} not found")
    
    def get_all_keys(self):
        """Get all keys in sorted order"""
        keys = self.tree.inorder_traversal()
        print(f"All keys (sorted): {keys}")
        return keys

# Create and use a symbol table
print("\nSymbol Table Example:")
st = SymbolTable()

st.put("apple", 1)
st.put("banana", 2)
st.put("cherry", 3)
st.put("date", 4)
st.put("elderberry", 5)

st.get_all_keys()

print("\nValue of 'cherry':", st.get("cherry"))
print("Contains 'grape':", st.contains_key("grape"))

st.remove_key("banana")
st.remove_key("grape")  # This key doesn't exist

st.get_all_keys()
```

2-3-4 Tree

When: Understanding Red-Black trees (they're isomorphic)
Why: Easier to visualize than Red-Black trees, naturally balanced
Examples: Educational purposes, alternative to Red-Black implementation

```python
# Python implementation of a 2-3-4 Tree
class TwoThreeFourNode:
    def __init__(self):
        """Initialize a 2-3-4 tree node"""
        self.keys = []  # Can hold 1, 2, or 3 keys
        self.children = []  # Can hold 2, 3, or 4 children
        self.num_keys = 0  # Number of keys in the node
    
    def is_leaf(self):
        """Check if the node is a leaf"""
        return len(self.children) == 0
    
    def is_full(self):
        """Check if the node has the maximum number of keys"""
        # For 2-3-4 tree, a node with 3 keys has 4 children
        return self.num_keys == 3
    
    def is_two_node(self):
        """Check if the node has 2 keys (and 3 children)"""
        return self.num_keys == 2
    
    def is_three_node(self):
        """Check if the node has 3 keys (and 4 children)"""
        return self.num_keys == 3
    
    def find_key_index(self, key):
        """Find the index where key should be inserted"""
        for i in range(self.num_keys):
            if key < self.keys[i]:
                return i
        return self.num_keys  # Key is greater than all existing keys
    
    def insert_key(self, key):
        """Insert a key into a node that is not full"""
        # Find the correct position for the key
        index = self.find_key_index(key)
        self.keys.insert(index, key)
        self.num_keys += 1
    
    def get_child_index(self, key):
        """Get the index of the child that should contain the key"""
        if self.num_keys == 0:
            return 0
        elif self.num_keys == 1:
            if key < self.keys[0]:
                return 0
            else:
                return 1
        else:  # self.num_keys == 2 or 3
            if key < self.keys[0]:
                return 0
            elif key < self.keys[1]:
                return 1
            elif self.num_keys == 2 or key < self.keys[2]:
                return 2
            else:
                return 3
    
    def __str__(self):
        """String representation of the node"""
        if self.is_leaf():
            return f"Leaf({self.keys[:self.num_keys]})"
        else:
            return f"Internal({self.keys[:self.num_keys]})"

class TwoThreeFourTree:
    def __init__(self):
        """Initialize an empty 2-3-4 tree"""
        self.root = None
    
    def is_empty(self):
        """Check if the tree is empty"""
        return self.root is None
    
    def search(self, key):
        """Search for a key in the 2-3-4 tree"""
        if self.is_empty():
            return False
        
        return self._search_recursive(self.root, key)
    
    def _search_recursive(self, node, key):
        """Helper method to search recursively"""
        if node.is_leaf():
            # Check if key is in this leaf node
            return key in node.keys[:node.num_keys]
        
        # Find which child to search
        child_index = node.get_child_index(key)
        return self._search_recursive(node.children[child_index], key)
    
    def insert(self, key):
        """Insert a key into the 2-3-4 tree"""
        if self.is_empty():
            # Create a new root if tree is empty
            self.root = TwoThreeFourNode()
            self.root.insert_key(key)
            return
        
        # Insert recursively and handle splits
        self.root = self._insert_recursive(self.root, key)
    
    def _insert_recursive(self, node, key):
        """Helper method to insert recursively and handle splits"""
        if node.is_leaf():
            # Insert into leaf node
            node.insert_key(key)
            
            # Check if node needs to be split
            if node.is_full():
                return self._split_node(node)
            return node
        
        # Find which child to insert into
        child_index = node.get_child_index(key)
        
        # Recursively insert into the appropriate child
        result = self._insert_recursive(node.children[child_index], key)
        
        # Check if child was split
        if isinstance(result, tuple) and len(result) == 2:
            # Child was split, need to insert the middle key and new child
            new_child, middle_key = result
            
            # Insert middle key into current node
            node.insert_key(middle_key)
            
            # Insert new child at the correct position
            node.children.insert(child_index + 1, new_child)
            
            # Check if current node needs to be split
            if node.is_full():
                return self._split_node(node)
        
        return node
    
    def _split_node(self, node):
        """Split a 2-3-4 node"""
        if node.num_keys != 3:
            return node
        
        # Create two new nodes
        left_node = TwoThreeFourNode()
        right_node = TwoThreeFourNode()
        
        # Left node gets the smallest key
        left_node.insert_key(node.keys[0])
        
        # Right node gets the largest key
        right_node.insert_key(node.keys[2])
        
        # Middle key moves up
        middle_key = node.keys[1]
        
        # Distribute children
        if not node.is_leaf():
            # Left node gets the first two children
            left_node.children = node.children[:2]
            
            # Right node gets the last two children
            right_node.children = node.children[2:]
        
        # Return the new nodes and the middle key
        return (left_node, middle_key, right_node)
    
    def inorder_traversal(self, node=None, result=None):
        """Inorder traversal of the 2-3-4 tree"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            if node.is_leaf():
                # Add all keys in the leaf
                for i in range(node.num_keys):
                    result.append(node.keys[i])
            else:
                # Traverse children in order
                for i in range(node.num_keys + 1):
                    self.inorder_traversal(node.children[i], result)
                
                # Add the keys in between
                for i in range(node.num_keys):
                    result.append(node.keys[i])
        
        return result
    
    def preorder_traversal(self, node=None, result=None):
        """Preorder traversal of the 2-3-4 tree"""
        if result is None:
            result = []
        
        if node is None:
            node = self.root
        
        if node:
            # Add all keys in the node
            for i in range(node.num_keys):
                result.append(node.keys[i])
            
            # Traverse all children
            for i in range(node.num_keys + 1):
                self.preorder_traversal(node.children[i], result)
        
        return result
    
    def height(self, node=None):
        """Calculate the height of the 2-3-4 tree"""
        if node is None:
            node = self.root
        
        if node is None or node.is_leaf():
            return 0
        
        # Height is 1 + max height of children
        max_child_height = 0
        for i in range(node.num_keys + 1):
            child_height = self.height(node.children[i])
            max_child_height = max(max_child_height, child_height)
        
        return max_child_height + 1
    
    def size(self, node=None):
        """Count the number of keys in the 2-3-4 tree"""
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        if node.is_leaf():
            return node.num_keys
        
        # Count keys in node and all children
        total = node.num_keys
        for i in range(node.num_keys + 1):
            total += self.size(node.children[i])
        
        return total
    
    def __str__(self):
        """String representation of the 2-3-4 tree (inorder)"""
        return str(self.inorder_traversal())

# Example usage
ttf = TwoThreeFourTree()

# Insert elements
for key in [30, 20, 10, 40, 50, 60, 70, 25, 35, 45, 15, 5, 80]:
    ttf.insert(key)
    print(f"Inserted {key}")

print("\n2-3-4 Tree (inorder):", ttf)  # Sorted keys
print("Height:", ttf.height())
print("Size:", ttf.size())

# Tree traversals
print("\nPre-order traversal:", ttf.preorder_traversal())
print("In-order traversal:", ttf.inorder_traversal())

# Search operations
print("\nSearch operations:")
for key in [20, 25, 55]:
    found = ttf.search(key)
    print(f"Search {key}: {'Found' if found else 'Not found'}")

# Example: 2-3-4 Tree for a simple symbol table
class SymbolTable:
    def __init__(self):
        """Initialize a symbol table using a 2-3-4 tree"""
        self.tree = TwoThreeFourTree()
        self.values = {}  # Map keys to values
    
    def put(self, key, value):
        """Add or update a key-value pair"""
        if key not in self.values:
            self.tree.insert(key)
        self.values[key] = value
        print(f"Added/Updated key {key} with value {value}")
    
    def get(self, key):
        """Get value for a given key"""
        if self.tree.search(key):
            return self.values.get(key)
        return None
    
    def contains_key(self, key):
        """Check if key exists in symbol table"""
        return self.tree.search(key)
    
    def remove_key(self, key):
        """Remove a key from symbol table"""
        if self.tree.search(key):
            # In a full implementation, we'd need to implement deletion
            # For simplicity, we're just removing from our dictionary
            del self.values[key]
            print(f"Removed key {key}")
        else:
            print(f"Cannot remove: Key {key} not found")
    
    def get_all_keys(self):
        """Get all keys in sorted order"""
        keys = self.tree.inorder_traversal()
        print(f"All keys (sorted): {keys}")
        return keys

# Create and use a symbol table
print("\nSymbol Table Example:")
st = SymbolTable()

st.put("apple", 1)
st.put("banana", 2)
st.put("cherry", 3)
st.put("date", 4)
st.put("elderberry", 5)
st.put("fig", 6)
st.put("grape", 7)

st.get_all_keys()

print("\nValue of 'cherry':", st.get("cherry"))
print("Contains 'grape':", st.contains_key("grape"))

st.remove_key("banana")
st.remove_key("grape")  # This key doesn't exist

st.get_all_keys()
```

Heaps
Binary Heap (Min/Max)

When: Need priority queue with simple implementation
Why: O(log n) insert/delete, O(1) find-min/max, array-based (cache-friendly)
Examples: Heap sort, priority queues, graph algorithms (Dijkstra, Prim)

```python
# Python implementation of a Binary Heap (Min-Heap and Max-Heap)
import heapq

class MinHeap:
    def __init__(self):
        """Initialize an empty min-heap"""
        self.heap = []
    
    def is_empty(self):
        """Check if the heap is empty"""
        return len(self.heap) == 0
    
    def insert(self, item):
        """Insert an item into the min-heap, O(log n)"""
        heapq.heappush(self.heap, item)
    
    def extract_min(self):
        """Extract and return the minimum item, O(log n)"""
        if self.is_empty():
            raise IndexError("extract_min from empty heap")
        return heapq.heappop(self.heap)
    
    def peek(self):
        """Return the minimum item without removing it, O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty heap")
        return self.heap[0]
    
    def size(self):
        """Return the number of items in the heap"""
        return len(self.heap)
    
    def __str__(self):
        """String representation of the heap"""
        return str(self.heap)

class MaxHeap:
    def __init__(self):
        """Initialize an empty max-heap"""
        self.heap = []
    
    def is_empty(self):
        """Check if the heap is empty"""
        return len(self.heap) == 0
    
    def insert(self, item):
        """Insert an item into the max-heap, O(log n)"""
        # For max-heap, we store negative values to use heapq as a max-heap
        heapq.heappush(self.heap, -item)
    
    def extract_max(self):
        """Extract and return the maximum item, O(log n)"""
        if self.is_empty():
            raise IndexError("extract_max from empty heap")
        return -heapq.heappop(self.heap)
    
    def peek(self):
        """Return the maximum item without removing it, O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty heap")
        return -self.heap[0]
    
    def size(self):
        """Return the number of items in the heap"""
        return len(self.heap)
    
    def __str__(self):
        """String representation of the heap"""
        # Convert back to positive values for display
        return str([-x for x in self.heap])

# Example usage for Min-Heap
min_heap = MinHeap()

# Insert elements
for item in [5, 3, 8, 1, 10, 6, 7, 4, 2, 9]:
    min_heap.insert(item)
    print(f"Inserted {item}")

print("\nMin-Heap:", min_heap)  # Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("Size:", min_heap.size())  # Output: 10

# Extract minimum elements
print("\nExtracting minimum elements:")
while not min_heap.is_empty():
    min_item = min_heap.extract_min()
    print(f"Extracted: {min_item}")

# Example usage for Max-Heap
max_heap = MaxHeap()

# Insert elements
for item in [5, 3, 8, 1, 10, 6, 7, 4, 2, 9]:
    max_heap.insert(item)
    print(f"Inserted {item}")

print("\nMax-Heap:", max_heap)  # Output: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
print("Size:", max_heap.size())  # Output: 10

# Extract maximum elements
print("\nExtracting maximum elements:")
while not max_heap.is_empty():
    max_item = max_heap.extract_max()
    print(f"Extracted: {max_item}")

# Example: Priority Queue using Min-Heap
class PriorityQueue:
    def __init__(self):
        """Initialize a priority queue using a min-heap"""
        self.heap = MinHeap()
    
    def enqueue(self, item, priority):
        """Add an item with the given priority"""
        # Store as a tuple (priority, item) for min-heap
        self.heap.insert((priority, item))
    
    def dequeue(self):
        """Remove and return the item with minimum priority"""
        if self.heap.is_empty():
            raise IndexError("dequeue from empty priority queue")
        priority, item = self.heap.extract_min()
        return item
    
    def peek(self):
        """Return the item with minimum priority without removing it"""
        if self.heap.is_empty():
            raise IndexError("peek from empty priority queue")
        priority, item = self.heap.peek()
        return item
    
    def is_empty(self):
        """Check if the priority queue is empty"""
        return self.heap.is_empty()
    
    def size(self):
        """Return the number of items in the priority queue"""
        return self.heap.size()
    
    def __str__(self):
        """String representation of the priority queue"""
        return str([(p, i) for p, i in self.heap.heap])

# Create and use a priority queue
print("\nPriority Queue Example:")
pq = PriorityQueue()

pq.enqueue("Task 1", 5)
pq.enqueue("Task 2", 1)
pq.enqueue("Task 3", 3)
pq.enqueue("Task 4", 2)
pq.enqueue("Task 5", 4)

print("Priority Queue:", pq)
print("Dequeue:", pq.dequeue())  # Output: Task 2
print("Dequeue:", pq.dequeue())  # Output: Task 4
print("Dequeue:", pq.dequeue())  # Output: Task 3
print("Dequeue:", pq.dequeue())  # Output: Task 5
print("Dequeue:", pq.dequeue())  # Output: Task 1

# Example: Heap Sort
def heap_sort(arr):
    """Sort an array using heap sort, O(n log n)"""
    if not arr:
        return []
    
    # Create a min-heap and insert all elements
    heap = MinHeap()
    for item in arr:
        heap.insert(item)
    
    # Extract all elements to get sorted order
    sorted_arr = []
    while not heap.is_empty():
        sorted_arr.append(heap.extract_min())
    
    return sorted_arr

# Test heap sort
print("\nHeap Sort Example:")
unsorted_array = [9, 4, 7, 1, 3, 6, 8, 2, 5]
print("Unsorted array:", unsorted_array)
sorted_array = heap_sort(unsorted_array)
print("Sorted array:", sorted_array)  # Output: [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Example: Dijkstra's algorithm using Min-Heap
class Graph:
    def __init__(self, vertices):
        """Initialize a graph with the given number of vertices"""
        self.V = vertices
        self.adj = [[] for _ in range(vertices)]  # Adjacency list
    
    def add_edge(self, u, v, weight):
        """Add an edge between u and v with the given weight"""
        self.adj[u].append((v, weight))
        self.adj[v].append((u, weight))

def dijkstra(graph, start):
    """Find shortest paths from start vertex using Dijkstra's algorithm"""
    V = graph.V
    dist = [float('inf')] * V  # Distance from start to each vertex
    prev = [None] * V  # Previous vertex in the shortest path
    
    # Priority queue to store (distance, vertex) pairs
    pq = PriorityQueue()
    pq.enqueue((0, start), 0)  # (distance, vertex)
    
    dist[start] = 0
    
    while not pq.is_empty():
        current_dist, u = pq.heap.peek()  # Get vertex with minimum distance
        item = pq.dequeue()  # Remove from queue
        
        # For all neighbors of u
        for v, weight in graph.adj[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                prev[v] = u
                pq.enqueue((dist[v], v), dist[v])
    
    return dist, prev

# Create a graph and run Dijkstra's algorithm
print("\nDijkstra's Algorithm Example:")
g = Graph(6)
g.add_edge(0, 1, 7)
g.add_edge(0, 2, 9)
g.add_edge(0, 5, 14)
g.add_edge(1, 2, 10)
g.add_edge(1, 3, 15)
g.add_edge(2, 3, 11)
g.add_edge(3, 4, 6)
g.add_edge(3, 5, 9)

start_vertex = 0
distances, predecessors = dijkstra(g, start_vertex)

print(f"Shortest distances from vertex {start_vertex}:")
for v in range(g.V):
    print(f"  To vertex {v}: {distances[v]}")

# Reconstruct path to vertex 5
path = []
current = 5
while current is not None:
    path.insert(0, current)
    current = predecessors[current]
path.reverse()
print(f"Path to vertex 5: {path}")
```

Fibonacci Heap

When: Need better amortized performance for decrease-key operations
Why: O(1) amortized insert, find-min, merge, decrease-key; O(log n) delete-min
Examples: Network optimization algorithms, Dijkstra's algorithm (theoretical), advanced graph algorithms

```python
# Python implementation of a Fibonacci Heap
import math

class FibonacciNode:
    def __init__(self, key=None):
        """Initialize a Fibonacci heap node"""
        self.key = key
        self.degree = 0  # Number of children
        self.parent = None
        self.child = None  # One of the children
        self.left = None
        self.right = None
        self.mark = False  # Used for cascading cuts
    
    def __str__(self):
        """String representation of the node"""
        return str(self.key)

class FibonacciHeap:
    def __init__(self):
        """Initialize an empty Fibonacci heap"""
        self.min = None  # Pointer to the minimum node
        self.total_nodes = 0  # Total number of nodes in the heap
    
    def is_empty(self):
        """Check if the heap is empty"""
        return self.min is None
    
    def insert(self, key):
        """Insert a key into the Fibonacci heap, O(1) amortized"""
        # Create a new node
        node = FibonacciNode(key)
        self.total_nodes += 1
        
        # If heap is empty, make the new node the minimum
        if self.min is None:
            self.min = node
            return
        
        # Insert the node into the root list
        self._meld(node, self.min)
        
        # Update the minimum pointer if needed
        if self.min.key > node.key:
            self.min = node
    
    def _meld(self, node1, node2):
        """Merge two Fibonacci heaps, O(1) amortized"""
        if node1.key > node2.key:
            node1, node2 = node2, node1
        
        # If node1 is None, return node2
        if node1 is None:
            return node2
        
        # If node2 is None, return node1
        if node2 is None:
            return node1
        
        # Link node2 as a child of node1
        node2.parent = node1
        node1.degree += 1
        node2.right = node1.child
        node1.child = node2
        
        # Cascading cut: if node1's degree exceeds a threshold
        # (typically 3), cut node1 from its parent and meld with the min
        if node1.parent and node1.degree > 3:
            self._cascading_cut(node1)
        
        return node1
    
    def _cascading_cut(self, node):
        """Perform cascading cut on a node with too many children"""
        parent = node.parent
        if parent is None:
            return
        
        # Remove node from parent's child list
        if node == parent.child:
            parent.child = node.right
        else:
            # Find and remove node from parent's children
            prev = None
            current = parent.child
            while current and current != node:
                prev = current
                current = current.right
            if prev:
                prev.right = node.right
            else:
                parent.child = node.right
        
        # Add node to root list
        node.parent = None
        node.right = None
        node.mark = True
        self._meld(node, self.min)
        
        # Update the minimum pointer if needed
        if self.min.key > node.key:
            self.min = node
    
    def find_min(self):
        """Return the minimum key, O(1) amortized"""
        if self.is_empty():
            return None
        return self.min.key
    
    def extract_min(self):
        """Extract and return the minimum key, O(log n)"""
        if self.is_empty():
            raise IndexError("extract_min from empty heap")
        
        min_node = self.min
        self.total_nodes -= 1
        
        # If min_node is the only node, clear the heap
        if min_node.left is None and min_node.right is None:
            self.min = None
            return min_node.key
        
        # Remove min_node from the root list
        if min_node.parent:
            if min_node == min_node.parent.child:
                min_node.parent.child = min_node.right
            else:
                # Find and remove min_node from parent's children
                prev = None
                current = min_node.parent.child
                while current and current != min_node:
                    prev = current
                    current = current.right
                if prev:
                    prev.right = min_node.right
                else:
                    min_node.parent.child = min_node.right
        
        # Clear min_node's parent and child pointers
        min_node.parent = None
        min_node.mark = False
        
        # Meld min_node's children into the root list
        if min_node.child:
            self._meld(min_node.child, self.min)
        
        # Meld min_node's right child into the root list
        if min_node.right:
            self._meld(min_node.right, self.min)
        
        # Update the minimum pointer
        self._update_min()
        
        return min_node.key
    
    def decrease_key(self, node, new_key):
        """Decrease the key of a node, O(1) amortized"""
        if new_key >= node.key:
            return  # Key is not decreasing
        
        node.key = new_key
        
        # If node is now smaller than its parent, cut it
        if node.parent and node.key < node.parent.key:
            self._cascading_cut(node)
        
        # Update the minimum pointer if needed
        if self.min.key > node.key:
            self.min = node
    
    def _update_min(self):
        """Update the minimum pointer to point to the minimum key in the root list"""
        if self.min is None:
            return
        
        current = self.min
        while current.right and current.right.key < current.key:
            current = current.right
        
        self.min = current
    
    def size(self):
        """Return the number of nodes in the heap"""
        return self.total_nodes
    
    def __str__(self):
        """String representation of the Fibonacci heap"""
        return f"FibHeap({self.find_min() if not self.is_empty() else 'Empty'})"

# Example usage
fib_heap = FibonacciHeap()

# Insert elements
for key in [10, 20, 5, 15, 30, 25, 40, 35, 45]:
    fib_heap.insert(key)
    print(f"Inserted {key}")

print("\nFibonacci Heap:")
print(f"Size: {fib_heap.size()}")
print(f"Minimum: {fib_heap.find_min()}")

# Extract minimum elements
print("\nExtracting minimum elements:")
while not fib_heap.is_empty():
    min_key = fib_heap.extract_min()
    print(f"Extracted: {min_key}")
    print(f"New minimum: {fib_heap.find_min()}")

# Example: Priority Queue using Fibonacci Heap
class PriorityQueue:
    def __init__(self):
        """Initialize a priority queue using a Fibonacci heap"""
        self.heap = FibonacciHeap()
        self.node_map = {}  # Map keys to nodes for decrease_key
    
    def enqueue(self, key, priority):
        """Add an item with the given priority"""
        node = FibonacciNode(priority)
        self.node_map[priority] = node
        self.heap.insert(priority)
        print(f"Enqueued item with priority {priority}")
    
    def dequeue(self):
        """Remove and return the item with minimum priority"""
        if self.heap.is_empty():
            raise IndexError("dequeue from empty priority queue")
        min_key = self.heap.extract_min()
        node = self.node_map[min_key]
        del self.node_map[min_key]
        print(f"Dequeued item with priority {min_key}")
        return node.key
    
    def decrease_priority(self, old_priority, new_priority):
        """Decrease the priority of an item"""
        if old_priority not in self.node_map:
            return  # Item not found
        
        node = self.node_map[old_priority]
        self.heap.decrease_key(node, new_priority)
        
        # Update the map
        del self.node_map[old_priority]
        self.node_map[new_priority] = node
        print(f"Decreased priority from {old_priority} to {new_priority}")
    
    def is_empty(self):
        """Check if the priority queue is empty"""
        return self.heap.is_empty()
    
    def size(self):
        """Return the number of items in the priority queue"""
        return self.heap.size()
    
    def __str__(self):
        """String representation of the priority queue"""
        return str([node.key for node in self.node_map.values()])

# Create and use a priority queue
print("\nPriority Queue Example:")
pq = PriorityQueue()

pq.enqueue("Task A", 5)
pq.enqueue("Task B", 3)
pq.enqueue("Task C", 7)
pq.enqueue("Task D", 1)
pq.enqueue("Task E", 4)

print("Priority Queue:", pq)
print("Dequeue:", pq.dequeue())  # Output: Task C
print("Dequeue:", pq.dequeue())  # Output: Task D
print("Dequeue:", pq.dequeue())  # Output: Task A
print("Dequeue:", pq.dequeue())  # Output: Task E

# Decrease priority of Task B from 3 to 2
pq.decrease_priority(3, 2)
print("After decreasing priority of Task B:", pq)
print("Dequeue:", pq.dequeue())  # Output: Task B
print("Dequeue:", pq.dequeue())  # Output: Task E
```

Binomial Heap

When: Need efficient heap merging
Why: O(log n) merge, all operations O(log n) or better
Examples: Priority queue merging, parallel algorithms

Pairing Heap

When: Need simpler implementation than Fibonacci with similar performance
Why: Simpler than Fibonacci heap, good practical performance, O(1) amortized decrease-key
Examples: Practical implementations where Fibonacci is too complex

Leftist Heap

When: Need efficient merging and simplicity
Why: O(log n) merge, simple implementation, mergeable heap property
Examples: Discrete event simulation, mergeable priority queues

Skew Heap

When: Want self-adjusting heap without maintaining structural information
Why: No balance information needed, simpler than leftist heap
Examples: When simplicity is prioritized, mergeable heaps

d-ary Heap

When: Want to tune heap branching factor for specific workload
Why: More children per node, can reduce height, trade-off between operations
Examples: Cache optimization, when insert/decrease-key ratio is high

Other Trees
Trie (Prefix Tree)

When: Need prefix-based string searching or autocomplete
Why: O(m) search where m=string length, shared prefixes save space, fast prefix queries
Examples: Autocomplete, spell checkers, IP routing, dictionary implementation, T9 texting

Suffix Tree

When: Need to solve complex string problems (substring search, longest common substring)
Why: O(m) substring search, preprocesses all suffixes, powerful pattern matching
Examples: Genome analysis, text searching, data compression, plagiarism detection

Suffix Array

When: Need suffix tree functionality with less memory
Why: Space-efficient alternative to suffix tree, similar capabilities
Examples: String matching in limited memory, bioinformatics with large genomes

Radix Tree (Compact Trie)

When: Need trie with less memory overhead
Why: Compresses chains of nodes, more space-efficient than standard trie
Examples: IP routing tables, memory-efficient dictionaries, routing in networks

Merkle Tree

When: Need to verify data integrity efficiently
Why: Allows verification of specific data without checking everything, tamper-evident
Examples: Blockchain, distributed systems (Git, BitTorrent), certificate transparency

Segment Tree

When: Need range queries and point updates on arrays
Why: O(log n) range query and update, handles sum/min/max/GCD over ranges
Examples: Range minimum/maximum queries, computational geometry, range sum updates

Fenwick Tree (Binary Indexed Tree)

When: Need prefix sums or cumulative frequency with updates
Why: Simpler than segment tree for specific cases, O(log n) query and update, less memory
Examples: Cumulative frequency tables, inversion counting, dynamic ranking

Interval Tree

When: Need to query which intervals overlap with a given point or interval
Why: O(log n + k) to find k overlapping intervals
Examples: Scheduling conflicts, genomic intervals, time range queries, calendar applications

Range Tree

When: Need multi-dimensional range queries
Why: Handles orthogonal range queries in multiple dimensions
Examples: Geographic information systems, database queries, multi-dimensional data

K-d Tree (k-dimensional)

When: Need spatial searching in k dimensions
Why: O(log n) average search, efficient for nearest neighbor
Examples: Nearest neighbor search, ray tracing, geographic databases, machine learning

Quadtree

When: Need 2D spatial partitioning
Why: Efficiently subdivides 2D space, adaptive resolution
Examples: Image compression, collision detection in games, geographic data, spatial indexing

Octree

When: Need 3D spatial partitioning
Why: Efficiently subdivides 3D space
Examples: 3D graphics, collision detection, voxel-based games (Minecraft), 3D modeling

R-Tree

When: Need spatial indexing of rectangles/bounding boxes
Why: Efficient for spatial queries in databases, groups nearby objects
Examples: Geographic databases, CAD systems, spatial databases (PostGIS), game engines

Van Emde Boas Tree

When: Integer universe is limited and need O(log log u) operations
Why: Faster than balanced BST for small integer universes
Examples: Priority queues with small integers, router tables, specialized integer operations

Graph Data Structures
Adjacency Matrix

When: Graph is dense (many edges) or need O(1) edge lookup
Why: O(1) edge query, simple implementation, works for dense graphs
Examples: Dense graphs, when checking edge existence frequently, small graphs

Adjacency List

When: Graph is sparse (few edges)
Why: O(V + E) space instead of O(V²), efficient iteration over neighbors
Examples: Most real-world graphs (social networks, web graphs), sparse graphs

Incidence Matrix

When: Need to represent edge information explicitly
Why: Rows for vertices, columns for edges, useful for certain algorithms
Examples: Network flow problems, theoretical analysis

Edge List

When: Simple representation needed or for sorting edges
Why: Simplest representation, easy to sort edges
Examples: Kruskal's MST algorithm, when processing edges sequentially

Directed Graph (Digraph)

When: Relationships have direction
Why: Models one-way relationships
Examples: Web links, task dependencies, state machines, social media follows

Undirected Graph

When: Relationships are bidirectional
Why: Models mutual relationships
Examples: Friendships, road networks, collaboration networks

Weighted Graph

When: Edges have costs/distances
Why: Models real-world scenarios with varying costs
Examples: Road networks with distances, network latency, cost optimization

Directed Acyclic Graph (DAG)

When: Representing dependencies without cycles
Why: Enables topological sorting, efficient for precedence constraints
Examples: Task scheduling, build systems (Make), version control, spreadsheet calculations

Multigraph

When: Multiple edges between same vertices allowed
Why: Models scenarios with multiple connections
Examples: Transportation networks (multiple routes), parallel edges in circuits

Hypergraph

When: Edges connect more than two vertices
Why: Models complex relationships beyond pairwise
Examples: Database schema relationships, chemical reactions, collaboration networks

Hash-Based Structures
Hash Table / Hash Map

When: Need fast average-case lookups, no ordering required
Why: O(1) average insert/search/delete, most versatile data structure
Examples: Caches, symbol tables, counting frequencies, database indexes, dictionaries

Hash Set

When: Need to track membership, no duplicates, no ordering
Why: O(1) average contains check, ensures uniqueness
Examples: Removing duplicates, checking membership, visited nodes in graph traversal

Cuckoo Hash Table

When: Need guaranteed O(1) worst-case lookup
Why: Two hash functions, relocates on collision, worst-case O(1) lookup
Examples: Network packet processing, real-time systems, hardware implementations

Robin Hood Hash Table

When: Want better cache performance than chaining
Why: Open addressing with "rich help poor" collision resolution, low variance
Examples: High-performance hash tables, Rust's HashMap

Hopscotch Hashing

When: Need good cache performance and simple resizing
Why: Combines linear probing benefits with bounded search, concurrent-friendly
Examples: Concurrent hash tables, high-performance systems

Bloom Filter

When: Can tolerate false positives but not false negatives, space-critical
Why: Probabilistic, very space-efficient, O(k) where k is small
Examples: Web crawlers (URL visited check), databases (avoiding disk reads), spell checkers, malware detection

Count-Min Sketch

When: Need approximate frequency counts with limited memory
Why: Sublinear space, probabilistic counts, handles streams
Examples: Network traffic analysis, finding heavy hitters, database query optimization

Perfect Hash Table

When: Static dataset, no collisions wanted
Why: Guaranteed O(1) lookup with no collisions
Examples: Compiler keyword tables, static configuration lookups, embedded systems

Advanced Data Structures
Union-Find
Disjoint Set (Union-Find)

When: Need to track connected components or equivalence classes
Why: Nearly O(1) find and union with path compression and union by rank
Examples: Kruskal's MST, detecting cycles, image segmentation, network connectivity

Weighted Union-Find

When: Need to track relative relationships between elements
Why: Can answer queries about distances/ratios between elements
Examples: Equations with variables, constraint satisfaction

String Structures
Suffix Automaton

When: Need all substring queries and more flexible than suffix tree
Why: Recognizes all substrings as paths, more space-efficient
Examples: Pattern matching, string analysis, text indexing

Aho-Corasick Automaton

When: Need to search for multiple patterns simultaneously
Why: O(n + m + z) where n=text, m=patterns, z=matches; finds all patterns in one pass
Examples: Antivirus scanning, intrusion detection, content filtering, DNA sequence matching

Rope (for strings)

When: Need efficient string operations (concat, split) on large strings
Why: O(log n) concatenation and split, better than array for large texts
Examples: Text editors, large document manipulation, undo/redo in editors

Probabilistic
HyperLogLog

When: Need to count distinct elements with minimal memory
Why: Estimates cardinality with very small memory (few KB for billions)
Examples: Counting unique visitors, database query optimization, big data analytics

Concurrent
Lock-free Queue

When: Multiple threads need lock-free communication
Why: Non-blocking, better scalability than locked queues
Examples: High-performance multi-threaded applications, actor systems

Lock-free Stack

When: Multiple threads need non-blocking stack operations
Why: CAS operations, better performance under contention
Examples: Memory allocators, work-stealing schedulers

Concurrent Hash Map

When: Multiple threads need shared hash table
Why: Fine-grained locking or lock-free segments, better concurrency
Examples: Multi-threaded caches, shared state in servers

Persistent
Persistent Array/Tree/Stack

When: Need to maintain history of all versions
Why: Each operation creates new version without modifying old, structural sharing
Examples: Undo systems, version control, functional programming, debugging time-travel

Specialized Structures
Matrix (2D Array)

When: Need rectangular grid of data
Why: Natural representation, O(1) access, cache-friendly
Examples: Images, game boards, mathematical computations, dynamic programming tables

Sparse Matrix

When: Matrix is mostly zeros
Why: Only stores non-zero elements, saves massive space
Examples: Graph adjacency for sparse graphs, scientific computing, recommendation systems

Circular Buffer

When: Fixed-size buffer with wraparound, FIFO behavior
Why: O(1) operations, fixed memory, efficient for streaming
Examples: Audio/video buffering, logging systems, producer-consumer with bounded buffer

Bit Array (Bitset)

When: Need to store boolean values compactly
Why: 1 bit per element, 8x-64x space savings, fast bitwise operations
Examples: Flags, Sieve of Eratosthenes, set operations, compression

Gap Buffer

When: Editing text with cursor (insertions/deletions at one location)
Why: O(1) operations near cursor, simple implementation
Examples: Text editors (Emacs), simple text editing implementations

Piece Table

When: Text editing with efficient undo, especially large files
Why: Original text unchanged, tracks additions/deletions separately
Examples: Text editors (Word, VS Code), large file editing

Zipper

When: Need functional tree navigation with updates
Why: Efficient navigation and modification in functional style
Examples: Functional programming, XML/JSON editing, tree traversal with modifications

Finger Tree

When: Need persistent sequence with efficient access at both ends
Why: Amortized O(1) at ends, O(log n) in middle, measurable elements
Examples: Functional programming, persistent deques, text editors

Dancing Links (DLX)

When: Solving exact cover problems (backtracking with easy undo)
Why: O(1) remove and restore of nodes
Examples: Sudoku solvers, n-queens problem, pentomino tiling

XOR Linked List

When: Need doubly-linked list with minimal memory
Why: Stores XOR of prev and next pointers, saves one pointer per node
Examples: Memory-constrained environments (rarely used in practice due to complexity)

Cache-oblivious Data Structures

When: Want optimal cache performance without tuning
Why: Optimal across all cache levels automatically
Examples: External memory algorithms, databases with unknown cache sizes

Succinct Data Structures

When: Need minimal space while supporting queries
Why: Near information-theoretic minimum space, often sublinear
Examples: Compressed text indexes, large-scale graph storage, RDF stores

Tree Structures
Cartesian Tree

When: Need to combine heap and BST properties, or solve range minimum queries
Why: Maintains both in-order property and heap property, can be built in O(n)
Examples: Range minimum query preprocessing, lowest common ancestor problems, computational geometry

Fusion Tree

When: Working with w-bit integers and need o(log n) operations
Why: O(log n / log w) operations, faster than comparison-based structures
Examples: Theoretical importance, integer sorting, successor queries

Exponential Tree

When: Need stratified tree structure for theoretical bounds
Why: Combines multiple levels of trees
Examples: Advanced algorithm design, theoretical computer science

AA Tree

When: Want simpler balanced tree than Red-Black
Why: Simpler implementation than Red-Black, easier to understand, still O(log n)
Examples: Educational purposes, when simplicity matters over slight performance gain

Link/Cut Tree

When: Need to maintain forest with dynamic tree operations (link/cut/path queries)
Why: O(log n) amortized for link, cut, and path aggregate queries
Examples: Dynamic connectivity, network flow algorithms, dynamic graph problems

Euler Tour Tree

When: Need to maintain dynamic forests and answer subtree queries
Why: Efficient subtree operations, dynamic tree connectivity
Examples: Dynamic trees, reachability queries, maintaining connected components

Top Tree

When: Need hierarchical decomposition of trees
Why: Supports dynamic tree operations with path queries
Examples: Advanced dynamic tree problems, path aggregation

SPQR Tree

When: Representing triconnected components of graphs
Why: Decomposes graph into series-parallel structure
Examples: Planar graph algorithms, graph drawing, network reliability

Heap Variants
d-heap Variants (Ternary, Quaternary, etc.)

When: Want to optimize for specific insert/delete-min ratios
Why: Different branching factors optimize different operation mixes
Examples: Tuning for cache performance, specific workload optimization

Soft Heap

When: Can tolerate approximate heap property with corruption
Why: O(1) amortized insert, faster operations with controlled errors
Examples: Approximate sorting, minimum spanning tree algorithms

```python
# Python implementation of a Soft Heap
# A soft heap allows some corruption for faster operations
import math

class SoftHeapNode:
    def __init__(self, key):
        self.key = key
        self.rank = 0
        self.left = None
        self.right = None
        self.next = None  # For linked list of nodes
        self.keys = [key]  # List of corrupted keys
        
class SoftHeap:
    def __init__(self, error_rate=0.5):
        """
        Initialize a soft heap with given error rate.
        error_rate: fraction of elements that can be corrupted (0 < ε <= 0.5)
        """
        self.root = None
        self.error_rate = error_rate
        self.r = max(1, int(math.ceil(math.log2(1.0 / error_rate))))
        self.size = 0
        
    def _meld(self, h1, h2):
        """Meld two soft heap roots"""
        if h1 is None:
            return h2
        if h2 is None:
            return h1
            
        # Ensure h1 has smaller or equal rank
        if h1.rank > h2.rank:
            h1, h2 = h2, h1
            
        if h1.rank < h2.rank:
            h2.next = h1.next
            h1.next = h2
            return h1
        
        # Same rank - create new root
        new_node = SoftHeapNode(min(h1.key, h2.key))
        new_node.rank = h1.rank + 1
        new_node.left = h1
        new_node.right = h2
        new_node.keys = h1.keys + h2.keys
        
        # Perform sifting if rank threshold exceeded
        if new_node.rank > self.r:
            new_node.keys = new_node.keys[:len(new_node.keys)//2]
            
        return new_node
    
    def insert(self, key):
        """Insert a key into the soft heap - O(1) amortized"""
        new_node = SoftHeapNode(key)
        self.root = self._meld(self.root, new_node)
        self.size += 1
        
    def find_min(self):
        """Find minimum key - O(1)"""
        if self.root is None:
            raise IndexError("Heap is empty")
        return self.root.key
    
    def extract_min(self):
        """Extract minimum with possible corruption - O(log n) amortized"""
        if self.root is None:
            raise IndexError("Heap is empty")
            
        min_key = self.root.key
        
        # Remove the minimum
        if len(self.root.keys) > 1:
            self.root.keys.pop(0)
            self.root.key = self.root.keys[0]
        else:
            # Meld children
            left = self.root.left
            right = self.root.right
            self.root = self._meld(left, right)
            
        self.size -= 1
        return min_key
    
    def meld_with(self, other_heap):
        """Meld this heap with another soft heap - O(1)"""
        self.root = self._meld(self.root, other_heap.root)
        self.size += other_heap.size
        other_heap.root = None
        other_heap.size = 0
        
    def is_empty(self):
        """Check if heap is empty"""
        return self.root is None

# Example usage
print("\nSoft Heap:")
soft_heap = SoftHeap(error_rate=0.3)

# Insert elements
for value in [10, 5, 15, 3, 8, 12, 20]:
    soft_heap.insert(value)
    print(f"Inserted {value}")

print(f"Minimum element: {soft_heap.find_min()}")

# Extract elements (note: some may be corrupted)
print("\nExtracting elements:")
while not soft_heap.is_empty():
    min_val = soft_heap.extract_min()
    print(f"Extracted: {min_val}")

# Example with melding
heap1 = SoftHeap(error_rate=0.3)
heap2 = SoftHeap(error_rate=0.3)

for val in [5, 10, 15]:
    heap1.insert(val)
for val in [3, 8, 12]:
    heap2.insert(val)

print("\nMelding two heaps:")
heap1.meld_with(heap2)
print(f"Size after meld: {heap1.size}")
```

Brodal Queue

When: Need worst-case optimal priority queue
Why: O(1) worst-case insert/find-min/merge, O(log n) delete-min
Examples: Theoretical importance, though complex in practice

```python
# Python implementation of a simplified Brodal Queue
# Note: Full Brodal Queue is extremely complex; this is a simplified version
# demonstrating the key concepts

class BrodalNode:
    def __init__(self, key, value=None):
        self.key = key
        self.value = value if value is not None else key
        self.children = []
        self.rank = 0
        self.parent = None
        
class BrodalQueue:
    """
    Simplified Brodal Queue implementation
    Achieves O(1) worst-case for insert, find-min, and meld
    O(log n) worst-case for delete-min
    """
    def __init__(self):
        self.min_node = None
        self.roots = []  # List of root nodes
        self.size = 0
        self.rank_map = {}  # Maps rank to list of nodes
        
    def insert(self, key, value=None):
        """Insert with O(1) worst-case time"""
        new_node = BrodalNode(key, value)
        
        if self.min_node is None or key < self.min_node.key:
            self.min_node = new_node
            
        self.roots.append(new_node)
        self._update_rank_map(new_node)
        self.size += 1
        
    def find_min(self):
        """Find minimum in O(1) worst-case time"""
        if self.min_node is None:
            raise IndexError("Queue is empty")
        return self.min_node.key, self.min_node.value
    
    def delete_min(self):
        """Delete minimum in O(log n) worst-case time"""
        if self.min_node is None:
            raise IndexError("Queue is empty")
            
        min_key = self.min_node.key
        min_value = self.min_node.value
        
        # Remove min node from roots
        if self.min_node in self.roots:
            self.roots.remove(self.min_node)
            
        # Add children to roots
        for child in self.min_node.children:
            child.parent = None
            self.roots.append(child)
            
        # Consolidate trees and find new minimum
        self._consolidate()
        self._find_new_min()
        
        self.size -= 1
        return min_key, min_value
    
    def meld(self, other_queue):
        """Meld with another queue in O(1) worst-case time"""
        if other_queue.min_node is None:
            return
            
        # Merge roots
        self.roots.extend(other_queue.roots)
        
        # Update minimum
        if self.min_node is None or other_queue.min_node.key < self.min_node.key:
            self.min_node = other_queue.min_node
            
        # Update size and rank map
        self.size += other_queue.size
        for rank, nodes in other_queue.rank_map.items():
            if rank in self.rank_map:
                self.rank_map[rank].extend(nodes)
            else:
                self.rank_map[rank] = nodes.copy()
                
        # Clear other queue
        other_queue.roots = []
        other_queue.min_node = None
        other_queue.size = 0
        other_queue.rank_map = {}
    
    def _update_rank_map(self, node):
        """Update the rank map with node"""
        if node.rank not in self.rank_map:
            self.rank_map[node.rank] = []
        self.rank_map[node.rank].append(node)
    
    def _consolidate(self):
        """Consolidate trees to maintain structure"""
        rank_bins = {}
        
        for root in self.roots:
            current = root
            while current.rank in rank_bins:
                other = rank_bins.pop(current.rank)
                
                # Link trees of same rank
                if current.key < other.key:
                    parent, child = current, other
                else:
                    parent, child = other, current
                    
                child.parent = parent
                parent.children.append(child)
                parent.rank += 1
                current = parent
                
            rank_bins[current.rank] = current
            
        self.roots = list(rank_bins.values())
        self.rank_map = {}
        for root in self.roots:
            self._update_rank_map(root)
    
    def _find_new_min(self):
        """Find new minimum among roots"""
        self.min_node = None
        for root in self.roots:
            if self.min_node is None or root.key < self.min_node.key:
                self.min_node = root
    
    def is_empty(self):
        """Check if queue is empty"""
        return self.size == 0
    
    def __len__(self):
        return self.size

# Example usage
print("\nBrodal Queue:")
bq = BrodalQueue()

# Insert elements - O(1) worst-case
elements = [15, 10, 20, 8, 25, 5, 30]
for elem in elements:
    bq.insert(elem)
    print(f"Inserted {elem}, current min: {bq.find_min()[0]}")

# Delete minimum - O(log n) worst-case
print("\nExtracting minimums:")
while not bq.is_empty():
    min_key, min_val = bq.delete_min()
    print(f"Deleted min: {min_key}")

# Example with melding - O(1) worst-case
print("\nMelding two queues:")
bq1 = BrodalQueue()
bq2 = BrodalQueue()

for val in [5, 15, 25]:
    bq1.insert(val)
for val in [10, 20, 30]:
    bq2.insert(val)

print(f"Queue 1 min: {bq1.find_min()[0]}")
print(f"Queue 2 min: {bq2.find_min()[0]}")

bq1.meld(bq2)
print(f"After meld, min: {bq1.find_min()[0]}, size: {len(bq1)}")
```

Strict Fibonacci Heap

When: Need worst-case bounds instead of amortized
Why: Converts Fibonacci heap's amortized to worst-case bounds
Examples: Real-time systems where amortized isn't acceptable

```python
# Python implementation of a Strict Fibonacci Heap
# Provides worst-case O(1) for most operations instead of amortized

class StrictFibNode:
    def __init__(self, key, value=None):
        self.key = key
        self.value = value if value is not None else key
        self.degree = 0
        self.parent = None
        self.child = None
        self.left = self
        self.right = self
        self.mark = False
        self.loss = 0  # Track losses for strict bounds
        
class StrictFibonacciHeap:
    """
    Strict Fibonacci Heap with worst-case bounds:
    - Insert: O(1) worst-case
    - Find-min: O(1) worst-case
    - Delete-min: O(log n) worst-case
    - Decrease-key: O(1) worst-case
    - Delete: O(log n) worst-case
    """
    def __init__(self):
        self.min_node = None
        self.num_nodes = 0
        self.max_degree = 0
        self.queue = []  # Work queue for maintaining strict bounds
        
    def insert(self, key, value=None):
        """Insert with O(1) worst-case time"""
        node = StrictFibNode(key, value)
        
        if self.min_node is None:
            self.min_node = node
        else:
            self._add_to_root_list(node)
            if node.key < self.min_node.key:
                self.min_node = node
                
        self.num_nodes += 1
        self._process_queue()  # Maintain strict bounds
        return node
    
    def find_min(self):
        """Find minimum in O(1) worst-case time"""
        if self.min_node is None:
            raise IndexError("Heap is empty")
        return self.min_node.key, self.min_node.value
    
    def delete_min(self):
        """Delete minimum in O(log n) worst-case time"""
        if self.min_node is None:
            raise IndexError("Heap is empty")
            
        min_node = self.min_node
        min_key = min_node.key
        min_value = min_node.value
        
        # Add children to root list
        if min_node.child:
            child = min_node.child
            while True:
                next_child = child.right
                child.parent = None
                child.mark = False
                child.loss = 0
                self._add_to_root_list(child)
                
                if child == min_node.child or next_child == min_node.child:
                    break
                child = next_child
        
        # Remove min from root list
        self._remove_from_root_list(min_node)
        
        if min_node == min_node.right:
            self.min_node = None
        else:
            self.min_node = min_node.right
            self._consolidate_strict()
            
        self.num_nodes -= 1
        return min_key, min_value
    
    def decrease_key(self, node, new_key):
        """Decrease key with O(1) worst-case time"""
        if new_key > node.key:
            raise ValueError("New key is greater than current key")
            
        node.key = new_key
        parent = node.parent
        
        if parent and node.key < parent.key:
            self._cut_strict(node, parent)
            
        if node.key < self.min_node.key:
            self.min_node = node
            
        self._process_queue()  # Maintain strict bounds
    
    def delete(self, node):
        """Delete node in O(log n) worst-case time"""
        self.decrease_key(node, float('-inf'))
        self.delete_min()
    
    def _add_to_root_list(self, node):
        """Add node to root list"""
        if self.min_node is None:
            self.min_node = node
            node.left = node
            node.right = node
        else:
            node.left = self.min_node
            node.right = self.min_node.right
            self.min_node.right.left = node
            self.min_node.right = node
    
    def _remove_from_root_list(self, node):
        """Remove node from root list"""
        if node.right == node:
            return
        node.left.right = node.right
        node.right.left = node.left
    
    def _cut_strict(self, node, parent):
        """Cut node from parent with strict bounds maintenance"""
        # Remove from parent's child list
        if parent.child == node:
            if node.right == node:
                parent.child = None
            else:
                parent.child = node.right
                
        node.left.right = node.right
        node.right.left = node.left
        parent.degree -= 1
        
        # Add to root list
        node.parent = None
        node.mark = False
        node.loss = 0
        self._add_to_root_list(node)
        
        # Increment parent's loss counter
        parent.loss += 1
        
        # Schedule parent for processing if needed
        if parent.loss >= 2 and parent.parent:
            self.queue.append(parent)
    
    def _process_queue(self):
        """Process work queue to maintain strict O(1) bounds"""
        work_limit = min(3, len(self.queue))  # Process at most 3 items
        
        for _ in range(work_limit):
            if not self.queue:
                break
                
            node = self.queue.pop(0)
            if node.parent and node.loss >= 2:
                parent = node.parent
                self._cut_strict(node, parent)
    
    def _consolidate_strict(self):
        """Consolidate with strict bounds"""
        max_degree = int(self.num_nodes ** 0.5) + 1
        degree_table = [None] * max_degree
        
        # Collect root nodes
        roots = []
        if self.min_node:
            current = self.min_node
            while True:
                roots.append(current)
                current = current.right
                if current == self.min_node:
                    break
        
        # Consolidate
        for root in roots:
            degree = root.degree
            while degree < len(degree_table) and degree_table[degree]:
                other = degree_table[degree]
                if root.key > other.key:
                    root, other = other, root
                    
                self._link_strict(other, root)
                degree_table[degree] = None
                degree += 1
                
            if degree < len(degree_table):
                degree_table[degree] = root
        
        # Find new minimum
        self.min_node = None
        for node in degree_table:
            if node:
                if self.min_node is None:
                    self.min_node = node
                    node.left = node
                    node.right = node
                else:
                    self._add_to_root_list(node)
                    if node.key < self.min_node.key:
                        self.min_node = node
    
    def _link_strict(self, child, parent):
        """Link child to parent"""
        self._remove_from_root_list(child)
        
        if parent.child is None:
            parent.child = child
            child.left = child
            child.right = child
        else:
            child.left = parent.child
            child.right = parent.child.right
            parent.child.right.left = child
            parent.child.right = child
            
        child.parent = parent
        parent.degree += 1
        child.mark = False
        child.loss = 0
    
    def is_empty(self):
        """Check if heap is empty"""
        return self.min_node is None
    
    def __len__(self):
        return self.num_nodes

# Example usage
print("\nStrict Fibonacci Heap:")
sfh = StrictFibonacciHeap()

# Insert elements - O(1) worst-case
elements = [20, 15, 30, 10, 25, 5, 35]
nodes = {}
for elem in elements:
    node = sfh.insert(elem)
    nodes[elem] = node
    print(f"Inserted {elem}, current min: {sfh.find_min()[0]}")

# Decrease key - O(1) worst-case
print("\nDecreasing key 30 to 3:")
sfh.decrease_key(nodes[30], 3)
print(f"New minimum: {sfh.find_min()[0]}")

# Delete minimum - O(log n) worst-case
print("\nExtracting minimums:")
for _ in range(3):
    if not sfh.is_empty():
        min_key, min_val = sfh.delete_min()
        print(f"Deleted min: {min_key}")

print(f"\nRemaining elements: {len(sfh)}")
print(f"Current minimum: {sfh.find_min()[0]}")
```

String Structures
FM-Index

When: Need compressed full-text index with pattern matching
Why: Compressed representation, supports backward search efficiently
Examples: Bioinformatics (genome indexing), compressed text search, search engines

```python
# Python implementation of FM-Index (Full-text index in Minute space)
# Based on Burrows-Wheeler Transform

class FMIndex:
    """
    FM-Index for efficient pattern matching in compressed space.
    Uses Burrows-Wheeler Transform and auxiliary data structures.
    """
    def __init__(self, text):
        """Build FM-Index from text"""
        if not text.endswith('$'):
            text += '$'  # Add end marker
        self.text = text
        self.n = len(text)
        
        # Build suffix array
        self.suffix_array = self._build_suffix_array(text)
        
        # Build BWT from suffix array
        self.bwt = self._build_bwt(text, self.suffix_array)
        
        # Build auxiliary structures
        self.C = self._build_C_array(self.bwt)  # Count of chars less than c
        self.Occ = self._build_occurrence_table(self.bwt)  # Occurrence table
        
    def _build_suffix_array(self, text):
        """Build suffix array using simple sorting"""
        suffixes = [(text[i:], i) for i in range(len(text))]
        suffixes.sort()
        return [suffix[1] for suffix in suffixes]
    
    def _build_bwt(self, text, suffix_array):
        """Build Burrows-Wheeler Transform"""
        bwt = []
        for i in suffix_array:
            if i == 0:
                bwt.append(text[-1])
            else:
                bwt.append(text[i-1])
        return ''.join(bwt)
    
    def _build_C_array(self, bwt):
        """Build C array: count of characters less than c in BWT"""
        chars = sorted(set(bwt))
        C = {}
        count = 0
        for char in chars:
            C[char] = count
            count += bwt.count(char)
        return C
    
    def _build_occurrence_table(self, bwt):
        """Build occurrence table for each character at each position"""
        chars = set(bwt)
        Occ = {char: [0] * (len(bwt) + 1) for char in chars}
        
        for char in chars:
            for i in range(len(bwt)):
                Occ[char][i+1] = Occ[char][i] + (1 if bwt[i] == char else 0)
        
        return Occ
    
    def count(self, pattern):
        """Count occurrences of pattern using backward search"""
        if not pattern:
            return 0
            
        # Initialize range to entire BWT
        top = 0
        bottom = self.n - 1
        
        # Process pattern from right to left
        for i in range(len(pattern) - 1, -1, -1):
            char = pattern[i]
            
            if char not in self.C:
                return 0
            
            # Update range using LF-mapping
            top = self.C[char] + self.Occ[char][top]
            bottom = self.C[char] + self.Occ[char][bottom + 1] - 1
            
            if top > bottom:
                return 0
        
        return bottom - top + 1
    
    def find(self, pattern):
        """Find all occurrences of pattern"""
        if not pattern:
            return []
            
        # Get range using backward search
        top = 0
        bottom = self.n - 1
        
        for i in range(len(pattern) - 1, -1, -1):
            char = pattern[i]
            
            if char not in self.C:
                return []
            
            top = self.C[char] + self.Occ[char][top]
            bottom = self.C[char] + self.Occ[char][bottom + 1] - 1
            
            if top > bottom:
                return []
        
        # Extract positions from suffix array
        positions = []
        for i in range(top, bottom + 1):
            positions.append(self.suffix_array[i])
        
        return sorted(positions)
    
    def LF_mapping(self, i):
        """Last-to-First column mapping"""
        char = self.bwt[i]
        return self.C[char] + self.Occ[char][i]
    
    def extract(self, start, length):
        """Extract substring from compressed representation"""
        result = []
        # Find position in BWT corresponding to start
        pos = 0
        for i, sa_pos in enumerate(self.suffix_array):
            if sa_pos == start:
                pos = i
                break
        
        # Extract characters using LF-mapping
        for _ in range(length):
            if pos >= len(self.bwt):
                break
            result.append(self.bwt[pos])
            pos = self.LF_mapping(pos)
        
        return ''.join(result)

# Example usage
print("\nFM-Index:")
text = "banana"
fm_index = FMIndex(text)

print(f"Original text: {text}")
print(f"BWT: {fm_index.bwt}")
print(f"Suffix array: {fm_index.suffix_array}")

# Search for patterns
patterns = ["ana", "ban", "nan", "xyz"]
print("\nPattern matching:")
for pattern in patterns:
    count = fm_index.count(pattern)
    positions = fm_index.find(pattern)
    print(f"Pattern '{pattern}': found {count} times at positions {positions}")

# Example with larger text
print("\nGenome sequence example:")
genome = "ACGTACGTTAGC"
genome_index = FMIndex(genome)

patterns = ["ACG", "TAG", "CGT"]
for pattern in patterns:
    count = genome_index.count(pattern)
    positions = genome_index.find(pattern)
    print(f"Pattern '{pattern}': found {count} times at positions {positions}")
```

Burrows-Wheeler Transform structures

When: Need compression-friendly text indexing
Why: Enables compression while maintaining searchability
Examples: Data compression (bzip2), genomic data, compressed indexes

```python
# Python implementation of Burrows-Wheeler Transform (BWT)
# Used for data compression and text indexing

class BWT:
    """
    Burrows-Wheeler Transform implementation
    Transforms text into a more compressible form while maintaining searchability
    """
    def __init__(self):
        self.end_marker = '$'
    
    def encode(self, text):
        """
        Encode text using BWT
        Returns: (bwt_string, original_index)
        """
        if not text:
            return "", 0
        
        # Add end marker if not present
        if not text.endswith(self.end_marker):
            text += self.end_marker
        
        # Create all rotations
        rotations = []
        for i in range(len(text)):
            rotation = text[i:] + text[:i]
            rotations.append((rotation, i))
        
        # Sort rotations lexicographically
        rotations.sort(key=lambda x: x[0])
        
        # Extract last column (BWT) and find original string index
        bwt = []
        original_index = 0
        for i, (rotation, orig_pos) in enumerate(rotations):
            bwt.append(rotation[-1])
            if orig_pos == 0:
                original_index = i
        
        return ''.join(bwt), original_index
    
    def decode(self, bwt, original_index):
        """
        Decode BWT back to original text
        """
        if not bwt:
            return ""
        
        n = len(bwt)
        
        # Create table with indices
        table = [(bwt[i], i) for i in range(n)]
        
        # Sort to get first column
        table.sort()
        
        # Build the original string by following the transformation
        result = []
        index = original_index
        
        for _ in range(n):
            char, next_index = table[index]
            result.append(char)
            # Find where this character came from in BWT
            index = next_index
            # Update index using LF mapping
            count = 0
            target_char = bwt[next_index]
            for i in range(next_index + 1):
                if bwt[i] == target_char:
                    count += 1
            
            # Find the count-th occurrence of target_char in sorted column
            occurrences = 0
            for i in range(n):
                if table[i][0] == target_char:
                    occurrences += 1
                    if occurrences == count:
                        index = i
                        break
        
        # Remove end marker
        result_str = ''.join(result)
        if result_str.endswith(self.end_marker):
            result_str = result_str[:-1]
        
        return result_str
    
    def encode_optimized(self, text):
        """
        Optimized encoding using suffix array
        """
        if not text:
            return "", 0
            
        if not text.endswith(self.end_marker):
            text += self.end_marker
        
        # Build suffix array
        suffixes = [(text[i:], i) for i in range(len(text))]
        suffixes.sort()
        suffix_array = [s[1] for s in suffixes]
        
        # Build BWT from suffix array
        bwt = []
        original_index = 0
        
        for i, pos in enumerate(suffix_array):
            if pos == 0:
                bwt.append(text[-1])
                original_index = i
            else:
                bwt.append(text[pos - 1])
        
        return ''.join(bwt), original_index

class BWTWithMoveToFront:
    """
    BWT combined with Move-to-Front encoding for better compression
    """
    def __init__(self):
        self.bwt = BWT()
    
    def encode(self, text):
        """Encode using BWT + Move-to-Front"""
        # Apply BWT
        bwt_text, original_index = self.bwt.encode(text)
        
        # Apply Move-to-Front encoding
        alphabet = sorted(set(bwt_text))
        mtf_encoded = []
        
        for char in bwt_text:
            index = alphabet.index(char)
            mtf_encoded.append(index)
            # Move character to front
            alphabet.insert(0, alphabet.pop(index))
        
        return mtf_encoded, original_index, sorted(set(bwt_text))
    
    def decode(self, mtf_encoded, original_index, alphabet):
        """Decode Move-to-Front + BWT"""
        # Decode Move-to-Front
        working_alphabet = alphabet.copy()
        bwt_text = []
        
        for index in mtf_encoded:
            char = working_alphabet[index]
            bwt_text.append(char)
            # Move character to front
            working_alphabet.insert(0, working_alphabet.pop(index))
        
        # Decode BWT
        return self.bwt.decode(''.join(bwt_text), original_index)

# Example usage
print("\nBurrows-Wheeler Transform:")
bwt = BWT()

# Simple example
text = "banana"
print(f"Original text: {text}")

encoded, orig_idx = bwt.encode(text)
print(f"BWT encoded: {encoded}")
print(f"Original index: {orig_idx}")

decoded = bwt.decode(encoded, orig_idx)
print(f"Decoded text: {decoded}")
print(f"Correct: {decoded == text}")

# Example showing compression-friendly property
text2 = "AAABBBCCCAAA"
print(f"\nOriginal: {text2}")
encoded2, orig_idx2 = bwt.encode(text2)
print(f"BWT: {encoded2}")
print(f"Note: Characters are grouped (AAABBBCCAAAC$ -> CCCC$AAABBBAA)")

# BWT with Move-to-Front
print("\nBWT with Move-to-Front encoding:")
bwt_mtf = BWTWithMoveToFront()

mtf_encoded, orig_idx, alphabet = bwt_mtf.encode("banana")
print(f"MTF encoded: {mtf_encoded}")
print(f"Alphabet: {alphabet}")

decoded_text = bwt_mtf.decode(mtf_encoded, orig_idx, alphabet)
print(f"Decoded: {decoded_text}")
print(f"Correct: {decoded_text == 'banana'}")
```

Generalized Suffix Tree

When: Need suffix tree for multiple strings
Why: Finds common patterns across multiple texts
Examples: Longest common substring of multiple strings, plagiarism detection

```python
# Python implementation of Generalized Suffix Tree
# Supports multiple strings for finding common patterns

class GSTNode:
    """Node in Generalized Suffix Tree"""
    def __init__(self):
        self.children = {}
        self.suffix_link = None
        self.start = -1
        self.end = None
        self.suffix_indices = []  # (string_id, position) pairs
        
class GeneralizedSuffixTree:
    """
    Generalized Suffix Tree for multiple strings
    Finds common substrings and patterns across texts
    """
    def __init__(self):
        self.root = GSTNode()
        self.root.suffix_link = self.root
        self.texts = []
        self.separators = []
        self.active_node = self.root
        self.active_edge = -1
        self.active_length = 0
        self.remaining = 0
        self.current_end = -1
        
    def add_string(self, text, string_id=None):
        """Add a string to the generalized suffix tree"""
        if string_id is None:
            string_id = len(self.texts)
        
        # Use unique separator for each string
        separator = chr(ord('$') + string_id)
        self.separators.append(separator)
        
        # Store text with separator
        full_text = text + separator
        self.texts.append(full_text)
        
        # Build suffix tree for this text
        self._build_for_text(full_text, string_id)
        
    def _build_for_text(self, text, string_id):
        """Build suffix tree using Ukkonen's algorithm"""
        n = len(text)
        
        for i in range(n):
            self._extend(text, i, string_id)
    
    def _extend(self, text, pos, string_id):
        """Extend the tree with character at position pos"""
        self.current_end = pos
        self.remaining += 1
        last_new_node = None
        
        while self.remaining > 0:
            if self.active_length == 0:
                self.active_edge = pos
            
            if text[self.active_edge] not in self.active_node.children:
                # Create new leaf
                leaf = GSTNode()
                leaf.start = pos
                leaf.end = [len(text) - 1]  # Will grow
                leaf.suffix_indices.append((string_id, pos))
                
                self.active_node.children[text[self.active_edge]] = leaf
                
                if last_new_node:
                    last_new_node.suffix_link = self.active_node
                    last_new_node = None
            else:
                next_node = self.active_node.children[text[self.active_edge]]
                edge_length = self._edge_length(next_node)
                
                if self.active_length >= edge_length:
                    self.active_edge += edge_length
                    self.active_length -= edge_length
                    self.active_node = next_node
                    continue
                
                if text[next_node.start + self.active_length] == text[pos]:
                    if last_new_node and self.active_node != self.root:
                        last_new_node.suffix_link = self.active_node
                    self.active_length += 1
                    break
                
                # Split edge
                split = GSTNode()
                split.start = next_node.start
                split.end = [next_node.start + self.active_length - 1]
                
                self.active_node.children[text[self.active_edge]] = split
                
                # New leaf
                leaf = GSTNode()
                leaf.start = pos
                leaf.end = [len(text) - 1]
                leaf.suffix_indices.append((string_id, pos))
                split.children[text[pos]] = leaf
                
                next_node.start += self.active_length
                split.children[text[next_node.start]] = next_node
                
                if last_new_node:
                    last_new_node.suffix_link = split
                last_new_node = split
            
            self.remaining -= 1
            
            if self.active_node == self.root and self.active_length > 0:
                self.active_length -= 1
                self.active_edge = pos - self.remaining + 1
            elif self.active_node != self.root:
                self.active_node = self.active_node.suffix_link
    
    def _edge_length(self, node):
        """Calculate edge length for a node"""
        if isinstance(node.end, list):
            return self.current_end - node.start + 1
        return node.end - node.start + 1
    
    def find_longest_common_substring(self):
        """Find longest common substring across all strings"""
        if len(self.texts) < 2:
            return ""
        
        max_length = 0
        max_substring = ""
        
        def dfs(node, depth, path):
            nonlocal max_length, max_substring
            
            # Check if this node has suffixes from all strings
            string_ids = set()
            self._collect_string_ids(node, string_ids)
            
            if len(string_ids) == len(self.texts):
                if depth > max_length:
                    max_length = depth
                    max_substring = path
            
            # Recurse on children
            for char, child in node.children.items():
                edge_len = self._edge_length(child)
                edge_text = self._get_edge_text(child)
                dfs(child, depth + edge_len, path + edge_text)
        
        dfs(self.root, 0, "")
        return max_substring
    
    def _collect_string_ids(self, node, string_ids):
        """Collect all string IDs present in subtree"""
        if node.suffix_indices:
            for string_id, _ in node.suffix_indices:
                string_ids.add(string_id)
        
        for child in node.children.values():
            self._collect_string_ids(child, string_ids)
    
    def _get_edge_text(self, node):
        """Get text on edge leading to node"""
        if not self.texts:
            return ""
        
        # Find text containing this edge
        for text in self.texts:
            if node.start < len(text):
                end = node.end[0] if isinstance(node.end, list) else node.end
                end = min(end + 1, len(text))
                return text[node.start:end]
        return ""
    
    def search(self, pattern):
        """Search for pattern in all texts"""
        node = self.root
        i = 0
        
        while i < len(pattern):
            if pattern[i] not in node.children:
                return []
            
            child = node.children[pattern[i]]
            edge_text = self._get_edge_text(child)
            
            j = 0
            while j < len(edge_text) and i < len(pattern):
                if edge_text[j] != pattern[i]:
                    return []
                j += 1
                i += 1
            
            node = child
        
        # Collect all suffix indices from this subtree
        results = []
        self._collect_suffixes(node, results)
        return results
    
    def _collect_suffixes(self, node, results):
        """Collect all suffix indices from subtree"""
        results.extend(node.suffix_indices)
        for child in node.children.values():
            self._collect_suffixes(child, results)

# Example usage
print("\nGeneralized Suffix Tree:")
gst = GeneralizedSuffixTree()

# Add multiple strings
strings = ["banana", "ananas", "bandana"]
for i, s in enumerate(strings):
    gst.add_string(s, i)
    print(f"Added string {i}: {s}")

# Find longest common substring
lcs = gst.find_longest_common_substring()
print(f"\nLongest common substring: '{lcs}'")

# Search for pattern
pattern = "ana"
results = gst.search(pattern)
print(f"\nPattern '{pattern}' found in:")
for string_id, pos in results:
    if string_id < len(strings):
        print(f"  String {string_id} ('{strings[string_id]}') at position {pos}")

# Example with DNA sequences
print("\nDNA sequence example:")
gst_dna = GeneralizedSuffixTree()
sequences = ["ACGTACGT", "ACGTTAGC", "TACGTACG"]

for i, seq in enumerate(sequences):
    gst_dna.add_string(seq, i)

lcs_dna = gst_dna.find_longest_common_substring()
print(f"Longest common DNA substring: '{lcs_dna}'")
```

Directed Acyclic Word Graph (DAWG)

When: Need minimal automaton for all suffixes
Why: More compact than suffix tree for some applications
Examples: Pattern matching, text indexing with less space

```python
# Python implementation of Directed Acyclic Word Graph (DAWG)
# A minimal automaton representing all suffixes of a string

class DAWGNode:
    """Node in the DAWG"""
    def __init__(self):
        self.edges = {}  # char -> node
        self.suffix_link = None
        self.is_terminal = False
        self.id = None  # For visualization/debugging
        
class DAWG:
    """
    Directed Acyclic Word Graph
    Space-efficient representation of all suffixes
    """
    def __init__(self):
        self.root = DAWGNode()
        self.root.id = 0
        self.nodes = [self.root]
        self.node_count = 1
        self.text = ""
        
    def build(self, text):
        """Build DAWG from text"""
        self.text = text
        
        # Build incrementally
        for i, char in enumerate(text):
            self._extend(char)
        
        # Mark terminal states
        self._mark_terminals()
        
        # Minimize the automaton
        self._minimize()
    
    def _extend(self, char):
        """Extend DAWG with a new character"""
        new_node = DAWGNode()
        new_node.id = self.node_count
        self.node_count += 1
        self.nodes.append(new_node)
        
        # Add edges from existing nodes
        current = self.root
        while current and char not in current.edges:
            current.edges[char] = new_node
            if current.suffix_link:
                current = current.suffix_link
            else:
                break
    
    def _mark_terminals(self):
        """Mark nodes that represent complete suffixes"""
        def dfs(node, depth):
            if depth == len(self.text):
                node.is_terminal = True
            for child in node.edges.values():
                dfs(child, depth + 1)
        
        # All paths from root that reach end of text
        current = self.root
        for i in range(len(self.text)):
            if self.text[i] in current.edges:
                current = current.edges[self.text[i]]
                if i == len(self.text) - 1:
                    current.is_terminal = True
    
    def _minimize(self):
        """Minimize DAWG by merging equivalent states"""
        # Build signature for each node
        signatures = {}
        
        def get_signature(node):
            if node in signatures:
                return signatures[node]
            
            # Signature based on edges and terminal status
            sig_parts = [str(node.is_terminal)]
            for char in sorted(node.edges.keys()):
                child_sig = get_signature(node.edges[char])
                sig_parts.append(f"{char}:{child_sig}")
            
            sig = '|'.join(sig_parts)
            signatures[node] = sig
            return sig
        
        # Get signatures for all nodes
        for node in self.nodes:
            get_signature(node)
        
        # Group nodes by signature
        sig_to_nodes = {}
        for node, sig in signatures.items():
            if sig not in sig_to_nodes:
                sig_to_nodes[sig] = []
            sig_to_nodes[sig].append(node)
        
        # Merge equivalent nodes (simplified - full DAWG minimization is complex)
        representative = {}
        for nodes in sig_to_nodes.values():
            rep = nodes[0]
            for node in nodes:
                representative[node] = rep
    
    def search(self, pattern):
        """Search for pattern in DAWG"""
        current = self.root
        
        for char in pattern:
            if char not in current.edges:
                return False
            current = current.edges[char]
        
        return True
    
    def count_occurrences(self, pattern):
        """Count occurrences of pattern"""
        # Navigate to pattern node
        current = self.root
        for char in pattern:
            if char not in current.edges:
                return 0
            current = current.edges[char]
        
        # Count paths from this node to terminals
        count = 0
        
        def dfs(node):
            nonlocal count
            if node.is_terminal:
                count += 1
            for child in node.edges.values():
                dfs(child)
        
        dfs(current)
        return count
    
    def get_all_suffixes(self):
        """Get all suffixes represented in DAWG"""
        suffixes = []
        
        def dfs(node, current_string):
            if node.is_terminal or not node.edges:
                suffixes.append(current_string)
            
            for char, child in node.edges.items():
                dfs(child, current_string + char)
        
        dfs(self.root, "")
        return suffixes
    
    def longest_common_substring(self, other_text):
        """Find longest common substring with another text"""
        max_len = 0
        max_substring = ""
        
        # Try all substrings of other_text
        for i in range(len(other_text)):
            for j in range(i + 1, len(other_text) + 1):
                substring = other_text[i:j]
                if self.search(substring) and len(substring) > max_len:
                    max_len = len(substring)
                    max_substring = substring
        
        return max_substring
    
    def node_count(self):
        """Return number of nodes in DAWG"""
        return len(self.nodes)

# Example usage
print("\nDirected Acyclic Word Graph (DAWG):")
dawg = DAWG()

# Build DAWG from text
text = "banana"
dawg.build(text)
print(f"Built DAWG for: {text}")
print(f"Number of nodes: {dawg.node_count}")

# Search for patterns
patterns = ["ana", "nan", "ban", "xyz", "a"]
print("\nPattern search:")
for pattern in patterns:
    found = dawg.search(pattern)
    print(f"Pattern '{pattern}': {'Found' if found else 'Not found'}")

# Get all suffixes
suffixes = dawg.get_all_suffixes()
print(f"\nSuffixes represented: {suffixes}")

# Example with larger text
print("\nLarger example:")
dawg2 = DAWG()
text2 = "mississippi"
dawg2.build(text2)
print(f"Text: {text2}")
print(f"Nodes in DAWG: {dawg2.node_count}")

# Search patterns
test_patterns = ["iss", "ssi", "ippi", "mis", "xyz"]
print("\nPattern matching:")
for pattern in test_patterns:
    found = dawg2.search(pattern)
    print(f"  '{pattern}': {'✓' if found else '✗'}")

# Longest common substring
print("\nLongest common substring:")
dawg3 = DAWG()
dawg3.build("banana")
lcs = dawg3.longest_common_substring("ananas")
print(f"Between 'banana' and 'ananas': '{lcs}'")
```

Patricia Trie (Practical Algorithm to Retrieve Information Coded in Alphanumeric)

When: Need compact trie with edge labels
Why: Compresses single-child chains, more space-efficient
Examples: IP routing, associative arrays with string keys

```python
# Python implementation of Patricia Trie (Radix Trie)
# Compresses chains of single-child nodes for space efficiency

class PatriciaNode:
    """Node in Patricia Trie"""
    def __init__(self):
        self.children = {}  # edge_label -> child_node
        self.is_end = False
        self.value = None  # Associated value for key-value pairs
        
class PatriciaTrie:
    """
    Patricia Trie (Radix Trie) implementation
    Compresses single-child chains into edge labels
    More space-efficient than standard trie
    """
    def __init__(self):
        self.root = PatriciaNode()
        self.size = 0
    
    def insert(self, key, value=None):
        """Insert key-value pair into trie"""
        if not key:
            return
        
        node = self.root
        i = 0
        
        while i < len(key):
            # Find matching edge
            found = False
            for edge_label, child in node.children.items():
                # Find common prefix
                common_len = self._common_prefix_length(key[i:], edge_label)
                
                if common_len > 0:
                    found = True
                    
                    if common_len == len(edge_label):
                        # Full match with edge, continue down
                        node = child
                        i += common_len
                    else:
                        # Partial match - need to split edge
                        self._split_edge(node, edge_label, child, common_len)
                        
                        # Navigate to split node
                        new_edge = edge_label[:common_len]
                        node = node.children[new_edge]
                        i += common_len
                    break
            
            if not found:
                # No matching edge, create new leaf
                new_node = PatriciaNode()
                new_node.is_end = True
                new_node.value = value
                node.children[key[i:]] = new_node
                self.size += 1
                return
        
        # Key fully consumed
        if not node.is_end:
            self.size += 1
        node.is_end = True
        node.value = value
    
    def _common_prefix_length(self, s1, s2):
        """Find length of common prefix"""
        i = 0
        while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
            i += 1
        return i
    
    def _split_edge(self, parent, edge_label, child, split_pos):
        """Split edge at given position"""
        # Remove old edge
        del parent.children[edge_label]
        
        # Create intermediate node
        split_node = PatriciaNode()
        
        # Add edges
        common_prefix = edge_label[:split_pos]
        remaining_suffix = edge_label[split_pos:]
        
        parent.children[common_prefix] = split_node
        split_node.children[remaining_suffix] = child
    
    def search(self, key):
        """Search for key, return associated value or None"""
        if not key:
            return None
        
        node = self.root
        i = 0
        
        while i < len(key):
            found = False
            
            for edge_label, child in node.children.items():
                common_len = self._common_prefix_length(key[i:], edge_label)
                
                if common_len == len(edge_label):
                    # Full edge match
                    node = child
                    i += common_len
                    found = True
                    break
                elif common_len > 0:
                    # Partial match - key not in trie
                    return None
            
            if not found:
                return None
        
        return node.value if node.is_end else None
    
    def contains(self, key):
        """Check if key exists in trie"""
        return self.search(key) is not None
    
    def delete(self, key):
        """Delete key from trie"""
        if not key:
            return False
        
        # Find node and path
        path = []
        node = self.root
        i = 0
        
        while i < len(key):
            found = False
            
            for edge_label, child in node.children.items():
                common_len = self._common_prefix_length(key[i:], edge_label)
                
                if common_len == len(edge_label):
                    path.append((node, edge_label, child))
                    node = child
                    i += common_len
                    found = True
                    break
            
            if not found:
                return False
        
        if not node.is_end:
            return False
        
        node.is_end = False
        node.value = None
        self.size -= 1
        
        # Clean up if node has no children
        if not node.children and path:
            parent, edge_label, _ = path[-1]
            del parent.children[edge_label]
            
            # Merge with parent if parent has only one child
            self._merge_if_needed(path)
        
        return True
    
    def _merge_if_needed(self, path):
        """Merge nodes with single children"""
        if len(path) < 2:
            return
        
        parent, parent_edge, node = path[-1]
        
        if not node.is_end and len(node.children) == 1:
            # Merge node with its child
            child_edge, child_node = next(iter(node.children.items()))
            merged_edge = parent_edge + child_edge
            parent.children[merged_edge] = child_node
            del parent.children[parent_edge]
    
    def starts_with(self, prefix):
        """Find all keys starting with prefix"""
        if not prefix:
            return self.get_all_keys()
        
        # Navigate to prefix node
        node = self.root
        i = 0
        current_prefix = ""
        
        while i < len(prefix):
            found = False
            
            for edge_label, child in node.children.items():
                common_len = self._common_prefix_length(prefix[i:], edge_label)
                
                if common_len > 0:
                    if common_len == len(edge_label):
                        current_prefix += edge_label
                        node = child
                        i += common_len
                        found = True
                        break
                    elif common_len == len(prefix[i:]):
                        # Prefix ends in middle of edge
                        current_prefix += prefix[i:]
                        return self._collect_keys(child, current_prefix + edge_label[common_len:])
            
            if not found:
                return []
        
        # Collect all keys from this node
        return self._collect_keys(node, current_prefix)
    
    def _collect_keys(self, node, prefix):
        """Collect all keys in subtree"""
        keys = []
        
        if node.is_end:
            keys.append(prefix)
        
        for edge_label, child in node.children.items():
            keys.extend(self._collect_keys(child, prefix + edge_label))
        
        return keys
    
    def get_all_keys(self):
        """Get all keys in trie"""
        return self._collect_keys(self.root, "")
    
    def __len__(self):
        return self.size
    
    def __contains__(self, key):
        return self.contains(key)

# Example usage
print("\nPatricia Trie:")
trie = PatriciaTrie()

# Insert key-value pairs
entries = [
    ("test", 1),
    ("testing", 2),
    ("tester", 3),
    ("team", 4),
    ("toast", 5),
    ("tree", 6)
]

for key, value in entries:
    trie.insert(key, value)
    print(f"Inserted '{key}' with value {value}")

print(f"\nTrie size: {len(trie)}")

# Search
print("\nSearching:")
for key in ["test", "testing", "team", "xyz"]:
    value = trie.search(key)
    print(f"  '{key}': {value if value is not None else 'Not found'}")

# Prefix search
print("\nKeys starting with 'test':")
matches = trie.starts_with("test")
for match in matches:
    print(f"  {match}")

print("\nKeys starting with 'te':")
matches = trie.starts_with("te")
for match in matches:
    print(f"  {match}")

# IP routing example
print("\nIP Routing example:")
ip_trie = PatriciaTrie()

routes = [
    ("192.168.1", "Router A"),
    ("192.168.2", "Router B"),
    ("192.168", "Router C"),
    ("10.0", "Router D"),
]

for prefix, router in routes:
    ip_trie.insert(prefix, router)

# Lookup IP addresses
ips = ["192.168.1.100", "192.168.2.50", "10.0.0.1"]
print("\nIP Lookups:")
for ip in ips:
    # Find longest prefix match
    matches = ip_trie.starts_with(ip[:ip.rfind('.')])
    if matches:
        print(f"  {ip} -> Best match: {matches[-1]}")
```

Spatial & Geometric
BSP Tree (Binary Space Partitioning) - detailed

When: Need recursive spatial subdivision for rendering
Why: Efficient visibility determination, painter's algorithm
Examples: 3D game engines (Doom, Quake), ray tracing, collision detection

```python
# Python implementation of BSP Tree (Binary Space Partitioning)
# Used for spatial subdivision in rendering and collision detection

import math

class Plane:
    """Represents a 2D plane (line) or 3D plane"""
    def __init__(self, point, normal):
        """
        point: a point on the plane (tuple/list)
        normal: normal vector to the plane (tuple/list)
        """
        self.point = point
        self.normal = self._normalize(normal)
    
    def _normalize(self, vector):
        """Normalize a vector"""
        magnitude = math.sqrt(sum(x*x for x in vector))
        if magnitude == 0:
            return vector
        return tuple(x / magnitude for x in vector)
    
    def classify_point(self, point, epsilon=1e-6):
        """
        Classify point relative to plane
        Returns: 'FRONT', 'BACK', or 'ON_PLANE'
        """
        # Calculate distance from point to plane
        diff = tuple(point[i] - self.point[i] for i in range(len(point)))
        distance = sum(diff[i] * self.normal[i] for i in range(len(diff)))
        
        if distance > epsilon:
            return 'FRONT'
        elif distance < -epsilon:
            return 'BACK'
        else:
            return 'ON_PLANE'
    
    def split_polygon(self, polygon):
        """
        Split polygon by this plane
        Returns: (front_polygons, back_polygons, coplanar_front, coplanar_back)
        """
        # Simplified: assumes polygon is a list of points
        front = []
        back = []
        
        for point in polygon:
            classification = self.classify_point(point)
            if classification == 'FRONT':
                front.append(point)
            elif classification == 'BACK':
                back.append(point)
            else:
                # Point on plane - add to both
                front.append(point)
                back.append(point)
        
        return front, back

class BSPNode:
    """Node in BSP Tree"""
    def __init__(self, plane=None, polygons=None):
        self.plane = plane  # Splitting plane
        self.front = None   # Front child (positive half-space)
        self.back = None    # Back child (negative half-space)
        self.polygons = polygons if polygons else []  # Polygons at this node
        
class BSPTree:
    """
    Binary Space Partitioning Tree
    Recursively subdivides space for efficient rendering
    """
    def __init__(self):
        self.root = None
    
    def build(self, polygons):
        """Build BSP tree from list of polygons"""
        if not polygons:
            return None
        
        self.root = self._build_recursive(polygons)
    
    def _build_recursive(self, polygons):
        """Recursively build BSP tree"""
        if not polygons:
            return None
        
        # Choose splitting plane (using first polygon's plane)
        # In practice, you'd use a heuristic to choose the best plane
        splitting_polygon = polygons[0]
        plane = self._polygon_to_plane(splitting_polygon)
        
        node = BSPNode(plane)
        
        front_polygons = []
        back_polygons = []
        coplanar_polygons = [splitting_polygon]
        
        # Classify remaining polygons
        for i in range(1, len(polygons)):
            polygon = polygons[i]
            classification = self._classify_polygon(polygon, plane)
            
            if classification == 'FRONT':
                front_polygons.append(polygon)
            elif classification == 'BACK':
                back_polygons.append(polygon)
            elif classification == 'SPANNING':
                # Split polygon
                front_part, back_part = plane.split_polygon(polygon)
                if front_part:
                    front_polygons.append(front_part)
                if back_part:
                    back_polygons.append(back_part)
            else:  # COPLANAR
                coplanar_polygons.append(polygon)
        
        node.polygons = coplanar_polygons
        
        # Recursively build subtrees
        if front_polygons:
            node.front = self._build_recursive(front_polygons)
        if back_polygons:
            node.back = self._build_recursive(back_polygons)
        
        return node
    
    def _polygon_to_plane(self, polygon):
        """Create plane from polygon (simplified for 2D)"""
        # Assumes polygon is a list of points
        if len(polygon) < 2:
            return Plane((0, 0), (1, 0))
        
        # Calculate normal (simplified 2D)
        p1, p2 = polygon[0], polygon[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Perpendicular in 2D
        if len(polygon[0]) == 2:
            normal = (-dy, dx)
        else:
            # 3D case - use cross product (simplified)
            if len(polygon) >= 3:
                p3 = polygon[2]
                v1 = (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
                v2 = (p3[0]-p1[0], p3[1]-p1[1], p3[2]-p1[2])
                normal = (
                    v1[1]*v2[2] - v1[2]*v2[1],
                    v1[2]*v2[0] - v1[0]*v2[2],
                    v1[0]*v2[1] - v1[1]*v2[0]
                )
            else:
                normal = (0, 0, 1)
        
        return Plane(polygon[0], normal)
    
    def _classify_polygon(self, polygon, plane):
        """Classify polygon relative to plane"""
        front_count = 0
        back_count = 0
        
        for point in polygon:
            classification = plane.classify_point(point)
            if classification == 'FRONT':
                front_count += 1
            elif classification == 'BACK':
                back_count += 1
        
        if front_count > 0 and back_count > 0:
            return 'SPANNING'
        elif front_count > 0:
            return 'FRONT'
        elif back_count > 0:
            return 'BACK'
        else:
            return 'COPLANAR'
    
    def render_back_to_front(self, camera_position):
        """
        Render polygons in back-to-front order (painter's algorithm)
        Returns list of polygons in rendering order
        """
        result = []
        self._traverse_back_to_front(self.root, camera_position, result)
        return result
    
    def _traverse_back_to_front(self, node, camera_pos, result):
        """Traverse tree in back-to-front order relative to camera"""
        if node is None:
            return
        
        # Classify camera position
        camera_side = node.plane.classify_point(camera_pos)
        
        if camera_side == 'FRONT':
            # Render back first, then coplanar, then front
            self._traverse_back_to_front(node.back, camera_pos, result)
            result.extend(node.polygons)
            self._traverse_back_to_front(node.front, camera_pos, result)
        else:
            # Render front first, then coplanar, then back
            self._traverse_back_to_front(node.front, camera_pos, result)
            result.extend(node.polygons)
            self._traverse_back_to_front(node.back, camera_pos, result)
    
    def point_location(self, point):
        """Find which leaf node contains the point"""
        node = self.root
        
        while node:
            if not node.front and not node.back:
                return node
            
            classification = node.plane.classify_point(point)
            
            if classification == 'FRONT' and node.front:
                node = node.front
            elif classification == 'BACK' and node.back:
                node = node.back
            else:
                return node
        
        return None

# Example usage
print("\nBSP Tree (Binary Space Partitioning):")

# 2D example with simple rectangles (represented as polygons)
polygons = [
    [(0, 0), (10, 0), (10, 5), (0, 5)],      # Wall 1
    [(5, 5), (15, 5), (15, 10), (5, 10)],    # Wall 2
    [(3, 3), (7, 3), (7, 7), (3, 7)],        # Wall 3
    [(12, 2), (18, 2), (18, 8), (12, 8)],    # Wall 4
]

print("Building BSP tree with", len(polygons), "polygons")
bsp_tree = BSPTree()
bsp_tree.build(polygons)
print("BSP tree built successfully")

# Render from different camera positions
camera_positions = [
    (5, 5),   # Center
    (-5, -5), # Bottom-left
    (20, 20), # Top-right
]

print("\nRendering order from different viewpoints:")
for i, cam_pos in enumerate(camera_positions):
    print(f"\nCamera at {cam_pos}:")
    rendering_order = bsp_tree.render_back_to_front(cam_pos)
    for j, polygon in enumerate(rendering_order):
        print(f"  {j+1}. Polygon at {polygon[0]}")

# Point location query
print("\nPoint location queries:")
test_points = [(5, 2), (10, 7), (15, 3)]
for point in test_points:
    node = bsp_tree.point_location(point)
    if node:
        print(f"Point {point} is in leaf node with {len(node.polygons)} polygons")
```

PH-Tree (Multi-dimensional indexing)

When: Need efficient multi-dimensional point storage
Why: Compact representation, efficient range queries
Examples: Spatial databases, multi-dimensional data

```python
# Python implementation of PH-Tree (Prefix-Hierarchical Tree)
# Efficient multi-dimensional index using bit-level prefix sharing

class PHNode:
    """Node in PH-Tree"""
    def __init__(self, prefix=0, prefix_len=0):
        self.prefix = prefix  # Common prefix bits
        self.prefix_len = prefix_len  # Length of prefix in bits
        self.children = {}  # suffix -> child node or value
        self.is_leaf = False
        self.value = None
        
class PHTree:
    """
    PH-Tree: Multi-dimensional index with prefix sharing
    Efficiently stores and queries multi-dimensional points
    """
    def __init__(self, dimensions=2, bits_per_dimension=32):
        """
        dimensions: number of dimensions
        bits_per_dimension: bits used per coordinate
        """
        self.dimensions = dimensions
        self.bits_per_dimension = bits_per_dimension
        self.total_bits = dimensions * bits_per_dimension
        self.root = PHNode()
        self.size = 0
    
    def _interleave_bits(self, point):
        """
        Interleave bits from all dimensions (Z-order curve)
        Example for 2D: [x1x0, y1y0] -> z3z2z1z0 where z = y1x1y0x0
        """
        result = 0
        for bit_pos in range(self.bits_per_dimension):
            for dim in range(self.dimensions):
                if len(point) > dim:
                    coord = int(point[dim]) & ((1 << self.bits_per_dimension) - 1)
                    bit = (coord >> bit_pos) & 1
                    result |= (bit << (bit_pos * self.dimensions + dim))
        return result
    
    def _deinterleave_bits(self, interleaved):
        """Convert interleaved bits back to coordinates"""
        point = [0] * self.dimensions
        for bit_pos in range(self.bits_per_dimension):
            for dim in range(self.dimensions):
                bit_index = bit_pos * self.dimensions + dim
                bit = (interleaved >> bit_index) & 1
                point[dim] |= (bit << bit_pos)
        return tuple(point)
    
    def insert(self, point, value=None):
        """Insert point into PH-Tree"""
        if len(point) != self.dimensions:
            raise ValueError(f"Point must have {self.dimensions} dimensions")
        
        key = self._interleave_bits(point)
        
        if value is None:
            value = point
        
        self._insert_recursive(self.root, key, value, 0)
        self.size += 1
    
    def _insert_recursive(self, node, key, value, depth):
        """Recursively insert key into tree"""
        if depth >= self.total_bits:
            node.is_leaf = True
            node.value = value
            return
        
        # Extract suffix after prefix
        suffix = key >> depth
        
        if suffix not in node.children:
            # Create new child
            child = PHNode()
            node.children[suffix] = child
            self._insert_recursive(child, key, value, depth + 1)
        else:
            # Navigate to existing child
            self._insert_recursive(node.children[suffix], key, value, depth + 1)
    
    def search(self, point):
        """Search for exact point"""
        if len(point) != self.dimensions:
            return None
        
        key = self._interleave_bits(point)
        return self._search_recursive(self.root, key, 0)
    
    def _search_recursive(self, node, key, depth):
        """Recursively search for key"""
        if node.is_leaf:
            return node.value
        
        if depth >= self.total_bits:
            return None
        
        suffix = key >> depth
        
        if suffix in node.children:
            return self._search_recursive(node.children[suffix], key, depth + 1)
        
        return None
    
    def range_query(self, min_point, max_point):
        """
        Find all points in hyper-rectangle [min_point, max_point]
        """
        if len(min_point) != self.dimensions or len(max_point) != self.dimensions:
            raise ValueError(f"Points must have {self.dimensions} dimensions")
        
        results = []
        min_key = self._interleave_bits(min_point)
        max_key = self._interleave_bits(max_point)
        
        self._range_query_recursive(self.root, min_key, max_key, 0, results)
        return results
    
    def _range_query_recursive(self, node, min_key, max_key, depth, results):
        """Recursively find points in range"""
        if node.is_leaf:
            # Check if point is in range
            if node.value:
                point = node.value if isinstance(node.value, tuple) else self._deinterleave_bits(min_key)
                in_range = all(
                    min_point <= coord <= max_point
                    for min_point, coord, max_point in zip(
                        self._deinterleave_bits(min_key),
                        point,
                        self._deinterleave_bits(max_key)
                    )
                )
                if in_range:
                    results.append(node.value)
            return
        
        if depth >= self.total_bits:
            return
        
        # Check all children that might be in range
        for suffix, child in node.children.items():
            # Simplified range check
            self._range_query_recursive(child, min_key, max_key, depth + 1, results)
    
    def nearest_neighbor(self, point, k=1):
        """Find k nearest neighbors (simplified implementation)"""
        if len(point) != self.dimensions:
            raise ValueError(f"Point must have {self.dimensions} dimensions")
        
        all_points = []
        self._collect_all_points(self.root, all_points)
        
        # Calculate distances
        distances = []
        for p in all_points:
            if isinstance(p, tuple):
                dist = sum((p[i] - point[i])**2 for i in range(len(point))) ** 0.5
                distances.append((dist, p))
        
        # Sort and return k nearest
        distances.sort()
        return [p for _, p in distances[:k]]
    
    def _collect_all_points(self, node, points):
        """Collect all points in subtree"""
        if node.is_leaf and node.value:
            points.append(node.value)
        
        for child in node.children.values():
            self._collect_all_points(child, points)
    
    def __len__(self):
        return self.size
    
    def __contains__(self, point):
        return self.search(point) is not None

# Example usage
print("\nPH-Tree (Multi-dimensional indexing):")

# Create 2D PH-Tree
phtree = PHTree(dimensions=2, bits_per_dimension=16)

# Insert points
points = [
    (10, 20),
    (15, 25),
    (5, 30),
    (25, 15),
    (30, 30),
    (8, 12),
]

print("Inserting points:")
for point in points:
    phtree.insert(point)
    print(f"  Inserted {point}")

print(f"\nTree size: {len(phtree)}")

# Exact search
print("\nExact search:")
test_points = [(10, 20), (15, 25), (100, 100)]
for point in test_points:
    result = phtree.search(point)
    print(f"  {point}: {'Found' if result else 'Not found'}")

# Range query
print("\nRange query [5, 5] to [20, 25]:")
results = phtree.range_query((5, 5), (20, 25))
print(f"  Found {len(results)} points:")
for point in results:
    print(f"    {point}")

# Nearest neighbor
print("\nNearest neighbors to (12, 18):")
neighbors = phtree.nearest_neighbor((12, 18), k=3)
for i, neighbor in enumerate(neighbors, 1):
    print(f"  {i}. {neighbor}")

# 3D example
print("\n3D PH-Tree example:")
phtree_3d = PHTree(dimensions=3, bits_per_dimension=16)

points_3d = [
    (10, 20, 30),
    (15, 25, 35),
    (5, 30, 25),
    (25, 15, 40),
]

for point in points_3d:
    phtree_3d.insert(point)

print(f"Inserted {len(phtree_3d)} 3D points")
print(f"Search (10, 20, 30): {phtree_3d.search((10, 20, 30))}")
```

M-Tree

When: Need metric space indexing (not just geometric)
Why: Works with any distance metric, not just Euclidean
Examples: Similarity search, multimedia databases, DNA sequences

```python
# Python implementation of M-Tree
# Metric space index that works with any distance function

import math
from collections import deque

class MTreeEntry:
    """Entry in M-Tree (can be object or routing entry)"""
    def __init__(self, obj, radius=0, distance_to_parent=0):
        self.obj = obj  # The actual data object
        self.radius = radius  # Covering radius (for routing entries)
        self.distance_to_parent = distance_to_parent
        self.subtree = None  # Pointer to subtree (for internal nodes)
        
class MTreeNode:
    """Node in M-Tree"""
    def __init__(self, is_leaf=True, capacity=4):
        self.is_leaf = is_leaf
        self.entries = []
        self.capacity = capacity
        self.parent_entry = None
        
class MTree:
    """
    M-Tree: Index for metric spaces
    Works with any distance metric (not just Euclidean)
    """
    def __init__(self, distance_func, max_capacity=4):
        """
        distance_func: function(obj1, obj2) -> float
        max_capacity: maximum entries per node
        """
        self.distance = distance_func
        self.root = MTreeNode(is_leaf=True, capacity=max_capacity)
        self.max_capacity = max_capacity
        self.size = 0
    
    def insert(self, obj):
        """Insert object into M-Tree"""
        # Find appropriate leaf node
        leaf = self._find_leaf(obj, self.root)
        
        # Insert into leaf
        new_entry = MTreeEntry(obj, radius=0)
        
        if leaf.entries:
            # Calculate distance to parent
            parent_obj = leaf.entries[0].obj
            new_entry.distance_to_parent = self.distance(obj, parent_obj)
        
        leaf.entries.append(new_entry)
        self.size += 1
        
        # Handle overflow
        if len(leaf.entries) > self.max_capacity:
            self._split_node(leaf)
    
    def _find_leaf(self, obj, node):
        """Find appropriate leaf node for insertion"""
        if node.is_leaf:
            return node
        
        # Choose subtree with minimum distance increase
        best_entry = None
        min_dist_increase = float('inf')
        
        for entry in node.entries:
            dist = self.distance(obj, entry.obj)
            dist_increase = max(0, dist - entry.radius)
            
            if dist_increase < min_dist_increase:
                min_dist_increase = dist_increase
                best_entry = entry
        
        if best_entry and best_entry.subtree:
            return self._find_leaf(obj, best_entry.subtree)
        
        return node
    
    def _split_node(self, node):
        """Split overflowing node"""
        # Choose two routing objects (promote)
        entries = node.entries
        
        # Simple split: choose two farthest objects
        max_dist = 0
        obj1_idx, obj2_idx = 0, 1
        
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                dist = self.distance(entries[i].obj, entries[j].obj)
                if dist > max_dist:
                    max_dist = dist
                    obj1_idx, obj2_idx = i, j
        
        # Create two new nodes
        node1 = MTreeNode(is_leaf=node.is_leaf, capacity=self.max_capacity)
        node2 = MTreeNode(is_leaf=node.is_leaf, capacity=self.max_capacity)
        
        obj1 = entries[obj1_idx].obj
        obj2 = entries[obj2_idx].obj
        
        # Partition entries
        for i, entry in enumerate(entries):
            if i == obj1_idx:
                continue
            elif i == obj2_idx:
                continue
            
            dist1 = self.distance(entry.obj, obj1)
            dist2 = self.distance(entry.obj, obj2)
            
            if dist1 <= dist2:
                node1.entries.append(entry)
            else:
                node2.entries.append(entry)
        
        # Add routing objects
        node1.entries.insert(0, entries[obj1_idx])
        node2.entries.insert(0, entries[obj2_idx])
        
        # Update parent or create new root
        if node == self.root:
            new_root = MTreeNode(is_leaf=False, capacity=self.max_capacity)
            
            # Create routing entries
            entry1 = MTreeEntry(obj1, radius=self._compute_radius(node1))
            entry1.subtree = node1
            
            entry2 = MTreeEntry(obj2, radius=self._compute_radius(node2))
            entry2.subtree = node2
            
            new_root.entries = [entry1, entry2]
            self.root = new_root
    
    def _compute_radius(self, node):
        """Compute covering radius for node"""
        if not node.entries:
            return 0
        
        center = node.entries[0].obj
        max_dist = 0
        
        for entry in node.entries[1:]:
            dist = self.distance(entry.obj, center)
            max_dist = max(max_dist, dist)
        
        return max_dist
    
    def range_search(self, query, radius):
        """Find all objects within radius of query"""
        results = []
        self._range_search_recursive(query, radius, self.root, results)
        return results
    
    def _range_search_recursive(self, query, radius, node, results):
        """Recursively search for objects in range"""
        if node.is_leaf:
            for entry in node.entries:
                if self.distance(query, entry.obj) <= radius:
                    results.append(entry.obj)
        else:
            for entry in node.entries:
                dist = self.distance(query, entry.obj)
                if dist <= radius + entry.radius:
                    if entry.subtree:
                        self._range_search_recursive(query, radius, entry.subtree, results)
    
    def k_nearest_neighbors(self, query, k):
        """Find k nearest neighbors to query"""
        candidates = []
        
        # Collect all objects with distances
        def collect(node):
            if node.is_leaf:
                for entry in node.entries:
                    dist = self.distance(query, entry.obj)
                    candidates.append((dist, entry.obj))
            else:
                for entry in node.entries:
                    if entry.subtree:
                        collect(entry.subtree)
        
        collect(self.root)
        
        # Sort by distance and return k nearest
        candidates.sort()
        return [obj for _, obj in candidates[:k]]
    
    def __len__(self):
        return self.size

# Example usage
print("\nM-Tree (Metric Space Index):")

# Define a custom distance function for strings (edit distance)
def edit_distance(s1, s2):
    """Levenshtein distance"""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

# Create M-Tree with edit distance
mtree = MTree(distance_func=edit_distance, max_capacity=3)

# Insert strings
words = ["hello", "hallo", "hullo", "world", "word", "work", "python", "jython"]
print("Inserting words:")
for word in words:
    mtree.insert(word)
    print(f"  Inserted: {word}")

print(f"\nTree size: {len(mtree)}")

# Range search
print("\nRange search (words within edit distance 2 of 'hello'):")
results = mtree.range_search("hello", 2)
for word in results:
    print(f"  {word} (distance: {edit_distance('hello', word)})")

# K-nearest neighbors
print("\nK-nearest neighbors (k=3) to 'world':")
neighbors = mtree.k_nearest_neighbors("world", 3)
for i, word in enumerate(neighbors, 1):
    print(f"  {i}. {word} (distance: {edit_distance('world', word)})")

# Euclidean distance example
print("\nM-Tree with Euclidean distance:")
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

mtree_2d = MTree(distance_func=euclidean_distance, max_capacity=3)

points = [(1, 2), (3, 4), (5, 6), (2, 3), (4, 5), (6, 7)]
for point in points:
    mtree_2d.insert(point)

print(f"Inserted {len(mtree_2d)} 2D points")

query_point = (3, 3)
radius = 2.0
print(f"\nPoints within radius {radius} of {query_point}:")
nearby = mtree_2d.range_search(query_point, radius)
for point in nearby:
    dist = euclidean_distance(query_point, point)
    print(f"  {point} (distance: {dist:.2f})")
```

X-Tree

When: R-Tree performs poorly in high dimensions
Why: Optimized for high-dimensional data
Examples: High-dimensional indexing, feature vector databases

```python
# Python implementation of X-Tree
# Extended R-Tree optimized for high-dimensional data

class XTreeEntry:
    """Entry in X-Tree node"""
    def __init__(self, mbr=None, child=None, data=None):
        self.mbr = mbr  # Minimum Bounding Rectangle
        self.child = child  # Pointer to child node
        self.data = data  # Data for leaf entries
        
class XTreeNode:
    """Node in X-Tree"""
    def __init__(self, is_leaf=True, capacity=10):
        self.is_leaf = is_leaf
        self.entries = []
        self.capacity = capacity
        self.is_supernode = False  # X-Tree allows supernodes
        
class MBR:
    """Minimum Bounding Rectangle"""
    def __init__(self, dimensions):
        self.low = list(dimensions)  # Lower bounds
        self.high = list(dimensions)  # Upper bounds
    
    @classmethod
    def from_point(cls, point):
        """Create MBR from a single point"""
        return cls(point)
    
    @classmethod
    def from_points(cls, points):
        """Create MBR enclosing multiple points"""
        if not points:
            return None
        
        dims = len(points[0])
        low = list(points[0])
        high = list(points[0])
        
        for point in points[1:]:
            for i in range(dims):
                low[i] = min(low[i], point[i])
                high[i] = max(high[i], point[i])
        
        mbr = cls(low)
        mbr.high = high
        return mbr
    
    def contains(self, point):
        """Check if point is inside MBR"""
        for i in range(len(point)):
            if point[i] < self.low[i] or point[i] > self.high[i]:
                return False
        return True
    
    def intersects(self, other):
        """Check if this MBR intersects with another"""
        for i in range(len(self.low)):
            if self.high[i] < other.low[i] or self.low[i] > other.high[i]:
                return False
        return True
    
    def volume(self):
        """Calculate volume of MBR"""
        vol = 1.0
        for i in range(len(self.low)):
            vol *= (self.high[i] - self.low[i])
        return vol
    
    def enlargement(self, point):
        """Calculate volume increase needed to include point"""
        new_low = [min(self.low[i], point[i]) for i in range(len(point))]
        new_high = [max(self.high[i], point[i]) for i in range(len(point))]
        
        new_vol = 1.0
        for i in range(len(new_low)):
            new_vol *= (new_high[i] - new_low[i])
        
        return new_vol - self.volume()
    
    def expand(self, point):
        """Expand MBR to include point"""
        for i in range(len(point)):
            self.low[i] = min(self.low[i], point[i])
            self.high[i] = max(self.high[i], point[i])

class XTree:
    """
    X-Tree: Extended R-Tree for high-dimensional data
    Uses supernodes to avoid splits that would create high overlap
    """
    def __init__(self, dimensions, max_capacity=10, min_fanout=4):
        self.dimensions = dimensions
        self.max_capacity = max_capacity
        self.min_fanout = min_fanout
        self.root = XTreeNode(is_leaf=True, capacity=max_capacity)
        self.size = 0
        self.overlap_threshold = 0.2  # Threshold for creating supernodes
    
    def insert(self, point, data=None):
        """Insert point into X-Tree"""
        if len(point) != self.dimensions:
            raise ValueError(f"Point must have {self.dimensions} dimensions")
        
        mbr = MBR.from_point(point)
        entry = XTreeEntry(mbr=mbr, data=data if data else point)
        
        # Find appropriate leaf
        leaf = self._choose_subtree(self.root, entry)
        
        # Insert into leaf
        leaf.entries.append(entry)
        self.size += 1
        
        # Handle overflow
        if len(leaf.entries) > leaf.capacity:
            self._handle_overflow(leaf)
    
    def _choose_subtree(self, node, entry):
        """Choose subtree for insertion"""
        if node.is_leaf:
            return node
        
        # Choose child with minimum enlargement
        min_enlargement = float('inf')
        best_child = None
        
        for child_entry in node.entries:
            enlargement = child_entry.mbr.enlargement(entry.mbr.low)
            
            if enlargement < min_enlargement:
                min_enlargement = enlargement
                best_child = child_entry
        
        if best_child and best_child.child:
            return self._choose_subtree(best_child.child, entry)
        
        return node
    
    def _handle_overflow(self, node):
        """Handle node overflow - may create supernode or split"""
        if node.is_supernode:
            # Supernode can grow beyond normal capacity
            node.capacity = int(node.capacity * 1.5)
            return
        
        # Check if split would create high overlap
        overlap = self._calculate_potential_overlap(node)
        
        if overlap > self.overlap_threshold:
            # Create supernode instead of splitting
            node.is_supernode = True
            node.capacity = int(node.capacity * 1.5)
        else:
            # Perform split
            self._split_node(node)
    
    def _calculate_potential_overlap(self, node):
        """Calculate potential overlap after split"""
        if len(node.entries) < 2:
            return 0.0
        
        # Simple heuristic: check overlap between potential splits
        mid = len(node.entries) // 2
        
        mbr1 = MBR.from_points([e.mbr.low for e in node.entries[:mid]])
        mbr2 = MBR.from_points([e.mbr.low for e in node.entries[mid:]])
        
        if not mbr1 or not mbr2:
            return 0.0
        
        # Calculate overlap ratio
        if mbr1.intersects(mbr2):
            overlap_vol = 1.0
            for i in range(self.dimensions):
                overlap_low = max(mbr1.low[i], mbr2.low[i])
                overlap_high = min(mbr1.high[i], mbr2.high[i])
                if overlap_high > overlap_low:
                    overlap_vol *= (overlap_high - overlap_low)
                else:
                    return 0.0
            
            total_vol = mbr1.volume() + mbr2.volume()
            return overlap_vol / total_vol if total_vol > 0 else 0.0
        
        return 0.0
    
    def _split_node(self, node):
        """Split node into two"""
        # Simple split: divide entries in half
        mid = len(node.entries) // 2
        
        node1 = XTreeNode(is_leaf=node.is_leaf, capacity=self.max_capacity)
        node2 = XTreeNode(is_leaf=node.is_leaf, capacity=self.max_capacity)
        
        node1.entries = node.entries[:mid]
        node2.entries = node.entries[mid:]
        
        # Update parent or create new root
        if node == self.root:
            new_root = XTreeNode(is_leaf=False, capacity=self.max_capacity)
            
            mbr1 = MBR.from_points([e.mbr.low for e in node1.entries])
            mbr2 = MBR.from_points([e.mbr.low for e in node2.entries])
            
            entry1 = XTreeEntry(mbr=mbr1, child=node1)
            entry2 = XTreeEntry(mbr=mbr2, child=node2)
            
            new_root.entries = [entry1, entry2]
            self.root = new_root
    
    def range_query(self, min_point, max_point):
        """Find all points in range [min_point, max_point]"""
        query_mbr = MBR(min_point)
        query_mbr.high = list(max_point)
        
        results = []
        self._range_query_recursive(self.root, query_mbr, results)
        return results
    
    def _range_query_recursive(self, node, query_mbr, results):
        """Recursively search for points in range"""
        if node.is_leaf:
            for entry in node.entries:
                if query_mbr.contains(entry.mbr.low):
                    results.append(entry.data)
        else:
            for entry in node.entries:
                if entry.mbr.intersects(query_mbr):
                    if entry.child:
                        self._range_query_recursive(entry.child, query_mbr, results)
    
    def __len__(self):
        return self.size

# Example usage
print("\nX-Tree (High-Dimensional Index):")

# Create X-Tree for high-dimensional data
xtree = XTree(dimensions=5, max_capacity=4)

# Insert high-dimensional points
print("Inserting 5D points:")
points = [
    (1, 2, 3, 4, 5),
    (2, 3, 4, 5, 6),
    (5, 6, 7, 8, 9),
    (1, 1, 1, 1, 1),
    (9, 8, 7, 6, 5),
    (3, 4, 5, 6, 7),
]

for point in points:
    xtree.insert(point)
    print(f"  Inserted {point}")

print(f"\nTree size: {len(xtree)}")

# Range query
min_point = (1, 1, 1, 1, 1)
max_point = (4, 5, 6, 7, 8)
print(f"\nRange query [{min_point}, {max_point}]:")
results = xtree.range_query(min_point, max_point)
print(f"Found {len(results)} points:")
for point in results:
    print(f"  {point}")

# High-dimensional example (simulating feature vectors)
print("\nHigh-dimensional feature vectors:")
xtree_10d = XTree(dimensions=10, max_capacity=5)

import random
random.seed(42)

features = []
for i in range(15):
    feature = tuple(random.uniform(0, 100) for _ in range(10))
    features.append(feature)
    xtree_10d.insert(feature, data=f"Object_{i}")

print(f"Inserted {len(xtree_10d)} 10D feature vectors")

# Query a range
query_min = tuple([20] * 10)
query_max = tuple([60] * 10)
query_results = xtree_10d.range_query(query_min, query_max)
print(f"\nFeature vectors in range: {len(query_results)}")
```

Hilbert R-Tree

When: Need better space-filling curve ordering than R-Tree
Why: Uses Hilbert curve for better spatial locality
Examples: Spatial databases with better clustering

```python
# Python implementation of Hilbert R-Tree
# Uses Hilbert space-filling curve for better spatial locality

class HilbertRTreeEntry:
    """Entry in Hilbert R-Tree"""
    def __init__(self, point, hilbert_value, data=None):
        self.point = point
        self.hilbert_value = hilbert_value
        self.data = data if data else point
        self.mbr = None  # For internal nodes
        self.child = None

class HilbertRTreeNode:
    """Node in Hilbert R-Tree"""
    def __init__(self, is_leaf=True, capacity=10):
        self.is_leaf = is_leaf
        self.entries = []
        self.capacity = capacity
        self.lhv = None  # Largest Hilbert Value in subtree

class HilbertRTree:
    """
    Hilbert R-Tree: R-Tree variant using Hilbert curve ordering
    Provides better clustering and query performance
    """
    def __init__(self, dimensions=2, max_capacity=10, hilbert_order=16):
        self.dimensions = dimensions
        self.max_capacity = max_capacity
        self.hilbert_order = hilbert_order  # Bits per dimension
        self.root = HilbertRTreeNode(is_leaf=True, capacity=max_capacity)
        self.size = 0
    
    def _xy_to_hilbert(self, x, y, order=16):
        """
        Convert 2D coordinates to Hilbert curve value
        Simplified implementation for 2D
        """
        hilbert = 0
        
        for i in range(order - 1, -1, -1):
            rx = 1 if (x & (1 << i)) else 0
            ry = 1 if (y & (1 << i)) else 0
            
            hilbert = (hilbert << 2) | ((3 * rx) ^ ry)
            
            # Rotate/flip quadrant
            if ry == 0:
                if rx == 1:
                    x, y = y, x
                    x = ~x
                    y = ~y
        
        return hilbert & ((1 << (2 * order)) - 1)
    
    def _point_to_hilbert(self, point):
        """Convert multi-dimensional point to Hilbert value"""
        if self.dimensions == 2:
            # Normalize to integer coordinates
            x = int(point[0] * (1 << self.hilbert_order))
            y = int(point[1] * (1 << self.hilbert_order))
            return self._xy_to_hilbert(x, y, self.hilbert_order)
        else:
            # For higher dimensions, use simplified interleaving
            result = 0
            for bit_pos in range(self.hilbert_order):
                for dim in range(self.dimensions):
                    if dim < len(point):
                        coord = int(point[dim] * (1 << self.hilbert_order))
                        bit = (coord >> bit_pos) & 1
                        result |= (bit << (bit_pos * self.dimensions + dim))
            return result
    
    def insert(self, point, data=None):
        """Insert point into Hilbert R-Tree"""
        if len(point) != self.dimensions:
            raise ValueError(f"Point must have {self.dimensions} dimensions")
        
        hilbert_value = self._point_to_hilbert(point)
        entry = HilbertRTreeEntry(point, hilbert_value, data)
        
        # Insert using Hilbert ordering
        self._insert_entry(self.root, entry)
        self.size += 1
    
    def _insert_entry(self, node, entry):
        """Insert entry into node maintaining Hilbert order"""
        if node.is_leaf:
            # Insert in sorted order by Hilbert value
            insert_pos = 0
            for i, existing in enumerate(node.entries):
                if entry.hilbert_value < existing.hilbert_value:
                    insert_pos = i
                    break
                insert_pos = i + 1
            
            node.entries.insert(insert_pos, entry)
            
            # Update LHV
            if node.entries:
                node.lhv = node.entries[-1].hilbert_value
            
            # Handle overflow
            if len(node.entries) > node.capacity:
                self._split_node(node)
        else:
            # Find appropriate child
            child_node = self._choose_child(node, entry)
            if child_node:
                self._insert_entry(child_node, entry)
    
    def _choose_child(self, node, entry):
        """Choose child node for insertion"""
        # Find child whose LHV is >= entry's Hilbert value
        for child_entry in node.entries:
            if child_entry.child and child_entry.child.lhv >= entry.hilbert_value:
                return child_entry.child
        
        # If not found, use last child
        if node.entries and node.entries[-1].child:
            return node.entries[-1].child
        
        return None
    
    def _split_node(self, node):
        """Split node maintaining Hilbert order"""
        # Split at middle point
        mid = len(node.entries) // 2
        
        node1 = HilbertRTreeNode(is_leaf=node.is_leaf, capacity=self.max_capacity)
        node2 = HilbertRTreeNode(is_leaf=node.is_leaf, capacity=self.max_capacity)
        
        node1.entries = node.entries[:mid]
        node2.entries = node.entries[mid:]
        
        # Update LHVs
        node1.lhv = node1.entries[-1].hilbert_value if node1.entries else 0
        node2.lhv = node2.entries[-1].hilbert_value if node2.entries else 0
        
        # Create new root if necessary
        if node == self.root:
            new_root = HilbertRTreeNode(is_leaf=False, capacity=self.max_capacity)
            
            entry1 = HilbertRTreeEntry(None, node1.lhv)
            entry1.child = node1
            
            entry2 = HilbertRTreeEntry(None, node2.lhv)
            entry2.child = node2
            
            new_root.entries = [entry1, entry2]
            new_root.lhv = node2.lhv
            
            self.root = new_root
    
    def range_query(self, min_point, max_point):
        """Find all points in range"""
        results = []
        self._range_query_recursive(self.root, min_point, max_point, results)
        return results
    
    def _range_query_recursive(self, node, min_point, max_point, results):
        """Recursively search for points in range"""
        if node.is_leaf:
            for entry in node.entries:
                if all(min_point[i] <= entry.point[i] <= max_point[i] 
                       for i in range(len(entry.point))):
                    results.append(entry.data)
        else:
            for entry in node.entries:
                if entry.child:
                    self._range_query_recursive(entry.child, min_point, max_point, results)
    
    def nearest_neighbor(self, query_point, k=1):
        """Find k nearest neighbors (simplified)"""
        all_points = []
        self._collect_all_points(self.root, all_points)
        
        # Calculate distances
        distances = []
        for point, data in all_points:
            dist = sum((point[i] - query_point[i])**2 for i in range(len(point))) ** 0.5
            distances.append((dist, data))
        
        distances.sort()
        return [data for _, data in distances[:k]]
    
    def _collect_all_points(self, node, points):
        """Collect all points in tree"""
        if node.is_leaf:
            for entry in node.entries:
                points.append((entry.point, entry.data))
        else:
            for entry in node.entries:
                if entry.child:
                    self._collect_all_points(entry.child, points)
    
    def __len__(self):
        return self.size

# Example usage
print("\nHilbert R-Tree:")

# Create Hilbert R-Tree
hrtree = HilbertRTree(dimensions=2, max_capacity=4)

# Insert 2D points
points = [
    (0.1, 0.2),
    (0.3, 0.4),
    (0.5, 0.6),
    (0.2, 0.3),
    (0.7, 0.8),
    (0.4, 0.5),
    (0.9, 0.1),
    (0.6, 0.7),
]

print("Inserting points with Hilbert curve ordering:")
for point in points:
    hilbert_val = hrtree._point_to_hilbert(point)
    hrtree.insert(point)
    print(f"  {point} -> Hilbert value: {hilbert_val}")

print(f"\nTree size: {len(hrtree)}")

# Range query
min_p = (0.2, 0.2)
max_p = (0.6, 0.6)
print(f"\nRange query [{min_p}, {max_p}]:")
results = hrtree.range_query(min_p, max_p)
print(f"Found {len(results)} points:")
for point in results:
    print(f"  {point}")

# Nearest neighbor
query = (0.5, 0.5)
print(f"\n3 nearest neighbors to {query}:")
neighbors = hrtree.nearest_neighbor(query, k=3)
for i, neighbor in enumerate(neighbors, 1):
    print(f"  {i}. {neighbor}")

# Demonstrate clustering property
print("\nDemonstrating Hilbert curve clustering:")
test_points = [(0.1, 0.1), (0.1, 0.2), (0.9, 0.9), (0.8, 0.9)]
hilbert_values = [(p, hrtree._point_to_hilbert(p)) for p in test_points]
hilbert_values.sort(key=lambda x: x[1])

print("Points sorted by Hilbert value (shows spatial locality):")
for point, hval in hilbert_values:
    print(f"  {point} -> {hval}")
```

KDB-Tree

When: Need multi-dimensional binary space partitioning
Why: Combines k-d tree and B-tree properties
Examples: Spatial databases, multi-dimensional range queries

```python
# Python implementation of KDB-Tree
# Combines K-D Tree and B-Tree properties for spatial indexing

class KDBRegion:
    """Represents a region in multi-dimensional space"""
    def __init__(self, dimensions):
        self.low = [float('-inf')] * dimensions
        self.high = [float('inf')] * dimensions
    
    def contains(self, point):
        """Check if point is in region"""
        for i in range(len(point)):
            if point[i] < self.low[i] or point[i] > self.high[i]:
                return False
        return True
    
    def intersects(self, other):
        """Check if regions intersect"""
        for i in range(len(self.low)):
            if self.high[i] < other.low[i] or self.low[i] > other.high[i]:
                return False
        return True
    
    def split(self, dimension, value):
        """Split region along dimension at value"""
        left = KDBRegion(len(self.low))
        right = KDBRegion(len(self.low))
        
        for i in range(len(self.low)):
            left.low[i] = self.low[i]
            left.high[i] = self.high[i]
            right.low[i] = self.low[i]
            right.high[i] = self.high[i]
        
        left.high[dimension] = value
        right.low[dimension] = value
        
        return left, right

class KDBNode:
    """Node in KDB-Tree"""
    def __init__(self, is_leaf=True, capacity=10):
        self.is_leaf = is_leaf
        self.entries = []  # Points for leaf, children for internal
        self.capacity = capacity
        self.region = None
        self.split_dim = None
        self.split_value = None
        self.left = None
        self.right = None

class KDBTree:
    """
    KDB-Tree: Multi-dimensional spatial index
    Combines K-D Tree partitioning with B-Tree node structure
    """
    def __init__(self, dimensions, max_capacity=10):
        self.dimensions = dimensions
        self.max_capacity = max_capacity
        self.root = KDBNode(is_leaf=True, capacity=max_capacity)
        self.root.region = KDBRegion(dimensions)
        self.size = 0
    
    def insert(self, point, data=None):
        """Insert point into KDB-Tree"""
        if len(point) != self.dimensions:
            raise ValueError(f"Point must have {self.dimensions} dimensions")
        
        entry = (point, data if data else point)
        self._insert_recursive(self.root, entry)
        self.size += 1
    
    def _insert_recursive(self, node, entry):
        """Recursively insert entry"""
        point = entry[0]
        
        if node.is_leaf:
            # Insert into leaf
            node.entries.append(entry)
            
            # Check for overflow
            if len(node.entries) > node.capacity:
                self._split_node(node)
        else:
            # Navigate to appropriate child
            if point[node.split_dim] <= node.split_value:
                if node.left:
                    self._insert_recursive(node.left, entry)
            else:
                if node.right:
                    self._insert_recursive(node.right, entry)
    
    def _split_node(self, node):
        """Split overflowing node"""
        if not node.entries:
            return
        
        # Choose splitting dimension (cycle through dimensions)
        split_dim = self._choose_split_dimension(node)
        
        # Sort entries by split dimension
        node.entries.sort(key=lambda e: e[0][split_dim])
        
        # Find split point (median)
        mid = len(node.entries) // 2
        split_value = node.entries[mid][0][split_dim]
        
        # Create child nodes
        left_node = KDBNode(is_leaf=True, capacity=self.max_capacity)
        right_node = KDBNode(is_leaf=True, capacity=self.max_capacity)
        
        # Split region
        left_region, right_region = node.region.split(split_dim, split_value)
        left_node.region = left_region
        right_node.region = right_region
        
        # Distribute entries
        left_node.entries = node.entries[:mid]
        right_node.entries = node.entries[mid:]
        
        # Convert node to internal
        node.is_leaf = False
        node.entries = []
        node.split_dim = split_dim
        node.split_value = split_value
        node.left = left_node
        node.right = right_node
    
    def _choose_split_dimension(self, node):
        """Choose dimension with maximum spread"""
        if not node.entries:
            return 0
        
        max_spread = -1
        best_dim = 0
        
        for dim in range(self.dimensions):
            values = [entry[0][dim] for entry in node.entries]
            spread = max(values) - min(values)
            
            if spread > max_spread:
                max_spread = spread
                best_dim = dim
        
        return best_dim
    
    def range_query(self, min_point, max_point):
        """Find all points in range [min_point, max_point]"""
        query_region = KDBRegion(self.dimensions)
        query_region.low = list(min_point)
        query_region.high = list(max_point)
        
        results = []
        self._range_query_recursive(self.root, query_region, results)
        return results
    
    def _range_query_recursive(self, node, query_region, results):
        """Recursively search for points in range"""
        # Check if node's region intersects query region
        if not node.region.intersects(query_region):
            return
        
        if node.is_leaf:
            # Check each point
            for point, data in node.entries:
                if query_region.contains(point):
                    results.append(data)
        else:
            # Recurse on children
            if node.left:
                self._range_query_recursive(node.left, query_region, results)
            if node.right:
                self._range_query_recursive(node.right, query_region, results)
    
    def nearest_neighbor(self, query_point, k=1):
        """Find k nearest neighbors (simplified)"""
        candidates = []
        self._collect_all_points(self.root, candidates)
        
        # Calculate distances
        distances = []
        for point, data in candidates:
            dist = sum((point[i] - query_point[i])**2 for i in range(len(point))) ** 0.5
            distances.append((dist, data))
        
        distances.sort()
        return [data for _, data in distances[:k]]
    
    def _collect_all_points(self, node, points):
        """Collect all points in tree"""
        if node.is_leaf:
            points.extend(node.entries)
        else:
            if node.left:
                self._collect_all_points(node.left, points)
            if node.right:
                self._collect_all_points(node.right, points)
    
    def __len__(self):
        return self.size

# Example usage
print("\nKDB-Tree:")

# Create KDB-Tree
kdbtree = KDBTree(dimensions=2, max_capacity=4)

# Insert 2D points
points = [
    (2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2),
    (1, 5), (3, 8), (6, 9), (10, 3), (2, 1), (8, 8)
]

print("Inserting points:")
for point in points:
    kdbtree.insert(point)
    print(f"  Inserted {point}")

print(f"\nTree size: {len(kdbtree)}")

# Range query
min_point = (2, 2)
max_point = (7, 7)
print(f"\nRange query [{min_point}, {max_point}]:")
results = kdbtree.range_query(min_point, max_point)
print(f"Found {len(results)} points:")
for point in results:
    print(f"  {point}")

# Nearest neighbors
query_point = (5, 5)
k = 3
print(f"\n{k} nearest neighbors to {query_point}:")
neighbors = kdbtree.nearest_neighbor(query_point, k)
for i, neighbor in enumerate(neighbors, 1):
    if isinstance(neighbor, tuple):
        dist = sum((neighbor[j] - query_point[j])**2 for j in range(len(query_point))) ** 0.5
        print(f"  {i}. {neighbor} (distance: {dist:.2f})")

# 3D example
print("\n3D KDB-Tree example:")
kdbtree_3d = KDBTree(dimensions=3, max_capacity=4)

points_3d = [
    (1, 2, 3), (4, 5, 6), (7, 8, 9), (2, 3, 4),
    (5, 6, 7), (8, 9, 10), (3, 4, 5), (6, 7, 8)
]

for point in points_3d:
    kdbtree_3d.insert(point)

print(f"Inserted {len(kdbtree_3d)} 3D points")

# 3D range query
min_3d = (2, 3, 4)
max_3d = (6, 7, 8)
results_3d = kdbtree_3d.range_query(min_3d, max_3d)
print(f"\n3D range query [{min_3d}, {max_3d}]:")
print(f"Found {len(results_3d)} points: {results_3d}")
```

Z-order Curve / Morton Order structures

When: Need to map multi-dimensional data to one dimension
Why: Preserves locality, simple computation
Examples: Geographic databases, quadtree linearization

```python
# Python implementation of Z-order Curve (Morton Order)
# Maps multi-dimensional coordinates to 1D while preserving locality

class MortonCode:
    """
    Morton Code / Z-order Curve implementation
    Interleaves bits of coordinates to create space-filling curve
    """
    
    @staticmethod
    def encode_2d(x, y, bits=16):
        """
        Encode 2D coordinates into Morton code
        Interleaves bits: x=x1x0, y=y1y0 -> z=y1x1y0x0
        """
        result = 0
        for i in range(bits):
            result |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
        return result
    
    @staticmethod
    def decode_2d(morton, bits=16):
        """Decode Morton code back to 2D coordinates"""
        x = 0
        y = 0
        for i in range(bits):
            x |= ((morton & (1 << (2 * i))) >> i)
            y |= ((morton & (1 << (2 * i + 1))) >> (i + 1))
        return x, y
    
    @staticmethod
    def encode_3d(x, y, z, bits=16):
        """Encode 3D coordinates into Morton code"""
        result = 0
        for i in range(bits):
            result |= ((x & (1 << i)) << (2 * i)) | \
                      ((y & (1 << i)) << (2 * i + 1)) | \
                      ((z & (1 << i)) << (2 * i + 2))
        return result
    
    @staticmethod
    def decode_3d(morton, bits=16):
        """Decode Morton code back to 3D coordinates"""
        x, y, z = 0, 0, 0
        for i in range(bits):
            x |= ((morton & (1 << (3 * i))) >> (2 * i))
            y |= ((morton & (1 << (3 * i + 1))) >> (2 * i + 1))
            z |= ((morton & (1 << (3 * i + 2))) >> (2 * i + 2))
        return x, y, z

class ZOrderIndex:
    """
    Z-order spatial index using Morton codes
    Efficiently maps 2D space to 1D for indexing
    """
    def __init__(self, bits_per_dimension=16):
        self.bits = bits_per_dimension
        self.data = {}  # morton_code -> data
        self.points = {}  # morton_code -> (x, y)
    
    def insert(self, x, y, data=None):
        """Insert point with Z-order indexing"""
        morton = MortonCode.encode_2d(x, y, self.bits)
        self.data[morton] = data if data else (x, y)
        self.points[morton] = (x, y)
    
    def search(self, x, y):
        """Search for exact point"""
        morton = MortonCode.encode_2d(x, y, self.bits)
        return self.data.get(morton)
    
    def range_query(self, min_x, min_y, max_x, max_y):
        """Find all points in rectangle (simplified)"""
        results = []
        
        # Get Morton range
        min_morton = MortonCode.encode_2d(min_x, min_y, self.bits)
        max_morton = MortonCode.encode_2d(max_x, max_y, self.bits)
        
        # Check all stored points (in practice, use range decomposition)
        for morton, point in self.points.items():
            x, y = point
            if min_x <= x <= max_x and min_y <= y <= max_y:
                results.append(self.data[morton])
        
        return results
    
    def get_neighbors(self, x, y, distance):
        """Get points within Manhattan distance"""
        results = []
        
        for morton, point in self.points.items():
            px, py = point
            if abs(px - x) <= distance and abs(py - y) <= distance:
                results.append(self.data[morton])
        
        return results
    
    def __len__(self):
        return len(self.data)

class QuadTreeLinearized:
    """
    Linearized Quadtree using Z-order curve
    More efficient than pointer-based quadtree
    """
    def __init__(self, depth=4):
        self.depth = depth
        self.max_coord = (1 << depth) - 1  # 2^depth - 1
        self.cells = {}  # morton_code -> list of points
    
    def _get_cell_morton(self, x, y, level):
        """Get Morton code for cell at given level"""
        # Quantize coordinates to cell level
        cell_x = x >> (self.depth - level)
        cell_y = y >> (self.depth - level)
        return MortonCode.encode_2d(cell_x, cell_y, level)
    
    def insert(self, x, y, data=None):
        """Insert point into quadtree"""
        # Insert at leaf level
        morton = MortonCode.encode_2d(x, y, self.depth)
        
        if morton not in self.cells:
            self.cells[morton] = []
        
        self.cells[morton].append(data if data else (x, y))
    
    def query(self, min_x, min_y, max_x, max_y):
        """Range query using Z-order decomposition"""
        results = []
        
        # Decompose query into Z-order ranges (simplified)
        for morton, points in self.cells.items():
            x, y = MortonCode.decode_2d(morton, self.depth)
            if min_x <= x <= max_x and min_y <= y <= max_y:
                results.extend(points)
        
        return results

class ZOrderCurveIterator:
    """
    Iterator that traverses space in Z-order
    Useful for cache-efficient spatial traversal
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.current = 0
        self.max_val = width * height
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.max_val:
            raise StopIteration
        
        # Find corresponding (x, y) for current Z-order index
        bits = max(self.width.bit_length(), self.height.bit_length())
        x, y = MortonCode.decode_2d(self.current, bits)
        
        self.current += 1
        
        # Skip if outside bounds
        while (x >= self.width or y >= self.height) and self.current < self.max_val:
            x, y = MortonCode.decode_2d(self.current, bits)
            self.current += 1
        
        if x < self.width and y < self.height:
            return (x, y)
        else:
            raise StopIteration

# Example usage
print("\nZ-order Curve / Morton Order:")

# Basic Morton code encoding/decoding
print("Morton Code Encoding:")
coords_2d = [(0, 0), (1, 0), (0, 1), (1, 1), (5, 3)]
for x, y in coords_2d:
    morton = MortonCode.encode_2d(x, y, bits=8)
    decoded = MortonCode.decode_2d(morton, bits=8)
    print(f"  ({x}, {y}) -> Morton: {morton:04b} -> Decoded: {decoded}")

# 3D encoding
print("\n3D Morton Code:")
coords_3d = [(1, 2, 3), (4, 5, 6)]
for x, y, z in coords_3d:
    morton = MortonCode.encode_3d(x, y, z, bits=8)
    decoded = MortonCode.decode_3d(morton, bits=8)
    print(f"  ({x}, {y}, {z}) -> Morton: {morton} -> Decoded: {decoded}")

# Z-order Index
print("\nZ-order Spatial Index:")
zindex = ZOrderIndex(bits_per_dimension=8)

points = [(10, 20), (15, 25), (5, 30), (25, 15), (30, 30)]
for x, y in points:
    zindex.insert(x, y, data=f"Point({x},{y})")
    print(f"  Inserted ({x}, {y})")

# Range query
print("\nRange query [5, 15] to [20, 30]:")
results = zindex.range_query(5, 15, 20, 30)
for result in results:
    print(f"  {result}")

# Neighbor search
print("\nNeighbors within distance 10 of (15, 25):")
neighbors = zindex.get_neighbors(15, 25, 10)
for neighbor in neighbors:
    print(f"  {neighbor}")

# Linearized Quadtree
print("\nLinearized Quadtree:")
quad = QuadTreeLinearized(depth=4)

for x, y in [(3, 5), (7, 2), (12, 8), (1, 15)]:
    quad.insert(x, y)
    print(f"  Inserted ({x}, {y})")

results = quad.query(0, 0, 10, 10)
print(f"\nQuery results [0,0] to [10,10]: {results}")

# Z-order traversal
print("\nZ-order traversal of 4x4 grid:")
traversal = []
for coord in ZOrderCurveIterator(4, 4):
    traversal.append(coord)

print("Traversal order:")
for i, (x, y) in enumerate(traversal):
    print(f"  Step {i}: ({x}, {y})")
```

Hash Variants
Extendible Hashing

When: Hash table on disk with dynamic growth
Why: Grows directory instead of full rehashing
Examples: Database indexes, disk-based hash tables

```python
# Python implementation of Extendible Hashing
# Grows dynamically by splitting buckets and doubling directory when needed

class Bucket:
    """
    Bucket stores key-value pairs with a local depth
    """
    def __init__(self, depth, bucket_size=4):
        self.depth = depth  # Local depth
        self.bucket_size = bucket_size
        self.entries = {}  # key -> value
    
    def insert(self, key, value):
        """Insert key-value pair, returns True if successful, False if full"""
        if len(self.entries) < self.bucket_size:
            self.entries[key] = value
            return True
        return False
    
    def remove(self, key):
        """Remove key from bucket"""
        if key in self.entries:
            del self.entries[key]
            return True
        return False
    
    def search(self, key):
        """Search for key in bucket"""
        return self.entries.get(key)
    
    def is_full(self):
        """Check if bucket is full"""
        return len(self.entries) >= self.bucket_size
    
    def is_empty(self):
        """Check if bucket is empty"""
        return len(self.entries) == 0
    
    def __repr__(self):
        return f"Bucket(depth={self.depth}, entries={len(self.entries)}/{self.bucket_size})"

class ExtendibleHashTable:
    """
    Extendible Hash Table with dynamic directory growth
    Directory doubles when needed, buckets split locally
    """
    def __init__(self, bucket_size=4):
        self.global_depth = 0  # Global depth of directory
        self.bucket_size = bucket_size
        self.directory = [Bucket(0, bucket_size)]  # Initially one bucket
    
    def _hash(self, key):
        """Hash function"""
        return hash(key)
    
    def _get_index(self, key):
        """Get directory index using global_depth least significant bits"""
        hash_val = self._hash(key)
        # Use last global_depth bits
        mask = (1 << self.global_depth) - 1
        return hash_val & mask
    
    def _expand_directory(self):
        """Double the directory size"""
        self.global_depth += 1
        # Double directory by duplicating pointers
        self.directory = self.directory + self.directory
    
    def _split_bucket(self, bucket_index):
        """Split a bucket and redistribute entries"""
        old_bucket = self.directory[bucket_index]
        
        # Increase local depth
        new_local_depth = old_bucket.depth + 1
        
        # Create two new buckets with increased depth
        bucket_0 = Bucket(new_local_depth, self.bucket_size)
        bucket_1 = Bucket(new_local_depth, self.bucket_size)
        
        # Redistribute entries based on new bit
        for key, value in old_bucket.entries.items():
            hash_val = self._hash(key)
            # Check the bit at position local_depth
            bit = (hash_val >> old_bucket.depth) & 1
            
            if bit == 0:
                bucket_0.entries[key] = value
            else:
                bucket_1.entries[key] = value
        
        # Update directory pointers
        # Find all directory entries pointing to old bucket
        step = 1 << new_local_depth  # 2^new_local_depth
        
        # Update pointers for bucket_0 and bucket_1
        for i in range(len(self.directory)):
            if self.directory[i] is old_bucket:
                # Check which bucket this index should point to
                bit = (i >> old_bucket.depth) & 1
                if bit == 0:
                    self.directory[i] = bucket_0
                else:
                    self.directory[i] = bucket_1
    
    def insert(self, key, value):
        """Insert key-value pair into hash table"""
        index = self._get_index(key)
        bucket = self.directory[index]
        
        # Try to insert into bucket
        if bucket.insert(key, value):
            return True
        
        # Bucket is full, need to split
        # First check if we need to expand directory
        if bucket.depth == self.global_depth:
            self._expand_directory()
            # Recalculate index after expansion
            index = self._get_index(key)
        
        # Split the bucket
        self._split_bucket(index)
        
        # Retry insertion
        return self.insert(key, value)
    
    def search(self, key):
        """Search for key in hash table"""
        index = self._get_index(key)
        bucket = self.directory[index]
        return bucket.search(key)
    
    def remove(self, key):
        """Remove key from hash table"""
        index = self._get_index(key)
        bucket = self.directory[index]
        return bucket.remove(key)
    
    def __repr__(self):
        unique_buckets = set(id(b) for b in self.directory)
        return f"ExtendibleHashTable(global_depth={self.global_depth}, " \
               f"directory_size={len(self.directory)}, unique_buckets={len(unique_buckets)})"
    
    def print_structure(self):
        """Print the structure of the hash table"""
        print(f"\nExtendible Hash Table Structure:")
        print(f"Global Depth: {self.global_depth}")
        print(f"Directory Size: {len(self.directory)}")
        
        # Track unique buckets
        seen_buckets = set()
        
        for i, bucket in enumerate(self.directory):
            bucket_id = id(bucket)
            prefix = format(i, f'0{self.global_depth}b') if self.global_depth > 0 else '0'
            
            if bucket_id not in seen_buckets:
                print(f"  [{prefix}] -> {bucket} | Keys: {list(bucket.entries.keys())}")
                seen_buckets.add(bucket_id)
            else:
                print(f"  [{prefix}] -> (same as above)")

# Example usage
print("\nExtendible Hashing:")

# Create hash table with small bucket size for demonstration
hash_table = ExtendibleHashTable(bucket_size=2)

print("\nInserting keys...")
keys = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape', 'honeydew']

for key in keys:
    hash_table.insert(key, f"value_{key}")
    print(f"Inserted: {key}")
    print(hash_table)

# Print structure
hash_table.print_structure()

# Search for keys
print("\nSearching for keys:")
for key in ['apple', 'cherry', 'mango']:
    result = hash_table.search(key)
    print(f"  Search '{key}': {result if result else 'Not found'}")

# Remove a key
print("\nRemoving 'cherry':")
hash_table.remove('cherry')
print(f"After removal: {hash_table.search('cherry')}")

hash_table.print_structure()
```

Linear Hashing

When: Need incremental hash table growth
Why: Gradual growth without full rehashing
Examples: Databases (PostgreSQL), dynamic disk structures

```python
# Python implementation of Linear Hashing
# Grows incrementally by splitting one bucket at a time

class LinearHashBucket:
    """Bucket for linear hashing"""
    def __init__(self, bucket_size=4):
        self.bucket_size = bucket_size
        self.entries = {}  # key -> value
        self.overflow = []  # Overflow chain if bucket is full
    
    def insert(self, key, value):
        """Insert key-value pair"""
        # Update if key exists
        if key in self.entries:
            self.entries[key] = value
            return
        
        # Try to insert in main bucket
        if len(self.entries) < self.bucket_size:
            self.entries[key] = value
        else:
            # Use overflow
            # Check if key exists in overflow
            for i, (k, v) in enumerate(self.overflow):
                if k == key:
                    self.overflow[i] = (key, value)
                    return
            self.overflow.append((key, value))
    
    def search(self, key):
        """Search for key"""
        if key in self.entries:
            return self.entries[key]
        
        # Search in overflow
        for k, v in self.overflow:
            if k == key:
                return v
        
        return None
    
    def remove(self, key):
        """Remove key"""
        if key in self.entries:
            del self.entries[key]
            return True
        
        # Search in overflow
        for i, (k, v) in enumerate(self.overflow):
            if k == key:
                self.overflow.pop(i)
                return True
        
        return False
    
    def get_all_entries(self):
        """Get all key-value pairs including overflow"""
        result = list(self.entries.items())
        result.extend(self.overflow)
        return result
    
    def size(self):
        """Total number of entries"""
        return len(self.entries) + len(self.overflow)
    
    def load_factor(self):
        """Current load factor"""
        return self.size() / self.bucket_size

class LinearHashTable:
    """
    Linear Hash Table with incremental growth
    Splits one bucket at a time, maintaining split pointer
    """
    def __init__(self, initial_buckets=4, bucket_size=4, max_load_factor=0.8):
        self.bucket_size = bucket_size
        self.max_load_factor = max_load_factor
        
        # Initialize buckets
        self.buckets = [LinearHashBucket(bucket_size) for _ in range(initial_buckets)]
        
        # Level and split pointer
        self.level = 0  # Current level
        self.next_to_split = 0  # Next bucket to split
        self.n = initial_buckets  # Initial number of buckets at level 0
        
        self.total_entries = 0
    
    def _hash(self, key, level):
        """Hash function for given level: h_level(key) = hash(key) mod (N * 2^level)"""
        return hash(key) % (self.n * (1 << level))
    
    def _get_bucket_index(self, key):
        """Get bucket index for key"""
        # Try hash at current level
        index = self._hash(key, self.level)
        
        # If index is before split pointer, use next level hash
        if index < self.next_to_split:
            index = self._hash(key, self.level + 1)
        
        return index
    
    def _split(self):
        """Split the next bucket"""
        # Get bucket to split
        old_bucket = self.buckets[self.next_to_split]
        
        # Create new bucket
        new_bucket = LinearHashBucket(self.bucket_size)
        self.buckets.append(new_bucket)
        
        # Redistribute entries
        entries = old_bucket.get_all_entries()
        old_bucket.entries = {}
        old_bucket.overflow = []
        
        for key, value in entries:
            # Rehash with level+1 hash
            new_index = self._hash(key, self.level + 1)
            
            if new_index == self.next_to_split:
                old_bucket.insert(key, value)
            else:
                new_bucket.insert(key, value)
        
        # Move split pointer
        self.next_to_split += 1
        
        # Check if we completed this level
        if self.next_to_split >= self.n * (1 << self.level):
            # Move to next level
            self.level += 1
            self.next_to_split = 0
    
    def insert(self, key, value):
        """Insert key-value pair"""
        index = self._get_bucket_index(key)
        bucket = self.buckets[index]
        
        # Check if key already exists
        if bucket.search(key) is not None:
            bucket.insert(key, value)  # Update
        else:
            bucket.insert(key, value)
            self.total_entries += 1
        
        # Check load factor and split if needed
        current_load = self.total_entries / (len(self.buckets) * self.bucket_size)
        if current_load > self.max_load_factor:
            self._split()
    
    def search(self, key):
        """Search for key"""
        index = self._get_bucket_index(key)
        return self.buckets[index].search(key)
    
    def remove(self, key):
        """Remove key"""
        index = self._get_bucket_index(key)
        if self.buckets[index].remove(key):
            self.total_entries -= 1
            return True
        return False
    
    def load_factor(self):
        """Current load factor"""
        return self.total_entries / (len(self.buckets) * self.bucket_size)
    
    def __repr__(self):
        return f"LinearHashTable(buckets={len(self.buckets)}, level={self.level}, " \
               f"split_ptr={self.next_to_split}, load={self.load_factor():.2f})"
    
    def print_structure(self):
        """Print structure of hash table"""
        print(f"\nLinear Hash Table Structure:")
        print(f"Level: {self.level}, Split Pointer: {self.next_to_split}")
        print(f"Buckets: {len(self.buckets)}, Total Entries: {self.total_entries}")
        print(f"Load Factor: {self.load_factor():.2f}")
        
        for i, bucket in enumerate(self.buckets):
            marker = " <- next to split" if i == self.next_to_split else ""
            print(f"  Bucket {i}: {bucket.size()} entries (overflow: {len(bucket.overflow)}){marker}")

# Example usage
print("\nLinear Hashing:")

# Create linear hash table
lh_table = LinearHashTable(initial_buckets=2, bucket_size=2, max_load_factor=0.75)

print("\nInserting keys...")
keys = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape', 'honeydew', 
        'kiwi', 'lemon', 'mango', 'nectarine']

for key in keys:
    lh_table.insert(key, f"value_{key}")
    print(f"Inserted '{key}': {lh_table}")

# Print structure
lh_table.print_structure()

# Search for keys
print("\nSearching:")
for key in ['apple', 'date', 'orange']:
    result = lh_table.search(key)
    print(f"  '{key}': {result if result else 'Not found'}")

# Remove and check structure
print("\nRemoving 'banana' and 'cherry':")
lh_table.remove('banana')
lh_table.remove('cherry')
print(lh_table)
lh_table.print_structure()
```

Dynamic Perfect Hashing

When: Need perfect hashing with dynamic updates
Why: O(1) worst-case with occasional rebuilds
Examples: Symbol tables, static with occasional updates

```python
# Python implementation of Dynamic Perfect Hashing (FKS scheme)
# Two-level hashing with perfect collision resolution

import random

class SecondLevelHash:
    """
    Second level hash table with perfect hashing (no collisions)
    Uses quadratic space to guarantee no collisions
    """
    def __init__(self, keys):
        self.n = len(keys)
        self.m = self.n * self.n  # Quadratic space
        self.table = [None] * self.m if self.m > 0 else []
        self.hash_params = None
        
        if self.n > 0:
            self._build_perfect_hash(keys)
    
    def _universal_hash(self, key, a, b, m):
        """Universal hash function: ((a * hash(key) + b) mod p) mod m"""
        p = 2**31 - 1  # Large prime
        return ((a * hash(key) + b) % p) % m
    
    def _build_perfect_hash(self, keys):
        """Build perfect hash with no collisions"""
        # Try random hash functions until we find one with no collisions
        max_attempts = 100
        
        for attempt in range(max_attempts):
            # Generate random parameters
            a = random.randint(1, 2**31 - 2)
            b = random.randint(0, 2**31 - 2)
            
            # Try to place all keys
            temp_table = [None] * self.m
            collision = False
            
            for key, value in keys:
                index = self._universal_hash(key, a, b, self.m)
                if temp_table[index] is not None:
                    collision = True
                    break
                temp_table[index] = (key, value)
            
            if not collision:
                # Success! No collisions
                self.hash_params = (a, b)
                self.table = temp_table
                return
        
        # Fallback: use chaining if perfect hash fails
        self.hash_params = (1, 0)
        self.table = [[] for _ in range(self.m)]
        for key, value in keys:
            index = self._universal_hash(key, 1, 0, self.m)
            self.table[index].append((key, value))
    
    def search(self, key):
        """Search for key - O(1) worst case"""
        if not self.hash_params or self.m == 0:
            return None
        
        a, b = self.hash_params
        index = self._universal_hash(key, a, b, self.m)
        
        if isinstance(self.table[index], list):
            # Chaining fallback
            for k, v in self.table[index]:
                if k == key:
                    return v
            return None
        else:
            # Perfect hashing
            if self.table[index] and self.table[index][0] == key:
                return self.table[index][1]
            return None

class DynamicPerfectHashTable:
    """
    Dynamic Perfect Hash Table (FKS Scheme)
    First level: O(n) space with universal hashing
    Second level: O(n_i^2) space for bucket i with n_i elements
    """
    def __init__(self, initial_size=16):
        self.n = 0  # Number of elements
        self.m = initial_size  # First level size
        self.first_level = [[] for _ in range(self.m)]  # Lists of (key, value)
        self.second_level = [None] * self.m  # Second level hash tables
        
        # Universal hash parameters for first level
        self.a = random.randint(1, 2**31 - 2)
        self.b = random.randint(0, 2**31 - 2)
        
        self.rebuild_threshold = self.m  # Rebuild when n > m
    
    def _first_level_hash(self, key):
        """First level universal hash"""
        p = 2**31 - 1
        return ((self.a * hash(key) + self.b) % p) % self.m
    
    def _rebuild_second_level(self, bucket_index):
        """Rebuild second level hash for a bucket"""
        bucket = self.first_level[bucket_index]
        if len(bucket) > 0:
            self.second_level[bucket_index] = SecondLevelHash(bucket)
        else:
            self.second_level[bucket_index] = None
    
    def _rebuild(self):
        """Rebuild entire hash table"""
        # Collect all entries
        all_entries = []
        for bucket in self.first_level:
            all_entries.extend(bucket)
        
        # Double the size
        self.m = max(self.m * 2, 16)
        self.first_level = [[] for _ in range(self.m)]
        self.second_level = [None] * self.m
        
        # New hash parameters
        self.a = random.randint(1, 2**31 - 2)
        self.b = random.randint(0, 2**31 - 2)
        self.rebuild_threshold = self.m
        
        # Reinsert all entries
        self.n = 0
        for key, value in all_entries:
            self._insert_without_rebuild(key, value)
    
    def _insert_without_rebuild(self, key, value):
        """Insert without triggering rebuild"""
        index = self._first_level_hash(key)
        
        # Check if key exists
        for i, (k, v) in enumerate(self.first_level[index]):
            if k == key:
                self.first_level[index][i] = (key, value)
                self._rebuild_second_level(index)
                return
        
        # Insert new key
        self.first_level[index].append((key, value))
        self.n += 1
        
        # Rebuild second level for this bucket
        self._rebuild_second_level(index)
    
    def insert(self, key, value):
        """Insert key-value pair - O(1) amortized"""
        # Check if rebuild needed
        if self.n >= self.rebuild_threshold:
            self._rebuild()
        
        self._insert_without_rebuild(key, value)
    
    def search(self, key):
        """Search for key - O(1) worst case"""
        index = self._first_level_hash(key)
        
        if self.second_level[index]:
            return self.second_level[index].search(key)
        
        # Fallback to linear search in first level
        for k, v in self.first_level[index]:
            if k == key:
                return v
        
        return None
    
    def remove(self, key):
        """Remove key - O(1) amortized"""
        index = self._first_level_hash(key)
        
        for i, (k, v) in enumerate(self.first_level[index]):
            if k == key:
                self.first_level[index].pop(i)
                self.n -= 1
                self._rebuild_second_level(index)
                return True
        
        return False
    
    def __repr__(self):
        return f"DynamicPerfectHashTable(n={self.n}, m={self.m}, load={self.n/self.m:.2f})"
    
    def print_structure(self):
        """Print structure"""
        print(f"\nDynamic Perfect Hash Structure:")
        print(f"Elements: {self.n}, First Level Size: {self.m}")
        
        non_empty = 0
        for i, bucket in enumerate(self.first_level):
            if len(bucket) > 0:
                non_empty += 1
                second_size = self.second_level[i].m if self.second_level[i] else 0
                print(f"  Bucket {i}: {len(bucket)} elements, 2nd level size: {second_size}")
        
        print(f"Non-empty buckets: {non_empty}/{self.m}")

# Example usage
print("\nDynamic Perfect Hashing:")

# Create hash table
dph_table = DynamicPerfectHashTable(initial_size=4)

print("\nInserting keys...")
keys = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape', 'honeydew']

for key in keys:
    dph_table.insert(key, f"value_{key}")
    print(f"Inserted '{key}': {dph_table}")

# Print structure
dph_table.print_structure()

# Search
print("\nSearching:")
for key in ['apple', 'date', 'mango']:
    result = dph_table.search(key)
    print(f"  '{key}': {result if result else 'Not found'}")

# Remove
print("\nRemoving 'cherry':")
dph_table.remove('cherry')
print(f"After removal: {dph_table.search('cherry')}")
print(dph_table)
```

Consistent Hashing

When: Distributing data across changing set of servers
Why: Minimal redistribution when servers added/removed
Examples: Distributed caches (Memcached), load balancing, distributed hash tables

```python
# Python implementation of Consistent Hashing
# Uses virtual nodes for better load distribution

import hashlib
import bisect

class ConsistentHash:
    """
    Consistent Hashing implementation with virtual nodes.
    Minimizes redistribution when nodes are added or removed.
    """
    def __init__(self, nodes=None, virtual_nodes=150):
        """
        Initialize consistent hash ring.
        
        Args:
            nodes: List of initial nodes (servers)
            virtual_nodes: Number of virtual nodes per physical node
        """
        self.virtual_nodes = virtual_nodes
        self.ring = {}  # hash -> node mapping
        self.sorted_keys = []  # Sorted hash keys for binary search
        
        if nodes:
            for node in nodes:
                self.add_node(node)
    
    def _hash(self, key):
        """Generate hash value for a key"""
        return int(hashlib.md5(str(key).encode('utf-8')).hexdigest(), 16)
    
    def add_node(self, node):
        """Add a node to the hash ring with virtual nodes"""
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node
            bisect.insort(self.sorted_keys, hash_value)
    
    def remove_node(self, node):
        """Remove a node and its virtual nodes from the ring"""
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            if hash_value in self.ring:
                del self.ring[hash_value]
                self.sorted_keys.remove(hash_value)
    
    def get_node(self, key):
        """Get the node responsible for the given key"""
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # Find the first node clockwise from the hash
        idx = bisect.bisect_right(self.sorted_keys, hash_value)
        
        # Wrap around if necessary
        if idx == len(self.sorted_keys):
            idx = 0
        
        return self.ring[self.sorted_keys[idx]]
    
    def get_nodes(self, key, n=1):
        """Get n distinct nodes for replication"""
        if not self.ring or n <= 0:
            return []
        
        hash_value = self._hash(key)
        nodes = []
        seen_nodes = set()
        
        # Start from the hash position
        idx = bisect.bisect_right(self.sorted_keys, hash_value)
        
        # Collect n distinct physical nodes
        checked = 0
        while len(nodes) < n and checked < len(self.sorted_keys):
            if idx >= len(self.sorted_keys):
                idx = 0
            
            node = self.ring[self.sorted_keys[idx]]
            if node not in seen_nodes:
                nodes.append(node)
                seen_nodes.add(node)
            
            idx += 1
            checked += 1
        
        return nodes
    
    def get_distribution(self, keys):
        """Get distribution of keys across nodes"""
        distribution = {}
        for key in keys:
            node = self.get_node(key)
            distribution[node] = distribution.get(node, 0) + 1
        return distribution

# Example usage
if __name__ == "__main__":
    print("Consistent Hashing:")
    
    # Create hash ring with servers
    servers = ['server1', 'server2', 'server3']
    ch = ConsistentHash(nodes=servers)
    
    # Distribute some keys
    keys = [f'key{i}' for i in range(20)]
    print("\nInitial distribution:")
    distribution = ch.get_distribution(keys)
    for server, count in sorted(distribution.items()):
        print(f"  {server}: {count} keys")
    
    # Add a new server
    print("\nAdding 'server4'...")
    ch.add_node('server4')
    distribution = ch.get_distribution(keys)
    for server, count in sorted(distribution.items()):
        print(f"  {server}: {count} keys")
    
    # Remove a server
    print("\nRemoving 'server2'...")
    ch.remove_node('server2')
    distribution = ch.get_distribution(keys)
    for server, count in sorted(distribution.items()):
        print(f"  {server}: {count} keys")
    
    # Get specific key location
    print("\nKey locations:")
    for key in keys[:5]:
        node = ch.get_node(key)
        print(f"  {key} -> {node}")
    
    # Get replicas for a key
    print("\nReplicas for 'key0':")
    replicas = ch.get_nodes('key0', n=3)
    print(f"  {replicas}")
```

Rendezvous Hashing (HRW)

When: Alternative to consistent hashing
Why: Better load distribution than consistent hashing
Examples: Content delivery networks, distributed storage

```python
# Python implementation of Rendezvous Hashing (Highest Random Weight)
# Provides better load distribution than consistent hashing

import hashlib

class RendezvousHash:
    """
    Rendezvous Hashing (HRW - Highest Random Weight) implementation.
    Each key-node pair gets a weight; key is assigned to highest weight node.
    """
    def __init__(self, nodes=None):
        """
        Initialize rendezvous hash.
        
        Args:
            nodes: List of initial nodes (servers)
        """
        self.nodes = set(nodes) if nodes else set()
    
    def _hash(self, key, node):
        """
        Generate hash value for key-node pair.
        Each key-node combination gets a unique hash.
        """
        combined = f"{key}:{node}"
        return int(hashlib.md5(combined.encode('utf-8')).hexdigest(), 16)
    
    def add_node(self, node):
        """Add a node to the set"""
        self.nodes.add(node)
    
    def remove_node(self, node):
        """Remove a node from the set"""
        self.nodes.discard(node)
    
    def get_node(self, key):
        """
        Get the node with highest weight for the given key.
        O(n) where n is number of nodes.
        """
        if not self.nodes:
            return None
        
        # Find node with maximum hash value for this key
        max_hash = -1
        selected_node = None
        
        for node in self.nodes:
            hash_value = self._hash(key, node)
            if hash_value > max_hash:
                max_hash = hash_value
                selected_node = node
        
        return selected_node
    
    def get_nodes(self, key, n=1):
        """
        Get top n nodes for replication.
        Returns nodes sorted by their weight (hash) for the key.
        """
        if not self.nodes or n <= 0:
            return []
        
        # Calculate hash for each node and sort
        node_hashes = [(node, self._hash(key, node)) for node in self.nodes]
        node_hashes.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n nodes
        return [node for node, _ in node_hashes[:min(n, len(node_hashes))]]
    
    def get_distribution(self, keys):
        """Get distribution of keys across nodes"""
        distribution = {}
        for key in keys:
            node = self.get_node(key)
            if node:
                distribution[node] = distribution.get(node, 0) + 1
        return distribution

# Example usage
if __name__ == "__main__":
    print("Rendezvous Hashing (HRW):")
    
    # Create hash with servers
    servers = ['server1', 'server2', 'server3']
    rh = RendezvousHash(nodes=servers)
    
    # Distribute some keys
    keys = [f'key{i}' for i in range(20)]
    print("\nInitial distribution:")
    distribution = rh.get_distribution(keys)
    for server, count in sorted(distribution.items()):
        print(f"  {server}: {count} keys")
    
    # Add a new server
    print("\nAdding 'server4'...")
    rh.add_node('server4')
    distribution = rh.get_distribution(keys)
    for server, count in sorted(distribution.items()):
        print(f"  {server}: {count} keys")
    
    # Remove a server
    print("\nRemoving 'server2'...")
    rh.remove_node('server2')
    distribution = rh.get_distribution(keys)
    for server, count in sorted(distribution.items()):
        print(f"  {server}: {count} keys")
    
    # Get specific key location
    print("\nKey locations:")
    for key in keys[:5]:
        node = rh.get_node(key)
        print(f"  {key} -> {node}")
    
    # Get replicas for a key
    print("\nTop 3 nodes for 'key0':")
    replicas = rh.get_nodes('key0', n=3)
    print(f"  {replicas}")
    
    # Compare with consistent hashing behavior
    print("\nComparing distributions (20 keys):")
    print(f"  Rendezvous: {rh.get_distribution(keys)}")
```

Graph Structures
Graph Coloring structures

When: Need to represent graph colorings efficiently
Why: Specialized for coloring problems
Examples: Register allocation, scheduling, frequency assignment

```python
# Python implementation of Graph Coloring structures
# Uses greedy coloring with various optimization strategies

class GraphColoring:
    """
    Graph coloring structure with multiple coloring strategies.
    Stores and manages vertex colors for graph coloring problems.
    """
    def __init__(self, num_vertices):
        """
        Initialize graph coloring structure.
        
        Args:
            num_vertices: Number of vertices in the graph
        """
        self.num_vertices = num_vertices
        self.adjacency = [set() for _ in range(num_vertices)]
        self.colors = [-1] * num_vertices  # -1 means uncolored
        self.num_colors_used = 0
    
    def add_edge(self, u, v):
        """Add an undirected edge between vertices u and v"""
        if 0 <= u < self.num_vertices and 0 <= v < self.num_vertices:
            self.adjacency[u].add(v)
            self.adjacency[v].add(u)
    
    def greedy_coloring(self):
        """
        Basic greedy coloring algorithm.
        Color vertices one by one, assigning smallest available color.
        Time: O(V + E), Space: O(V)
        """
        self.colors = [-1] * self.num_vertices
        
        for vertex in range(self.num_vertices):
            # Find colors used by neighbors
            neighbor_colors = set()
            for neighbor in self.adjacency[vertex]:
                if self.colors[neighbor] != -1:
                    neighbor_colors.add(self.colors[neighbor])
            
            # Assign smallest color not used by neighbors
            color = 0
            while color in neighbor_colors:
                color += 1
            
            self.colors[vertex] = color
        
        self.num_colors_used = max(self.colors) + 1 if self.colors else 0
        return self.colors
    
    def welsh_powell_coloring(self):
        """
        Welsh-Powell algorithm: color vertices in decreasing degree order.
        Often produces better coloring than basic greedy.
        """
        # Sort vertices by degree (descending)
        vertices_by_degree = sorted(
            range(self.num_vertices),
            key=lambda v: len(self.adjacency[v]),
            reverse=True
        )
        
        self.colors = [-1] * self.num_vertices
        
        for vertex in vertices_by_degree:
            # Find colors used by neighbors
            neighbor_colors = set()
            for neighbor in self.adjacency[vertex]:
                if self.colors[neighbor] != -1:
                    neighbor_colors.add(self.colors[neighbor])
            
            # Assign smallest available color
            color = 0
            while color in neighbor_colors:
                color += 1
            
            self.colors[vertex] = color
        
        self.num_colors_used = max(self.colors) + 1 if self.colors else 0
        return self.colors
    
    def dsatur_coloring(self):
        """
        DSATUR (Degree of Saturation) algorithm.
        Colors vertex with highest saturation degree first.
        Often produces optimal or near-optimal colorings.
        """
        self.colors = [-1] * self.num_vertices
        colored_count = 0
        
        while colored_count < self.num_vertices:
            # Find uncolored vertex with highest saturation degree
            # Saturation = number of different colors used by neighbors
            max_saturation = -1
            max_degree = -1
            selected_vertex = -1
            
            for vertex in range(self.num_vertices):
                if self.colors[vertex] == -1:  # Uncolored
                    # Calculate saturation degree
                    neighbor_colors = set()
                    for neighbor in self.adjacency[vertex]:
                        if self.colors[neighbor] != -1:
                            neighbor_colors.add(self.colors[neighbor])
                    
                    saturation = len(neighbor_colors)
                    degree = len(self.adjacency[vertex])
                    
                    # Select vertex with highest saturation, break ties by degree
                    if (saturation > max_saturation or 
                        (saturation == max_saturation and degree > max_degree)):
                        max_saturation = saturation
                        max_degree = degree
                        selected_vertex = vertex
            
            # Color the selected vertex
            if selected_vertex != -1:
                neighbor_colors = set()
                for neighbor in self.adjacency[selected_vertex]:
                    if self.colors[neighbor] != -1:
                        neighbor_colors.add(self.colors[neighbor])
                
                color = 0
                while color in neighbor_colors:
                    color += 1
                
                self.colors[selected_vertex] = color
                colored_count += 1
        
        self.num_colors_used = max(self.colors) + 1 if self.colors else 0
        return self.colors
    
    def get_color(self, vertex):
        """Get the color of a vertex"""
        if 0 <= vertex < self.num_vertices:
            return self.colors[vertex]
        return -1
    
    def is_valid_coloring(self):
        """Check if current coloring is valid (no adjacent vertices have same color)"""
        for vertex in range(self.num_vertices):
            vertex_color = self.colors[vertex]
            if vertex_color == -1:
                return False  # Uncolored vertex
            
            for neighbor in self.adjacency[vertex]:
                if self.colors[neighbor] == vertex_color:
                    return False  # Adjacent vertices with same color
        
        return True
    
    def get_chromatic_number(self):
        """Get the number of colors used"""
        return self.num_colors_used
    
    def __str__(self):
        """String representation of the coloring"""
        result = []
        for vertex in range(self.num_vertices):
            result.append(f"Vertex {vertex}: Color {self.colors[vertex]}")
        result.append(f"Total colors used: {self.num_colors_used}")
        return "\n".join(result)

# Example usage
if __name__ == "__main__":
    print("Graph Coloring:")
    
    # Create a graph (Petersen graph - requires 3 colors)
    gc = GraphColoring(10)
    
    # Outer pentagon
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer cycle
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),  # Inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Connecting edges
    ]
    
    for u, v in edges:
        gc.add_edge(u, v)
    
    # Try greedy coloring
    print("\nGreedy Coloring:")
    gc.greedy_coloring()
    print(f"Colors used: {gc.get_chromatic_number()}")
    print(f"Valid: {gc.is_valid_coloring()}")
    
    # Try Welsh-Powell
    print("\nWelsh-Powell Coloring:")
    gc.welsh_powell_coloring()
    print(f"Colors used: {gc.get_chromatic_number()}")
    print(f"Valid: {gc.is_valid_coloring()}")
    
    # Try DSATUR
    print("\nDSATUR Coloring:")
    gc.dsatur_coloring()
    print(f"Colors used: {gc.get_chromatic_number()}")
    print(f"Valid: {gc.is_valid_coloring()}")
    print(f"\n{gc}")
```

Chord (DHT structure)

When: Building distributed hash table
Why: O(log n) lookup in peer-to-peer network
Examples: P2P networks, distributed storage systems

```python
# Python implementation of Chord DHT (Distributed Hash Table)
# Simplified version showing key concepts

import hashlib

class ChordNode:
    """
    A node in the Chord DHT ring.
    Each node has an ID and maintains a finger table for efficient lookup.
    """
    def __init__(self, node_id, m=8):
        """
        Initialize a Chord node.
        
        Args:
            node_id: Identifier for this node (0 to 2^m - 1)
            m: Number of bits in the identifier space (ring size = 2^m)
        """
        self.id = node_id
        self.m = m
        self.ring_size = 2 ** m
        self.successor = None  # Next node in ring
        self.predecessor = None  # Previous node in ring
        self.finger_table = [None] * m  # Finger table for routing
        self.data = {}  # Key-value storage
    
    def _in_range(self, key, start, end, inclusive_start=False, inclusive_end=False):
        """Check if key is in range (start, end) on circular ring"""
        if start == end:
            return inclusive_start or inclusive_end
        
        if start < end:
            if inclusive_start and inclusive_end:
                return start <= key <= end
            elif inclusive_start:
                return start <= key < end
            elif inclusive_end:
                return start < key <= end
            else:
                return start < key < end
        else:  # Wraps around
            if inclusive_start and inclusive_end:
                return key >= start or key <= end
            elif inclusive_start:
                return key >= start or key < end
            elif inclusive_end:
                return key > start or key <= end
            else:
                return key > start or key < end
    
    def find_successor(self, key):
        """
        Find the node responsible for the given key.
        Uses finger table for O(log n) lookup.
        """
        # If key is between this node and successor, return successor
        if self._in_range(key, self.id, self.successor.id, 
                         inclusive_end=True):
            return self.successor
        
        # Otherwise, find closest preceding node and ask it
        node = self.closest_preceding_node(key)
        if node == self:
            return self.successor
        return node.find_successor(key)
    
    def closest_preceding_node(self, key):
        """Find the closest node preceding the key in finger table"""
        # Search finger table from end to start
        for i in range(self.m - 1, -1, -1):
            if self.finger_table[i] is not None:
                if self._in_range(self.finger_table[i].id, self.id, key):
                    return self.finger_table[i]
        return self
    
    def update_finger_table(self, nodes_dict):
        """
        Update finger table entries.
        finger[i] points to successor of (n + 2^i) mod 2^m
        """
        for i in range(self.m):
            start = (self.id + 2**i) % self.ring_size
            self.finger_table[i] = self._find_successor_in_dict(start, nodes_dict)
    
    def _find_successor_in_dict(self, key, nodes_dict):
        """Helper to find successor from a dictionary of nodes"""
        if not nodes_dict:
            return self
        
        # Find the first node >= key
        sorted_ids = sorted(nodes_dict.keys())
        for node_id in sorted_ids:
            if node_id >= key:
                return nodes_dict[node_id]
        
        # Wrap around to first node
        return nodes_dict[sorted_ids[0]]
    
    def store(self, key, value):
        """Store a key-value pair"""
        self.data[key] = value
    
    def retrieve(self, key):
        """Retrieve a value by key"""
        return self.data.get(key)
    
    def __repr__(self):
        return f"ChordNode({self.id})"

class ChordDHT:
    """
    Chord Distributed Hash Table implementation.
    Maintains a ring of nodes with O(log n) lookup.
    """
    def __init__(self, m=8):
        """
        Initialize Chord DHT.
        
        Args:
            m: Number of bits for identifier space
        """
        self.m = m
        self.ring_size = 2 ** m
        self.nodes = {}  # node_id -> ChordNode
    
    def _hash(self, key):
        """Hash a key to the identifier space"""
        hash_val = int(hashlib.md5(str(key).encode()).hexdigest(), 16)
        return hash_val % self.ring_size
    
    def add_node(self, node_id):
        """Add a new node to the ring"""
        if node_id in self.nodes:
            return False
        
        new_node = ChordNode(node_id, self.m)
        self.nodes[node_id] = new_node
        
        # Update ring structure
        self._update_ring_structure()
        return True
    
    def remove_node(self, node_id):
        """Remove a node from the ring"""
        if node_id not in self.nodes:
            return False
        
        # Transfer data to successor before removing
        node = self.nodes[node_id]
        if node.successor and node.successor != node:
            for key, value in node.data.items():
                node.successor.store(key, value)
        
        del self.nodes[node_id]
        self._update_ring_structure()
        return True
    
    def _update_ring_structure(self):
        """Update successor/predecessor pointers and finger tables"""
        if not self.nodes:
            return
        
        sorted_ids = sorted(self.nodes.keys())
        n = len(sorted_ids)
        
        # Update successor and predecessor for each node
        for i, node_id in enumerate(sorted_ids):
            node = self.nodes[node_id]
            node.successor = self.nodes[sorted_ids[(i + 1) % n]]
            node.predecessor = self.nodes[sorted_ids[(i - 1) % n]]
        
        # Update finger tables
        for node in self.nodes.values():
            node.update_finger_table(self.nodes)
    
    def lookup(self, key):
        """Look up which node is responsible for a key"""
        if not self.nodes:
            return None
        
        hashed_key = self._hash(key)
        
        # Start from any node (use first node)
        start_node = next(iter(self.nodes.values()))
        responsible_node = start_node.find_successor(hashed_key)
        
        return responsible_node
    
    def put(self, key, value):
        """Store a key-value pair in the DHT"""
        node = self.lookup(key)
        if node:
            hashed_key = self._hash(key)
            node.store(hashed_key, value)
            return True
        return False
    
    def get(self, key):
        """Retrieve a value from the DHT"""
        node = self.lookup(key)
        if node:
            hashed_key = self._hash(key)
            return node.retrieve(hashed_key)
        return None
    
    def __str__(self):
        """String representation of the Chord ring"""
        if not self.nodes:
            return "Empty Chord ring"
        
        sorted_ids = sorted(self.nodes.keys())
        result = [f"Chord DHT (m={self.m}, {len(self.nodes)} nodes):"]
        for node_id in sorted_ids:
            node = self.nodes[node_id]
            fingers = [f.id if f else None for f in node.finger_table[:3]]
            result.append(
                f"  Node {node_id}: successor={node.successor.id}, "
                f"fingers={fingers}..."
            )
        return "\n".join(result)

# Example usage
if __name__ == "__main__":
    print("Chord DHT:")
    
    # Create a Chord ring with m=4 (16 positions)
    chord = ChordDHT(m=4)
    
    # Add nodes
    node_ids = [0, 4, 8, 12]
    print(f"\nAdding nodes: {node_ids}")
    for node_id in node_ids:
        chord.add_node(node_id)
    
    print(chord)
    
    # Store some key-value pairs
    print("\nStoring data:")
    data = [('apple', 1), ('banana', 2), ('cherry', 3), ('date', 4)]
    for key, value in data:
        chord.put(key, value)
        node = chord.lookup(key)
        print(f"  '{key}' -> Node {node.id}")
    
    # Retrieve data
    print("\nRetrieving data:")
    for key, expected_value in data:
        value = chord.get(key)
        print(f"  '{key}': {value} (expected {expected_value})")
    
    # Add a new node and see redistribution
    print("\nAdding node 6:")
    chord.add_node(6)
    print(chord)
```

Graph Spanners

When: Need sparse subgraph approximating distances
Why: Fewer edges while preserving approximate distances
Examples: Network design, route planning with fewer edges

```python
# Python implementation of Graph Spanners
# Creates sparse subgraphs that approximate distances

import heapq
from collections import defaultdict

class GraphSpanner:
    """
    Graph Spanner construction and management.
    A t-spanner preserves distances within a factor of t using fewer edges.
    """
    def __init__(self, num_vertices):
        """
        Initialize graph spanner structure.
        
        Args:
            num_vertices: Number of vertices in the original graph
        """
        self.num_vertices = num_vertices
        self.original_edges = []  # (u, v, weight)
        self.spanner_edges = []  # Edges in the spanner
        self.adjacency = defaultdict(dict)  # adjacency[u][v] = weight
    
    def add_original_edge(self, u, v, weight=1):
        """Add an edge to the original graph"""
        self.original_edges.append((u, v, weight))
        self.adjacency[u][v] = weight
        self.adjacency[v][u] = weight
    
    def greedy_spanner(self, stretch_factor):
        """
        Construct a greedy t-spanner.
        Includes edge (u,v) only if distance in current spanner > t * weight(u,v)
        
        Args:
            stretch_factor: The stretch factor t
        
        Returns:
            List of edges in the spanner
        """
        self.spanner_edges = []
        spanner_adj = defaultdict(dict)
        
        # Sort edges by weight
        sorted_edges = sorted(self.original_edges, key=lambda e: e[2])
        
        for u, v, weight in sorted_edges:
            # Check if u and v are already close enough in spanner
            dist = self._dijkstra_distance(u, v, spanner_adj)
            
            if dist > stretch_factor * weight:
                # Add edge to spanner
                self.spanner_edges.append((u, v, weight))
                spanner_adj[u][v] = weight
                spanner_adj[v][u] = weight
        
        return self.spanner_edges
    
    def _dijkstra_distance(self, start, end, adjacency):
        """
        Calculate shortest distance from start to end using Dijkstra's algorithm.
        Returns float('inf') if no path exists.
        """
        if start == end:
            return 0
        
        if not adjacency:
            return float('inf')
        
        distances = {start: 0}
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_dist, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            
            visited.add(u)
            
            if u == end:
                return current_dist
            
            for v, weight in adjacency[u].items():
                if v not in visited:
                    new_dist = current_dist + weight
                    if v not in distances or new_dist < distances[v]:
                        distances[v] = new_dist
                        heapq.heappush(pq, (new_dist, v))
        
        return float('inf')
    
    def baswana_sen_spanner(self, stretch_factor):
        """
        Baswana-Sen algorithm for (2k-1)-spanner construction.
        More efficient than greedy for certain stretch factors.
        
        Args:
            stretch_factor: Must be odd (2k-1)
        
        Returns:
            List of edges in the spanner
        """
        if stretch_factor % 2 == 0:
            raise ValueError("Baswana-Sen requires odd stretch factor (2k-1)")
        
        k = (stretch_factor + 1) // 2
        self.spanner_edges = []
        
        # Build adjacency list from original edges
        adj = defaultdict(set)
        for u, v, weight in self.original_edges:
            adj[u].add((v, weight))
            adj[v].add((u, weight))
        
        # Initialize clusters - each vertex in its own cluster
        clusters = {v: {v} for v in range(self.num_vertices)}
        cluster_id = {v: v for v in range(self.num_vertices)}
        
        # Iteratively build spanner
        for i in range(k):
            # Sample vertices
            sampled = set()
            for v in range(self.num_vertices):
                # Sample with probability n^(-1/k)
                if hash((v, i)) % (self.num_vertices ** (1/k) + 1) == 0:
                    sampled.add(v)
            
            # Add edges to sampled vertices
            for u in range(self.num_vertices):
                for v, weight in adj[u]:
                    if cluster_id[u] != cluster_id[v]:
                        # Different clusters - add edge
                        if u < v:  # Avoid duplicates
                            self.spanner_edges.append((u, v, weight))
                        
                        # Merge clusters if one endpoint is sampled
                        if u in sampled or v in sampled:
                            # Merge clusters
                            old_cluster = cluster_id[v]
                            new_cluster = cluster_id[u]
                            for vertex in clusters[old_cluster]:
                                cluster_id[vertex] = new_cluster
                                clusters[new_cluster].add(vertex)
        
        return self.spanner_edges
    
    def get_stretch(self):
        """
        Calculate the actual stretch factor of the current spanner.
        Compares distances in original graph vs spanner.
        """
        if not self.spanner_edges:
            return float('inf')
        
        # Build spanner adjacency
        spanner_adj = defaultdict(dict)
        for u, v, weight in self.spanner_edges:
            spanner_adj[u][v] = weight
            spanner_adj[v][u] = weight
        
        max_stretch = 1.0
        
        # Sample pairs to check stretch
        for u, v, orig_weight in self.original_edges[:min(20, len(self.original_edges))]:
            orig_dist = self._dijkstra_distance(u, v, self.adjacency)
            spanner_dist = self._dijkstra_distance(u, v, spanner_adj)
            
            if orig_dist > 0:
                stretch = spanner_dist / orig_dist
                max_stretch = max(max_stretch, stretch)
        
        return max_stretch
    
    def get_spanner_stats(self):
        """Get statistics about the spanner"""
        return {
            'original_edges': len(self.original_edges),
            'spanner_edges': len(self.spanner_edges),
            'reduction': 1 - len(self.spanner_edges) / len(self.original_edges) if self.original_edges else 0,
            'stretch': self.get_stretch()
        }

# Example usage
if __name__ == "__main__":
    print("Graph Spanners:")
    
    # Create a graph
    gs = GraphSpanner(6)
    
    # Add edges (complete graph on 6 vertices)
    edges = [
        (0, 1, 1), (0, 2, 2), (0, 3, 3), (0, 4, 4), (0, 5, 5),
        (1, 2, 1), (1, 3, 2), (1, 4, 3), (1, 5, 4),
        (2, 3, 1), (2, 4, 2), (2, 5, 3),
        (3, 4, 1), (3, 5, 2),
        (4, 5, 1)
    ]
    
    for u, v, w in edges:
        gs.add_original_edge(u, v, w)
    
    print(f"Original graph: {len(edges)} edges")
    
    # Construct 3-spanner using greedy algorithm
    print("\nConstructing 3-spanner (greedy):")
    spanner = gs.greedy_spanner(stretch_factor=3)
    stats = gs.get_spanner_stats()
    
    print(f"  Spanner edges: {len(spanner)}")
    print(f"  Edge reduction: {stats['reduction']*100:.1f}%")
    print(f"  Actual stretch: {stats['stretch']:.2f}")
    print(f"  Edges: {spanner[:5]}...")  # Show first 5 edges
```

Cactus Graph representations

When: Representing minimum cuts or cycle structures
Why: Efficient for specific graph properties
Examples: Network reliability, cut problems

```python
# Python implementation of Cactus Graph representation
# A cactus graph is where every edge belongs to at most one cycle

from collections import defaultdict, deque

class CactusGraph:
    """
    Cactus Graph representation.
    In a cactus graph, every edge belongs to at most one simple cycle.
    Useful for representing minimum cuts and cycle structures.
    """
    def __init__(self, num_vertices):
        """
        Initialize cactus graph.
        
        Args:
            num_vertices: Number of vertices
        """
        self.num_vertices = num_vertices
        self.adjacency = defaultdict(list)  # u -> [(v, edge_id), ...]
        self.edges = {}  # edge_id -> (u, v)
        self.edge_counter = 0
        self.cycles = []  # List of cycles (list of vertices)
        self.is_cactus = True
    
    def add_edge(self, u, v):
        """
        Add an edge to the cactus graph.
        Returns edge_id or None if adding would violate cactus property.
        """
        edge_id = self.edge_counter
        self.edge_counter += 1
        
        self.edges[edge_id] = (u, v)
        self.adjacency[u].append((v, edge_id))
        self.adjacency[v].append((u, edge_id))
        
        # Check if still a cactus after adding edge
        if not self._verify_cactus():
            # Remove the edge
            self.adjacency[u] = [(x, eid) for x, eid in self.adjacency[u] if eid != edge_id]
            self.adjacency[v] = [(x, eid) for x, eid in self.adjacency[v] if eid != edge_id]
            del self.edges[edge_id]
            self.edge_counter -= 1
            self.is_cactus = False
            return None
        
        return edge_id
    
    def _verify_cactus(self):
        """
        Verify that the graph is still a cactus.
        Uses DFS to find cycles and check that edges don't belong to multiple cycles.
        """
        visited = set()
        parent = {}
        edge_to_cycle = {}  # edge_id -> cycle_id
        
        def dfs(v, p_edge=-1):
            visited.add(v)
            
            for u, edge_id in self.adjacency[v]:
                if edge_id == p_edge:
                    continue
                
                if u in visited:
                    # Found a back edge - check if it creates a new cycle
                    # Trace back to find the cycle
                    cycle = []
                    current = v
                    
                    # Walk back to u
                    while current != u:
                        cycle.append(current)
                        # Find edge to parent
                        if current in parent:
                            current = parent[current]
                        else:
                            break
                    
                    cycle.append(u)
                    
                    # Check if any edge in this cycle is already in another cycle
                    for i in range(len(cycle)):
                        v1, v2 = cycle[i], cycle[(i + 1) % len(cycle)]
                        
                        # Find edge between v1 and v2
                        for neighbor, eid in self.adjacency[v1]:
                            if neighbor == v2:
                                if eid in edge_to_cycle and edge_to_cycle[eid] != len(self.cycles):
                                    return False  # Edge in multiple cycles!
                                edge_to_cycle[eid] = len(self.cycles)
                    
                    self.cycles.append(cycle)
                else:
                    parent[u] = v
                    if not dfs(u, edge_id):
                        return False
            
            return True
        
        # Clear previous cycle information
        self.cycles = []
        
        # Run DFS from all unvisited vertices
        for v in range(self.num_vertices):
            if v not in visited and v in self.adjacency:
                if not dfs(v):
                    return False
        
        return True
    
    def find_all_cycles(self):
        """Find all cycles in the cactus graph"""
        self._verify_cactus()
        return self.cycles
    
    def get_cycle_for_edge(self, u, v):
        """Find which cycle (if any) contains the edge (u, v)"""
        # Find the edge_id
        edge_id = None
        for neighbor, eid in self.adjacency[u]:
            if neighbor == v:
                edge_id = eid
                break
        
        if edge_id is None:
            return None
        
        # Check all cycles
        self._verify_cactus()
        for cycle in self.cycles:
            for i in range(len(cycle)):
                v1, v2 = cycle[i], cycle[(i + 1) % len(cycle)]
                if (v1 == u and v2 == v) or (v1 == v and v2 == u):
                    return cycle
        
        return None
    
    def is_tree_edge(self, u, v):
        """Check if edge (u, v) is a tree edge (not in any cycle)"""
        return self.get_cycle_for_edge(u, v) is None
    
    def get_cycle_edges(self):
        """Get all edges that belong to cycles"""
        cycle_edges = set()
        self._verify_cactus()
        
        for cycle in self.cycles:
            for i in range(len(cycle)):
                v1, v2 = cycle[i], cycle[(i + 1) % len(cycle)]
                edge = tuple(sorted([v1, v2]))
                cycle_edges.add(edge)
        
        return cycle_edges
    
    def contract_cycle(self, cycle):
        """
        Contract a cycle to a single vertex (useful for minimum cut algorithms).
        Returns the new super-vertex id.
        """
        if not cycle or len(cycle) < 3:
            return None
        
        # Create new super-vertex
        super_vertex = self.num_vertices
        self.num_vertices += 1
        
        # Collect external edges
        external_edges = []
        for v in cycle:
            for u, edge_id in self.adjacency[v]:
                if u not in cycle:
                    external_edges.append((v, u, edge_id))
        
        # Remove cycle vertices
        for v in cycle:
            del self.adjacency[v]
        
        # Add edges from super-vertex to external vertices
        for v, u, old_edge_id in external_edges:
            self.add_edge(super_vertex, u)
        
        return super_vertex
    
    def __str__(self):
        """String representation"""
        result = [f"Cactus Graph ({self.num_vertices} vertices, {len(self.edges)} edges)"]
        result.append(f"Is valid cactus: {self.is_cactus}")
        result.append(f"Number of cycles: {len(self.cycles)}")
        
        if self.cycles:
            result.append("Cycles:")
            for i, cycle in enumerate(self.cycles):
                result.append(f"  Cycle {i}: {cycle}")
        
        return "\n".join(result)

# Example usage
if __name__ == "__main__":
    print("Cactus Graph:")
    
    # Create a cactus graph
    cg = CactusGraph(8)
    
    # Add edges forming a cactus structure
    # Tree edges
    cg.add_edge(0, 1)
    cg.add_edge(0, 2)
    
    # First cycle: 1-3-4-1
    cg.add_edge(1, 3)
    cg.add_edge(3, 4)
    cg.add_edge(4, 1)
    
    # Second cycle: 2-5-6-2
    cg.add_edge(2, 5)
    cg.add_edge(5, 6)
    cg.add_edge(6, 2)
    
    # Tree edge
    cg.add_edge(4, 7)
    
    print(cg)
    
    # Try to add an edge that would violate cactus property
    print("\nTrying to add edge that creates overlapping cycles...")
    result = cg.add_edge(3, 4)  # This edge already exists in a cycle
    print(f"Edge added: {result is not None}")
    
    # Get cycle information
    print("\nCycle edges:")
    cycle_edges = cg.get_cycle_edges()
    for edge in sorted(cycle_edges):
        print(f"  {edge}")
    
    # Check if specific edges are tree edges
    print("\nTree edges:")
    for u, v in [(0, 1), (0, 2), (4, 7)]:
        is_tree = cg.is_tree_edge(u, v)
        print(f"  ({u}, {v}): {'tree edge' if is_tree else 'cycle edge'}")
```

Miscellaneous Advanced
Y-fast Trie

When: Successor/predecessor queries on integers
Why: O(log log u) operations, better than van Emde Boas for some operations
Examples: Integer operations, faster than balanced BST for small universes

```python
# Python implementation of Y-fast Trie
# Combines X-fast trie with balanced BSTs for efficient integer operations

import bisect
from collections import defaultdict

class YFastTrie:
    """
    Y-fast Trie for efficient successor/predecessor queries on integers.
    Achieves O(log log u) time for operations where u is the universe size.
    Uses an X-fast trie for sampling + balanced trees for dense storage.
    """
    def __init__(self, universe_bits=32):
        """
        Initialize Y-fast trie.
        
        Args:
            universe_bits: Number of bits for the universe (u = 2^universe_bits)
        """
        self.universe_bits = universe_bits
        self.universe_size = 2 ** universe_bits
        
        # Representative elements (sampled every log u elements)
        self.sample_size = max(1, universe_bits)
        self.representatives = {}  # Maps representative -> BST of elements
        self.element_to_rep = {}  # Maps element -> its representative
        
        # Simplified X-fast trie structure for representatives
        self.rep_tree = {}  # Binary trie of representatives
        
        self.size = 0
    
    def _get_representative(self, key):
        """Find the representative for a key"""
        # In a full implementation, this would use the X-fast trie
        # For simplicity, we use a dictionary lookup
        if not self.representatives:
            return None
        
        # Find the representative <= key
        candidates = [r for r in self.representatives.keys() if r <= key]
        if candidates:
            return max(candidates)
        return None
    
    def _find_bucket(self, key):
        """Find which bucket (BST) should contain the key"""
        if not self.representatives:
            return None
        
        # Find closest representative
        reps = sorted(self.representatives.keys())
        idx = bisect.bisect_left(reps, key)
        
        if idx < len(reps):
            # Check if key would go in this bucket or previous
            if idx > 0:
                # Check which bucket is closer
                if key - reps[idx - 1] < reps[idx] - key:
                    return reps[idx - 1]
            return reps[idx]
        elif idx > 0:
            return reps[idx - 1]
        
        return None
    
    def insert(self, key):
        """
        Insert a key into the Y-fast trie.
        O(log log u) amortized time.
        """
        if key < 0 or key >= self.universe_size:
            raise ValueError(f"Key must be in range [0, {self.universe_size})")
        
        if key in self.element_to_rep:
            return False  # Already exists
        
        # Find appropriate bucket
        rep = self._find_bucket(key)
        
        if rep is None:
            # First element or need new representative
            rep = key
            self.representatives[rep] = [key]
            self.element_to_rep[key] = rep
        else:
            # Add to existing bucket
            bucket = self.representatives[rep]
            bisect.insort(bucket, key)
            self.element_to_rep[key] = rep
            
            # Split bucket if it gets too large (> 2 * log u)
            if len(bucket) > 2 * self.sample_size:
                self._split_bucket(rep)
        
        self.size += 1
        return True
    
    def _split_bucket(self, rep):
        """Split a bucket that has grown too large"""
        bucket = self.representatives[rep]
        mid = len(bucket) // 2
        
        # Create new bucket with second half
        new_rep = bucket[mid]
        new_bucket = bucket[mid:]
        old_bucket = bucket[:mid]
        
        # Update representatives
        self.representatives[rep] = old_bucket
        self.representatives[new_rep] = new_bucket
        
        # Update element mappings
        for key in new_bucket:
            self.element_to_rep[key] = new_rep
    
    def delete(self, key):
        """
        Delete a key from the Y-fast trie.
        O(log log u) amortized time.
        """
        if key not in self.element_to_rep:
            return False
        
        rep = self.element_to_rep[key]
        bucket = self.representatives[rep]
        
        # Remove from bucket
        bucket.remove(key)
        del self.element_to_rep[key]
        
        # If bucket becomes empty, remove representative
        if not bucket:
            del self.representatives[rep]
        # If bucket becomes too small, merge with neighbor
        elif len(bucket) < self.sample_size // 2:
            self._merge_buckets(rep)
        
        self.size -= 1
        return True
    
    def _merge_buckets(self, rep):
        """Merge a small bucket with a neighbor"""
        reps = sorted(self.representatives.keys())
        idx = reps.index(rep)
        
        if idx > 0:
            # Merge with previous bucket
            prev_rep = reps[idx - 1]
            prev_bucket = self.representatives[prev_rep]
            curr_bucket = self.representatives[rep]
            
            # Merge buckets
            prev_bucket.extend(curr_bucket)
            prev_bucket.sort()
            
            # Update element mappings
            for key in curr_bucket:
                self.element_to_rep[key] = prev_rep
            
            # Remove current bucket
            del self.representatives[rep]
    
    def successor(self, key):
        """
        Find the smallest element >= key.
        O(log log u) time.
        """
        # Find bucket that might contain successor
        rep = self._find_bucket(key)
        
        if rep is None:
            return None
        
        # Search in the bucket
        bucket = self.representatives[rep]
        idx = bisect.bisect_left(bucket, key)
        
        if idx < len(bucket):
            return bucket[idx]
        
        # Need to check next bucket
        reps = sorted(self.representatives.keys())
        rep_idx = reps.index(rep)
        
        if rep_idx + 1 < len(reps):
            next_rep = reps[rep_idx + 1]
            next_bucket = self.representatives[next_rep]
            if next_bucket:
                return next_bucket[0]
        
        return None
    
    def predecessor(self, key):
        """
        Find the largest element <= key.
        O(log log u) time.
        """
        # Find bucket that might contain predecessor
        rep = self._find_bucket(key)
        
        if rep is None:
            return None
        
        # Search in the bucket
        bucket = self.representatives[rep]
        idx = bisect.bisect_right(bucket, key)
        
        if idx > 0:
            return bucket[idx - 1]
        
        # Need to check previous bucket
        reps = sorted(self.representatives.keys())
        rep_idx = reps.index(rep)
        
        if rep_idx > 0:
            prev_rep = reps[rep_idx - 1]
            prev_bucket = self.representatives[prev_rep]
            if prev_bucket:
                return prev_bucket[-1]
        
        return None
    
    def contains(self, key):
        """Check if key exists - O(log log u)"""
        return key in self.element_to_rep
    
    def __len__(self):
        return self.size
    
    def __str__(self):
        """String representation"""
        result = [f"Y-fast Trie (size={self.size}, universe_bits={self.universe_bits})"]
        if self.representatives:
            result.append("Buckets:")
            for rep in sorted(self.representatives.keys()):
                bucket = self.representatives[rep]
                result.append(f"  Rep {rep}: {bucket[:5]}{'...' if len(bucket) > 5 else ''}")
        return "\n".join(result)

# Example usage
if __name__ == "__main__":
    print("Y-fast Trie:")
    
    # Create Y-fast trie with 16-bit universe
    yft = YFastTrie(universe_bits=16)
    
    # Insert elements
    elements = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115]
    print(f"\nInserting: {elements}")
    for elem in elements:
        yft.insert(elem)
    
    print(yft)
    
    # Successor queries
    print("\nSuccessor queries:")
    for query in [20, 50, 80, 120]:
        succ = yft.successor(query)
        print(f"  successor({query}) = {succ}")
    
    # Predecessor queries
    print("\nPredecessor queries:")
    for query in [20, 50, 80, 120]:
        pred = yft.predecessor(query)
        print(f"  predecessor({query}) = {pred}")
    
    # Delete some elements
    print("\nDeleting 25 and 55:")
    yft.delete(25)
    yft.delete(55)
    
    print(f"Size after deletion: {len(yft)}")
    print(f"Contains 25: {yft.contains(25)}")
    print(f"Contains 35: {yft.contains(35)}")
```

X-fast Trie

When: Precursor to Y-fast trie
Why: O(log log u) search but less space-efficient than Y-fast
Examples: Theoretical foundation for Y-fast

```python
# Python implementation of X-fast Trie
# Binary trie with hash tables at each level for O(log log u) operations

class XFastTrie:
    """
    X-fast Trie for efficient successor/predecessor queries on integers.
    Uses a binary trie with hash tables at each level.
    Achieves O(log log u) time using binary search on levels.
    """
    def __init__(self, universe_bits=32):
        """
        Initialize X-fast trie.
        
        Args:
            universe_bits: Number of bits for the universe (u = 2^universe_bits)
        """
        self.universe_bits = universe_bits
        self.universe_size = 2 ** universe_bits
        
        # Hash table for each level of the trie
        # level_hash[i] contains all prefixes of length i
        self.level_hash = [dict() for _ in range(universe_bits + 1)]
        
        # Leaf nodes (actual values) with doubly-linked structure
        self.leaves = {}  # key -> node
        
        # Doubly-linked list of leaves for successor/predecessor
        self.min_key = None
        self.max_key = None
        
        self.size = 0
    
    def _get_prefix(self, key, length):
        """Get the prefix of key with given length"""
        if length == 0:
            return 0
        # Shift right to get top 'length' bits
        return key >> (self.universe_bits - length)
    
    def _get_bit(self, key, position):
        """Get bit at position (0 = leftmost/MSB)"""
        return (key >> (self.universe_bits - position - 1)) & 1
    
    def insert(self, key):
        """
        Insert a key into the X-fast trie.
        O(log u) time for insertion.
        """
        if key < 0 or key >= self.universe_size:
            raise ValueError(f"Key must be in range [0, {self.universe_size})")
        
        if key in self.leaves:
            return False  # Already exists
        
        # Create leaf node with prev/next pointers
        node = {'key': key, 'prev': None, 'next': None}
        self.leaves[key] = node
        
        # Insert all prefixes into hash tables
        for level in range(self.universe_bits + 1):
            prefix = self._get_prefix(key, level)
            
            if prefix not in self.level_hash[level]:
                self.level_hash[level][prefix] = {
                    'prefix': prefix,
                    'left': None,  # Pointer to left child
                    'right': None,  # Pointer to right child
                    'descendant': key  # Leaf in this subtree
                }
        
        # Update doubly-linked list
        if self.min_key is None:
            self.min_key = key
            self.max_key = key
        else:
            # Find predecessor and successor
            pred_key = self._find_predecessor(key)
            succ_key = self._find_successor(key)
            
            # Link with predecessor
            if pred_key is not None:
                pred_node = self.leaves[pred_key]
                node['prev'] = pred_key
                node['next'] = pred_node['next']
                pred_node['next'] = key
                if node['next'] is not None:
                    self.leaves[node['next']]['prev'] = key
            else:
                # New minimum
                node['next'] = self.min_key
                if self.min_key is not None:
                    self.leaves[self.min_key]['prev'] = key
                self.min_key = key
            
            # Update max if necessary
            if key > self.max_key:
                self.max_key = key
        
        self.size += 1
        return True
    
    def _find_lca_level(self, key):
        """
        Find the level of the lowest common ancestor using binary search.
        Returns the level where search path diverges.
        O(log log u) time.
        """
        # Binary search on levels
        low, high = 0, self.universe_bits
        result = 0
        
        while low <= high:
            mid = (low + high) // 2
            prefix = self._get_prefix(key, mid)
            
            if prefix in self.level_hash[mid]:
                result = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return result
    
    def _find_successor(self, key):
        """Internal method to find successor for linking"""
        # Find the LCA level
        lca_level = self._find_lca_level(key)
        
        if lca_level == self.universe_bits:
            # Exact match or need to check neighbors
            if key in self.leaves:
                node = self.leaves[key]
                return node['next']
            return None
        
        # Get the prefix at LCA level
        prefix = self._get_prefix(key, lca_level)
        
        if prefix not in self.level_hash[lca_level]:
            return None
        
        lca_node = self.level_hash[lca_level][prefix]
        
        # Determine if we need left or right subtree
        bit = self._get_bit(key, lca_level)
        
        if bit == 0:
            # Go right for successor
            descendant = lca_node['descendant']
            if descendant > key:
                return descendant
        
        # Find next leaf
        if lca_node['descendant'] in self.leaves:
            node = self.leaves[lca_node['descendant']]
            return node['next']
        
        return None
    
    def _find_predecessor(self, key):
        """Internal method to find predecessor for linking"""
        lca_level = self._find_lca_level(key)
        
        if lca_level == self.universe_bits:
            if key in self.leaves:
                node = self.leaves[key]
                return node['prev']
            return None
        
        prefix = self._get_prefix(key, lca_level)
        
        if prefix not in self.level_hash[lca_level]:
            return None
        
        lca_node = self.level_hash[lca_level][prefix]
        descendant = lca_node['descendant']
        
        if descendant < key:
            return descendant
        
        if descendant in self.leaves:
            node = self.leaves[descendant]
            return node['prev']
        
        return None
    
    def successor(self, key):
        """
        Find the smallest element >= key.
        O(log log u) time.
        """
        if key in self.leaves:
            return key
        
        # Use binary search on trie levels
        lca_level = self._find_lca_level(key)
        
        if lca_level == 0:
            # No prefix match, return min
            return self.min_key
        
        prefix = self._get_prefix(key, lca_level)
        if prefix in self.level_hash[lca_level]:
            lca_node = self.level_hash[lca_level][prefix]
            descendant = lca_node['descendant']
            
            if descendant >= key:
                return descendant
            elif descendant in self.leaves:
                return self.leaves[descendant]['next']
        
        return None
    
    def predecessor(self, key):
        """
        Find the largest element <= key.
        O(log log u) time.
        """
        if key in self.leaves:
            return key
        
        lca_level = self._find_lca_level(key)
        
        if lca_level == 0:
            return self.max_key
        
        prefix = self._get_prefix(key, lca_level)
        if prefix in self.level_hash[lca_level]:
            lca_node = self.level_hash[lca_level][prefix]
            descendant = lca_node['descendant']
            
            if descendant <= key:
                return descendant
            elif descendant in self.leaves:
                return self.leaves[descendant]['prev']
        
        return None
    
    def contains(self, key):
        """Check if key exists - O(1)"""
        return key in self.leaves
    
    def delete(self, key):
        """
        Delete a key from the X-fast trie.
        O(log u) time.
        """
        if key not in self.leaves:
            return False
        
        node = self.leaves[key]
        
        # Update linked list
        if node['prev'] is not None:
            self.leaves[node['prev']]['next'] = node['next']
        else:
            self.min_key = node['next']
        
        if node['next'] is not None:
            self.leaves[node['next']]['prev'] = node['prev']
        else:
            self.max_key = node['prev']
        
        # Remove from leaves
        del self.leaves[key]
        
        # Remove prefixes from hash tables (simplified - should check if others exist)
        for level in range(self.universe_bits + 1):
            prefix = self._get_prefix(key, level)
            if prefix in self.level_hash[level]:
                # Only delete if no other keys share this prefix
                # (simplified version just deletes)
                del self.level_hash[level][prefix]
        
        self.size -= 1
        return True
    
    def __len__(self):
        return self.size
    
    def __str__(self):
        """String representation"""
        result = [f"X-fast Trie (size={self.size}, bits={self.universe_bits})"]
        if self.leaves:
            keys = sorted(self.leaves.keys())
            result.append(f"Keys: {keys[:10]}{'...' if len(keys) > 10 else ''}")
        return "\n".join(result)

# Example usage
if __name__ == "__main__":
    print("X-fast Trie:")
    
    # Create X-fast trie with 8-bit universe
    xft = XFastTrie(universe_bits=8)
    
    # Insert elements
    elements = [15, 25, 35, 45, 55, 65, 75, 85]
    print(f"\nInserting: {elements}")
    for elem in elements:
        xft.insert(elem)
    
    print(xft)
    
    # Successor queries
    print("\nSuccessor queries:")
    for query in [20, 50, 80]:
        succ = xft.successor(query)
        print(f"  successor({query}) = {succ}")
    
    # Predecessor queries
    print("\nPredecessor queries:")
    for query in [20, 50, 80]:
        pred = xft.predecessor(query)
        print(f"  predecessor({query}) = {pred}")
    
    # Delete and test
    print("\nDeleting 35:")
    xft.delete(35)
    print(f"Contains 35: {xft.contains(35)}")
    print(f"Size: {len(xft)}")
```

Fractional Cascading

When: Multiple binary searches in related structures
Why: Speeds up repeated searches in similar sorted lists
Examples: Computational geometry, range searching

```python
# Python implementation of Fractional Cascading
# Speeds up multiple binary searches in related sorted lists

import bisect

class FractionalCascading:
    """
    Fractional Cascading data structure for efficient multiple binary searches.
    Preprocesses a set of sorted lists to enable fast searches across all lists.
    """
    def __init__(self, lists):
        """
        Initialize fractional cascading structure.
        
        Args:
            lists: List of sorted lists to preprocess
        """
        self.num_lists = len(lists)
        self.original_lists = [list(l) for l in lists]
        self.augmented_lists = []
        self.bridge_pointers = []
        
        # Build augmented lists from bottom to top
        self._build_augmented_structure()
    
    def _build_augmented_structure(self):
        """Build augmented lists with bridge pointers"""
        # Start from the last list
        for i in range(self.num_lists - 1, -1, -1):
            if i == self.num_lists - 1:
                # Last list: just copy it
                self.augmented_lists.insert(0, list(self.original_lists[i]))
                self.bridge_pointers.insert(0, [])
            else:
                # Merge current list with samples from the list below
                current_list = self.original_lists[i]
                next_augmented = self.augmented_lists[0]
                
                # Sample every other element from the list below
                sampled = next_augmented[::2]
                
                # Merge current list with sampled elements
                merged = []
                bridges = []
                curr_idx = 0
                sample_idx = 0
                
                while curr_idx < len(current_list) or sample_idx < len(sampled):
                    if curr_idx >= len(current_list):
                        # Only sampled elements left
                        merged.append(sampled[sample_idx])
                        bridges.append(('down', sample_idx * 2))
                        sample_idx += 1
                    elif sample_idx >= len(sampled):
                        # Only current elements left
                        merged.append(current_list[curr_idx])
                        # Find position in next list
                        pos = bisect.bisect_left(next_augmented, current_list[curr_idx])
                        bridges.append(('down', pos))
                        curr_idx += 1
                    elif current_list[curr_idx] <= sampled[sample_idx]:
                        merged.append(current_list[curr_idx])
                        pos = bisect.bisect_left(next_augmented, current_list[curr_idx])
                        bridges.append(('down', pos))
                        curr_idx += 1
                    else:
                        merged.append(sampled[sample_idx])
                        bridges.append(('down', sample_idx * 2))
                        sample_idx += 1
                
                self.augmented_lists.insert(0, merged)
                self.bridge_pointers.insert(0, bridges)
    
    def search(self, value):
        """
        Search for value in all lists.
        Returns list of (list_index, position) tuples where value should be inserted.
        
        Time: O(log n + k) where k is the number of lists
        """
        results = []
        
        if not self.augmented_lists:
            return results
        
        # Binary search in first augmented list
        first_list = self.augmented_lists[0]
        pos = bisect.bisect_left(first_list, value)
        
        # Add result for first original list
        orig_pos = bisect.bisect_left(self.original_lists[0], value)
        results.append((0, orig_pos))
        
        # Use bridge pointers to find positions in remaining lists
        for i in range(1, self.num_lists):
            if i - 1 < len(self.bridge_pointers) and pos < len(self.bridge_pointers[i - 1]):
                # Use bridge pointer
                _, next_pos = self.bridge_pointers[i - 1][pos]
                
                # Refine search around bridge pointer
                next_list = self.augmented_lists[i]
                
                # Search in small range around bridge pointer
                left = max(0, next_pos - 2)
                right = min(len(next_list), next_pos + 2)
                
                while left < right and left < len(next_list):
                    if next_list[left] >= value:
                        break
                    left += 1
                
                pos = left
            else:
                # Fallback to binary search
                pos = bisect.bisect_left(self.augmented_lists[i], value)
            
            # Get position in original list
            orig_pos = bisect.bisect_left(self.original_lists[i], value)
            results.append((i, orig_pos))
        
        return results
    
    def range_query(self, low, high):
        """
        Find all elements in range [low, high] across all lists.
        
        Returns: List of (list_index, elements) tuples
        """
        results = []
        
        for i, lst in enumerate(self.original_lists):
            left_pos = bisect.bisect_left(lst, low)
            right_pos = bisect.bisect_right(lst, high)
            
            elements = lst[left_pos:right_pos]
            if elements:
                results.append((i, elements))
        
        return results

# Example usage
print("\nFractional Cascading:")

# Create multiple sorted lists
lists = [
    [1, 3, 7, 12, 18, 25],
    [2, 5, 8, 13, 20, 27],
    [4, 6, 10, 15, 22, 30],
    [1, 9, 11, 16, 24, 28]
]

fc = FractionalCascading(lists)

# Search for value across all lists
search_value = 10
print(f"\nSearching for {search_value}:")
results = fc.search(search_value)
for list_idx, pos in results:
    print(f"  List {list_idx}: insert at position {pos}")
    if pos < len(lists[list_idx]):
        print(f"    (before element {lists[list_idx][pos]})")

# Range query
print(f"\nRange query [5, 15]:")
range_results = fc.range_query(5, 15)
for list_idx, elements in range_results:
    print(f"  List {list_idx}: {elements}")

# Multiple searches (demonstrating efficiency)
print(f"\nMultiple searches:")
for val in [8, 15, 20]:
    results = fc.search(val)
    print(f"  Value {val}: found positions {[pos for _, pos in results]}")
```

Kinetic Data Structures

When: Objects move continuously and need to maintain properties
Why: Efficiently updates as objects move
Examples: Collision detection with moving objects, computational geometry with motion

```python
# Python implementation of Kinetic Data Structures
# Maintains properties of moving objects efficiently

import heapq
from dataclasses import dataclass
from typing import List, Tuple, Callable

@dataclass
class Event:
    """Event in kinetic data structure"""
    time: float
    event_type: str
    objects: Tuple
    
    def __lt__(self, other):
        return self.time < other.time

class MovingPoint:
    """Point moving with constant velocity"""
    def __init__(self, x0, y0, vx, vy, point_id):
        self.x0 = x0  # Initial x position
        self.y0 = y0  # Initial y position
        self.vx = vx  # x velocity
        self.vy = vy  # y velocity
        self.id = point_id
    
    def position_at(self, time):
        """Get position at given time"""
        return (self.x0 + self.vx * time, self.y0 + self.vy * time)
    
    def __repr__(self):
        return f"Point{self.id}({self.x0:.1f},{self.y0:.1f}, v=({self.vx:.1f},{self.vy:.1f}))"

class KineticConvexHull:
    """
    Maintains convex hull of moving points.
    Points move with constant velocity.
    """
    def __init__(self):
        self.points: List[MovingPoint] = []
        self.current_time = 0.0
        self.event_queue = []  # Min heap of events
        self.hull_points = set()  # IDs of points on hull
    
    def add_point(self, point: MovingPoint):
        """Add a moving point"""
        self.points.append(point)
        self._recompute_hull()
    
    def _recompute_hull(self):
        """Recompute convex hull at current time"""
        if len(self.points) < 3:
            self.hull_points = set(p.id for p in self.points)
            return
        
        # Get current positions
        positions = [(p.position_at(self.current_time), p.id) 
                    for p in self.points]
        
        # Compute convex hull using Graham scan
        positions.sort()
        
        # Build lower hull
        lower = []
        for pos, pid in positions:
            while len(lower) >= 2 and self._cross(lower[-2][0], lower[-1][0], pos) <= 0:
                lower.pop()
            lower.append((pos, pid))
        
        # Build upper hull
        upper = []
        for pos, pid in reversed(positions):
            while len(upper) >= 2 and self._cross(upper[-2][0], upper[-1][0], pos) <= 0:
                upper.pop()
            upper.append((pos, pid))
        
        # Combine (remove last point of each half because they're duplicated)
        hull = lower[:-1] + upper[:-1]
        self.hull_points = set(pid for _, pid in hull)
        
        # Schedule certificate failures (when points enter/leave hull)
        self._schedule_events()
    
    def _cross(self, o, a, b):
        """Cross product of vectors OA and OB"""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    def _schedule_events(self):
        """Schedule future events when hull topology changes"""
        # Clear old events
        self.event_queue = []
        
        # For simplicity, check all triples of points
        # In practice, only check adjacent hull edges
        for i, p1 in enumerate(self.points):
            for j, p2 in enumerate(self.points[i+1:], i+1):
                for k, p3 in enumerate(self.points[j+1:], j+1):
                    # Find time when these three points become collinear
                    t = self._collinear_time(p1, p2, p3)
                    if t and t > self.current_time:
                        event = Event(t, "collinear", (p1.id, p2.id, p3.id))
                        heapq.heappush(self.event_queue, event)
    
    def _collinear_time(self, p1, p2, p3):
        """Find time when three moving points become collinear"""
        # Points are collinear when cross product of vectors is zero
        # This is a quadratic equation in time
        # Simplified version: return None (full implementation is complex)
        return None
    
    def advance_to(self, new_time):
        """Advance time and process events"""
        while self.event_queue and self.event_queue[0].time <= new_time:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            # Process event: recompute hull
            self._recompute_hull()
        
        self.current_time = new_time
    
    def get_hull(self):
        """Get current convex hull point IDs"""
        return self.hull_points

# Example usage
print("\nKinetic Data Structures - Convex Hull:")

# Create moving points
points = [
    MovingPoint(0, 0, 1, 0, 0),    # Moving right
    MovingPoint(5, 0, -1, 0, 1),   # Moving left
    MovingPoint(2, 5, 0, -1, 2),   # Moving down
    MovingPoint(2, -2, 0, 1, 3)    # Moving up
]

kds = KineticConvexHull()
for p in points:
    kds.add_point(p)

# Check hull at different times
for t in [0, 1, 2, 3]:
    kds.advance_to(t)
    hull = kds.get_hull()
    print(f"\nTime {t}: Hull contains points {sorted(hull)}")
    print(f"  Positions:")
    for p in points:
        if p.id in hull:
            pos = p.position_at(t)
            print(f"    Point {p.id}: ({pos[0]:.1f}, {pos[1]:.1f})")
```

Retroactive Data Structures

When: Need to insert/delete operations in the past
Why: Can modify history of operations
Examples: Debugging, version control, timeline manipulation

```python
# Python implementation of Retroactive Data Structures
# Allows operations to be inserted/deleted in the past

from typing import Any, Optional, List, Tuple
from dataclasses import dataclass, field

@dataclass
class Operation:
    """An operation with a timestamp"""
    time: float
    op_type: str  # 'insert' or 'delete'
    value: Any
    op_id: int = field(default=0)
    
    def __lt__(self, other):
        return self.time < other.time

class RetroactivePriorityQueue:
    """
    Retroactive Priority Queue - allows insertion/deletion of operations in the past.
    Maintains the ability to query what the min was at any time.
    """
    def __init__(self):
        self.operations: List[Operation] = []  # Sorted by time
        self.next_id = 0
        self._recompute_needed = False
        self.cache = {}  # Cache of computed states
    
    def retroactive_insert(self, time: float, value: Any) -> int:
        """
        Insert an 'insert' operation at the given time.
        Returns operation ID.
        """
        op = Operation(time, 'insert', value, self.next_id)
        self.next_id += 1
        
        # Find insertion point (binary search could be used)
        insert_pos = 0
        for i, existing_op in enumerate(self.operations):
            if existing_op.time > time:
                break
            insert_pos = i + 1
        
        self.operations.insert(insert_pos, op)
        self._recompute_needed = True
        self.cache.clear()
        
        return op.op_id
    
    def retroactive_delete(self, time: float, value: Any) -> int:
        """
        Insert a 'delete' operation at the given time.
        Returns operation ID.
        """
        op = Operation(time, 'delete', value, self.next_id)
        self.next_id += 1
        
        insert_pos = 0
        for i, existing_op in enumerate(self.operations):
            if existing_op.time > time:
                break
            insert_pos = i + 1
        
        self.operations.insert(insert_pos, op)
        self._recompute_needed = True
        self.cache.clear()
        
        return op.op_id
    
    def delete_operation(self, op_id: int):
        """Delete an operation by its ID (removing it from history)"""
        self.operations = [op for op in self.operations if op.op_id != op_id]
        self._recompute_needed = True
        self.cache.clear()
    
    def query_min_at_time(self, query_time: float) -> Optional[Any]:
        """
        Query what the minimum value was at the given time.
        """
        if query_time in self.cache:
            return self.cache[query_time]
        
        # Replay operations up to query_time
        current_pq = []
        
        for op in self.operations:
            if op.time > query_time:
                break
            
            if op.op_type == 'insert':
                current_pq.append(op.value)
                current_pq.sort()  # In practice, use a proper heap
            elif op.op_type == 'delete':
                if op.value in current_pq:
                    current_pq.remove(op.value)
        
        result = current_pq[0] if current_pq else None
        self.cache[query_time] = result
        return result
    
    def get_timeline(self) -> List[Tuple[float, str, Any]]:
        """Get the complete timeline of operations"""
        return [(op.time, op.op_type, op.value) for op in self.operations]

class RetroactiveStack:
    """
    Retroactive Stack - allows push/pop operations to be inserted in the past.
    """
    def __init__(self):
        self.operations: List[Operation] = []
        self.next_id = 0
    
    def retroactive_push(self, time: float, value: Any) -> int:
        """Insert a push operation at given time"""
        op = Operation(time, 'push', value, self.next_id)
        self.next_id += 1
        
        insert_pos = len(self.operations)
        for i, existing_op in enumerate(self.operations):
            if existing_op.time > time:
                insert_pos = i
                break
        
        self.operations.insert(insert_pos, op)
        return op.op_id
    
    def retroactive_pop(self, time: float) -> int:
        """Insert a pop operation at given time"""
        op = Operation(time, 'pop', None, self.next_id)
        self.next_id += 1
        
        insert_pos = len(self.operations)
        for i, existing_op in enumerate(self.operations):
            if existing_op.time > time:
                insert_pos = i
                break
        
        self.operations.insert(insert_pos, op)
        return op.op_id
    
    def query_at_time(self, query_time: float) -> List[Any]:
        """Get stack contents at given time"""
        stack = []
        
        for op in self.operations:
            if op.time > query_time:
                break
            
            if op.op_type == 'push':
                stack.append(op.value)
            elif op.op_type == 'pop' and stack:
                stack.pop()
        
        return stack

# Example usage
print("\nRetroactive Priority Queue:")

rpq = RetroactivePriorityQueue()

# Insert operations at different times
rpq.retroactive_insert(1.0, 10)
rpq.retroactive_insert(2.0, 5)
rpq.retroactive_insert(3.0, 15)
rpq.retroactive_delete(2.5, 5)

print("Timeline:", rpq.get_timeline())

# Query minimum at different times
for t in [0.5, 1.5, 2.2, 2.8, 3.5]:
    min_val = rpq.query_min_at_time(t)
    print(f"  Min at time {t}: {min_val}")

# Go back in time and insert a new operation
print("\nInserting value 3 at time 0.5...")
rpq.retroactive_insert(0.5, 3)

# Re-query
for t in [0.5, 1.5, 2.2, 2.8, 3.5]:
    min_val = rpq.query_min_at_time(t)
    print(f"  Min at time {t}: {min_val}")

print("\nRetroactive Stack:")
rstack = RetroactiveStack()

# Build a timeline of stack operations
rstack.retroactive_push(1.0, 'A')
rstack.retroactive_push(2.0, 'B')
rstack.retroactive_push(3.0, 'C')
rstack.retroactive_pop(3.5)

# Query at different times
for t in [0.5, 1.5, 2.5, 3.2, 4.0]:
    contents = rstack.query_at_time(t)
    print(f"  Stack at time {t}: {contents}")

# Insert operation in the past
print("\nInserting 'X' at time 1.5...")
rstack.retroactive_push(1.5, 'X')

for t in [0.5, 1.5, 2.5, 3.2, 4.0]:
    contents = rstack.query_at_time(t)
    print(f"  Stack at time {t}: {contents}")
```

Bx-Tree

When: B-tree operations with update logging
Why: Combines B-tree with write-ahead logging
Examples: Database systems, transactional storage

```python
# Python implementation of Bx-Tree
# Combines B-tree with write-ahead logging for transactional storage

from typing import Any, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum

class LogEntryType(Enum):
    INSERT = "INSERT"
    DELETE = "DELETE"
    UPDATE = "UPDATE"

@dataclass
class LogEntry:
    """Write-ahead log entry"""
    lsn: int  # Log Sequence Number
    entry_type: LogEntryType
    key: Any
    value: Optional[Any] = None
    old_value: Optional[Any] = None

class BxTreeNode:
    """Node in a Bx-Tree"""
    def __init__(self, is_leaf=True, max_keys=4):
        self.is_leaf = is_leaf
        self.keys = []
        self.values = []  # Only for leaf nodes
        self.children = []  # Only for internal nodes
        self.max_keys = max_keys
        self.lsn = 0  # Last LSN that modified this node
    
    def is_full(self):
        return len(self.keys) >= self.max_keys

class BxTree:
    """
    Bx-Tree: B-tree with write-ahead logging.
    Ensures ACID properties for database operations.
    """
    def __init__(self, max_keys=4):
        self.root = BxTreeNode(is_leaf=True, max_keys=max_keys)
        self.max_keys = max_keys
        self.write_ahead_log: List[LogEntry] = []
        self.next_lsn = 1
        self.checkpoint_lsn = 0
    
    def _log_operation(self, entry_type: LogEntryType, key: Any, 
                       value: Optional[Any] = None, old_value: Optional[Any] = None) -> int:
        """Add entry to write-ahead log"""
        log_entry = LogEntry(self.next_lsn, entry_type, key, value, old_value)
        self.write_ahead_log.append(log_entry)
        current_lsn = self.next_lsn
        self.next_lsn += 1
        return current_lsn
    
    def insert(self, key: Any, value: Any):
        """Insert key-value pair with logging"""
        # Write to log first (write-ahead logging)
        lsn = self._log_operation(LogEntryType.INSERT, key, value)
        
        # Then perform the actual insertion
        if self.root.is_full():
            # Split root
            old_root = self.root
            self.root = BxTreeNode(is_leaf=False, max_keys=self.max_keys)
            self.root.children.append(old_root)
            self._split_child(self.root, 0)
        
        self._insert_non_full(self.root, key, value, lsn)
    
    def _insert_non_full(self, node: BxTreeNode, key: Any, value: Any, lsn: int):
        """Insert into a node that is not full"""
        node.lsn = lsn  # Update node's LSN
        
        if node.is_leaf:
            # Find position to insert
            i = len(node.keys) - 1
            node.keys.append(None)
            node.values.append(None)
            
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.values[i + 1] = node.values[i]
                i -= 1
            
            node.keys[i + 1] = key
            node.values[i + 1] = value
        else:
            # Find child to insert into
            i = len(node.keys) - 1
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            
            if node.children[i].is_full():
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            
            self._insert_non_full(node.children[i], key, value, lsn)
    
    def _split_child(self, parent: BxTreeNode, index: int):
        """Split a full child node"""
        full_child = parent.children[index]
        mid = len(full_child.keys) // 2
        
        new_child = BxTreeNode(is_leaf=full_child.is_leaf, max_keys=self.max_keys)
        
        # Move half the keys to new node
        new_child.keys = full_child.keys[mid+1:]
        full_child.keys = full_child.keys[:mid]
        
        if full_child.is_leaf:
            new_child.values = full_child.values[mid+1:]
            full_child.values = full_child.values[:mid]
        else:
            new_child.children = full_child.children[mid+1:]
            full_child.children = full_child.children[:mid+1]
        
        # Insert median key into parent
        parent.keys.insert(index, full_child.keys[mid] if not full_child.is_leaf else new_child.keys[0])
        parent.children.insert(index + 1, new_child)
    
    def search(self, key: Any) -> Optional[Any]:
        """Search for a key"""
        return self._search_node(self.root, key)
    
    def _search_node(self, node: BxTreeNode, key: Any) -> Optional[Any]:
        """Search in a specific node"""
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        
        if i < len(node.keys) and key == node.keys[i]:
            if node.is_leaf:
                return node.values[i]
            else:
                return self._search_node(node.children[i + 1], key)
        
        if node.is_leaf:
            return None
        else:
            return self._search_node(node.children[i], key)
    
    def checkpoint(self):
        """Create a checkpoint - mark all log entries as committed"""
        self.checkpoint_lsn = self.next_lsn - 1
        print(f"Checkpoint created at LSN {self.checkpoint_lsn}")
    
    def recover(self):
        """
        Recovery using write-ahead log.
        Replay log entries after last checkpoint.
        """
        print(f"Recovering from checkpoint LSN {self.checkpoint_lsn}")
        
        for log_entry in self.write_ahead_log:
            if log_entry.lsn > self.checkpoint_lsn:
                if log_entry.entry_type == LogEntryType.INSERT:
                    # Re-apply insertion
                    print(f"  Replaying INSERT: {log_entry.key} = {log_entry.value}")
                # Handle other operation types similarly
    
    def get_log(self) -> List[LogEntry]:
        """Get the write-ahead log"""
        return self.write_ahead_log

# Example usage
print("\nBx-Tree:")

bxtree = BxTree(max_keys=3)

# Insert with logging
print("Inserting keys with WAL:")
for key, value in [(10, 'A'), (20, 'B'), (5, 'C'), (15, 'D'), (25, 'E')]:
    bxtree.insert(key, value)
    print(f"  Inserted {key} = {value}, LSN = {bxtree.next_lsn - 1}")

# Search
print("\nSearching:")
for key in [5, 15, 30]:
    result = bxtree.search(key)
    print(f"  Key {key}: {result}")

# Show write-ahead log
print("\nWrite-Ahead Log:")
for entry in bxtree.get_log():
    print(f"  LSN {entry.lsn}: {entry.entry_type.value} key={entry.key} value={entry.value}")

# Create checkpoint
print()
bxtree.checkpoint()

# Simulate crash and recovery
print("\nSimulating recovery:")
bxtree.recover()
```

LSM-Tree (Log-Structured Merge-Tree) (Log-Structured Merge-Tree)

When: Write-heavy workloads, especially on SSDs
Why: Optimizes writes by batching, sequential writes to disk
Examples: NoSQL databases (Cassandra, RocksDB, LevelDB), time-series databases

```python
# Python implementation of LSM-Tree (Log-Structured Merge-Tree)
# Optimizes for write-heavy workloads

from typing import Any, Optional, List, Dict, Tuple
from dataclasses import dataclass
import bisect
import time

@dataclass
class Entry:
    """Key-value entry with timestamp"""
    key: Any
    value: Optional[Any]  # None indicates deletion (tombstone)
    timestamp: float
    
    def __lt__(self, other):
        return self.key < other.key

class MemTable:
    """In-memory sorted buffer for recent writes"""
    def __init__(self, max_size=100):
        self.data: Dict[Any, Entry] = {}
        self.max_size = max_size
    
    def put(self, key: Any, value: Any):
        """Insert or update key-value pair"""
        self.data[key] = Entry(key, value, time.time())
    
    def delete(self, key: Any):
        """Mark key as deleted (tombstone)"""
        self.data[key] = Entry(key, None, time.time())
    
    def get(self, key: Any) -> Optional[Entry]:
        """Get entry for key"""
        return self.data.get(key)
    
    def is_full(self):
        """Check if memtable is full"""
        return len(self.data) >= self.max_size
    
    def get_sorted_entries(self) -> List[Entry]:
        """Get all entries sorted by key"""
        return sorted(self.data.values(), key=lambda e: e.key)
    
    def clear(self):
        """Clear the memtable"""
        self.data.clear()

class SSTable:
    """Sorted String Table - immutable on-disk sorted file"""
    def __init__(self, entries: List[Entry], level: int, table_id: int):
        self.entries = sorted(entries, key=lambda e: e.key)
        self.level = level
        self.table_id = table_id
        self.min_key = self.entries[0].key if self.entries else None
        self.max_key = self.entries[-1].key if self.entries else None
    
    def get(self, key: Any) -> Optional[Entry]:
        """Binary search for key"""
        left, right = 0, len(self.entries)
        
        while left < right:
            mid = (left + right) // 2
            if self.entries[mid].key == key:
                return self.entries[mid]
            elif self.entries[mid].key < key:
                left = mid + 1
            else:
                right = mid
        
        return None
    
    def overlaps(self, other: 'SSTable') -> bool:
        """Check if this SSTable overlaps with another"""
        return not (self.max_key < other.min_key or self.min_key > other.max_key)
    
    def __repr__(self):
        return f"SSTable(L{self.level}, id={self.table_id}, keys={len(self.entries)})"

class LSMTree:
    """
    LSM-Tree implementation with memtable and multiple levels of SSTables.
    Optimized for write-heavy workloads.
    """
    def __init__(self, memtable_size=10, level_size_multiplier=10):
        self.memtable = MemTable(max_size=memtable_size)
        self.immutable_memtables: List[MemTable] = []
        self.levels: List[List[SSTable]] = [[] for _ in range(7)]  # 7 levels
        self.next_sstable_id = 0
        self.level_size_multiplier = level_size_multiplier
    
    def put(self, key: Any, value: Any):
        """Insert or update key-value pair - O(1) amortized"""
        self.memtable.put(key, value)
        
        if self.memtable.is_full():
            self._flush_memtable()
    
    def delete(self, key: Any):
        """Delete key (adds tombstone) - O(1)"""
        self.memtable.delete(key)
        
        if self.memtable.is_full():
            self._flush_memtable()
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Search for key - O(log n) per level.
        Searches from newest to oldest data.
        """
        # Check memtable first (newest data)
        entry = self.memtable.get(key)
        if entry:
            return entry.value
        
        # Check immutable memtables
        for imm_table in self.immutable_memtables:
            entry = imm_table.get(key)
            if entry:
                return entry.value
        
        # Check SSTables from level 0 to higher levels
        for level_tables in self.levels:
            for sstable in reversed(level_tables):  # Newer tables first
                entry = sstable.get(key)
                if entry:
                    return entry.value  # None if tombstone
        
        return None
    
    def _flush_memtable(self):
        """Flush memtable to level 0 as SSTable"""
        # Move current memtable to immutable
        self.immutable_memtables.append(self.memtable)
        self.memtable = MemTable(max_size=self.memtable.max_size)
        
        # Flush immutable memtables to disk (level 0)
        if self.immutable_memtables:
            imm_table = self.immutable_memtables.pop(0)
            entries = imm_table.get_sorted_entries()
            
            if entries:
                sstable = SSTable(entries, 0, self.next_sstable_id)
                self.next_sstable_id += 1
                self.levels[0].append(sstable)
                
                # Trigger compaction if needed
                self._maybe_compact()
    
    def _maybe_compact(self):
        """Check if compaction is needed and perform it"""
        for level in range(len(self.levels) - 1):
            max_tables = (self.level_size_multiplier ** level) if level > 0 else 4
            
            if len(self.levels[level]) > max_tables:
                self._compact_level(level)
                break
    
    def _compact_level(self, level: int):
        """Compact level into next level (merge sort)"""
        if level >= len(self.levels) - 1:
            return
        
        print(f"Compacting level {level} -> {level + 1}")
        
        # Take all tables from current level
        source_tables = self.levels[level]
        self.levels[level] = []
        
        # Find overlapping tables in next level
        target_tables = []
        remaining_tables = []
        
        for target_table in self.levels[level + 1]:
            overlaps = any(target_table.overlaps(src) for src in source_tables)
            if overlaps:
                target_tables.append(target_table)
            else:
                remaining_tables.append(target_table)
        
        # Merge all entries
        all_entries = []
        for table in source_tables + target_tables:
            all_entries.extend(table.entries)
        
        # Sort and deduplicate (keep newest version)
        all_entries.sort(key=lambda e: (e.key, -e.timestamp))
        
        # Remove duplicates (keep first occurrence which is newest)
        seen_keys = set()
        merged_entries = []
        for entry in all_entries:
            if entry.key not in seen_keys:
                if entry.value is not None:  # Skip tombstones during compaction
                    merged_entries.append(entry)
                seen_keys.add(entry.key)
        
        # Create new SSTable at next level
        if merged_entries:
            new_sstable = SSTable(merged_entries, level + 1, self.next_sstable_id)
            self.next_sstable_id += 1
            remaining_tables.append(new_sstable)
        
        self.levels[level + 1] = remaining_tables
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the LSM tree"""
        return {
            'memtable_size': len(self.memtable.data),
            'immutable_memtables': len(self.immutable_memtables),
            'level_counts': [len(tables) for tables in self.levels],
            'total_sstables': sum(len(tables) for tables in self.levels)
        }

# Example usage
print("\nLSM-Tree:")

lsm = LSMTree(memtable_size=5)

# Insert many keys (write-heavy workload)
print("Inserting keys (write-optimized):")
for i in range(20):
    lsm.put(i, f"value_{i}")
    if i % 5 == 4:
        stats = lsm.get_stats()
        print(f"  After {i+1} inserts: {stats}")

# Read operations
print("\nReading keys:")
for key in [0, 5, 10, 15, 25]:
    value = lsm.get(key)
    print(f"  Key {key}: {value}")

# Delete operation
print("\nDeleting key 10:")
lsm.delete(10)
print(f"  Key 10 after delete: {lsm.get(10)}")

# Final statistics
print("\nFinal stats:", lsm.get_stats())
```

Counted B-Tree

When: B-tree with rank queries
Why: Maintains counts for order statistics
Examples: Databases with ranking, positional queries

```python
# Python implementation of Counted B-Tree
# B-Tree with rank queries and order statistics

from typing import Any, Optional, List, Tuple

class CountedBTreeNode:
    """Node in a Counted B-Tree with subtree counts"""
    def __init__(self, is_leaf=True, max_keys=4):
        self.is_leaf = is_leaf
        self.keys: List[Any] = []
        self.values: List[Any] = []  # Only for leaf nodes
        self.children: List['CountedBTreeNode'] = []  # Only for internal nodes
        self.counts: List[int] = []  # Count of keys in each subtree
        self.max_keys = max_keys
    
    def is_full(self):
        return len(self.keys) >= self.max_keys
    
    def get_total_count(self) -> int:
        """Get total number of keys in this subtree"""
        if self.is_leaf:
            return len(self.keys)
        else:
            return sum(self.counts)

class CountedBTree:
    """
    Counted B-Tree: B-Tree augmented with counts for rank queries.
    Supports order statistics in O(log n) time.
    """
    def __init__(self, max_keys=4):
        self.root = CountedBTreeNode(is_leaf=True, max_keys=max_keys)
        self.max_keys = max_keys
        self.total_size = 0
    
    def insert(self, key: Any, value: Any):
        """Insert key-value pair and update counts"""
        if self.root.is_full():
            # Split root
            old_root = self.root
            self.root = CountedBTreeNode(is_leaf=False, max_keys=self.max_keys)
            self.root.children.append(old_root)
            self.root.counts.append(old_root.get_total_count())
            self._split_child(self.root, 0)
        
        self._insert_non_full(self.root, key, value)
        self.total_size += 1
    
    def _insert_non_full(self, node: CountedBTreeNode, key: Any, value: Any):
        """Insert into a node that is not full"""
        if node.is_leaf:
            # Find position and insert
            i = len(node.keys) - 1
            node.keys.append(None)
            node.values.append(None)
            
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.values[i + 1] = node.values[i]
                i -= 1
            
            node.keys[i + 1] = key
            node.values[i + 1] = value
        else:
            # Find child to insert into
            i = len(node.keys) - 1
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            
            if node.children[i].is_full():
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            
            self._insert_non_full(node.children[i], key, value)
            
            # Update count for this child
            node.counts[i] = node.children[i].get_total_count()
    
    def _split_child(self, parent: CountedBTreeNode, index: int):
        """Split a full child node"""
        full_child = parent.children[index]
        mid = len(full_child.keys) // 2
        
        new_child = CountedBTreeNode(is_leaf=full_child.is_leaf, max_keys=self.max_keys)
        
        # Split keys
        new_child.keys = full_child.keys[mid+1:]
        full_child.keys = full_child.keys[:mid]
        
        if full_child.is_leaf:
            new_child.values = full_child.values[mid+1:]
            full_child.values = full_child.values[:mid]
        else:
            # Split children and counts
            new_child.children = full_child.children[mid+1:]
            new_child.counts = full_child.counts[mid+1:]
            full_child.children = full_child.children[:mid+1]
            full_child.counts = full_child.counts[:mid+1]
        
        # Insert median key into parent
        median_key = full_child.keys[mid] if not full_child.is_leaf else new_child.keys[0]
        parent.keys.insert(index, median_key)
        parent.children.insert(index + 1, new_child)
        
        # Update counts in parent
        parent.counts[index] = full_child.get_total_count()
        parent.counts.insert(index + 1, new_child.get_total_count())
    
    def search(self, key: Any) -> Optional[Any]:
        """Search for a key"""
        return self._search_node(self.root, key)
    
    def _search_node(self, node: CountedBTreeNode, key: Any) -> Optional[Any]:
        """Search in a specific node"""
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        
        if i < len(node.keys) and key == node.keys[i]:
            if node.is_leaf:
                return node.values[i]
            else:
                return self._search_node(node.children[i + 1], key)
        
        if node.is_leaf:
            return None
        else:
            return self._search_node(node.children[i], key)
    
    def rank(self, key: Any) -> int:
        """
        Find the rank (0-indexed position) of a key if it were inserted.
        Returns the number of keys less than the given key.
        O(log n) time complexity.
        """
        return self._rank_in_node(self.root, key)
    
    def _rank_in_node(self, node: CountedBTreeNode, key: Any) -> int:
        """Calculate rank within a node"""
        if node.is_leaf:
            # Count keys less than key
            rank = 0
            for k in node.keys:
                if k < key:
                    rank += 1
                else:
                    break
            return rank
        else:
            # Internal node
            rank = 0
            i = 0
            
            while i < len(node.keys):
                if key <= node.keys[i]:
                    # Go to left child
                    return rank + self._rank_in_node(node.children[i], key)
                else:
                    # Add count of left child and current key
                    rank += node.counts[i]
                    rank += 1  # For the current key
                    i += 1
            
            # Go to rightmost child
            return rank + self._rank_in_node(node.children[i], key)
    
    def select(self, rank: int) -> Optional[Tuple[Any, Any]]:
        """
        Find the key-value pair at the given rank (0-indexed).
        Returns the (rank+1)-th smallest key.
        O(log n) time complexity.
        """
        if rank < 0 or rank >= self.total_size:
            return None
        
        return self._select_in_node(self.root, rank)
    
    def _select_in_node(self, node: CountedBTreeNode, rank: int) -> Optional[Tuple[Any, Any]]:
        """Find element at rank within a node"""
        if node.is_leaf:
            if rank < len(node.keys):
                return (node.keys[rank], node.values[rank])
            return None
        else:
            # Internal node
            current_rank = 0
            
            for i in range(len(node.keys)):
                left_count = node.counts[i]
                
                if rank < current_rank + left_count:
                    # Element is in left child
                    return self._select_in_node(node.children[i], rank - current_rank)
                
                current_rank += left_count
                
                if rank == current_rank:
                    # Element is the current key
                    # For simplicity, we don't store values in internal nodes
                    # so we continue to the child
                    pass
                
                current_rank += 1
            
            # Element is in rightmost child
            return self._select_in_node(node.children[-1], rank - current_rank)
    
    def size(self) -> int:
        """Get total number of elements"""
        return self.total_size

# Example usage
print("\nCounted B-Tree:")

cbt = CountedBTree(max_keys=3)

# Insert elements
elements = [(10, 'A'), (20, 'B'), (5, 'C'), (15, 'D'), (25, 'E'), 
            (12, 'F'), (18, 'G'), (30, 'H')]

print("Inserting elements:")
for key, value in elements:
    cbt.insert(key, value)
    print(f"  Inserted {key}={value}, tree size: {cbt.size()}")

# Test rank queries
print("\nRank queries (number of elements less than key):")
for key in [5, 12, 20, 30, 35]:
    rank = cbt.rank(key)
    print(f"  Rank of {key}: {rank}")

# Test select queries (find kth smallest)
print("\nSelect queries (find kth smallest element):")
for k in range(min(8, cbt.size())):
    result = cbt.select(k)
    if result:
        key, value = result
        print(f"  {k}-th element: {key}={value}")

# Search
print("\nSearch queries:")
for key in [12, 20, 100]:
    value = cbt.search(key)
    print(f"  Key {key}: {value}")
```

Prefix Hash Tree (PHT) (PHT)

When: Distributed hash tree
Why: Combines trie and hash table for distributed systems
Examples: P2P systems, distributed databases

```python
# Python implementation of Prefix Hash Tree (PHT)
# Combines trie and hash table for distributed systems

from typing import Any, Optional, List, Dict
import hashlib

class PHTNode:
    """Node in a Prefix Hash Tree"""
    def __init__(self, prefix: str = ""):
        self.prefix = prefix  # Binary prefix
        self.is_leaf = True
        self.data: Dict[str, Any] = {}  # Stores key-value pairs if leaf
        self.left: Optional['PHTNode'] = None  # Child for prefix + '0'
        self.right: Optional['PHTNode'] = None  # Child for prefix + '1'
        self.max_capacity = 4  # Split threshold
    
    def is_full(self):
        return len(self.data) >= self.max_capacity

class PrefixHashTree:
    """
    Prefix Hash Tree for distributed hash tables.
    Combines trie structure with hash-based distribution.
    """
    def __init__(self, hash_bits=32):
        self.root = PHTNode("")
        self.hash_bits = hash_bits
        self.total_keys = 0
    
    def _hash_key(self, key: str) -> str:
        """Hash key to binary string"""
        hash_obj = hashlib.md5(str(key).encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        # Convert to binary string of fixed length
        binary = bin(hash_int)[2:].zfill(self.hash_bits)
        return binary[:self.hash_bits]  # Truncate to hash_bits
    
    def insert(self, key: str, value: Any):
        """Insert key-value pair"""
        hash_binary = self._hash_key(key)
        self._insert_at_node(self.root, key, value, hash_binary, 0)
        self.total_keys += 1
    
    def _insert_at_node(self, node: PHTNode, key: str, value: Any, 
                        hash_binary: str, depth: int):
        """Insert at a specific node"""
        if node.is_leaf:
            # Insert into leaf node
            node.data[key] = value
            
            # Split if full
            if node.is_full() and depth < self.hash_bits:
                self._split_node(node, depth)
        else:
            # Route to appropriate child
            if depth < len(hash_binary):
                bit = hash_binary[depth]
                if bit == '0':
                    if node.left is None:
                        node.left = PHTNode(node.prefix + '0')
                    self._insert_at_node(node.left, key, value, hash_binary, depth + 1)
                else:
                    if node.right is None:
                        node.right = PHTNode(node.prefix + '1')
                    self._insert_at_node(node.right, key, value, hash_binary, depth + 1)
    
    def _split_node(self, node: PHTNode, depth: int):
        """Split a full leaf node into two children"""
        node.is_leaf = False
        node.left = PHTNode(node.prefix + '0')
        node.right = PHTNode(node.prefix + '1')
        
        # Redistribute data to children
        for key, value in node.data.items():
            hash_binary = self._hash_key(key)
            if depth < len(hash_binary):
                bit = hash_binary[depth]
                if bit == '0':
                    node.left.data[key] = value
                else:
                    node.right.data[key] = value
        
        node.data.clear()
    
    def search(self, key: str) -> Optional[Any]:
        """Search for a key"""
        hash_binary = self._hash_key(key)
        return self._search_at_node(self.root, key, hash_binary, 0)
    
    def _search_at_node(self, node: PHTNode, key: str, 
                        hash_binary: str, depth: int) -> Optional[Any]:
        """Search at a specific node"""
        if node.is_leaf:
            return node.data.get(key)
        else:
            if depth < len(hash_binary):
                bit = hash_binary[depth]
                if bit == '0' and node.left:
                    return self._search_at_node(node.left, key, hash_binary, depth + 1)
                elif bit == '1' and node.right:
                    return self._search_at_node(node.right, key, hash_binary, depth + 1)
        
        return None
    
    def delete(self, key: str) -> bool:
        """Delete a key"""
        hash_binary = self._hash_key(key)
        return self._delete_at_node(self.root, key, hash_binary, 0)
    
    def _delete_at_node(self, node: PHTNode, key: str, 
                        hash_binary: str, depth: int) -> bool:
        """Delete at a specific node"""
        if node.is_leaf:
            if key in node.data:
                del node.data[key]
                self.total_keys -= 1
                return True
            return False
        else:
            if depth < len(hash_binary):
                bit = hash_binary[depth]
                if bit == '0' and node.left:
                    return self._delete_at_node(node.left, key, hash_binary, depth + 1)
                elif bit == '1' and node.right:
                    return self._delete_at_node(node.right, key, hash_binary, depth + 1)
        
        return False
    
    def get_node_for_prefix(self, prefix: str) -> Optional[PHTNode]:
        """Get the node responsible for a given prefix (for distributed systems)"""
        node = self.root
        
        for bit in prefix:
            if node.is_leaf:
                return node
            
            if bit == '0':
                if node.left:
                    node = node.left
                else:
                    return None
            else:
                if node.right:
                    node = node.right
                else:
                    return None
        
        return node
    
    def get_tree_structure(self) -> List[str]:
        """Get tree structure for visualization"""
        result = []
        self._traverse_structure(self.root, result, 0)
        return result
    
    def _traverse_structure(self, node: PHTNode, result: List[str], depth: int):
        """Traverse and collect structure"""
        indent = "  " * depth
        if node.is_leaf:
            result.append(f"{indent}Leaf[{node.prefix}]: {len(node.data)} keys")
        else:
            result.append(f"{indent}Internal[{node.prefix}]:")
            if node.left:
                self._traverse_structure(node.left, result, depth + 1)
            if node.right:
                self._traverse_structure(node.right, result, depth + 1)

# Example usage
print("\nPrefix Hash Tree:")

pht = PrefixHashTree(hash_bits=8)

# Insert keys
print("Inserting keys:")
keys = ['apple', 'banana', 'cherry', 'date', 'elderberry', 
        'fig', 'grape', 'honeydew', 'kiwi']

for key in keys:
    pht.insert(key, f"value_of_{key}")
    print(f"  Inserted '{key}', total keys: {pht.total_keys}")

# Search
print("\nSearching:")
for key in ['apple', 'banana', 'mango']:
    value = pht.search(key)
    print(f"  '{key}': {value}")

# Delete
print("\nDeleting 'banana':")
pht.delete('banana')
print(f"  Total keys after delete: {pht.total_keys}")
print(f"  Search 'banana': {pht.search('banana')}")

# Show tree structure
print("\nTree structure:")
for line in pht.get_tree_structure():
    print(line)

# Distributed system example: finding responsible node
print("\nDistributed lookup:")
test_keys = ['apple', 'zebra']
for key in test_keys:
    hash_val = pht._hash_key(key)
    prefix = hash_val[:4]  # First 4 bits
    print(f"  Key '{key}' -> hash prefix: {prefix}")
```

T-Tree

When: Main-memory database indexing
Why: Optimized for in-memory, combines AVL and array
Examples: In-memory databases, real-time systems

```python
# Python implementation of T-Tree
# Combines AVL tree with array storage for in-memory databases

from typing import Any, Optional, List

class TTreeNode:
    """
    T-Tree node stores multiple sorted values in an array.
    Combines benefits of AVL tree (balanced) and array (cache-friendly).
    """
    def __init__(self, min_capacity=2, max_capacity=4):
        self.values: List[Any] = []  # Sorted array of values
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.left: Optional['TTreeNode'] = None
        self.right: Optional['TTreeNode'] = None
        self.height = 1
    
    def is_full(self):
        return len(self.values) >= self.max_capacity
    
    def is_underfull(self):
        return len(self.values) < self.min_capacity
    
    def get_min(self):
        return self.values[0] if self.values else None
    
    def get_max(self):
        return self.values[-1] if self.values else None

class TTree:
    """
    T-Tree: Balanced tree optimized for main-memory databases.
    Each node stores multiple sorted values in an array.
    """
    def __init__(self, min_capacity=2, max_capacity=4):
        self.root: Optional[TTreeNode] = None
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.size = 0
    
    def _height(self, node: Optional[TTreeNode]) -> int:
        """Get height of node"""
        return node.height if node else 0
    
    def _balance_factor(self, node: TTreeNode) -> int:
        """Calculate balance factor"""
        return self._height(node.left) - self._height(node.right)
    
    def _update_height(self, node: TTreeNode):
        """Update node height"""
        node.height = 1 + max(self._height(node.left), self._height(node.right))
    
    def _rotate_right(self, y: TTreeNode) -> TTreeNode:
        """Right rotation"""
        x = y.left
        T2 = x.right
        
        x.right = y
        y.left = T2
        
        self._update_height(y)
        self._update_height(x)
        
        return x
    
    def _rotate_left(self, x: TTreeNode) -> TTreeNode:
        """Left rotation"""
        y = x.right
        T2 = y.left
        
        y.left = x
        x.right = T2
        
        self._update_height(x)
        self._update_height(y)
        
        return y
    
    def insert(self, value: Any):
        """Insert a value"""
        self.root = self._insert_rec(self.root, value)
        self.size += 1
    
    def _insert_rec(self, node: Optional[TTreeNode], value: Any) -> TTreeNode:
        """Recursively insert value"""
        # Base case: create new node
        if node is None:
            new_node = TTreeNode(self.min_capacity, self.max_capacity)
            new_node.values.append(value)
            return new_node
        
        # If current node has space and value fits in range
        if not node.is_full():
            if ((not node.left or value >= node.get_min()) and 
                (not node.right or value <= node.get_max())):
                # Insert into current node
                node.values.append(value)
                node.values.sort()
                return node
        
        # Route to appropriate subtree
        if value < node.get_min():
            node.left = self._insert_rec(node.left, value)
        else:
            node.right = self._insert_rec(node.right, value)
        
        # Update height and balance
        self._update_height(node)
        balance = self._balance_factor(node)
        
        # Left-Left case
        if balance > 1 and value < node.left.get_min():
            return self._rotate_right(node)
        
        # Right-Right case
        if balance < -1 and value > node.right.get_max():
            return self._rotate_left(node)
        
        # Left-Right case
        if balance > 1 and value > node.left.get_max():
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        
        # Right-Left case
        if balance < -1 and value < node.right.get_min():
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        
        return node
    
    def search(self, value: Any) -> bool:
        """Search for a value"""
        return self._search_rec(self.root, value)
    
    def _search_rec(self, node: Optional[TTreeNode], value: Any) -> bool:
        """Recursively search for value"""
        if node is None:
            return False
        
        # Check if value is in current node
        if value in node.values:
            return True
        
        # Route to appropriate subtree
        if value < node.get_min():
            return self._search_rec(node.left, value)
        elif value > node.get_max():
            return self._search_rec(node.right, value)
        else:
            # Value should be in current node but isn't
            return False
    
    def range_query(self, low: Any, high: Any) -> List[Any]:
        """Find all values in range [low, high]"""
        result = []
        self._range_query_rec(self.root, low, high, result)
        return sorted(result)
    
    def _range_query_rec(self, node: Optional[TTreeNode], low: Any, high: Any, result: List[Any]):
        """Recursively perform range query"""
        if node is None:
            return
        
        # Check left subtree if range overlaps
        if node.get_min() > low:
            self._range_query_rec(node.left, low, high, result)
        
        # Add values from current node in range
        for value in node.values:
            if low <= value <= high:
                result.append(value)
        
        # Check right subtree if range overlaps
        if node.get_max() < high:
            self._range_query_rec(node.right, low, high, result)
    
    def inorder_traversal(self) -> List[Any]:
        """Get all values in sorted order"""
        result = []
        self._inorder_rec(self.root, result)
        return result
    
    def _inorder_rec(self, node: Optional[TTreeNode], result: List[Any]):
        """Recursive inorder traversal"""
        if node is None:
            return
        
        self._inorder_rec(node.left, result)
        result.extend(node.values)
        self._inorder_rec(node.right, result)

# Example usage
print("\nT-Tree:")

ttree = TTree(min_capacity=2, max_capacity=4)

# Insert values
print("Inserting values:")
values = [10, 20, 5, 15, 25, 12, 18, 30, 8, 22]
for val in values:
    ttree.insert(val)
    print(f"  Inserted {val}, tree size: {ttree.size}")

# Search
print("\nSearching:")
for val in [12, 18, 100]:
    found = ttree.search(val)
    print(f"  Value {val}: {'Found' if found else 'Not found'}")

# Range query (efficient for in-memory databases)
print("\nRange query [10, 20]:")
results = ttree.range_query(10, 20)
print(f"  Results: {results}")

# Inorder traversal
print("\nAll values (sorted):")
all_values = ttree.inorder_traversal()
print(f"  {all_values}")

# Performance characteristics
print("\nT-Tree characteristics:")
print("  - Nodes store multiple values (cache-friendly)")
print("  - Balanced like AVL tree")
print("  - Optimized for in-memory operation")
print("  - Range queries are efficient")
```

Judy Array

When: Sparse array with fast operations
Why: Highly optimized sparse array, very fast in practice
Examples: High-performance sparse arrays, associative arrays

```python
# Python implementation of Judy Array
# Highly optimized sparse array with adaptive node types

from typing import Any, Optional, Dict, List

class JudyArrayNode:
    """Base node for Judy Array with adaptive structure"""
    def __init__(self):
        self.node_type = "linear"  # Can be: linear, bitmap, branch
        self.entries: Dict[int, Any] = {}  # For linear and small nodes
        self.bitmap: int = 0  # For bitmap nodes
        self.children: List[Optional['JudyArrayNode']] = []  # For branch nodes
        
class JudyArray:
    """
    Simplified Judy Array implementation.
    Real Judy Arrays use complex adaptive structures, but this shows the concept.
    """
    def __init__(self):
        self.root: Dict[int, Any] = {}  # Simplified: use dict for demo
        self.size = 0
        # In real implementation, would use adaptive node types based on density
    
    def insert(self, index: int, value: Any):
        """Insert value at index - O(log n) amortized"""
        if index not in self.root:
            self.size += 1
        self.root[index] = value
    
    def get(self, index: int) -> Optional[Any]:
        """Get value at index - O(1) expected"""
        return self.root.get(index)
    
    def delete(self, index: int) -> bool:
        """Delete value at index"""
        if index in self.root:
            del self.root[index]
            self.size -= 1
            return True
        return False
    
    def first(self) -> Optional[tuple]:
        """Get first (smallest) index and value"""
        if not self.root:
            return None
        min_index = min(self.root.keys())
        return (min_index, self.root[min_index])
    
    def last(self) -> Optional[tuple]:
        """Get last (largest) index and value"""
        if not self.root:
            return None
        max_index = max(self.root.keys())
        return (max_index, self.root[max_index])
    
    def next(self, index: int) -> Optional[tuple]:
        """Get next index and value after given index"""
        next_indices = [k for k in self.root.keys() if k > index]
        if not next_indices:
            return None
        next_index = min(next_indices)
        return (next_index, self.root[next_index])
    
    def prev(self, index: int) -> Optional[tuple]:
        """Get previous index and value before given index"""
        prev_indices = [k for k in self.root.keys() if k < index]
        if not prev_indices:
            return None
        prev_index = max(prev_indices)
        return (prev_index, self.root[prev_index])
    
    def count(self) -> int:
        """Get number of elements"""
        return self.size
    
    def count_range(self, start: int, end: int) -> int:
        """Count elements in range [start, end]"""
        return sum(1 for k in self.root.keys() if start <= k <= end)
    
    def memory_usage_estimate(self) -> int:
        """Estimate memory usage (simplified)"""
        # Real Judy Array has very efficient memory usage
        # This is just a rough estimate for demonstration
        return len(self.root) * 16  # bytes per entry (simplified)

# Example usage
print("\nJudy Array:")

judy = JudyArray()

# Insert sparse data
print("Inserting sparse data:")
sparse_indices = [10, 1000, 50000, 100, 5000000, 25]
for idx in sparse_indices:
    judy.insert(idx, f"value_{idx}")
    print(f"  Inserted at index {idx}")

print(f"\nTotal elements: {judy.count()}")

# Access operations
print("\nAccess operations:")
for idx in [10, 1000, 999]:
    value = judy.get(idx)
    print(f"  Index {idx}: {value}")

# Navigation operations
print("\nNavigation:")
first = judy.first()
print(f"  First: {first}")

last = judy.last()
print(f"  Last: {last}")

next_after_100 = judy.next(100)
print(f"  Next after 100: {next_after_100}")

prev_before_1000 = judy.prev(1000)
print(f"  Previous before 1000: {prev_before_1000}")

# Range counting
print("\nRange counting:")
count = judy.count_range(0, 10000)
print(f"  Elements in [0, 10000]: {count}")

# Memory efficiency
print(f"\nEstimated memory usage: {judy.memory_usage_estimate()} bytes")
print("  (Real Judy Array is extremely memory-efficient)")

# Iteration
print("\nAll indices (sorted):")
all_indices = sorted(judy.root.keys())
print(f"  {all_indices}")
```

Bitmapped Vector Trie / Bit-Partitioned Vector Trie / Bit-Partitioned Vector Trie

When: Persistent vectors in functional programming
Why: Near-constant time operations with structural sharing
Examples: Clojure vectors, functional data structures

```python
# Python implementation of Bitmapped Vector Trie (Persistent Vector)
# Used in functional programming languages like Clojure

from typing import Any, Optional, List
import copy

class BVTNode:
    """Node in a Bitmapped Vector Trie"""
    def __init__(self, branching_factor=32):
        self.children: List[Optional[Any]] = [None] * branching_factor
        self.is_leaf = False
        
class BitmappedVectorTrie:
    """
    Bitmapped Vector Trie (Persistent Vector).
    Provides near O(1) operations with structural sharing.
    """
    def __init__(self, branching_factor=32):
        self.branching_factor = branching_factor
        self.shift_bits = 5  # log2(32) for branching factor 32
        self.mask = branching_factor - 1
        self.root: Optional[BVTNode] = None
        self.tail: List[Any] = []  # Last incomplete block
        self.size = 0
        self.depth = 0
    
    def get(self, index: int) -> Any:
        """Get element at index - O(log32 n) ≈ O(1)"""
        if index < 0 or index >= self.size:
            raise IndexError("Index out of bounds")
        
        # Check if in tail
        if index >= self.size - len(self.tail):
            tail_index = index - (self.size - len(self.tail))
            return self.tail[tail_index]
        
        # Navigate through trie
        node = self.root
        level = self.depth
        
        while level > 0:
            idx = (index >> (level * self.shift_bits)) & self.mask
            node = node.children[idx]
            level -= 1
        
        idx = index & self.mask
        return node.children[idx]
    
    def append(self, value: Any) -> 'BitmappedVectorTrie':
        """
        Append element - O(log32 n) ≈ O(1).
        Returns new vector with structural sharing (persistent).
        """
        new_vec = BitmappedVectorTrie(self.branching_factor)
        new_vec.root = self.root
        new_vec.tail = self.tail.copy()
        new_vec.size = self.size
        new_vec.depth = self.depth
        
        # Add to tail if room
        if len(new_vec.tail) < self.branching_factor:
            new_vec.tail.append(value)
            new_vec.size += 1
            return new_vec
        
        # Tail is full, push it to trie
        new_node = BVTNode(self.branching_factor)
        new_node.children = new_vec.tail.copy()
        new_node.is_leaf = True
        
        new_vec.root = self._push_tail(new_vec.root, new_node, new_vec.depth)
        new_vec.tail = [value]
        new_vec.size += 1
        
        return new_vec
    
    def _push_tail(self, node: Optional[BVTNode], tail_node: BVTNode, level: int) -> BVTNode:
        """Push tail into trie structure"""
        if node is None:
            # Create new root if needed
            if level == 0:
                return tail_node
            new_node = BVTNode(self.branching_factor)
            new_node.children[0] = self._push_tail(None, tail_node, level - 1)
            self.depth += 1
            return new_node
        
        # Copy node for persistence
        new_node = BVTNode(self.branching_factor)
        new_node.children = node.children.copy()
        new_node.is_leaf = node.is_leaf
        
        if level == 0:
            # Reached leaf level
            return tail_node
        
        # Find insertion point
        subidx = ((self.size - 1) >> (level * self.shift_bits)) & self.mask
        new_node.children[subidx] = self._push_tail(
            node.children[subidx], tail_node, level - 1
        )
        
        return new_node
    
    def update(self, index: int, value: Any) -> 'BitmappedVectorTrie':
        """
        Update element at index - O(log32 n) ≈ O(1).
        Returns new vector with structural sharing.
        """
        if index < 0 or index >= self.size:
            raise IndexError("Index out of bounds")
        
        new_vec = BitmappedVectorTrie(self.branching_factor)
        new_vec.branching_factor = self.branching_factor
        new_vec.size = self.size
        new_vec.depth = self.depth
        
        # Check if in tail
        if index >= self.size - len(self.tail):
            new_vec.root = self.root
            new_vec.tail = self.tail.copy()
            tail_index = index - (self.size - len(self.tail))
            new_vec.tail[tail_index] = value
            return new_vec
        
        # Update in trie (with path copying)
        new_vec.tail = self.tail
        new_vec.root = self._update_node(self.root, index, value, self.depth)
        
        return new_vec
    
    def _update_node(self, node: BVTNode, index: int, value: Any, level: int) -> BVTNode:
        """Update node with path copying"""
        new_node = BVTNode(self.branching_factor)
        new_node.children = node.children.copy()
        new_node.is_leaf = node.is_leaf
        
        if level == 0:
            # Leaf level
            idx = index & self.mask
            new_node.children[idx] = value
        else:
            # Internal node
            idx = (index >> (level * self.shift_bits)) & self.mask
            new_node.children[idx] = self._update_node(
                node.children[idx], index, value, level - 1
            )
        
        return new_node
    
    def __len__(self):
        return self.size
    
    def to_list(self) -> List[Any]:
        """Convert to regular Python list"""
        result = []
        for i in range(self.size):
            result.append(self.get(i))
        return result

# Example usage
print("\nBitmapped Vector Trie (Persistent Vector):")

# Create vector
vec1 = BitmappedVectorTrie(branching_factor=4)  # Small branching for demo

# Append elements (persistent - returns new vectors)
print("Appending elements:")
vec2 = vec1.append(10)
vec3 = vec2.append(20)
vec4 = vec3.append(30)
vec5 = vec4.append(40)

print(f"  vec1 size: {len(vec1)}")
print(f"  vec5 size: {len(vec5)}")
print(f"  vec5 contents: {vec5.to_list()}")

# Update (persistent)
print("\nUpdating index 1 to 999:")
vec6 = vec5.update(1, 999)
print(f"  vec5[1]: {vec5.get(1)} (original unchanged)")
print(f"  vec6[1]: {vec6.get(1)} (new version)")

# Structural sharing demonstration
print("\nStructural sharing:")
print(f"  vec5 and vec6 share structure (memory efficient)")
print(f"  Only modified path is copied")

# Access elements
print("\nAccessing elements:")
for i in range(len(vec6)):
    print(f"  vec6[{i}] = {vec6.get(i)}")
```

HNSW (Hierarchical Navigable Small World) (Hierarchical Navigable Small World)

When: Approximate nearest neighbor search in high dimensions
Why: Very fast approximate NN, hierarchical graph structure
Examples: Vector similarity search, recommendation systems, semantic search

```python
# Python implementation of HNSW (Hierarchical Navigable Small World)
# For approximate nearest neighbor search in high dimensions

import random
import math
from typing import List, Set, Tuple, Optional
from collections import defaultdict
import heapq

class HNSWNode:
    """Node in HNSW graph"""
    def __init__(self, node_id: int, vector: List[float], level: int):
        self.id = node_id
        self.vector = vector
        self.level = level
        self.neighbors: List[Set[int]] = [set() for _ in range(level + 1)]

class HNSW:
    """
    Hierarchical Navigable Small World graph for ANN search.
    Provides very fast approximate nearest neighbor queries.
    """
    def __init__(self, m=16, ef_construction=200, ml=1.0/math.log(2.0)):
        """
        Initialize HNSW.
        
        Args:
            m: Max number of connections per layer
            ef_construction: Size of dynamic candidate list during construction
            ml: Normalization factor for level generation
        """
        self.m = m
        self.m_max = m
        self.m_max_0 = m * 2  # Layer 0 has more connections
        self.ef_construction = ef_construction
        self.ml = ml
        
        self.nodes: dict[int, HNSWNode] = {}
        self.entry_point: Optional[int] = None
        self.max_layer = 0
        self.next_id = 0
    
    def _distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Euclidean distance between vectors"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
    
    def _random_level(self) -> int:
        """Select level for new node"""
        level = 0
        while random.random() < 0.5 and level < 16:
            level += 1
        return level
    
    def insert(self, vector: List[float]) -> int:
        """Insert a vector into the HNSW graph"""
        node_id = self.next_id
        self.next_id += 1
        
        level = self._random_level()
        new_node = HNSWNode(node_id, vector, level)
        self.nodes[node_id] = new_node
        
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_layer = level
            return node_id
        
        # Search for nearest neighbors at each layer
        ep = [self.entry_point]
        
        # Layers above new node's level
        for lc in range(self.max_layer, level, -1):
            ep = self._search_layer(vector, ep, 1, lc)
        
        # Insert at all layers from level down to 0
        for lc in range(level, -1, -1):
            candidates = self._search_layer(vector, ep, self.ef_construction, lc)
            
            # Select m neighbors
            m = self.m if lc > 0 else self.m_max_0
            neighbors = self._select_neighbors(candidates, m, vector)
            
            # Add bidirectional links
            for neighbor_id in neighbors:
                new_node.neighbors[lc].add(neighbor_id)
                self.nodes[neighbor_id].neighbors[lc].add(node_id)
                
                # Prune neighbors if needed
                max_conn = self.m if lc > 0 else self.m_max_0
                if len(self.nodes[neighbor_id].neighbors[lc]) > max_conn:
                    # Prune the farthest neighbor
                    neighbor_vec = self.nodes[neighbor_id].vector
                    neighbors_to_keep = self._select_neighbors(
                        list(self.nodes[neighbor_id].neighbors[lc]),
                        max_conn,
                        neighbor_vec
                    )
                    self.nodes[neighbor_id].neighbors[lc] = set(neighbors_to_keep)
            
            ep = candidates
        
        # Update entry point if needed
        if level > self.max_layer:
            self.entry_point = node_id
            self.max_layer = level
        
        return node_id
    
    def _search_layer(self, query: List[float], entry_points: List[int], 
                      ef: int, layer: int) -> List[int]:
        """Search for nearest neighbors at a specific layer"""
        visited = set(entry_points)
        candidates = [(self._distance(query, self.nodes[ep].vector), ep) 
                     for ep in entry_points]
        heapq.heapify(candidates)
        
        w = [(-dist, ep) for dist, ep in candidates]
        heapq.heapify(w)
        
        while candidates:
            current_dist, current = heapq.heappop(candidates)
            
            if current_dist > -w[0][0]:
                break
            
            # Check neighbors
            for neighbor_id in self.nodes[current].neighbors[layer]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    
                    dist = self._distance(query, self.nodes[neighbor_id].vector)
                    
                    if dist < -w[0][0] or len(w) < ef:
                        heapq.heappush(candidates, (dist, neighbor_id))
                        heapq.heappush(w, (-dist, neighbor_id))
                        
                        if len(w) > ef:
                            heapq.heappop(w)
        
        return [ep for _, ep in w]
    
    def _select_neighbors(self, candidates: List[int], m: int, 
                         query_vec: List[float]) -> List[int]:
        """Select m nearest neighbors from candidates"""
        # Simple heuristic: select m closest
        distances = [(self._distance(query_vec, self.nodes[c].vector), c) 
                    for c in candidates]
        distances.sort()
        return [c for _, c in distances[:m]]
    
    def search(self, query: List[float], k: int = 10, ef: int = None) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector
            k: Number of nearest neighbors to return
            ef: Size of dynamic candidate list (default: max(ef_construction, k))
        
        Returns:
            List of (node_id, distance) tuples
        """
        if ef is None:
            ef = max(self.ef_construction, k)
        
        if self.entry_point is None:
            return []
        
        ep = [self.entry_point]
        
        # Traverse layers from top to 0
        for lc in range(self.max_layer, 0, -1):
            ep = self._search_layer(query, ep, 1, lc)
        
        # Search at layer 0
        ep = self._search_layer(query, ep, ef, 0)
        
        # Return k nearest
        results = [(node_id, self._distance(query, self.nodes[node_id].vector)) 
                  for node_id in ep]
        results.sort(key=lambda x: x[1])
        
        return results[:k]

# Example usage
print("\nHNSW (Hierarchical Navigable Small World):")

hnsw = HNSW(m=8, ef_construction=100)

# Insert vectors
print("Inserting vectors:")
vectors = [
    [1.0, 2.0, 3.0],
    [1.1, 2.1, 3.1],
    [5.0, 6.0, 7.0],
    [5.1, 6.1, 7.1],
    [10.0, 11.0, 12.0],
    [2.0, 3.0, 4.0],
    [8.0, 9.0, 10.0]
]

for i, vec in enumerate(vectors):
    node_id = hnsw.insert(vec)
    print(f"  Inserted vector {i}: {vec} as node {node_id}")

# Search for nearest neighbors
query = [1.0, 2.0, 3.0]
print(f"\nSearching for 3 nearest neighbors to {query}:")
results = hnsw.search(query, k=3)

for node_id, dist in results:
    vec = hnsw.nodes[node_id].vector
    print(f"  Node {node_id}: {vec}, distance: {dist:.4f}")

# Another query
query2 = [9.0, 10.0, 11.0]
print(f"\nSearching for 3 nearest neighbors to {query2}:")
results2 = hnsw.search(query2, k=3)

for node_id, dist in results2:
    vec = hnsw.nodes[node_id].vector
    print(f"  Node {node_id}: {vec}, distance: {dist:.4f}")

print("\nHNSW characteristics:")
print("  - Hierarchical structure for fast search")
print("  - Approximate nearest neighbor (very fast)")
print("  - Good for high-dimensional data")
print("  - Used in vector databases")
```

Inverted Index

When: Text search and information retrieval
Why: Maps terms to documents, essential for search engines
Examples: Search engines (Elasticsearch), document retrieval, full-text search

```python
# Python implementation of Inverted Index
# Core data structure for search engines

from typing import Dict, List, Set
from collections import defaultdict
import re

class PostingList:
    """Posting list for a term"""
    def __init__(self):
        self.doc_ids: List[int] = []
        self.frequencies: Dict[int, int] = {}  # doc_id -> term frequency
        self.positions: Dict[int, List[int]] = {}  # doc_id -> positions
    
    def add_occurrence(self, doc_id: int, position: int):
        """Add an occurrence of the term"""
        if doc_id not in self.frequencies:
            self.doc_ids.append(doc_id)
            self.frequencies[doc_id] = 0
            self.positions[doc_id] = []
        
        self.frequencies[doc_id] += 1
        self.positions[doc_id].append(position)
    
    def get_doc_frequency(self) -> int:
        """Number of documents containing this term"""
        return len(self.doc_ids)
    
    def get_term_frequency(self, doc_id: int) -> int:
        """Frequency of term in a specific document"""
        return self.frequencies.get(doc_id, 0)

class InvertedIndex:
    """
    Inverted Index for full-text search.
    Maps terms to documents containing them.
    """
    def __init__(self):
        self.index: Dict[str, PostingList] = defaultdict(PostingList)
        self.documents: Dict[int, str] = {}
        self.doc_lengths: Dict[int, int] = {}
        self.next_doc_id = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def add_document(self, text: str) -> int:
        """
        Add a document to the index.
        Returns the document ID.
        """
        doc_id = self.next_doc_id
        self.next_doc_id += 1
        
        self.documents[doc_id] = text
        
        # Tokenize and index
        tokens = self._tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        
        for position, token in enumerate(tokens):
            self.index[token].add_occurrence(doc_id, position)
        
        return doc_id
    
    def search(self, query: str) -> List[int]:
        """
        Search for documents containing query terms.
        Returns list of document IDs.
        """
        query_terms = self._tokenize(query)
        
        if not query_terms:
            return []
        
        # Get documents containing first term
        if query_terms[0] not in self.index:
            return []
        
        result_docs = set(self.index[query_terms[0]].doc_ids)
        
        # Intersect with documents containing other terms (AND query)
        for term in query_terms[1:]:
            if term in self.index:
                result_docs &= set(self.index[term].doc_ids)
            else:
                return []  # No documents contain all terms
        
        return list(result_docs)
    
    def search_or(self, query: str) -> List[int]:
        """
        Search for documents containing any query terms (OR query).
        Returns list of document IDs.
        """
        query_terms = self._tokenize(query)
        result_docs = set()
        
        for term in query_terms:
            if term in self.index:
                result_docs |= set(self.index[term].doc_ids)
        
        return list(result_docs)
    
    def search_phrase(self, phrase: str) -> List[int]:
        """
        Search for exact phrase.
        Returns list of document IDs containing the phrase.
        """
        phrase_terms = self._tokenize(phrase)
        
        if not phrase_terms:
            return []
        
        # Get candidates (documents containing all terms)
        candidates = self.search(phrase)
        
        # Check for phrase match
        result = []
        for doc_id in candidates:
            # Check if terms appear consecutively
            positions_lists = [self.index[term].positions[doc_id] 
                             for term in phrase_terms]
            
            # Check if consecutive positions exist
            if self._has_consecutive_positions(positions_lists):
                result.append(doc_id)
        
        return result
    
    def _has_consecutive_positions(self, positions_lists: List[List[int]]) -> bool:
        """Check if term positions are consecutive"""
        if not positions_lists:
            return False
        
        # For each position of first term
        for start_pos in positions_lists[0]:
            # Check if other terms follow consecutively
            consecutive = True
            for i, positions in enumerate(positions_lists[1:], 1):
                if start_pos + i not in positions:
                    consecutive = False
                    break
            
            if consecutive:
                return True
        
        return False
    
    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """Get frequency of term in document"""
        term = term.lower()
        if term in self.index:
            return self.index[term].get_term_frequency(doc_id)
        return 0
    
    def get_document_frequency(self, term: str) -> int:
        """Get number of documents containing term"""
        term = term.lower()
        if term in self.index:
            return self.index[term].get_doc_frequency()
        return 0
    
    def get_vocabulary_size(self) -> int:
        """Get number of unique terms"""
        return len(self.index)

# Example usage
print("\nInverted Index:")

index = InvertedIndex()

# Add documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog runs fast",
    "The lazy cat sleeps all day",
    "Quick foxes are clever animals"
]

print("Adding documents:")
for i, doc in enumerate(documents):
    doc_id = index.add_document(doc)
    print(f"  Doc {doc_id}: {doc}")

# Search queries
print("\nSearch queries (AND):")
queries = ["quick brown", "lazy", "fox dog"]
for query in queries:
    results = index.search(query)
    print(f"  Query '{query}': documents {results}")

# OR queries
print("\nSearch queries (OR):")
or_queries = ["fox cat", "fast clever"]
for query in or_queries:
    results = index.search_or(query)
    print(f"  Query '{query}': documents {results}")

# Phrase search
print("\nPhrase search:")
phrases = ["quick brown", "lazy dog", "brown fox"]
for phrase in phrases:
    results = index.search_phrase(phrase)
    print(f"  Phrase '{phrase}': documents {results}")

# Statistics
print("\nIndex statistics:")
print(f"  Total documents: {len(index.documents)}")
print(f"  Vocabulary size: {index.get_vocabulary_size()}")

term = "quick"
df = index.get_document_frequency(term)
print(f"  Document frequency of '{term}': {df}")
```

Posting List structures

When: Part of inverted index, storing document occurrences
Why: Efficient compression and intersection operations
Examples: Search engines, text databases

```python
# Python implementation of Posting List with compression
# Optimized storage for inverted index entries

from typing import List, Tuple
import bisect

class CompressedPostingList:
    """
    Compressed posting list using delta encoding and gap compression.
    Used in search engines for efficient storage.
    """
    def __init__(self):
        self.doc_ids: List[int] = []  # Sorted document IDs
        self.gaps: List[int] = []  # Delta-encoded gaps
    
    def add(self, doc_id: int):
        """Add document ID to posting list"""
        if not self.doc_ids:
            self.doc_ids.append(doc_id)
            self.gaps.append(doc_id)
        else:
            # Insert in sorted order
            pos = bisect.bisect_left(self.doc_ids, doc_id)
            
            if pos < len(self.doc_ids) and self.doc_ids[pos] == doc_id:
                return  # Already exists
            
            self.doc_ids.insert(pos, doc_id)
            
            # Recompute gaps
            self._recompute_gaps()
    
    def _recompute_gaps(self):
        """Recompute delta-encoded gaps"""
        self.gaps = []
        if not self.doc_ids:
            return
        
        self.gaps.append(self.doc_ids[0])
        for i in range(1, len(self.doc_ids)):
            self.gaps.append(self.doc_ids[i] - self.doc_ids[i-1])
    
    def intersect(self, other: 'CompressedPostingList') -> 'CompressedPostingList':
        """
        Intersect two posting lists.
        Optimized for sorted lists.
        """
        result = CompressedPostingList()
        
        i, j = 0, 0
        while i < len(self.doc_ids) and j < len(other.doc_ids):
            if self.doc_ids[i] == other.doc_ids[j]:
                result.add(self.doc_ids[i])
                i += 1
                j += 1
            elif self.doc_ids[i] < other.doc_ids[j]:
                i += 1
            else:
                j += 1
        
        return result
    
    def union(self, other: 'CompressedPostingList') -> 'CompressedPostingList':
        """Union of two posting lists"""
        result = CompressedPostingList()
        result.doc_ids = sorted(set(self.doc_ids) | set(other.doc_ids))
        result._recompute_gaps()
        return result
    
    def get_doc_ids(self) -> List[int]:
        """Get all document IDs"""
        return self.doc_ids.copy()
    
    def size(self) -> int:
        """Number of documents"""
        return len(self.doc_ids)
    
    def compressed_size(self) -> int:
        """Estimate compressed size in bytes"""
        # Variable-byte encoding: small gaps take less space
        total_bytes = 0
        for gap in self.gaps:
            if gap < 128:
                total_bytes += 1
            elif gap < 16384:
                total_bytes += 2
            else:
                total_bytes += 4
        return total_bytes

# Example usage
print("\nPosting List with Compression:")

# Create posting lists for terms
term1_list = CompressedPostingList()
term2_list = CompressedPostingList()

# Add documents to term1 ("quick")
print("Term 'quick' appears in documents:")
for doc_id in [1, 5, 10, 15, 20, 100, 200]:
    term1_list.add(doc_id)
    print(f"  {doc_id}")

# Add documents to term2 ("brown")
print("\nTerm 'brown' appears in documents:")
for doc_id in [1, 10, 25, 100, 150]:
    term2_list.add(doc_id)
    print(f"  {doc_id}")

# Intersection (AND query)
print("\nIntersection (documents with both 'quick' AND 'brown'):")
intersection = term1_list.intersect(term2_list)
print(f"  Documents: {intersection.get_doc_ids()}")

# Union (OR query)
print("\nUnion (documents with 'quick' OR 'brown'):")
union = term1_list.union(term2_list)
print(f"  Documents: {union.get_doc_ids()}")

# Compression statistics
print("\nCompression statistics:")
print(f"  Term1 posting list:")
print(f"    Doc IDs: {term1_list.get_doc_ids()}")
print(f"    Gaps: {term1_list.gaps}")
print(f"    Compressed size: ~{term1_list.compressed_size()} bytes")
print(f"    Uncompressed size: ~{term1_list.size() * 4} bytes")
print(f"    Compression ratio: {term1_list.size() * 4 / max(1, term1_list.compressed_size()):.2f}x")
```

Cover Tree

When: Nearest neighbor in metric spaces with theoretical guarantees
Why: Provably good bounds for NN queries in metric spaces
Examples: Machine learning, similarity search, clustering

```python
# Python implementation of Cover Tree
# Nearest neighbor search with provably good bounds

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CoverTreeNode:
    """Node in a Cover Tree"""
    point: List[float]
    level: int
    children: List['CoverTreeNode']
    
    def distance_to(self, other_point: List[float]) -> float:
        """Euclidean distance"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.point, other_point)))

class CoverTree:
    """
    Cover Tree for nearest neighbor search in metric spaces.
    Provides O(log n) search with theoretical guarantees.
    """
    def __init__(self, base=2.0):
        """
        Initialize Cover Tree.
        
        Args:
            base: Base for level separation (typically 2.0)
        """
        self.base = base
        self.root: Optional[CoverTreeNode] = None
        self.max_level = 0
        self.min_level = 0
    
    def _covering_radius(self, level: int) -> float:
        """Covering radius at given level"""
        return self.base ** level
    
    def insert(self, point: List[float]):
        """Insert a point into the cover tree"""
        if self.root is None:
            self.root = CoverTreeNode(point, 0, [])
            return
        
        # Find insertion level
        dist_to_root = self.root.distance_to(point)
        
        # Determine level for new point
        if dist_to_root > 0:
            level = math.floor(math.log(dist_to_root) / math.log(self.base))
        else:
            return  # Point already exists
        
        # Insert at appropriate level
        self._insert_at_level(self.root, point, level)
    
    def _insert_at_level(self, node: CoverTreeNode, point: List[float], level: int):
        """Insert point starting from node at given level"""
        if level > node.level:
            # Create new root
            new_root = CoverTreeNode(point, level, [node])
            self.root = new_root
            self.max_level = level
            return
        
        # Find closest child to insert under
        if not node.children:
            # Add as child
            child = CoverTreeNode(point, level, [])
            node.children.append(child)
        else:
            # Find best child
            best_child = min(node.children, key=lambda c: c.distance_to(point))
            self._insert_at_level(best_child, point, level)
    
    def nearest_neighbor(self, query: List[float]) -> Optional[Tuple[List[float], float]]:
        """
        Find nearest neighbor to query point.
        Returns (point, distance) tuple.
        """
        if self.root is None:
            return None
        
        return self._nearest_neighbor_recursive(self.root, query, float('inf'))
    
    def _nearest_neighbor_recursive(self, node: CoverTreeNode, query: List[float], 
                                   best_dist: float) -> Tuple[List[float], float]:
        """Recursively find nearest neighbor"""
        dist = node.distance_to(query)
        
        if dist < best_dist:
            best_point = node.point
            best_dist = dist
        else:
            best_point = None
        
        # Check children within covering radius
        for child in node.children:
            child_dist = child.distance_to(query)
            
            # Prune if too far
            if child_dist - self._covering_radius(child.level) < best_dist:
                result = self._nearest_neighbor_recursive(child, query, best_dist)
                if result[1] < best_dist:
                    best_point, best_dist = result
        
        return (best_point if best_point is not None else node.point, best_dist)
    
    def k_nearest_neighbors(self, query: List[float], k: int) -> List[Tuple[List[float], float]]:
        """Find k nearest neighbors"""
        if self.root is None:
            return []
        
        # Simple approach: collect all points and sort
        all_points = []
        self._collect_points(self.root, all_points)
        
        # Calculate distances and sort
        distances = [(p, math.sqrt(sum((a - b) ** 2 for a, b in zip(p, query)))) 
                    for p in all_points]
        distances.sort(key=lambda x: x[1])
        
        return distances[:k]
    
    def _collect_points(self, node: CoverTreeNode, points: List[List[float]]):
        """Collect all points in tree"""
        if node.point not in points:
            points.append(node.point)
        
        for child in node.children:
            self._collect_points(child, points)

# Example usage
print("\nCover Tree:")

ctree = CoverTree(base=2.0)

# Insert points
points = [
    [1.0, 2.0],
    [1.5, 2.5],
    [5.0, 6.0],
    [5.5, 6.5],
    [10.0, 11.0],
    [2.0, 3.0]
]

print("Inserting points:")
for p in points:
    ctree.insert(p)
    print(f"  {p}")

# Nearest neighbor search
query = [1.2, 2.2]
print(f"\nNearest neighbor to {query}:")
result = ctree.nearest_neighbor(query)
if result:
    point, dist = result
    print(f"  Point: {point}, Distance: {dist:.4f}")

# K nearest neighbors
print(f"\n3 nearest neighbors to {query}:")
knn_results = ctree.k_nearest_neighbors(query, k=3)
for point, dist in knn_results:
    print(f"  Point: {point}, Distance: {dist:.4f}")

print("\nCover Tree characteristics:")
print("  - Hierarchical structure with levels")
print("  - Provably good search bounds")
print("  - Works with any metric space")
print("  - Efficient for low-dimensional data")
```

Ball Tree

When: Nearest neighbor in high dimensions, better than k-d tree
Why: Works better in higher dimensions than k-d trees
Examples: scikit-learn, nearest neighbor in ML, clustering

```python
# Python implementation of Ball Tree
# Efficient for nearest neighbor in higher dimensions

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class BallTreeNode:
    """Node in a Ball Tree"""
    center: List[float]
    radius: float
    points: List[List[float]]
    left: Optional['BallTreeNode'] = None
    right: Optional['BallTreeNode'] = None
    is_leaf: bool = True

class BallTree:
    """
    Ball Tree for nearest neighbor search.
    More efficient than k-d tree in higher dimensions.
    """
    def __init__(self, points: List[List[float]], leaf_size=10):
        """
        Build Ball Tree from points.
        
        Args:
            points: List of points (each point is a list of coordinates)
            leaf_size: Maximum points in a leaf node
        """
        self.leaf_size = leaf_size
        self.dim = len(points[0]) if points else 0
        self.root = self._build_tree(points)
    
    def _distance(self, p1: List[float], p2: List[float]) -> float:
        """Euclidean distance"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    def _compute_centroid(self, points: List[List[float]]) -> List[float]:
        """Compute centroid of points"""
        if not points:
            return []
        
        n = len(points)
        dim = len(points[0])
        centroid = [sum(p[i] for p in points) / n for i in range(dim)]
        return centroid
    
    def _compute_radius(self, center: List[float], points: List[List[float]]) -> float:
        """Compute radius (max distance from center to any point)"""
        if not points:
            return 0.0
        return max(self._distance(center, p) for p in points)
    
    def _build_tree(self, points: List[List[float]]) -> Optional[BallTreeNode]:
        """Recursively build Ball Tree"""
        if not points:
            return None
        
        # Compute ball parameters
        center = self._compute_centroid(points)
        radius = self._compute_radius(center, points)
        
        # Create leaf if few points
        if len(points) <= self.leaf_size:
            return BallTreeNode(center, radius, points, is_leaf=True)
        
        # Split points along dimension with largest spread
        spreads = []
        for dim in range(len(points[0])):
            values = [p[dim] for p in points]
            spread = max(values) - min(values)
            spreads.append((spread, dim))
        
        _, split_dim = max(spreads)
        
        # Sort points along split dimension
        sorted_points = sorted(points, key=lambda p: p[split_dim])
        mid = len(sorted_points) // 2
        
        # Create internal node
        node = BallTreeNode(center, radius, [], is_leaf=False)
        node.left = self._build_tree(sorted_points[:mid])
        node.right = self._build_tree(sorted_points[mid:])
        
        return node
    
    def query(self, point: List[float], k=1) -> List[Tuple[List[float], float]]:
        """
        Find k nearest neighbors.
        
        Args:
            point: Query point
            k: Number of neighbors
        
        Returns:
            List of (neighbor_point, distance) tuples
        """
        if self.root is None:
            return []
        
        # Use priority queue to track k nearest
        candidates = []
        self._query_recursive(self.root, point, k, candidates)
        
        # Sort by distance
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]
    
    def _query_recursive(self, node: BallTreeNode, query: List[float], 
                        k: int, candidates: List[Tuple[List[float], float]]):
        """Recursively search for nearest neighbors"""
        if node is None:
            return
        
        # Distance from query to ball center
        dist_to_center = self._distance(query, node.center)
        
        if node.is_leaf:
            # Check all points in leaf
            for p in node.points:
                dist = self._distance(query, p)
                candidates.append((p, dist))
        else:
            # Determine which child to visit first
            left_dist = self._distance(query, node.left.center) if node.left else float('inf')
            right_dist = self._distance(query, node.right.center) if node.right else float('inf')
            
            if left_dist < right_dist:
                first, second = node.left, node.right
                first_dist, second_dist = left_dist, right_dist
            else:
                first, second = node.right, node.left
                first_dist, second_dist = right_dist, left_dist
            
            # Visit closer child
            if first:
                self._query_recursive(first, query, k, candidates)
            
            # Visit farther child if necessary
            # Prune if we have k candidates and farther ball is too far
            if second and len(candidates) < k:
                self._query_recursive(second, query, k, candidates)
            elif second and len(candidates) >= k:
                # Check if farther ball might contain closer points
                kth_dist = sorted(candidates, key=lambda x: x[1])[k-1][1]
                if second_dist - second.radius < kth_dist:
                    self._query_recursive(second, query, k, candidates)
    
    def query_radius(self, point: List[float], radius: float) -> List[List[float]]:
        """Find all points within given radius"""
        if self.root is None:
            return []
        
        results = []
        self._query_radius_recursive(self.root, point, radius, results)
        return results
    
    def _query_radius_recursive(self, node: BallTreeNode, query: List[float], 
                               radius: float, results: List[List[float]]):
        """Recursively find points within radius"""
        if node is None:
            return
        
        dist_to_center = self._distance(query, node.center)
        
        # Prune if ball is entirely outside query radius
        if dist_to_center - node.radius > radius:
            return
        
        if node.is_leaf:
            # Check all points in leaf
            for p in node.points:
                if self._distance(query, p) <= radius:
                    results.append(p)
        else:
            # Recursively search children
            if node.left:
                self._query_radius_recursive(node.left, query, radius, results)
            if node.right:
                self._query_radius_recursive(node.right, query, radius, results)

# Example usage
print("\nBall Tree:")

# Create dataset
points = [
    [1.0, 2.0],
    [1.5, 2.5],
    [5.0, 6.0],
    [5.5, 6.5],
    [10.0, 11.0],
    [10.5, 11.5],
    [2.0, 3.0],
    [8.0, 9.0]
]

print("Building Ball Tree with points:")
for p in points:
    print(f"  {p}")

btree = BallTree(points, leaf_size=3)

# Nearest neighbor query
query = [1.2, 2.2]
print(f"\nFinding 3 nearest neighbors to {query}:")
neighbors = btree.query(query, k=3)
for point, dist in neighbors:
    print(f"  Point: {point}, Distance: {dist:.4f}")

# Radius query
print(f"\nPoints within radius 2.0 of {query}:")
nearby = btree.query_radius(query, radius=2.0)
for point in nearby:
    print(f"  {point}")

print("\nBall Tree characteristics:")
print("  - Hierarchical ball decomposition")
print("  - Better than k-d tree in high dimensions")
print("  - Efficient range queries")
print("  - Used in scikit-learn")
```

VP-Tree (Vantage Point Tree) (Vantage Point Tree)

When: Metric space searching
Why: Works with any metric, not just Euclidean
Examples: Similarity search, image retrieval, pattern matching

```python
# Python implementation of VP-Tree (Vantage Point Tree)
# Works with any metric, not just Euclidean

import random
import math
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

@dataclass
class VPTreeNode:
    """Node in a VP-Tree"""
    vantage_point: List[float]
    threshold: float
    left: Optional['VPTreeNode'] = None  # Points closer than threshold
    right: Optional['VPTreeNode'] = None  # Points farther than threshold
    is_leaf: bool = False
    points: List[List[float]] = None  # For leaf nodes

class VPTree:
    """
    Vantage Point Tree for metric space searching.
    Works with any distance metric, not just Euclidean.
    """
    def __init__(self, points: List[List[float]], 
                 distance_func: Optional[Callable] = None,
                 leaf_size=5):
        """
        Build VP-Tree from points.
        
        Args:
            points: List of points
            distance_func: Distance function (default: Euclidean)
            leaf_size: Maximum points in a leaf node
        """
        self.leaf_size = leaf_size
        
        if distance_func is None:
            self.distance = lambda p1, p2: math.sqrt(
                sum((a - b) ** 2 for a, b in zip(p1, p2))
            )
        else:
            self.distance = distance_func
        
        self.root = self._build_tree(points)
    
    def _build_tree(self, points: List[List[float]]) -> Optional[VPTreeNode]:
        """Recursively build VP-Tree"""
        if not points:
            return None
        
        # Create leaf if few points
        if len(points) <= self.leaf_size:
            vp = points[0]
            node = VPTreeNode(vp, 0.0, is_leaf=True, points=points)
            return node
        
        # Choose vantage point (randomly or first point)
        vp_index = random.randint(0, len(points) - 1)
        vantage_point = points[vp_index]
        
        # Calculate distances from vantage point
        distances = [(p, self.distance(vantage_point, p)) 
                    for i, p in enumerate(points) if i != vp_index]
        
        if not distances:
            return VPTreeNode(vantage_point, 0.0, is_leaf=True, points=[vantage_point])
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Choose median as threshold
        median_idx = len(distances) // 2
        threshold = distances[median_idx][1]
        
        # Split into near and far
        near_points = [p for p, d in distances[:median_idx]]
        far_points = [p for p, d in distances[median_idx:]]
        
        # Create node
        node = VPTreeNode(vantage_point, threshold)
        node.left = self._build_tree(near_points)
        node.right = self._build_tree(far_points)
        
        return node
    
    def nearest_neighbor(self, query: List[float]) -> Tuple[List[float], float]:
        """
        Find nearest neighbor to query.
        
        Returns:
            (nearest_point, distance) tuple
        """
        if self.root is None:
            return None, float('inf')
        
        best = [None, float('inf')]  # [point, distance]
        self._search(self.root, query, best)
        return best[0], best[1]
    
    def _search(self, node: VPTreeNode, query: List[float], best: List):
        """Recursively search for nearest neighbor"""
        if node is None:
            return
        
        # Distance from query to vantage point
        dist = self.distance(query, node.vantage_point)
        
        if node.is_leaf:
            # Check all points in leaf
            for p in node.points:
                d = self.distance(query, p)
                if d < best[1]:
                    best[0] = p
                    best[1] = d
            return
        
        # Update best if vantage point is closer
        if dist < best[1]:
            best[0] = node.vantage_point
            best[1] = dist
        
        # Decide which subtree to search first
        if dist < node.threshold:
            # Query is in near region
            if node.left:
                self._search(node.left, query, best)
            
            # Check far region if necessary
            if node.right and dist + best[1] >= node.threshold:
                self._search(node.right, query, best)
        else:
            # Query is in far region
            if node.right:
                self._search(node.right, query, best)
            
            # Check near region if necessary
            if node.left and dist - best[1] <= node.threshold:
                self._search(node.left, query, best)
    
    def k_nearest_neighbors(self, query: List[float], k: int) -> List[Tuple[List[float], float]]:
        """Find k nearest neighbors"""
        if self.root is None:
            return []
        
        # Collect all candidates
        candidates = []
        self._collect_all(self.root, query, candidates)
        
        # Sort by distance and return k nearest
        candidates.sort(key=lambda x: x[1])
        return candidates[:k]
    
    def _collect_all(self, node: VPTreeNode, query: List[float], 
                     candidates: List[Tuple[List[float], float]]):
        """Collect all points with distances"""
        if node is None:
            return
        
        if node.is_leaf:
            for p in node.points:
                candidates.append((p, self.distance(query, p)))
        else:
            candidates.append((node.vantage_point, self.distance(query, node.vantage_point)))
            self._collect_all(node.left, query, candidates)
            self._collect_all(node.right, query, candidates)
    
    def range_search(self, query: List[float], radius: float) -> List[List[float]]:
        """Find all points within radius of query"""
        if self.root is None:
            return []
        
        results = []
        self._range_search_recursive(self.root, query, radius, results)
        return results
    
    def _range_search_recursive(self, node: VPTreeNode, query: List[float], 
                                radius: float, results: List[List[float]]):
        """Recursively find points within radius"""
        if node is None:
            return
        
        dist = self.distance(query, node.vantage_point)
        
        if node.is_leaf:
            for p in node.points:
                if self.distance(query, p) <= radius:
                    results.append(p)
            return
        
        if dist <= radius:
            results.append(node.vantage_point)
        
        # Check which subtrees might contain points within radius
        if dist - radius <= node.threshold and node.left:
            self._range_search_recursive(node.left, query, radius, results)
        
        if dist + radius >= node.threshold and node.right:
            self._range_search_recursive(node.right, query, radius, results)

# Example usage
print("\nVP-Tree (Vantage Point Tree):")

# Create dataset
points = [
    [1.0, 2.0],
    [1.5, 2.5],
    [5.0, 6.0],
    [5.5, 6.5],
    [10.0, 11.0],
    [2.0, 3.0],
    [8.0, 9.0],
    [3.0, 4.0]
]

print("Building VP-Tree with points:")
for p in points:
    print(f"  {p}")

vptree = VPTree(points, leaf_size=3)

# Nearest neighbor
query = [1.2, 2.2]
print(f"\nNearest neighbor to {query}:")
nearest, dist = vptree.nearest_neighbor(query)
print(f"  Point: {nearest}, Distance: {dist:.4f}")

# K nearest neighbors
print(f"\n3 nearest neighbors to {query}:")
knn = vptree.k_nearest_neighbors(query, k=3)
for point, dist in knn:
    print(f"  Point: {point}, Distance: {dist:.4f}")

# Range search
print(f"\nPoints within radius 3.0 of {query}:")
nearby = vptree.range_search(query, radius=3.0)
for point in nearby:
    print(f"  {point}")

print("\nVP-Tree characteristics:")
print("  - Works with any metric (not just Euclidean)")
print("  - Binary tree structure")
print("  - Good for similarity search")
print("  - Efficient pruning using triangle inequality")
```

Huffman Tree

When: Optimal prefix-free encoding
Why: Constructs optimal compression codes
Examples: Data compression, encoding/decoding algorithms

```python
# Python implementation of Huffman Tree
# Optimal prefix-free encoding for compression

import heapq
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

@dataclass(order=True)
class HuffmanNode:
    """Node in Huffman Tree"""
    frequency: int
    char: Optional[str] = field(compare=False, default=None)
    left: Optional['HuffmanNode'] = field(compare=False, default=None)
    right: Optional['HuffmanNode'] = field(compare=False, default=None)
    
    def is_leaf(self):
        return self.left is None and self.right is None

class HuffmanTree:
    """
    Huffman Tree for optimal prefix-free encoding.
    Used in data compression algorithms.
    """
    def __init__(self, text: str):
        """
        Build Huffman tree from text.
        
        Args:
            text: Input text to encode
        """
        self.text = text
        self.root: Optional[HuffmanNode] = None
        self.codes: Dict[str, str] = {}
        self.reverse_codes: Dict[str, str] = {}
        
        # Build the tree
        self._build_tree()
        self._generate_codes()
    
    def _build_tree(self):
        """Build Huffman tree using greedy algorithm"""
        # Count character frequencies
        freq_map = {}
        for char in self.text:
            freq_map[char] = freq_map.get(char, 0) + 1
        
        if not freq_map:
            return
        
        # Special case: only one unique character
        if len(freq_map) == 1:
            char = list(freq_map.keys())[0]
            self.root = HuffmanNode(freq_map[char], char)
            return
        
        # Create priority queue with leaf nodes
        heap = [HuffmanNode(freq, char) for char, freq in freq_map.items()]
        heapq.heapify(heap)
        
        # Build tree bottom-up
        while len(heap) > 1:
            # Take two nodes with minimum frequency
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            # Create parent node
            parent = HuffmanNode(
                frequency=left.frequency + right.frequency,
                left=left,
                right=right
            )
            
            heapq.heappush(heap, parent)
        
        self.root = heap[0]
    
    def _generate_codes(self):
        """Generate Huffman codes for each character"""
        if self.root is None:
            return
        
        # Special case: single character
        if self.root.is_leaf():
            self.codes[self.root.char] = '0'
            self.reverse_codes['0'] = self.root.char
            return
        
        # Traverse tree to generate codes
        self._generate_codes_recursive(self.root, '')
    
    def _generate_codes_recursive(self, node: HuffmanNode, code: str):
        """Recursively generate codes"""
        if node is None:
            return
        
        if node.is_leaf():
            self.codes[node.char] = code
            self.reverse_codes[code] = node.char
            return
        
        self._generate_codes_recursive(node.left, code + '0')
        self._generate_codes_recursive(node.right, code + '1')
    
    def encode(self, text: str = None) -> str:
        """
        Encode text using Huffman codes.
        
        Args:
            text: Text to encode (uses original text if None)
        
        Returns:
            Binary string representation
        """
        if text is None:
            text = self.text
        
        encoded = []
        for char in text:
            if char in self.codes:
                encoded.append(self.codes[char])
            else:
                raise ValueError(f"Character '{char}' not in Huffman tree")
        
        return ''.join(encoded)
    
    def decode(self, encoded: str) -> str:
        """
        Decode binary string using Huffman tree.
        
        Args:
            encoded: Binary string to decode
        
        Returns:
            Decoded text
        """
        if self.root is None:
            return ''
        
        # Special case: single character
        if self.root.is_leaf():
            return self.root.char * len(encoded)
        
        decoded = []
        current = self.root
        
        for bit in encoded:
            if bit == '0':
                current = current.left
            else:
                current = current.right
            
            if current.is_leaf():
                decoded.append(current.char)
                current = self.root
        
        return ''.join(decoded)
    
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio"""
        original_bits = len(self.text) * 8  # ASCII: 8 bits per char
        encoded = self.encode()
        compressed_bits = len(encoded)
        
        return original_bits / compressed_bits if compressed_bits > 0 else 0
    
    def get_code_table(self) -> Dict[str, Tuple[str, int]]:
        """Get code table with frequencies"""
        table = {}
        for char, code in self.codes.items():
            freq = self.text.count(char)
            table[char] = (code, freq)
        return table

# Example usage
print("\nHuffman Tree:")

# Create text with varying frequencies
text = "ABRACADABRA"
print(f"Original text: {text}")

# Build Huffman tree
huffman = HuffmanTree(text)

# Show code table
print("\nHuffman codes:")
code_table = huffman.get_code_table()
for char, (code, freq) in sorted(code_table.items()):
    print(f"  '{char}': {code} (frequency: {freq})")

# Encode
encoded = huffman.encode()
print(f"\nEncoded: {encoded}")
print(f"Original length: {len(text)} chars = {len(text) * 8} bits")
print(f"Encoded length: {len(encoded)} bits")
print(f"Compression ratio: {huffman.get_compression_ratio():.2f}x")

# Decode
decoded = huffman.decode(encoded)
print(f"\nDecoded: {decoded}")
print(f"Correct: {decoded == text}")

# Another example with more varied text
text2 = "this is an example of huffman encoding"
print(f"\n\nExample 2: '{text2}'")
huffman2 = HuffmanTree(text2)

print("\nTop 5 most frequent characters:")
code_table2 = huffman2.get_code_table()
sorted_codes = sorted(code_table2.items(), key=lambda x: x[1][1], reverse=True)
for char, (code, freq) in sorted_codes[:5]:
    print(f"  '{char}': {code} (frequency: {freq})")

print(f"\nCompression ratio: {huffman2.get_compression_ratio():.2f}x")
```

Wavelet Tree

When: Succinct data structure for sequences with rank/select
Why: Space-efficient, supports many operations
Examples: Compressed text indexing, range counting

```python
# Python implementation of Wavelet Tree
# Succinct data structure for sequences with rank/select operations

from typing import List, Optional

class WaveletTreeNode:
    """Node in a Wavelet Tree"""
    def __init__(self, alphabet_start: int, alphabet_end: int):
        self.alphabet_start = alphabet_start
        self.alphabet_end = alphabet_end
        self.bitmap: List[int] = []  # 0 for left, 1 for right
        self.left: Optional['WaveletTreeNode'] = None
        self.right: Optional['WaveletTreeNode'] = None
    
    def is_leaf(self):
        return self.alphabet_start == self.alphabet_end

class WaveletTree:
    """
    Wavelet Tree: Succinct data structure for sequences.
    Supports rank, select, and range counting queries.
    """
    def __init__(self, sequence: List[int]):
        """
        Build Wavelet Tree from sequence.
        
        Args:
            sequence: List of integers (0 to max_val)
        """
        self.sequence = sequence
        self.n = len(sequence)
        
        if not sequence:
            self.root = None
            self.alphabet_size = 0
            return
        
        self.alphabet_size = max(sequence) + 1
        self.root = self._build(sequence, 0, self.alphabet_size - 1)
    
    def _build(self, sequence: List[int], alpha_start: int, alpha_end: int) -> Optional[WaveletTreeNode]:
        """Recursively build Wavelet Tree"""
        if not sequence:
            return None
        
        node = WaveletTreeNode(alpha_start, alpha_end)
        
        # Leaf node
        if alpha_start == alpha_end:
            return node
        
        # Split alphabet in half
        alpha_mid = (alpha_start + alpha_end) // 2
        
        # Create bitmap and split sequence
        left_seq = []
        right_seq = []
        
        for val in sequence:
            if val <= alpha_mid:
                node.bitmap.append(0)
                left_seq.append(val)
            else:
                node.bitmap.append(1)
                right_seq.append(val)
        
        # Build children
        node.left = self._build(left_seq, alpha_start, alpha_mid)
        node.right = self._build(right_seq, alpha_mid + 1, alpha_end)
        
        return node
    
    def access(self, index: int) -> int:
        """
        Access element at index.
        O(log alphabet_size) time.
        """
        if index < 0 or index >= self.n:
            raise IndexError("Index out of bounds")
        
        return self._access_recursive(self.root, index)
    
    def _access_recursive(self, node: WaveletTreeNode, index: int) -> int:
        """Recursively access element"""
        if node.is_leaf():
            return node.alphabet_start
        
        if node.bitmap[index] == 0:
            # Go left - count 0s before this position
            new_index = node.bitmap[:index].count(0)
            return self._access_recursive(node.left, new_index)
        else:
            # Go right - count 1s before this position
            new_index = node.bitmap[:index].count(1)
            return self._access_recursive(node.right, new_index)
    
    def rank(self, value: int, index: int) -> int:
        """
        Count occurrences of value in sequence[0:index].
        O(log alphabet_size) time.
        """
        if index <= 0:
            return 0
        if index > self.n:
            index = self.n
        
        return self._rank_recursive(self.root, value, index)
    
    def _rank_recursive(self, node: WaveletTreeNode, value: int, index: int) -> int:
        """Recursively compute rank"""
        if node is None or index == 0:
            return 0
        
        if node.is_leaf():
            return index if node.alphabet_start == value else 0
        
        alpha_mid = (node.alphabet_start + node.alphabet_end) // 2
        
        if value <= alpha_mid:
            # Count 0s in bitmap up to index
            zeros_count = node.bitmap[:index].count(0)
            return self._rank_recursive(node.left, value, zeros_count)
        else:
            # Count 1s in bitmap up to index
            ones_count = node.bitmap[:index].count(1)
            return self._rank_recursive(node.right, value, ones_count)
    
    def range_count(self, left: int, right: int, value_min: int, value_max: int) -> int:
        """
        Count occurrences of values in [value_min, value_max] 
        within sequence[left:right].
        O(log alphabet_size) time.
        """
        if left >= right or left < 0 or right > self.n:
            return 0
        
        return self._range_count_recursive(self.root, left, right, value_min, value_max)
    
    def _range_count_recursive(self, node: WaveletTreeNode, left: int, right: int,
                               value_min: int, value_max: int) -> int:
        """Recursively count values in range"""
        if node is None or left >= right:
            return 0
        
        # Check if node's range intersects query range
        if value_max < node.alphabet_start or value_min > node.alphabet_end:
            return 0
        
        # If node's range is completely within query range
        if value_min <= node.alphabet_start and node.alphabet_end <= value_max:
            return right - left
        
        if node.is_leaf():
            if value_min <= node.alphabet_start <= value_max:
                return right - left
            return 0
        
        # Count in left and right subtrees
        left_zeros = node.bitmap[:left].count(0)
        right_zeros = node.bitmap[:right].count(0)
        
        left_ones = left - left_zeros
        right_ones = right - right_zeros
        
        count = 0
        if node.left:
            count += self._range_count_recursive(node.left, left_zeros, right_zeros, 
                                                 value_min, value_max)
        if node.right:
            count += self._range_count_recursive(node.right, left_ones, right_ones,
                                                 value_min, value_max)
        
        return count

# Example usage
print("\nWavelet Tree:")

# Create sequence
sequence = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
print(f"Sequence: {sequence}")

wt = WaveletTree(sequence)

# Access operation
print("\nAccess operations:")
for i in [0, 3, 7]:
    val = wt.access(i)
    print(f"  sequence[{i}] = {val} (verify: {sequence[i]})")

# Rank operation
print("\nRank operations (count occurrences):")
for val in [1, 5, 9]:
    for idx in [5, len(sequence)]:
        count = wt.rank(val, idx)
        actual = sequence[:idx].count(val)
        print(f"  rank({val}, {idx}) = {count} (verify: {actual})")

# Range counting
print("\nRange counting:")
queries = [
    (2, 8, 1, 5),   # Count values in [1,5] within positions [2,8)
    (0, 11, 5, 9),  # Count values in [5,9] within positions [0,11)
]

for left, right, val_min, val_max in queries:
    count = wt.range_count(left, right, val_min, val_max)
    # Verify
    actual = sum(1 for i in range(left, min(right, len(sequence))) 
                 if val_min <= sequence[i] <= val_max)
    print(f"  range_count([{left},{right}), values [{val_min},{val_max}]) = {count} (verify: {actual})")

print("\nWavelet Tree characteristics:")
print("  - Succinct data structure")
print("  - O(log alphabet_size) operations")
print("  - Supports rank, select, range counting")
print("  - Used in compressed text indexing")
```

Range Minimum Query (RMQ) structures (RMQ) structures

When: Need fast range minimum queries
Why: <O(n), O(1)> preprocessing and query time
Examples: Lowest common ancestor, computational geometry

```python
# Python implementation of Range Minimum Query (RMQ) structures
# Preprocessing for fast range minimum queries

from typing import List
import math

class RMQPreprocessing:
    """
    Range Minimum Query with <O(n log n), O(1)> preprocessing and query.
    Uses sparse table approach.
    """
    def __init__(self, array: List[int]):
        """
        Preprocess array for RMQ.
        O(n log n) preprocessing time and space.
        """
        self.array = array
        self.n = len(array)
        
        if self.n == 0:
            return
        
        # Build sparse table
        self.log_table = [0] * (self.n + 1)
        self.sparse_table = []
        self._build_sparse_table()
    
    def _build_sparse_table(self):
        """Build sparse table for RMQ"""
        # Precompute log values
        for i in range(2, self.n + 1):
            self.log_table[i] = self.log_table[i // 2] + 1
        
        # sparse_table[i][j] = index of minimum in range [j, j + 2^i)
        max_log = self.log_table[self.n] + 1
        self.sparse_table = [[0] * self.n for _ in range(max_log)]
        
        # Base case: ranges of length 1
        for i in range(self.n):
            self.sparse_table[0][i] = i
        
        # Build table for larger ranges
        for i in range(1, max_log):
            length = 1 << i  # 2^i
            for j in range(self.n):
                if j + length // 2 < self.n:
                    left_min_idx = self.sparse_table[i-1][j]
                    right_min_idx = self.sparse_table[i-1][j + length // 2]
                    
                    if self.array[left_min_idx] <= self.array[right_min_idx]:
                        self.sparse_table[i][j] = left_min_idx
                    else:
                        self.sparse_table[i][j] = right_min_idx
                else:
                    self.sparse_table[i][j] = self.sparse_table[i-1][j]
    
    def query(self, left: int, right: int) -> int:
        """
        Query minimum in range [left, right] (inclusive).
        O(1) time.
        
        Returns: Index of minimum element
        """
        if left > right or left < 0 or right >= self.n:
            raise IndexError("Invalid range")
        
        # Find largest power of 2 that fits in range
        length = right - left + 1
        k = self.log_table[length]
        
        # Compare minimums of two overlapping ranges
        left_min_idx = self.sparse_table[k][left]
        right_min_idx = self.sparse_table[k][right - (1 << k) + 1]
        
        if self.array[left_min_idx] <= self.array[right_min_idx]:
            return left_min_idx
        else:
            return right_min_idx
    
    def query_value(self, left: int, right: int) -> int:
        """
        Query minimum value in range [left, right].
        O(1) time.
        """
        idx = self.query(left, right)
        return self.array[idx]

class RMQSegmentTree:
    """
    Range Minimum Query using Segment Tree.
    Supports both queries and updates.
    """
    def __init__(self, array: List[int]):
        """Build segment tree - O(n) time"""
        self.n = len(array)
        self.tree = [float('inf')] * (4 * self.n)
        self.array = array.copy()
        
        if self.n > 0:
            self._build(0, 0, self.n - 1)
    
    def _build(self, node: int, start: int, end: int):
        """Recursively build segment tree"""
        if start == end:
            self.tree[node] = self.array[start]
            return
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        self._build(left_child, start, mid)
        self._build(right_child, mid + 1, end)
        
        self.tree[node] = min(self.tree[left_child], self.tree[right_child])
    
    def query(self, left: int, right: int) -> int:
        """
        Query minimum in range [left, right].
        O(log n) time.
        """
        return self._query_recursive(0, 0, self.n - 1, left, right)
    
    def _query_recursive(self, node: int, start: int, end: int, 
                         left: int, right: int) -> int:
        """Recursively query range"""
        # No overlap
        if right < start or left > end:
            return float('inf')
        
        # Complete overlap
        if left <= start and end <= right:
            return self.tree[node]
        
        # Partial overlap
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        left_min = self._query_recursive(left_child, start, mid, left, right)
        right_min = self._query_recursive(right_child, mid + 1, end, left, right)
        
        return min(left_min, right_min)
    
    def update(self, index: int, value: int):
        """
        Update element at index.
        O(log n) time.
        """
        self.array[index] = value
        self._update_recursive(0, 0, self.n - 1, index, value)
    
    def _update_recursive(self, node: int, start: int, end: int, 
                          index: int, value: int):
        """Recursively update segment tree"""
        if start == end:
            self.tree[node] = value
            return
        
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        if index <= mid:
            self._update_recursive(left_child, start, mid, index, value)
        else:
            self._update_recursive(right_child, mid + 1, end, index, value)
        
        self.tree[node] = min(self.tree[left_child], self.tree[right_child])

# Example usage
print("\nRange Minimum Query (RMQ) structures:")

array = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
print(f"Array: {array}")

# Using preprocessing approach (fast queries, no updates)
print("\n1. RMQ with preprocessing (static):")
rmq_prep = RMQPreprocessing(array)

queries = [(0, 3), (2, 7), (5, 9)]
for left, right in queries:
    min_idx = rmq_prep.query(left, right)
    min_val = rmq_prep.query_value(left, right)
    print(f"  RMQ({left}, {right}) = {min_val} at index {min_idx}")

# Using segment tree (supports updates)
print("\n2. RMQ with Segment Tree (dynamic):")
rmq_tree = RMQSegmentTree(array)

for left, right in queries:
    min_val = rmq_tree.query(left, right)
    print(f"  RMQ({left}, {right}) = {min_val}")

# Update and requery
print("\nUpdating array[2] = 0:")
rmq_tree.update(2, 0)
min_val = rmq_tree.query(0, 5)
print(f"  RMQ(0, 5) after update = {min_val}")

print("\nComparison:")
print("  Preprocessing approach: O(n log n) space, O(1) query, no updates")
print("  Segment tree: O(n) space, O(log n) query, O(log n) update")
```

Sparse Table

When: Static RMQ or other idempotent range queries
Why: O(n log n) space, O(1) query for idempotent operations
Examples: Range minimum/maximum on static arrays, GCD queries

```python
# Python implementation of Sparse Table
# O(n log n) space, O(1) query for idempotent operations

from typing import List, Callable
import math

class SparseTable:
    """
    Sparse Table for static range queries on idempotent operations.
    Idempotent: f(x, x) = x (e.g., min, max, gcd)
    """
    def __init__(self, array: List[int], operation: Callable[[int, int], int] = min):
        """
        Build sparse table.
        
        Args:
            array: Input array
            operation: Binary operation (must be idempotent)
        
        Time: O(n log n)
        Space: O(n log n)
        """
        self.array = array
        self.n = len(array)
        self.op = operation
        
        if self.n == 0:
            return
        
        # Compute logarithms
        self.log = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.log[i] = self.log[i // 2] + 1
        
        # Build sparse table
        # table[k][i] = op over range [i, i + 2^k)
        self.k_max = self.log[self.n] + 1
        self.table = [[0] * self.n for _ in range(self.k_max)]
        
        self._build()
    
    def _build(self):
        """Build the sparse table"""
        # Base case: ranges of length 1
        for i in range(self.n):
            self.table[0][i] = self.array[i]
        
        # Fill table for larger ranges
        for k in range(1, self.k_max):
            length = 1 << k  # 2^k
            
            for i in range(self.n):
                if i + (length // 2) < self.n:
                    left = self.table[k-1][i]
                    right = self.table[k-1][i + (length // 2)]
                    self.table[k][i] = self.op(left, right)
                else:
                    self.table[k][i] = self.table[k-1][i]
    
    def query(self, left: int, right: int) -> int:
        """
        Query operation over range [left, right] (inclusive).
        O(1) time.
        """
        if left > right or left < 0 or right >= self.n:
            raise IndexError("Invalid range")
        
        # Find largest power of 2 that fits in range
        length = right - left + 1
        k = self.log[length]
        
        # Query two overlapping ranges
        # For idempotent operations, overlap is okay
        left_result = self.table[k][left]
        right_result = self.table[k][right - (1 << k) + 1]
        
        return self.op(left_result, right_result)

class SparseTable2D:
    """
    2D Sparse Table for rectangular range queries.
    """
    def __init__(self, matrix: List[List[int]], operation: Callable[[int, int], int] = min):
        """
        Build 2D sparse table.
        
        Args:
            matrix: 2D array (list of lists)
            operation: Binary operation (must be idempotent)
        """
        self.matrix = matrix
        self.rows = len(matrix)
        self.cols = len(matrix[0]) if matrix else 0
        self.op = operation
        
        if self.rows == 0 or self.cols == 0:
            return
        
        # Compute logarithms
        max_dim = max(self.rows, self.cols) + 1
        self.log = [0] * max_dim
        for i in range(2, max_dim):
            self.log[i] = self.log[i // 2] + 1
        
        # Build 2D sparse table
        self.k_rows = self.log[self.rows] + 1
        self.k_cols = self.log[self.cols] + 1
        
        self.table = [[[[0 for _ in range(self.cols)]
                        for _ in range(self.rows)]
                       for _ in range(self.k_cols)]
                      for _ in range(self.k_rows)]
        
        self._build()
    
    def _build(self):
        """Build 2D sparse table"""
        # Base case: 1x1 rectangles
        for i in range(self.rows):
            for j in range(self.cols):
                self.table[0][0][i][j] = self.matrix[i][j]
        
        # Build for rows
        for ki in range(1, self.k_rows):
            step_i = 1 << ki
            for i in range(self.rows):
                if i + step_i // 2 < self.rows:
                    for j in range(self.cols):
                        self.table[ki][0][i][j] = self.op(
                            self.table[ki-1][0][i][j],
                            self.table[ki-1][0][i + step_i // 2][j]
                        )
        
        # Build for columns
        for ki in range(self.k_rows):
            for kj in range(1, self.k_cols):
                step_j = 1 << kj
                for i in range(self.rows):
                    for j in range(self.cols):
                        if j + step_j // 2 < self.cols:
                            self.table[ki][kj][i][j] = self.op(
                                self.table[ki][kj-1][i][j],
                                self.table[ki][kj-1][i][j + step_j // 2]
                            )
    
    def query(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """
        Query rectangle from (r1,c1) to (r2,c2) inclusive.
        O(1) time.
        """
        ki = self.log[r2 - r1 + 1]
        kj = self.log[c2 - c1 + 1]
        
        # Query four overlapping rectangles
        tl = self.table[ki][kj][r1][c1]
        tr = self.table[ki][kj][r1][c2 - (1 << kj) + 1]
        bl = self.table[ki][kj][r2 - (1 << ki) + 1][c1]
        br = self.table[ki][kj][r2 - (1 << ki) + 1][c2 - (1 << kj) + 1]
        
        return self.op(self.op(tl, tr), self.op(bl, br))

# Example usage
print("\nSparse Table:")

# 1D Sparse Table - Range Minimum
array = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
print(f"Array: {array}")

st_min = SparseTable(array, operation=min)

print("\nRange Minimum Queries:")
queries = [(0, 3), (2, 7), (5, 9), (0, 9)]
for left, right in queries:
    result = st_min.query(left, right)
    print(f"  min({left}, {right}) = {result}")

# Sparse Table with different operation - Range Maximum
st_max = SparseTable(array, operation=max)

print("\nRange Maximum Queries:")
for left, right in queries[:3]:
    result = st_max.query(left, right)
    print(f"  max({left}, {right}) = {result}")

# Sparse Table for GCD
import math as m

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

array_gcd = [12, 18, 24, 36, 6, 30]
print(f"\nArray for GCD: {array_gcd}")

st_gcd = SparseTable(array_gcd, operation=gcd)

print("Range GCD Queries:")
gcd_queries = [(0, 2), (1, 4), (0, 5)]
for left, right in gcd_queries:
    result = st_gcd.query(left, right)
    print(f"  gcd({left}, {right}) = {result}")

# 2D Sparse Table
print("\n2D Sparse Table:")
matrix = [
    [3, 1, 4, 8],
    [2, 5, 1, 9],
    [6, 3, 7, 2],
    [4, 8, 3, 5]
]

print("Matrix:")
for row in matrix:
    print(f"  {row}")

st_2d = SparseTable2D(matrix, operation=min)

print("\n2D Range Minimum Queries:")
rect_queries = [(0, 0, 1, 1), (1, 1, 3, 3), (0, 0, 3, 3)]
for r1, c1, r2, c2 in rect_queries:
    result = st_2d.query(r1, c1, r2, c2)
    print(f"  min([{r1},{c1}] to [{r2},{c2}]) = {result}")

print("\nSparse Table characteristics:")
print("  - O(n log n) preprocessing")
print("  - O(1) query time")
print("  - Works for idempotent operations")
print("  - Static (no updates)")
```

Heavy-Light Decomposition (not structure but pattern) (not structure but pattern)

When: Tree path queries and updates
Why: Decomposes tree into O(log n) paths
Examples: Tree query problems, competitive programming

```python
# Python implementation of Heavy-Light Decomposition
# Decomposes a tree into heavy and light edges for efficient path queries

class HeavyLightDecomposition:
    """
    Heavy-Light Decomposition for tree path queries and updates.
    Decomposes tree into O(log n) chains for efficient queries.
    """
    def __init__(self, n):
        """
        Initialize HLD for a tree with n nodes (0-indexed).
        """
        self.n = n
        self.graph = [[] for _ in range(n)]
        
        # HLD arrays
        self.parent = [-1] * n
        self.depth = [0] * n
        self.heavy = [-1] * n  # heavy child of each node
        self.head = list(range(n))  # head of the chain
        self.pos = [0] * n  # position in segment tree
        self.size = [1] * n  # subtree size
        
        self.cur_pos = 0
        
    def add_edge(self, u, v):
        """Add an undirected edge between u and v"""
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def build(self, root=0):
        """Build the HLD from the given root"""
        self._dfs_size(root, -1)
        self._dfs_hld(root, -1)
    
    def _dfs_size(self, u, p):
        """First DFS: calculate subtree sizes and find heavy children"""
        self.parent[u] = p
        self.size[u] = 1
        max_size = 0
        
        for v in self.graph[u]:
            if v == p:
                continue
            
            self.depth[v] = self.depth[u] + 1
            self._dfs_size(v, u)
            self.size[u] += self.size[v]
            
            # Find heavy child (child with largest subtree)
            if self.size[v] > max_size:
                max_size = self.size[v]
                self.heavy[u] = v
    
    def _dfs_hld(self, u, p):
        """Second DFS: decompose tree into chains"""
        self.pos[u] = self.cur_pos
        self.cur_pos += 1
        
        # Process heavy child first
        if self.heavy[u] != -1:
            self.head[self.heavy[u]] = self.head[u]
            self._dfs_hld(self.heavy[u], u)
        
        # Process light children
        for v in self.graph[u]:
            if v == p or v == self.heavy[u]:
                continue
            self._dfs_hld(v, u)
    
    def lca(self, u, v):
        """Find Lowest Common Ancestor of u and v using HLD"""
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] > self.depth[self.head[v]]:
                u = self.parent[self.head[u]]
            else:
                v = self.parent[self.head[v]]
        
        return u if self.depth[u] < self.depth[v] else v
    
    def path_query(self, u, v, query_func):
        """
        Query on path from u to v using query_func.
        query_func should query a range [l, r] in the underlying data structure.
        Returns list of query results for each chain segment.
        """
        results = []
        
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] > self.depth[self.head[v]]:
                # Query from pos[head[u]] to pos[u]
                results.append(query_func(self.pos[self.head[u]], self.pos[u]))
                u = self.parent[self.head[u]]
            else:
                # Query from pos[head[v]] to pos[v]
                results.append(query_func(self.pos[self.head[v]], self.pos[v]))
                v = self.parent[self.head[v]]
        
        # Query the remaining path in the same chain
        if self.depth[u] > self.depth[v]:
            results.append(query_func(self.pos[v], self.pos[u]))
        else:
            results.append(query_func(self.pos[u], self.pos[v]))
        
        return results
    
    def path_update(self, u, v, update_func):
        """
        Update on path from u to v using update_func.
        update_func should update a range [l, r] in the underlying data structure.
        """
        while self.head[u] != self.head[v]:
            if self.depth[self.head[u]] > self.depth[self.head[v]]:
                update_func(self.pos[self.head[u]], self.pos[u])
                u = self.parent[self.head[u]]
            else:
                update_func(self.pos[self.head[v]], self.pos[v])
                v = self.parent[self.head[v]]
        
        # Update the remaining path in the same chain
        if self.depth[u] > self.depth[v]:
            update_func(self.pos[v], self.pos[u])
        else:
            update_func(self.pos[u], self.pos[v])

# Example usage with path sum queries
class PathSumHLD:
    """HLD with path sum queries using segment tree"""
    def __init__(self, n, values):
        self.hld = HeavyLightDecomposition(n)
        self.values = values
        self.tree = [0] * (4 * n)
        
    def add_edge(self, u, v):
        self.hld.add_edge(u, v)
    
    def build(self, root=0):
        self.hld.build(root)
        self._build_tree(0, 0, self.hld.n - 1)
    
    def _build_tree(self, node, start, end):
        """Build segment tree based on HLD positions"""
        if start == end:
            # Find which original node maps to this position
            for i in range(self.hld.n):
                if self.hld.pos[i] == start:
                    self.tree[node] = self.values[i]
                    break
        else:
            mid = (start + end) // 2
            self._build_tree(2*node + 1, start, mid)
            self._build_tree(2*node + 2, mid + 1, end)
            self.tree[node] = self.tree[2*node + 1] + self.tree[2*node + 2]
    
    def _query_tree(self, node, start, end, l, r):
        """Query segment tree"""
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        
        mid = (start + end) // 2
        left = self._query_tree(2*node + 1, start, mid, l, r)
        right = self._query_tree(2*node + 2, mid + 1, end, l, r)
        return left + right
    
    def path_sum(self, u, v):
        """Calculate sum on path from u to v"""
        query_func = lambda l, r: self._query_tree(0, 0, self.hld.n - 1, l, r)
        results = self.hld.path_query(u, v, query_func)
        return sum(results)
    
    def lca(self, u, v):
        """Find LCA"""
        return self.hld.lca(u, v)

# Example usage
print("\nHeavy-Light Decomposition:")

# Create a tree: 0-1-2-3-4 with branch at 1 to 5
#       0
#       |
#       1
#      / \
#     2   5
#     |
#     3
#     |
#     4
n = 6
values = [1, 2, 3, 4, 5, 6]  # node values
hld = PathSumHLD(n, values)

# Add edges
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5)]
for u, v in edges:
    hld.add_edge(u, v)

hld.build(root=0)

print(f"Tree with {n} nodes")
print(f"Node values: {values}")

# Path queries
queries = [(0, 4), (0, 5), (2, 5), (3, 5)]
print("\nPath sum queries:")
for u, v in queries:
    result = hld.path_sum(u, v)
    print(f"  Path sum from {u} to {v}: {result}")

# LCA queries
print("\nLCA queries:")
lca_queries = [(4, 5), (2, 5), (3, 4)]
for u, v in lca_queries:
    lca_node = hld.lca(u, v)
    print(f"  LCA of {u} and {v}: {lca_node}")

# Example showing decomposition benefits
print("\nHLD Properties:")
print(f"  Max chain depth: O(log n)")
print(f"  Path query complexity: O(log²n)")
print(f"  Enables efficient tree path operations")
```

Centroid Decomposition structures

When: Tree distance and path queries
Why: Recursively decomposes tree by centroids
Examples: Tree algorithms, distance queries on trees

```python
# Python implementation of Centroid Decomposition
# Recursively decomposes tree into centroids for efficient distance queries

from collections import defaultdict, deque

class CentroidDecomposition:
    """
    Centroid Decomposition for tree distance and path queries.
    Decomposes tree recursively by finding centroids.
    """
    def __init__(self, n):
        """
        Initialize centroid decomposition for a tree with n nodes (0-indexed).
        """
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.removed = [False] * n
        self.subtree_size = [0] * n
        
        # Centroid tree structure
        self.centroid_parent = [-1] * n
        self.centroid_children = [[] for _ in range(n)]
        self.root = -1
        
    def add_edge(self, u, v):
        """Add an undirected edge between u and v"""
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def build(self, root=0):
        """Build the centroid decomposition"""
        self.root = self._decompose(root, -1)
        return self.root
    
    def _get_subtree_size(self, node, parent):
        """Calculate subtree size (excluding removed nodes)"""
        self.subtree_size[node] = 1
        for neighbor in self.graph[node]:
            if neighbor != parent and not self.removed[neighbor]:
                self.subtree_size[node] += self._get_subtree_size(neighbor, node)
        return self.subtree_size[node]
    
    def _find_centroid(self, node, parent, tree_size):
        """Find centroid of the tree/subtree"""
        for neighbor in self.graph[node]:
            if neighbor != parent and not self.removed[neighbor]:
                if self.subtree_size[neighbor] > tree_size // 2:
                    return self._find_centroid(neighbor, node, tree_size)
        return node
    
    def _decompose(self, node, parent_centroid):
        """Recursively decompose tree by centroids"""
        # Get size of current component
        tree_size = self._get_subtree_size(node, -1)
        
        # Find centroid of current component
        centroid = self._find_centroid(node, -1, tree_size)
        
        # Mark centroid as removed
        self.removed[centroid] = True
        
        # Set parent in centroid tree
        self.centroid_parent[centroid] = parent_centroid
        if parent_centroid != -1:
            self.centroid_children[parent_centroid].append(centroid)
        
        # Recursively decompose each subtree
        for neighbor in self.graph[centroid]:
            if not self.removed[neighbor]:
                self._decompose(neighbor, centroid)
        
        return centroid
    
    def get_distance(self, u, v):
        """Get distance between two nodes using BFS"""
        if u == v:
            return 0
        
        visited = [False] * self.n
        queue = deque([(u, 0)])
        visited[u] = True
        
        while queue:
            node, dist = queue.popleft()
            
            for neighbor in self.graph[node]:
                if not visited[neighbor]:
                    if neighbor == v:
                        return dist + 1
                    visited[neighbor] = True
                    queue.append((neighbor, dist + 1))
        
        return -1  # Not connected
    
    def print_centroid_tree(self):
        """Print the centroid tree structure"""
        print("Centroid Tree:")
        print(f"  Root: {self.root}")
        
        def dfs_print(node, depth=0):
            indent = "  " * depth
            children = self.centroid_children[node]
            print(f"{indent}Centroid {node}: children = {children}")
            for child in children:
                dfs_print(child, depth + 1)
        
        if self.root != -1:
            dfs_print(self.root)

class CentroidDecompositionWithQueries(CentroidDecomposition):
    """
    Extended centroid decomposition with distance queries.
    Supports efficient k-th nearest neighbor queries.
    """
    def __init__(self, n):
        super().__init__(n)
        # Store distances from each centroid to all nodes in its subtree
        self.distances = [defaultdict(int) for _ in range(n)]
    
    def build(self, root=0):
        """Build centroid decomposition and precompute distances"""
        self.root = self._decompose_with_distances(root, -1)
        return self.root
    
    def _decompose_with_distances(self, node, parent_centroid):
        """Decompose and compute distances from centroids"""
        tree_size = self._get_subtree_size(node, -1)
        centroid = self._find_centroid(node, -1, tree_size)
        
        # Compute distances from this centroid to all nodes in component
        self._compute_distances(centroid, centroid, -1, 0)
        
        self.removed[centroid] = True
        self.centroid_parent[centroid] = parent_centroid
        if parent_centroid != -1:
            self.centroid_children[parent_centroid].append(centroid)
        
        for neighbor in self.graph[centroid]:
            if not self.removed[neighbor]:
                self._decompose_with_distances(neighbor, centroid)
        
        return centroid
    
    def _compute_distances(self, centroid, node, parent, dist):
        """Compute distances from centroid to all nodes in its subtree"""
        self.distances[centroid][node] = dist
        
        for neighbor in self.graph[node]:
            if neighbor != parent and not self.removed[neighbor]:
                self._compute_distances(centroid, neighbor, node, dist + 1)
    
    def query_distance(self, u, v):
        """Query distance between u and v using centroid tree"""
        result = float('inf')
        
        # Find LCA in centroid tree and sum distances
        u_ancestors = []
        v_ancestors = []
        
        # Get ancestors of u in centroid tree
        curr = u
        while curr != -1:
            u_ancestors.append(curr)
            curr = self.centroid_parent[curr]
        
        # Get ancestors of v in centroid tree
        curr = v
        while curr != -1:
            v_ancestors.append(curr)
            curr = self.centroid_parent[curr]
        
        # Find LCA and calculate distance
        for u_anc in u_ancestors:
            for v_anc in v_ancestors:
                if u_anc == v_anc:
                    # Found common ancestor
                    dist = self.distances[u_anc][u] + self.distances[v_anc][v]
                    result = min(result, dist)
        
        return result

# Example usage
print("\nCentroid Decomposition:")

# Create a tree
#       0
#      /|\
#     1 2 3
#    /|   |\
#   4 5   6 7
n = 8
cd = CentroidDecomposition(n)

# Build tree
edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (3, 6), (3, 7)]
for u, v in edges:
    cd.add_edge(u, v)

print(f"Original tree with {n} nodes")
print(f"Edges: {edges}")

# Build centroid decomposition
root = cd.build(root=0)
print(f"\nCentroid tree root: {root}")

# Print centroid tree structure
cd.print_centroid_tree()

# Example with distance queries
print("\nCentroid Decomposition with Distance Queries:")
cd_query = CentroidDecompositionWithQueries(n)

for u, v in edges:
    cd_query.add_edge(u, v)

cd_query.build(root=0)

# Test distance queries
distance_queries = [(4, 7), (5, 6), (1, 3), (4, 5)]
print("\nDistance queries:")
for u, v in distance_queries:
    dist = cd_query.query_distance(u, v)
    print(f"  Distance from {u} to {v}: {dist}")

# Show benefits
print("\nCentroid Decomposition Properties:")
print("  - Tree height: O(log n)")
print("  - Distance queries: O(log n)")
print("  - Good for k-th nearest neighbor problems")
```

Disjoint Sparse Table

When: Non-overlapping range queries
Why: O(n) space instead of O(n log n)
Examples: Range queries with non-overlapping constraint

```python
# Python implementation of Disjoint Sparse Table
# O(n) space for non-overlapping range queries

from typing import List, Callable
import math

class DisjointSparseTable:
    """
    Disjoint Sparse Table for static range queries.
    Uses O(n) space instead of O(n log n) by using disjoint intervals.
    Works with any associative operation (not just idempotent).
    """
    def __init__(self, array: List[int], operation: Callable[[int, int], int] = min):
        """
        Build disjoint sparse table.
        
        Args:
            array: Input array
            operation: Binary associative operation (e.g., min, max, sum, gcd)
        """
        self.n = len(array)
        self.array = array
        self.op = operation
        
        if self.n == 0:
            self.levels = 0
            self.table = []
            return
        
        # Calculate number of levels needed
        self.levels = math.ceil(math.log2(self.n)) if self.n > 1 else 1
        
        # Initialize table: table[level][i] stores result for disjoint interval
        self.table = [[None] * self.n for _ in range(self.levels)]
        
        # Build the table
        self._build()
    
    def _build(self):
        """Build the disjoint sparse table"""
        # Level 0: single elements
        for i in range(self.n):
            self.table[0][i] = self.array[i]
        
        # Build higher levels
        for level in range(1, self.levels):
            block_size = 1 << level  # 2^level
            
            # Process each block at this level
            for block_start in range(0, self.n, block_size * 2):
                # Find middle of the block
                mid = min(block_start + block_size, self.n)
                
                # Build prefix from mid going left
                if mid - 1 >= block_start:
                    self.table[level][mid - 1] = self.array[mid - 1]
                    for i in range(mid - 2, block_start - 1, -1):
                        self.table[level][i] = self.op(
                            self.array[i],
                            self.table[level][i + 1]
                        )
                
                # Build suffix from mid going right
                if mid < self.n:
                    self.table[level][mid] = self.array[mid]
                    for i in range(mid + 1, min(mid + block_size, self.n)):
                        self.table[level][i] = self.op(
                            self.table[level][i - 1],
                            self.array[i]
                        )
    
    def query(self, left: int, right: int) -> int:
        """
        Query range [left, right] inclusive.
        O(1) time complexity.
        """
        if left < 0 or right >= self.n or left > right:
            raise IndexError(f"Invalid range [{left}, {right}]")
        
        if left == right:
            return self.array[left]
        
        # Find the highest level where left and right are in different blocks
        level = math.floor(math.log2(left ^ right))
        
        # Combine results from both sides
        return self.op(self.table[level][left], self.table[level][right])
    
    def __len__(self):
        return self.n

# Example usage
print("\nDisjoint Sparse Table:")

# Create array
array = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
print(f"Array: {array}")

# Disjoint Sparse Table for Range Minimum Query
dst_min = DisjointSparseTable(array, operation=min)

print("\nRange Minimum Queries:")
queries = [(0, 5), (2, 7), (1, 3), (5, 9), (0, 9)]
for left, right in queries:
    result = dst_min.query(left, right)
    print(f"  min({left}, {right}) = {result}")

# Disjoint Sparse Table for Range Maximum Query
dst_max = DisjointSparseTable(array, operation=max)

print("\nRange Maximum Queries:")
for left, right in queries[:3]:
    result = dst_max.query(left, right)
    print(f"  max({left, right}) = {result}")

# Disjoint Sparse Table for Range Sum
dst_sum = DisjointSparseTable(array, operation=lambda a, b: a + b)

print("\nRange Sum Queries:")
for left, right in queries[:3]:
    result = dst_sum.query(left, right)
    print(f"  sum({left}, {right}) = {result}")

# GCD example
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

array_gcd = [12, 18, 24, 36, 48, 60]
print(f"\nArray for GCD: {array_gcd}")

dst_gcd = DisjointSparseTable(array_gcd, operation=gcd)

print("Range GCD Queries:")
gcd_queries = [(0, 2), (1, 4), (0, 5), (2, 5)]
for left, right in gcd_queries:
    result = dst_gcd.query(left, right)
    print(f"  gcd({left}, {right}) = {result}")

# Space comparison
print("\nSpace Complexity Comparison:")
n = len(array)
standard_sparse_table_space = n * math.ceil(math.log2(n)) if n > 0 else 0
disjoint_sparse_table_space = n * math.ceil(math.log2(n)) if n > 0 else 0
print(f"  Standard Sparse Table: O(n log n) ≈ {standard_sparse_table_space} elements")
print(f"  Disjoint Sparse Table: O(n log n) construction, but more cache-friendly")
print(f"  Both support O(1) queries")
print(f"  Disjoint version works with non-idempotent operations (e.g., sum)")
```

Practical Modern Structures
Copy-on-Write (COW) B-Tree

When: Need snapshots and versioning in databases
Why: Allows cheap snapshots, multi-version concurrency
Examples: BTRFS filesystem, ZFS, modern databases

```python
# Python implementation of Copy-on-Write (COW) B-Tree
# Enables efficient snapshots and multi-version concurrency

import copy
from typing import Any, Optional, List, Tuple

class COWBTreeNode:
    """Node in a COW B-Tree"""
    def __init__(self, t, is_leaf=True):
        """
        Args:
            t: Minimum degree (node has at least t-1 keys)
            is_leaf: Whether this is a leaf node
        """
        self.keys = []
        self.children = []
        self.is_leaf = is_leaf
        self.t = t
        self.version = 0  # Version number for this node
        
    def is_full(self):
        """Check if node is full"""
        return len(self.keys) >= 2 * self.t - 1
    
    def clone(self):
        """Create a copy of this node (Copy-on-Write)"""
        new_node = COWBTreeNode(self.t, self.is_leaf)
        new_node.keys = self.keys.copy()
        new_node.children = self.children.copy()  # Shallow copy - children are shared
        new_node.version = self.version + 1
        return new_node

class COWBTree:
    """
    Copy-on-Write B-Tree implementation.
    Supports efficient snapshots without copying entire tree.
    """
    def __init__(self, t=3):
        """
        Initialize COW B-Tree.
        
        Args:
            t: Minimum degree (minimum t-1 keys per node)
        """
        self.t = t
        self.root = COWBTreeNode(t, is_leaf=True)
        self.version = 0
        self.snapshots = {}  # version -> root mapping
        
    def search(self, key, node=None) -> Optional[Any]:
        """Search for a key in the tree"""
        if node is None:
            node = self.root
        
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        
        if i < len(node.keys) and key == node.keys[i]:
            return key
        
        if node.is_leaf:
            return None
        
        return self.search(key, node.children[i])
    
    def insert(self, key):
        """Insert a key into the tree (creates new version)"""
        self.version += 1
        
        if self.root.is_full():
            # Root is full, create new root
            new_root = COWBTreeNode(self.t, is_leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        
        self.root = self._insert_non_full(self.root, key)
    
    def _insert_non_full(self, node, key):
        """Insert into a non-full node (returns new node due to COW)"""
        # Clone the node (Copy-on-Write)
        new_node = node.clone()
        
        if new_node.is_leaf:
            # Insert key in sorted order
            i = len(new_node.keys) - 1
            new_node.keys.append(None)
            
            while i >= 0 and key < new_node.keys[i]:
                new_node.keys[i + 1] = new_node.keys[i]
                i -= 1
            
            new_node.keys[i + 1] = key
        else:
            # Find child to insert into
            i = len(new_node.keys) - 1
            while i >= 0 and key < new_node.keys[i]:
                i -= 1
            i += 1
            
            # Check if child is full
            if new_node.children[i].is_full():
                self._split_child(new_node, i)
                if key > new_node.keys[i]:
                    i += 1
            
            # Recursively insert into child (COW)
            new_node.children[i] = self._insert_non_full(new_node.children[i], key)
        
        return new_node
    
    def _split_child(self, parent, index):
        """Split a full child (modifies parent in-place during insertion)"""
        t = self.t
        full_child = parent.children[index]
        new_child = COWBTreeNode(t, is_leaf=full_child.is_leaf)
        
        # Move second half of keys to new child
        mid_index = t - 1
        new_child.keys = full_child.keys[mid_index + 1:]
        full_child.keys = full_child.keys[:mid_index]
        
        # Move second half of children if not leaf
        if not full_child.is_leaf:
            new_child.children = full_child.children[mid_index + 1:]
            full_child.children = full_child.children[:mid_index + 1]
        
        # Insert middle key into parent
        parent.keys.insert(index, full_child.keys[mid_index])
        full_child.keys = full_child.keys[:mid_index]
        
        # Insert new child into parent
        parent.children.insert(index + 1, new_child)
    
    def snapshot(self) -> int:
        """Create a snapshot of current tree state"""
        snapshot_version = self.version
        self.snapshots[snapshot_version] = self.root
        return snapshot_version
    
    def restore_snapshot(self, snapshot_version: int) -> bool:
        """Restore tree to a previous snapshot"""
        if snapshot_version not in self.snapshots:
            return False
        
        self.root = self.snapshots[snapshot_version]
        self.version = snapshot_version
        return True
    
    def get_snapshot_versions(self) -> List[int]:
        """Get all available snapshot versions"""
        return sorted(self.snapshots.keys())
    
    def inorder_traversal(self, node=None) -> List[Any]:
        """Return keys in sorted order"""
        if node is None:
            node = self.root
        
        result = []
        
        for i in range(len(node.keys)):
            if not node.is_leaf:
                result.extend(self.inorder_traversal(node.children[i]))
            result.append(node.keys[i])
        
        if not node.is_leaf:
            result.extend(self.inorder_traversal(node.children[-1]))
        
        return result

# Example usage
print("\nCopy-on-Write (COW) B-Tree:")

# Create COW B-Tree
cow_tree = COWBTree(t=3)

# Insert initial data
print("Inserting initial data: [10, 20, 5, 6, 12, 30]")
for key in [10, 20, 5, 6, 12, 30]:
    cow_tree.insert(key)

print(f"Tree contents: {cow_tree.inorder_traversal()}")
print(f"Version: {cow_tree.version}")

# Create snapshot 1
snapshot1 = cow_tree.snapshot()
print(f"\nCreated snapshot {snapshot1}")

# Insert more data
print("\nInserting more data: [7, 17, 25]")
for key in [7, 17, 25]:
    cow_tree.insert(key)

print(f"Tree contents: {cow_tree.inorder_traversal()}")
print(f"Version: {cow_tree.version}")

# Create snapshot 2
snapshot2 = cow_tree.snapshot()
print(f"\nCreated snapshot {snapshot2}")

# Insert even more data
print("\nInserting more data: [3, 40, 50]")
for key in [3, 40, 50]:
    cow_tree.insert(key)

print(f"Tree contents: {cow_tree.inorder_traversal()}")
print(f"Version: {cow_tree.version}")

# Restore to snapshot 1
print(f"\nRestoring to snapshot {snapshot1}...")
cow_tree.restore_snapshot(snapshot1)
print(f"Tree contents after restore: {cow_tree.inorder_traversal()}")
print(f"Version: {cow_tree.version}")

# Restore to snapshot 2
print(f"\nRestoring to snapshot {snapshot2}...")
cow_tree.restore_snapshot(snapshot2)
print(f"Tree contents after restore: {cow_tree.inorder_traversal()}")
print(f"Version: {cow_tree.version}")

# Show all available snapshots
print(f"\nAvailable snapshots: {cow_tree.get_snapshot_versions()}")

# Demonstrate multi-version concurrency
print("\nMulti-version concurrency example:")
print("  - Multiple versions coexist in memory")
print("  - Snapshots share unchanged nodes (space-efficient)")
print("  - No locks needed for read operations")
print("  - Used in BTRFS, ZFS, and modern databases")
```

Fractal Tree Index

When: Need better write performance than B-tree
Why: Batches updates, better write amplification
Examples: Databases optimizing for writes

```python
# Python implementation of Fractal Tree Index
# Batches updates in buffers for better write performance

from collections import deque
from typing import Any, Optional, List, Tuple

class FractalTreeNode:
    """Node in a Fractal Tree Index"""
    def __init__(self, t, is_leaf=True):
        """
        Args:
            t: Minimum degree (branching factor)
            is_leaf: Whether this is a leaf node
        """
        self.keys = []
        self.children = []
        self.is_leaf = is_leaf
        self.t = t
        
        # Message buffer - stores pending operations
        self.buffer = []  # List of (operation, key, value) tuples
        self.buffer_size_limit = 2 * t  # When to flush buffer
        
    def is_full(self):
        """Check if node is full"""
        return len(self.keys) >= 2 * self.t - 1
    
    def buffer_is_full(self):
        """Check if message buffer is full"""
        return len(self.buffer) >= self.buffer_size_limit

class FractalTreeIndex:
    """
    Fractal Tree Index implementation.
    Batches updates in buffers for better write performance.
    Reduces write amplification compared to traditional B-trees.
    """
    def __init__(self, t=3):
        """
        Initialize Fractal Tree Index.
        
        Args:
            t: Minimum degree (branching factor)
        """
        self.t = t
        self.root = FractalTreeNode(t, is_leaf=True)
        self.data = {}  # Actual key-value storage for simplicity
        
    def insert(self, key, value=None):
        """Insert a key-value pair"""
        if value is None:
            value = key
        
        # Add insert message to root's buffer
        self.root.buffer.append(('insert', key, value))
        
        # Check if we need to flush the buffer
        if self.root.buffer_is_full():
            self._flush_buffer(self.root)
        
        # If root is full after flushing, split it
        if self.root.is_full():
            new_root = FractalTreeNode(self.t, is_leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
    
    def delete(self, key):
        """Delete a key"""
        # Add delete message to root's buffer
        self.root.buffer.append(('delete', key, None))
        
        if self.root.buffer_is_full():
            self._flush_buffer(self.root)
    
    def search(self, key) -> Optional[Any]:
        """Search for a key (may need to flush buffers)"""
        return self._search_with_flush(self.root, key)
    
    def _search_with_flush(self, node, key) -> Optional[Any]:
        """Search with buffer flushing"""
        # Flush buffer to ensure up-to-date results
        if node.buffer:
            self._flush_buffer(node)
        
        if node.is_leaf:
            return self.data.get(key)
        
        # Find appropriate child
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        
        return self._search_with_flush(node.children[i], key)
    
    def _flush_buffer(self, node):
        """Flush messages from node's buffer to children or apply to data"""
        if not node.buffer:
            return
        
        if node.is_leaf:
            # Apply messages to actual data
            for op, key, value in node.buffer:
                if op == 'insert':
                    self.data[key] = value
                elif op == 'delete':
                    self.data.pop(key, None)
            node.buffer.clear()
        else:
            # Push messages to appropriate children
            for op, key, value in node.buffer:
                # Find child to push message to
                i = 0
                while i < len(node.keys) and key > node.keys[i]:
                    i += 1
                
                child = node.children[i]
                child.buffer.append((op, key, value))
                
                # Recursively flush if child's buffer is full
                if child.buffer_is_full():
                    self._flush_buffer(child)
            
            node.buffer.clear()
    
    def _split_child(self, parent, index):
        """Split a full child"""
        t = self.t
        full_child = parent.children[index]
        new_child = FractalTreeNode(t, is_leaf=full_child.is_leaf)
        
        # Split keys
        mid_index = t - 1
        parent.keys.insert(index, full_child.keys[mid_index])
        new_child.keys = full_child.keys[mid_index + 1:]
        full_child.keys = full_child.keys[:mid_index]
        
        # Split children if not leaf
        if not full_child.is_leaf:
            new_child.children = full_child.children[mid_index + 1:]
            full_child.children = full_child.children[:mid_index + 1]
        
        # Split buffer messages
        new_child.buffer = full_child.buffer[len(full_child.buffer)//2:]
        full_child.buffer = full_child.buffer[:len(full_child.buffer)//2]
        
        parent.children.insert(index + 1, new_child)
    
    def force_flush_all(self):
        """Force flush all buffers (for testing/demonstration)"""
        self._force_flush_recursive(self.root)
    
    def _force_flush_recursive(self, node):
        """Recursively flush all buffers"""
        if node.buffer:
            self._flush_buffer(node)
        
        if not node.is_leaf:
            for child in node.children:
                self._force_flush_recursive(child)
    
    def get_all_keys(self) -> List[Any]:
        """Get all keys (forces flush)"""
        self.force_flush_all()
        return sorted(self.data.keys())
    
    def range_query(self, start_key, end_key) -> List[Tuple[Any, Any]]:
        """Range query (forces flush)"""
        self.force_flush_all()
        result = []
        for key in sorted(self.data.keys()):
            if start_key <= key <= end_key:
                result.append((key, self.data[key]))
        return result

# Example usage
print("\nFractal Tree Index:")

# Create Fractal Tree
ft = FractalTreeIndex(t=3)

# Insert data (messages are batched)
print("Inserting data with batched writes:")
data = [(10, 'A'), (20, 'B'), (5, 'C'), (15, 'D'), (25, 'E'), 
        (30, 'F'), (12, 'G'), (8, 'H'), (18, 'I'), (22, 'J')]

for key, value in data:
    ft.insert(key, value)
    print(f"  Inserted ({key}, '{value}') - buffered in memory")

# Search without explicit flush
print("\nSearching (auto-flush on access):")
search_keys = [10, 15, 25]
for key in search_keys:
    value = ft.search(key)
    print(f"  Key {key}: {value}")

# Get all keys (forces complete flush)
print("\nAll keys (after full flush):")
all_keys = ft.get_all_keys()
print(f"  {all_keys}")

# Range query
print("\nRange query [10, 20]:")
range_result = ft.range_query(10, 20)
for key, value in range_result:
    print(f"  {key}: {value}")

# Delete operations (also batched)
print("\nDeleting keys [5, 25, 30]:")
for key in [5, 25, 30]:
    ft.delete(key)
    print(f"  Deleted {key} - buffered")

# Force flush and show results
ft.force_flush_all()
print("\nAfter deletions:")
all_keys = ft.get_all_keys()
print(f"  Remaining keys: {all_keys}")

# Show benefits
print("\nFractal Tree Benefits:")
print("  - Write amplification: O(1/B log_B N) vs O(log_B N) for B-tree")
print("  - Batches updates in node buffers")
print("  - Better for write-heavy workloads")
print("  - Used in TokuDB and other write-optimized databases")
print("  - Trades some read latency for much better write throughput")
```

ART (Adaptive Radix Tree)

When: In-memory indexing with strings/integers
Why: Adaptive node sizes, very cache-efficient
Examples: In-memory databases, key-value stores

```python
# Python implementation of ART (Adaptive Radix Tree)
# Uses adaptive node sizes for cache efficiency

from typing import Any, Optional, List

class ARTNode:
    """Base class for ART nodes"""
    def __init__(self):
        self.prefix = []  # Common prefix
        self.prefix_len = 0
        
class ARTNode4(ARTNode):
    """Node with up to 4 children (most memory-efficient)"""
    def __init__(self):
        super().__init__()
        self.keys = [None] * 4
        self.children = [None] * 4
        self.num_children = 0
        
class ARTNode16(ARTNode):
    """Node with up to 16 children"""
    def __init__(self):
        super().__init__()
        self.keys = [None] * 16
        self.children = [None] * 16
        self.num_children = 0
        
class ARTNode48(ARTNode):
    """Node with up to 48 children (uses index array)"""
    def __init__(self):
        super().__init__()
        self.indexes = [None] * 256  # Maps byte value to child index
        self.children = [None] * 48
        self.num_children = 0
        
class ARTNode256(ARTNode):
    """Node with up to 256 children (full direct mapping)"""
    def __init__(self):
        super().__init__()
        self.children = [None] * 256
        self.num_children = 0

class ARTLeaf:
    """Leaf node storing actual key-value pair"""
    def __init__(self, key, value):
        self.key = key
        self.value = value

class AdaptiveRadixTree:
    """
    Adaptive Radix Tree (ART) implementation.
    Uses different node types based on number of children for cache efficiency.
    """
    def __init__(self):
        self.root = None
        self.size = 0
    
    def _key_to_bytes(self, key) -> List[int]:
        """Convert key to list of bytes"""
        if isinstance(key, str):
            return [ord(c) for c in key]
        elif isinstance(key, int):
            return list(key.to_bytes((key.bit_length() + 7) // 8, 'big'))
        else:
            return list(str(key).encode())
    
    def insert(self, key, value):
        """Insert a key-value pair"""
        key_bytes = self._key_to_bytes(key)
        
        if self.root is None:
            self.root = ARTLeaf(key, value)
            self.size += 1
            return
        
        self.root = self._insert_recursive(self.root, key_bytes, value, 0)
    
    def _insert_recursive(self, node, key_bytes, value, depth):
        """Recursively insert into the tree"""
        if isinstance(node, ARTLeaf):
            # Leaf node - need to expand
            if node.key == key_bytes:
                # Update existing key
                node.value = value
                return node
            
            # Create new node and split
            new_node = ARTNode4()
            leaf1 = node
            leaf2 = ARTLeaf(key_bytes, value)
            
            # Find common prefix
            leaf1_bytes = self._key_to_bytes(leaf1.key) if isinstance(leaf1.key, (str, int)) else leaf1.key
            common_prefix_len = 0
            while (depth + common_prefix_len < len(leaf1_bytes) and 
                   depth + common_prefix_len < len(key_bytes) and
                   leaf1_bytes[depth + common_prefix_len] == key_bytes[depth + common_prefix_len]):
                common_prefix_len += 1
            
            new_node.prefix = key_bytes[depth:depth + common_prefix_len]
            new_node.prefix_len = common_prefix_len
            
            # Add both leaves as children
            if depth + common_prefix_len < len(leaf1_bytes):
                self._add_child(new_node, leaf1_bytes[depth + common_prefix_len], leaf1)
            if depth + common_prefix_len < len(key_bytes):
                self._add_child(new_node, key_bytes[depth + common_prefix_len], leaf2)
            
            self.size += 1
            return new_node
        
        # Check prefix match
        if node.prefix_len > 0:
            prefix_match = 0
            while (prefix_match < node.prefix_len and 
                   depth + prefix_match < len(key_bytes) and
                   node.prefix[prefix_match] == key_bytes[depth + prefix_match]):
                prefix_match += 1
            
            if prefix_match < node.prefix_len:
                # Prefix mismatch - need to split node
                return self._split_node(node, key_bytes, value, depth, prefix_match)
        
        # Navigate to appropriate child
        depth += node.prefix_len
        
        if depth >= len(key_bytes):
            # Key ends here - add as leaf
            return node
        
        next_byte = key_bytes[depth]
        child = self._find_child(node, next_byte)
        
        if child is None:
            # No child exists - add new leaf
            new_leaf = ARTLeaf(key_bytes, value)
            self._add_child(node, next_byte, new_leaf)
            self.size += 1
        else:
            # Recursively insert into child
            new_child = self._insert_recursive(child, key_bytes, value, depth + 1)
            self._update_child(node, next_byte, new_child)
        
        return node
    
    def _find_child(self, node, byte_val):
        """Find child for given byte value"""
        if isinstance(node, ARTNode4) or isinstance(node, ARTNode16):
            for i in range(node.num_children):
                if node.keys[i] == byte_val:
                    return node.children[i]
            return None
        elif isinstance(node, ARTNode48):
            idx = node.indexes[byte_val]
            return node.children[idx] if idx is not None else None
        elif isinstance(node, ARTNode256):
            return node.children[byte_val]
        return None
    
    def _add_child(self, node, byte_val, child):
        """Add child to node (may trigger growth)"""
        if isinstance(node, ARTNode4):
            if node.num_children < 4:
                node.keys[node.num_children] = byte_val
                node.children[node.num_children] = child
                node.num_children += 1
            else:
                # Grow to Node16
                new_node = self._grow_node(node, ARTNode16)
                self._add_child(new_node, byte_val, child)
                return new_node
        elif isinstance(node, ARTNode16):
            if node.num_children < 16:
                node.keys[node.num_children] = byte_val
                node.children[node.num_children] = child
                node.num_children += 1
            else:
                # Grow to Node48
                new_node = self._grow_node(node, ARTNode48)
                self._add_child(new_node, byte_val, child)
                return new_node
        elif isinstance(node, ARTNode48):
            if node.num_children < 48:
                # Find empty slot
                for i in range(48):
                    if node.children[i] is None:
                        node.children[i] = child
                        node.indexes[byte_val] = i
                        node.num_children += 1
                        break
            else:
                # Grow to Node256
                new_node = self._grow_node(node, ARTNode256)
                self._add_child(new_node, byte_val, child)
                return new_node
        elif isinstance(node, ARTNode256):
            node.children[byte_val] = child
            node.num_children += 1
        
        return node
    
    def _update_child(self, node, byte_val, child):
        """Update existing child"""
        if isinstance(node, ARTNode4) or isinstance(node, ARTNode16):
            for i in range(node.num_children):
                if node.keys[i] == byte_val:
                    node.children[i] = child
                    return
        elif isinstance(node, ARTNode48):
            idx = node.indexes[byte_val]
            if idx is not None:
                node.children[idx] = child
        elif isinstance(node, ARTNode256):
            node.children[byte_val] = child
    
    def _grow_node(self, old_node, new_type):
        """Grow node to larger type"""
        new_node = new_type()
        new_node.prefix = old_node.prefix
        new_node.prefix_len = old_node.prefix_len
        
        # Copy children
        if isinstance(old_node, ARTNode4):
            for i in range(old_node.num_children):
                if isinstance(new_node, ARTNode16):
                    new_node.keys[i] = old_node.keys[i]
                    new_node.children[i] = old_node.children[i]
            new_node.num_children = old_node.num_children
        
        return new_node
    
    def _split_node(self, node, key_bytes, value, depth, prefix_match):
        """Split node due to prefix mismatch"""
        new_node = ARTNode4()
        new_node.prefix = node.prefix[:prefix_match]
        new_node.prefix_len = prefix_match
        
        # Adjust old node's prefix
        node.prefix = node.prefix[prefix_match + 1:]
        node.prefix_len -= prefix_match + 1
        
        # Add old node as child
        self._add_child(new_node, node.prefix[0] if node.prefix else 0, node)
        
        # Add new leaf
        new_leaf = ARTLeaf(key_bytes, value)
        self._add_child(new_node, key_bytes[depth + prefix_match], new_leaf)
        
        self.size += 1
        return new_node
    
    def search(self, key) -> Optional[Any]:
        """Search for a key"""
        key_bytes = self._key_to_bytes(key)
        return self._search_recursive(self.root, key_bytes, 0)
    
    def _search_recursive(self, node, key_bytes, depth):
        """Recursively search for key"""
        if node is None:
            return None
        
        if isinstance(node, ARTLeaf):
            leaf_bytes = self._key_to_bytes(node.key) if isinstance(node.key, (str, int)) else node.key
            return node.value if leaf_bytes == key_bytes else None
        
        # Check prefix
        if node.prefix_len > 0:
            for i in range(node.prefix_len):
                if depth + i >= len(key_bytes) or node.prefix[i] != key_bytes[depth + i]:
                    return None
        
        depth += node.prefix_len
        
        if depth >= len(key_bytes):
            return None
        
        child = self._find_child(node, key_bytes[depth])
        return self._search_recursive(child, key_bytes, depth + 1)
    
    def __len__(self):
        return self.size

# Example usage
print("\nAdaptive Radix Tree (ART):")

# Create ART
art = AdaptiveRadixTree()

# Insert string keys
print("Inserting string keys:")
data = [
    ("apple", 1), ("application", 2), ("apply", 3),
    ("banana", 4), ("band", 5), ("can", 6),
    ("cat", 7), ("dog", 8)
]

for key, value in data:
    art.insert(key, value)
    print(f"  Inserted ('{key}', {value})")

print(f"\nTree size: {len(art)}")

# Search for keys
print("\nSearching:")
search_keys = ["apple", "cat", "application", "zebra", "ban"]
for key in search_keys:
    result = art.search(key)
    status = f"found: {result}" if result is not None else "not found"
    print(f"  '{key}': {status}")

# Insert integer keys
print("\nInserting integer keys:")
int_art = AdaptiveRadixTree()
for i in [10, 20, 30, 15, 25, 5]:
    int_art.insert(i, i * 10)
    print(f"  Inserted ({i}, {i * 10})")

print("\nSearching integer keys:")
for key in [10, 15, 25, 100]:
    result = int_art.search(key)
    status = f"found: {result}" if result is not None else "not found"
    print(f"  {key}: {status}")

# Show adaptive nature
print("\nART Properties:")
print("  - Node types: Node4, Node16, Node48, Node256")
print("  - Grows dynamically based on number of children")
print("  - Very cache-efficient (small nodes fit in cache lines)")
print("  - Used in: Redis, in-memory databases, key-value stores")
print("  - Better than hash tables for ordered operations")
print("  - Better than B-trees for string keys")
```

Masstree

When: High-performance concurrent B-tree variant
Why: Optimized for modern multicore systems
Examples: Concurrent in-memory databases

```python
# Python implementation of Masstree
# High-performance concurrent B+tree variant optimized for modern multicore systems

from typing import Any, Optional, List, Tuple
import threading

class MasstreeNode:
    """Node in a Masstree"""
    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.keys = []
        self.values = []  # For leaf nodes
        self.children = []  # For internal nodes
        self.next_leaf = None  # Link to next leaf (for range scans)
        self.lock = threading.RLock()  # Fine-grained locking
        self.version = 0  # For optimistic concurrency control
        
class MasstreeLayer:
    """
    A layer in the Masstree (each layer is a B+tree).
    Masstree uses multiple layers for long keys.
    """
    def __init__(self, order=4):
        self.order = order  # Max children per node
        self.root = MasstreeNode(is_leaf=True)
        self.lock = threading.RLock()
        
    def _split_leaf(self, node):
        """Split a full leaf node"""
        mid = len(node.keys) // 2
        
        new_node = MasstreeNode(is_leaf=True)
        new_node.keys = node.keys[mid:]
        new_node.values = node.values[mid:]
        new_node.next_leaf = node.next_leaf
        
        node.keys = node.keys[:mid]
        node.values = node.values[:mid]
        node.next_leaf = new_node
        
        return node.keys[-1], new_node
    
    def _split_internal(self, node):
        """Split a full internal node"""
        mid = len(node.keys) // 2
        
        new_node = MasstreeNode(is_leaf=False)
        split_key = node.keys[mid]
        
        new_node.keys = node.keys[mid + 1:]
        new_node.children = node.children[mid + 1:]
        
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]
        
        return split_key, new_node
    
    def insert(self, key, value):
        """Insert key-value pair"""
        with self.lock:
            if len(self.root.keys) >= self.order - 1:
                # Root is full, create new root
                old_root = self.root
                self.root = MasstreeNode(is_leaf=False)
                
                if old_root.is_leaf:
                    split_key, new_node = self._split_leaf(old_root)
                else:
                    split_key, new_node = self._split_internal(old_root)
                
                self.root.keys = [split_key]
                self.root.children = [old_root, new_node]
            
            self._insert_non_full(self.root, key, value)
    
    def _insert_non_full(self, node, key, value):
        """Insert into a non-full node"""
        with node.lock:
            node.version += 1
            
            if node.is_leaf:
                # Insert into leaf
                i = 0
                while i < len(node.keys) and key > node.keys[i]:
                    i += 1
                
                if i < len(node.keys) and key == node.keys[i]:
                    # Update existing key
                    node.values[i] = value
                else:
                    # Insert new key
                    node.keys.insert(i, key)
                    node.values.insert(i, value)
            else:
                # Find child to insert into
                i = 0
                while i < len(node.keys) and key > node.keys[i]:
                    i += 1
                
                child = node.children[i]
                
                # Check if child is full
                if len(child.keys) >= self.order - 1:
                    # Split child
                    if child.is_leaf:
                        split_key, new_child = self._split_leaf(child)
                    else:
                        split_key, new_child = self._split_internal(child)
                    
                    # Insert split key into current node
                    node.keys.insert(i, split_key)
                    node.children.insert(i + 1, new_child)
                    
                    # Determine which child to insert into
                    if key > split_key:
                        i += 1
                        child = new_child
                
                self._insert_non_full(child, key, value)
    
    def search(self, key) -> Optional[Any]:
        """Search for a key"""
        return self._search_node(self.root, key)
    
    def _search_node(self, node, key) -> Optional[Any]:
        """Search in a node"""
        # Optimistic read (no lock for reads)
        version_before = node.version
        
        if node.is_leaf:
            # Search in leaf
            for i, k in enumerate(node.keys):
                if k == key:
                    value = node.values[i]
                    # Verify version hasn't changed
                    if node.version == version_before:
                        return value
                    # Retry if version changed
                    return self._search_node(node, key)
            return None
        else:
            # Find child to search
            i = 0
            while i < len(node.keys) and key > node.keys[i]:
                i += 1
            
            child = node.children[i]
            
            # Verify version before following pointer
            if node.version != version_before:
                # Retry if version changed
                return self._search_node(node, key)
            
            return self._search_node(child, key)
    
    def range_query(self, start_key, end_key) -> List[Tuple[Any, Any]]:
        """Range query [start_key, end_key]"""
        result = []
        
        # Find starting leaf
        leaf = self._find_leaf(self.root, start_key)
        
        # Scan leaves
        while leaf is not None:
            with leaf.lock:
                for i, key in enumerate(leaf.keys):
                    if start_key <= key <= end_key:
                        result.append((key, leaf.values[i]))
                    elif key > end_key:
                        return result
                
                leaf = leaf.next_leaf
        
        return result
    
    def _find_leaf(self, node, key):
        """Find leaf node that should contain key"""
        if node.is_leaf:
            return node
        
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        
        return self._find_leaf(node.children[i], key)

class Masstree:
    """
    Masstree: High-performance concurrent key-value store.
    Optimized for modern multicore systems with cache-conscious design.
    """
    def __init__(self, order=4):
        """
        Initialize Masstree.
        
        Args:
            order: Branching factor for B+tree nodes
        """
        self.order = order
        # For simplicity, using single layer
        # Real Masstree uses multiple layers for long keys
        self.layer = MasstreeLayer(order)
        self.size = 0
        self.lock = threading.Lock()
    
    def insert(self, key, value):
        """Insert key-value pair (thread-safe)"""
        self.layer.insert(key, value)
        with self.lock:
            self.size += 1
    
    def search(self, key) -> Optional[Any]:
        """Search for key (lock-free read)"""
        return self.layer.search(key)
    
    def range_query(self, start_key, end_key) -> List[Tuple[Any, Any]]:
        """Range query (returns sorted list of (key, value) pairs)"""
        return self.layer.range_query(start_key, end_key)
    
    def __len__(self):
        return self.size

# Example usage
print("\nMasstree:")

# Create Masstree
mt = Masstree(order=4)

# Insert data
print("Inserting data:")
data = [(10, 'A'), (20, 'B'), (5, 'C'), (15, 'D'), (25, 'E'),
        (30, 'F'), (12, 'G'), (18, 'H'), (22, 'I'), (8, 'J')]

for key, value in data:
    mt.insert(key, value)
    print(f"  Inserted ({key}, '{value}')")

print(f"\nTree size: {len(mt)}")

# Search operations
print("\nSearch operations (lock-free reads):")
search_keys = [10, 15, 25, 100]
for key in search_keys:
    value = mt.search(key)
    status = f"found: '{value}'" if value else "not found"
    print(f"  Key {key}: {status}")

# Range query
print("\nRange query [10, 22]:")
range_result = mt.range_query(10, 22)
for key, value in range_result:
    print(f"  {key}: '{value}'")

# Update operation (just insert with same key)
print("\nUpdating key 15:")
mt.insert(15, 'D_updated')
value = mt.search(15)
print(f"  Key 15: '{value}'")

# Concurrent operations example
print("\nConcurrent operations (thread-safe):")
import concurrent.futures

def insert_batch(start, count):
    for i in range(start, start + count):
        mt.insert(i + 100, f'V{i}')
    return count

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(insert_batch, 0, 5),
        executor.submit(insert_batch, 5, 5),
        executor.submit(insert_batch, 10, 5)
    ]
    concurrent.futures.wait(futures)

print(f"  After concurrent inserts, tree size: {len(mt)}")

# Range query on newly inserted data
print("\nRange query [100, 110]:")
range_result = mt.range_query(100, 110)
print(f"  Found {len(range_result)} keys")

# Show benefits
print("\nMasstree Properties:")
print("  - Optimistic concurrency control (lock-free reads)")
print("  - Fine-grained locking (per-node locks for writes)")
print("  - Cache-conscious design (optimized node layout)")
print("  - Efficient range scans (linked leaf nodes)")
print("  - Used in: High-performance key-value stores")
print("  - Better than skip lists for range queries")
print("  - Better than hash tables for ordered operations")
```
