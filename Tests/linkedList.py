class Node:
    def __init__(self, data):
        self.data  = data 
        self.next = None
        return # what does this mean?

    def has_value(self, value):
        if self.data == value:
            return True
        return False

class SingleLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None 
        
    def append(self, item):
        # add an item in the list 
        if not isinstance(item, Node):
            item = Node(item)

        if self.head is None:
            self.head = item
        else:
            self.tail.next = item
        
        self.tail = item
        return 

    def list_length(self):
        count = 0
        current_node = self.head

        while current_node is not None:
            count +=1
            current_node = current_node.next
        return count