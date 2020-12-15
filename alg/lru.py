# coding=utf8


class LinkedNode(object):
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRU(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.map = {}
        self.head = LinkedNode(None, None)
        self.tail = LinkedNode(None, None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key):
        if key not in self.map:
            return None
        node = self.map.get(key)
        self.move_to_head(node)
        return node.value

    def put(self, key, value):
        if key in self.map:
            node = self.map.get(key)
            node.value = value
            self.move_to_head(node)
        else:
            new_node = LinkedNode(key, value)
            if len(self.map) >= self.capacity:
                to_del = self.remove_tail()
                self.map.pop(to_del.key)
            self.add_to_head(new_node)
            self.map[key] = new_node

    def move_to_head(self, node):
        tmp_prev = node.prev
        tmp_next = node.next
        tmp_prev.next = tmp_next
        tmp_next.prev = tmp_prev
        head_next = self.head.next
        self.head.next = node
        node.prev = self.head
        node.next = head_next
        head_next.prev = node

    def add_to_head(self, node):
        head_next = self.head.next
        self.head.next = node
        node.prev = self.head
        node.next = head_next
        head_next.prev = node

    def remove_tail(self):
        to_del = self.tail.prev
        to_del.prev.next = self.tail
        self.tail.prev = to_del.prev
        to_del.prev = None
        to_del.next = None
        return to_del
