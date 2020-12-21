# coding=utf8


class Node(object):
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.time = 1
        self.prev = None
        self.next = None


class LFU(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.key_map = dict()
        self.time_map = dict()
        self.min_freq = 1

    def get(self, key):
        if key not in self.key_map:
            return -1
        node = self.key_map.get(key)
        pre_time = node.time
        new_time = pre_time + 1
        if new_time not in self.time_map:
            self.time_map[new_time] = self._init_link_node()
        head, tail = self.time_map[new_time]
        node.time = new_time
        self._delete_node(node)
        self.insert_node(head, node)
        if pre_time == self.min_freq:
            if self._is_empty(pre_time):
                self.min_freq = new_time
        return node.val

    def put(self, key, val):
        if key not in self.key_map:
            if len(self.key_map) >= self.capacity and self.capacity>0:
                head, tail = self.time_map[self.min_freq]
                node = self._delete_by_tail(tail)
                self.key_map.pop(node.key)
                if self._is_empty(self.min_freq):
                    self.time_map.pop(self.min_freq)

            if self.capacity == 0:
                return

            new_node = Node(key, val)
            self.key_map[key] = new_node
            time = 1
            if time not in self.time_map:
                self.time_map[time] = self._init_link_node()
            head, tail = self.time_map[time]
            self.insert_node(head, new_node)
            if self.min_freq > time:
                self.min_freq = time
        else:
            node = self.key_map[key]
            old_time = node.time
            new_time = old_time + 1
            if new_time not in self.time_map:
                self.time_map[new_time] = self._init_link_node()
            head, tail = self.time_map[new_time]
            node.time = new_time
            node.val = val
            self._delete_node(node)
            self.insert_node(head, node)
            if old_time == self.min_freq:
                if self._is_empty(old_time):
                    self.min_freq = new_time

    def _init_link_node(self):
        head = Node(None, None)
        tail = Node(None, None)
        head.prev = None
        head.next = tail
        tail.head = head
        tail.next = None
        return head, tail

    def insert_node(self, head, new_node):
        pre_next = head.next
        head.next = new_node
        new_node.prev = head
        pre_next.prev = new_node
        new_node.next = pre_next

    def _delete_node(self, node):
        pre_node = node.prev
        next_node = node.next
        if pre_node is None or next_node is None:
            return
        node.prev = None
        node.next = None
        pre_node.next = next_node
        next_node.prev = pre_node
        return node

    def _delete_by_tail(self, tail):
        to_del = tail.prev
        return self._delete_node(to_del)

    def _is_empty(self, time):
        if time not in self.time_map:
            return True
        head, tail = self.time_map[time]
        return head.next == tail
