# -*- coding: utf-8 -*-
# Author      : ShiFan
# Created Date: 2024/11/30 19:01
import array
import bisect
import inspect
import math
from collections import deque
from threading import RLock
from typing import Any, Iterable, Optional, Union


DEBUG = 0


class BPTDebug:

    def __enter__(self):
        if DEBUG > 1:
            stack = inspect.stack()
            snap_frame = stack[1]
            func_name = snap_frame.function
            f_locals = snap_frame.frame.f_locals
            obj = f_locals.pop("self")
            kwargs = ", ".join(f"{k}: {v}" for k,v in f_locals.items())
            snap_stack = f"{func_name}({kwargs}):\n"
            snap_stack += "=" * 80
            snap_stack += f"\n{obj}"
            del stack
            print(snap_stack)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if DEBUG > 1:
            stack = inspect.stack()
            frame = stack[1]
            func_name = frame.function
            f_locals = frame.frame.f_locals
            obj = f_locals.pop("self")
            kwargs = ", ".join(f"{k}: {v}" for k,v in f_locals.items())
            snap_stack = f"now for {func_name}({kwargs}):\n"
            snap_stack += "=" * 80
            snap_stack += f"\n{obj}"
            del stack
            print(snap_stack)


# BPTRLock = BPTDebug()
BPTRLock = RLock()


class InternalNode:
    """
    使用线程递归锁
    """
    NodeType = "Internal"

    def __init__(
        self, tree: "BPTree",
        children: list[Union["InternalNode", "LeafageNode"]],
        parent: "InternalNode"=None
    ):
        self.tree = tree
        self.keys: array.array[int] = array.array("q")
        if children:
            self.children = children
            left = self.children[0]
            if isinstance(left, InternalNode):
                left.parent = self
            else:
                left.internal = self
            for children in self.children[1:]:
                if isinstance(children, InternalNode):
                    children.parent = self
                    self.keys.append(children.left.keys[0])
                elif isinstance(children, LeafageNode):
                    children.internal = self
                    self.keys.append(children.keys[0])
        else:
            self.children: list[Union["InternalNode", "LeafageNode"]] = list()
        if parent:
            self.parent = parent
        else:
            self.parent: Optional["InternalNode"] = None

    @property
    def left(self):
        with BPTRLock:
            node = self
            while 1:
                try:
                    node = node.children[0]
                    if isinstance(node, LeafageNode):
                        return node
                except IndexError:
                    return None

    def balanced(self):
        """
        子节点数量超出限制，触发分裂
        子节点数量不够半满，触发合并
        """
        parent = self.parent
        if not self.children:
            parent.remove_children(self)
        if not parent and len(self.keys) == 0:
            _child = self.children[-1]
            self.tree.root = _child
            if isinstance(_child, InternalNode):
                _child.parent = None
            else:
                _child.internal = None
            return
        if len(self.keys) < self.tree.half_full and parent:
            self.merge()
        elif len(self.keys) > self.tree.branching_factor + 1:
            self.split()
        if parent:
            return parent.balanced()

    def add_pointer(self, pointer: int) -> Optional[int]:
        with BPTRLock:
            left_children = self.children[0]
            if pointer < left_children.keys[0]:
                child_index = 0
            elif self.keys and pointer < self.keys[0]:
                child_index = 1
            else:
                child_index = bisect.bisect(self.keys, pointer) + 1
            if child_index:
                self.keys.insert(child_index - 1, pointer)
            else:
                if isinstance(left_children, InternalNode):
                    self.keys.insert(0, left_children.left.keys[0])
                else:
                    self.keys.insert(0, left_children.keys[0])
                if self.parent:
                    self.parent.update_pointer(
                        self.keys[1], pointer
                    )
            return child_index

    def remove_pointer(self, pointer: int):
        with BPTRLock:
            if pointer in self.keys:
                old_index = self.keys.index(pointer)
                self.keys.remove(pointer)
                return old_index
            if self.parent:
                self.parent.remove_pointer(pointer)
                return -1

    def update_pointer(self, old_pointer, new_pointer):
        with BPTRLock:
            if old_pointer in self.keys:
                index = self.keys.index(old_pointer)
                self.keys[index] = new_pointer
            if self.parent:
                self.parent.update_pointer(old_pointer, new_pointer)

    def _add_children(self,  child: Union["InternalNode", "LeafageNode"]):
        with BPTRLock:
            if isinstance(child, LeafageNode):
                child_index = self.add_pointer(child.keys[0])
            else:
                child_index = self.add_pointer(child.left.keys[0])
            self.children.insert(child_index, child)
            if isinstance(child, LeafageNode):
                child.internal = self
            elif isinstance(child, InternalNode):
                child.parent = self

    def add_children(self, child: Union["InternalNode", "LeafageNode"]):
        self._add_children(child)

    def _remove_children(self, child: Union["InternalNode", "LeafageNode"]):
        with BPTRLock:
            child_index = self.children.index(child)
            pointer = None
            # next_child = self.children[child_index + 1]
            self.children.remove(child)
            if isinstance(child, LeafageNode):
                child.internal = None
                pointer = child.keys[0]
            else:
                child.parent = None
                if child_left := child.left:
                    pointer = child_left.keys[0]
            if child_index == 0:
                new_pointer = self.keys.pop(0)
                if self.parent:
                    self.parent.update_pointer(pointer, new_pointer)
            elif pointer is not None:
                self.remove_pointer(pointer)
            # if not self.keys:
            #     self.keys.append(next_child.left.keys[0])

    def remove_children(self, child: Union["InternalNode", "LeafageNode"]):
        """
        """
        self._remove_children(child)

    def split(self):
        """
        分裂自身，从右边拆分出新节点，避免需要更新索引指针
        """
        with BPTRLock:
            right_children = self.children[self.tree.half_full + 1:]
            for child in right_children:
                self._remove_children(child)
            right_node = InternalNode(
                self.tree,
                parent=self.parent, children=right_children
            )
            if not self.parent:
                new_parent = InternalNode(
                    self.tree,
                    children=[self, right_node]
                )
                self.tree.root = self.parent = new_parent
                right_node.parent = new_parent
            else:
                self.parent.add_children(right_node)

    def merge(self):
        """
        键数量不到半满
        避免更新边界，尽可能从右边往左边合并
        """
        with BPTRLock:
            if not self.parent or len(self.parent.children) == 1:
                return
            index = self.parent.children.index(self)
            if index > 0:
                left = self.parent.children[index - 1]
                try:
                    right = self.parent.children[index + 1]
                except IndexError:
                    right = None
            else:
                left = None
                right = self.parent.children[1]
            if left and (
                    len(left.children) + len(self.children) <=
                    self.tree.branching_factor + 1
            ):
                dest = "left"
            elif right and (
                    len(right.children) + len(self.children) <=
                    self.tree.branching_factor + 1
            ):
                dest = "right"
            else:
                dest = None
            if dest == "left":
                self.parent._remove_children(self)
                for child in self.children:
                    # self._remove_children(child)
                    left._add_children(child)
            elif dest == "right":
                self.parent._remove_children(right)
                for child in right.children:
                    # right._remove_children(child)
                    self._add_children(child)

    def __repr__(self):
        debug = (
            f"<{self.NodeType}: "
            f"[{', '.join(str(k) for k in self.keys)}]({len(self.keys)}) "
            f"children({len(self.children)})>"
        )
        return debug


class LeafageNode:
    """
    使用线程递归锁
    """
    NodeType = "Leafage"

    def __init__(
        self, tree: "BPTree",
        keys: Iterable,
        datas: list[Any],
        internal: "InternalNode"=None,
        previous_leafage: "LeafageNode" = None,
        next_leafage: "LeafageNode" = None
    ):
        """
        :param tree: BPlusTree
        :param keys: 外部数据的索引
        :param datas: 已索引的外部数据，可以看作缓存
        :param internal: 关联的上级内部节点
        :param next_leafage: 链接的下一个叶子节点
        """
        self.tree = tree
        self.max_num = self.tree.branching_factor
        self.keys: array.array[int] = array.array("q")
        if keys:
            self.keys.extend(keys)
        if internal:
            self.internal = internal
        else:
            self.internal: Optional[InternalNode] = None
        self.datas = datas
        if previous_leafage:
            self.previous_leafage = previous_leafage
            previous_leafage.next_leafage = self
        else:
            self.previous_leafage: Optional["LeafageNode"] = None
        if next_leafage:
            self.next_leafage = next_leafage
            next_leafage.previous_leafage = self
        else:
            self.next_leafage: Optional["LeafageNode"] = None

    def balanced(self):
        internal = self.internal
        if not self.datas:
            if self.previous_leafage:
                self.previous_leafage.next_leafage = self.next_leafage
            if self.next_leafage:
                self.next_leafage.previous_leafage = self.previous_leafage
            if not internal:
                self.keys.pop()
                return
            internal.remove_children(self)
        if len(self.keys) < self.tree.half_full and internal:
            self.merge()
        elif len(self.keys) > self.tree.branching_factor + 1:
            self.split()
        if internal:
            internal.balanced()

    def add_data(self, key, data):
        with BPTRLock:
            if key in self.keys:
                raise RuntimeError(f"{key} is exists!")
            if not self.keys:
                index = 0
                old_key = None
            elif key > self.keys[-1]:
                index = len(self.keys)
                old_key = None
            else:
                index = bisect.bisect(self.keys, key)
                old_key = self.keys[index]
            self.keys.insert(index, key)
            self.datas.insert(index, data)
            if (
                    old_key is not None
                    and index == 0
                    and self.internal
            ):
                self.internal.update_pointer(old_key, key)

    def remove_data(self, key):
        with BPTRLock:
            index = self.keys.index(key)
            self.datas.pop(index)
            if self.datas:
                self.keys.pop(index)
                if index == 0 and self.internal:
                    self.internal.update_pointer(key, self.keys[0])

    def update_data(self, key, data):
        with BPTRLock:
            index = self.keys.index(key)
            self.datas[index] = data

    def split(self):
        """
        往右边拆分自身，调用上级内部节点的 add_children 方法添加新拆分出来的叶子节点
        """
        with BPTRLock:
            self.keys, right_keys = (
                self.keys[:self.tree.half_full],
                self.keys[self.tree.half_full:]
            )
            self.datas, right_datas = (
                self.datas[:self.tree.half_full],
                self.datas[self.tree.half_full:]
            )
            self.next_leafage = LeafageNode(
                self.tree, right_keys, right_datas,
                previous_leafage=self,
                next_leafage=self.next_leafage
            )
            if not self.internal:
                internal = InternalNode(
                    self.tree,
                    [self, self.next_leafage]
                )
                self.tree.root = self.internal = internal
                self.next_leafage.internal = internal
            else:
                self.internal.add_children(self.next_leafage)

    def merge(self):
        """
        找到距离最近的邻近叶子节点，合并过去，从上级内部节点删除合并前位于右边的叶子节点
        """
        with BPTRLock:
            inf = float("inf")
            key_num = len(self.keys)
            if (
                    previous_l := self.previous_leafage
            ) and (
                    previous_l.internal is self.internal
            ):
                distance = self.keys[0] - previous_l.keys[-1]
                previous_key_num = len(previous_l.keys)
            else:
                distance = previous_key_num = inf
            if (
                    next_l := self.next_leafage
            ) and (
                    next_l.internal is self.internal
            ):
                right_distance = next_l.keys[0] - self.keys[-1]
                next_key_num = len(next_l.keys)
            else:
                right_distance = next_key_num = inf

            def left_merge():
                if self.internal:
                    self.internal.remove_children(self)
                previous_l.keys.extend(self.keys)
                previous_l.datas.extend(self.datas)
                previous_l.next_leafage = next_l
                if previous_l.next_leafage:
                    previous_l.next_leafage.previous_leafage = previous_l
                self.previous_leafage = self.next_leafage = None

            def right_merge():
                if self.internal:
                    self.internal.remove_children(next_l)
                self.keys.extend(next_l.keys)
                self.datas.extend(next_l.datas)
                self.next_leafage = next_l.next_leafage
                if self.next_leafage:
                    self.next_leafage.previous_leafage = self
                next_l.previous_leafage = None
                next_l.next_leafage = None

            if (distance == right_distance == inf) and self.internal:
                if self.tree.root is self.internal:
                    self.tree.root = self
                return
            elif (
                    previous_key_num + key_num <= self.tree.branching_factor
                    and next_key_num + key_num <= self.tree.branching_factor
            ):
                if distance < right_distance:
                    left_merge()
                else:
                    right_merge()
            elif previous_key_num + key_num <= self.tree.branching_factor:
                left_merge()
            elif next_key_num + key_num <= self.tree.branching_factor:
                right_merge()

    def __repr__(self):
        debug = (
            f"<{self.NodeType}: "
            f"[{', '.join(str(k) for k in self.keys)}] "
            f"datas({len(self.datas)})>"
        )
        return debug


class BPTree:
    """
    当前线程使用递归锁

    branching factor = 3
    half full = ceil((3 + 1) / 2) = floor(3 / 2) + 1= 2
    min num children = (half full)
    max num children (branching factor) + 1
    min num keys = (min num children) - 1
    max num keys = (max num children) - 1
    keys = (children) - 1


    root, leafage    [5]
                      |
    data             d5

    root, internal                  [5, 9, 39]
                              /         |             \
    leafage     [1, 2, 3] => [5, 6] => [9, 10, 16] => [39, 40, 42]
                 |  |  |      |  |      |   |   |      |   |   |
    data        d1 d2 d3     d5  d6    d9 d10 d16     d39 d40 d42

    ====>> add data d50

    root, internal                  [5, 9, 39]
                              /         |             \
    leafage     [1, 2, 3] => [5, 6] => [9, 10, 16] => [39, 40, 42, 50](split)
                 |  |  |      |  |      |   |   |      |   |   |   |
    data        d1 d2 d3     d5  d6    d9 d10 d16     d39 d40 d42 d50

    ====>> split leafage, parent add pointer

    root, internal                  [5, 9, 39, 50]
                              /         |             \
    leafage     [1, 2, 3] => [5, 6] => [9, 10, 16] => [39, 40, 42] => [50]
                 |  |  |      |  |      |   |   |      |   |   |        |
    data        d1 d2 d3     d5  d6    d9 d10 d16     d39 d40  d42     d50

    ====>> root split

    root, internal              [5, 9]                        [50]
                             /     |   \                    /     \
    leafage     [1, 2, 3] => [5, 6] => [9, 10, 16] => [39, 40, 42] => [50]
                 |  |  |      |  |      |   |   |      |   |   |        |
    data        d1 d2 d3     d5  d6    d9 d10 d16     d39 d40  d42     d50

    ====>> key = children - 1

    root, internal              [5, 9]                        [50]
                             /     \   \                    /     \
    leafage     [1, 2, 3] => [5, 6] => [9, 10, 16] => [39, 40, 42] => [50]
                 |  |  |      |  |      |   |   |      |   |   |        |
    data        d1 d2 d3     d5  d6    d9 d10 d16     d39 d40  d42     d50

    ====>> add root

    root                                         [39]
                                         /                 \
    internal                    [5, 9]                        [50]
                             /     \   \                    /     \
    leafage     [1, 2, 3] => [5, 6] => [9, 10, 16] => [39, 40, 42] => [50]
                 |  |  |      |  |      |  |   |       |   |   |       |
    data        d1 d2 d3     d5  d6    d9 d10 d16     d39 d40  d42    d50


    ====>> delete data d5, parent change pointer

    root                                       [39]
                                      /                 \
    internal                 [6, 9]                        [50]
                          /     /   \                    /     \
    leafage     [1, 2, 3] => [6] => [9, 10, 16] => [39, 40, 42] => [50]
                 |  |  |      |      |  |   |       |   |   |       |
    data        d1 d2 d3      d6    d9 d10 d16     d39 d40  d42    d50

    ====>> merge leafage, parent remove pointer

    root                                    [39]
                                     /               \
    internal                 [6, 9]                    [50]
                           /     \                   /     \
    leafage     [1, 2, 3, 6] => [9, 10, 16] => [39, 40, 42] => [50]
                 |  |  |  |      |  |   |       |   |   |       |
    data        d1 d2 d3  d6    d9 d10 d16     d39 d40  d42    d50

    ====>> split leafage, parent add pointer

    root                                      [39]
                                       /                 \
    internal                   [3, 9]                      [50]
                             /       \                   /     \
    leafage     [1, 2] => [3, 6] => [9, 10, 16] => [39, 40, 42] => [50]
                 |  |     |  |      |  |   |       |   |    |      |
    data        d1 d2    d3  d6    d9 d10 d16     d39 d40   d42   d50

    """

    def __init__(
        self,
        branching_factor: int,
        root_node: Union[InternalNode, LeafageNode, None] = None,
    ):
        self.branching_factor = branching_factor
        self.half_full = math.floor(self.branching_factor / 2) + 1
        self.root = root_node

    def nearest_search(self, key: int) -> LeafageNode:
        """
        返回索引键范围包含该键的叶子节点
        """
        with BPTRLock:
            if not self.root:
                raise RuntimeError("Empty Tree")
            if isinstance(self.root, LeafageNode):
                return self.root
            node = self.root
            while 1:
                for index, pointer in enumerate(node.keys):
                    if key < pointer:
                        node = node.children[index]
                        break
                else:
                    node = node.children[-1]
                if isinstance(node, LeafageNode):
                    break
            return node

    def range_query(self, start: int, end: int) -> deque[tuple]:
        with BPTRLock:
            datas = deque()
            leafage = self.nearest_search(start)
            while 1:
                if start > leafage.keys[-1]:
                    if leafage.next_leafage:
                        leafage = leafage.next_leafage
                        continue
                    break
                elif end < leafage.keys[0]:
                    break
                elif start < leafage.keys[-1]:
                    for index, key in enumerate(leafage.keys):
                        if start <= key <= end:
                            datas.append((key, leafage.datas[index]))
            return datas

    def get(self, key: int) -> Any:
        with BPTRLock:
            leafage = self.nearest_search(key)
            if key in leafage.keys:
                index = leafage.keys.index(key)
                return leafage.datas[index]
            raise KeyError(f"not found {key} from {leafage}")

    def insert(self, key: int, data):
        with BPTRLock:
            if self.root:
                leafage = self.nearest_search(key)
                if (
                        key > leafage.keys[-1]
                        and len(leafage.keys) == self.branching_factor
                        and (next_leafage := leafage.next_leafage)
                        and len(next_leafage.keys) < self.branching_factor
                ):
                    leafage = next_leafage
                try:
                    leafage.add_data(key, data)
                    leafage.balanced()
                except Exception as e:
                    raise RuntimeError(
                        f"add {key} to {leafage}({leafage.internal}) failed."
                        f"\n{self}"
                    ) from e
            else:
                leafage = LeafageNode(self, [key], [data])
                self.root = leafage

    def remove(self, key: int):
        with BPTRLock:
            leafage = self.nearest_search(key)
            leafage.remove_data(key)
            leafage.balanced()

    def update(self, key: int, data):
        with BPTRLock:
            leafage = self.nearest_search(key)
            leafage.update_data(key, data)

    def __repr__(self):
        g = ""
        if not self.root:
            return g
        nodes = [self.root]
        while nodes:
            info = []
            new_nodes = []
            for node in nodes:
                info.append(str(node))
                if isinstance(node, InternalNode):
                    new_nodes.extend(node.children)
            info = "    ".join(info)
            g += f"{info}\n"
            nodes = new_nodes
        return g
