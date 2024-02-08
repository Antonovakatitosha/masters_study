from collections import deque
import enum
import copy


class MEMBER(enum.Enum):
    WOLF = 'волк'
    GOAT = 'коза'
    CABBAGE = 'капуста'
    BOATMAN = 'лодочник'


MEMBERS = {member for member in MEMBER}


class Shore:
    def __init__(self, members=None):

        if members is None: self._members = set()
        elif not members.issubset(MEMBERS): raise ValueError(f"Не берегу могут находиться только {MEMBERS}")
        else: self._members = members

    @property
    def members(self):
        return self._members

    @property
    def is_boatman(self):
        return MEMBER.BOATMAN in self._members

    def move_to_shore(self, member=None):
        if MEMBER.BOATMAN in self._members:
            raise ValueError( f"лодочник был на этом берегу, лодка не могла приплыть пустой")
        elif member and member not in MEMBERS:
            raise ValueError(f"Нельзя перевезти {member.value} на берег")
        if member in self._members:
            raise ValueError(f"{member.value} уже на этом берегу")
        # if member == MEMBER.BOATMAN: raise ValueError(f"Лодочник по умолчанию прибывает на берег на лодке")

        if member: self._members.add(member)
        if member != MEMBER.BOATMAN: self._members.add(MEMBER.BOATMAN)

    def move_from_shore(self, member=None):
        if MEMBER.BOATMAN not in self._members:
            raise ValueError(f"лодочника нет на этом берегу, лодка не может отплыть")
        elif member and (member not in MEMBERS or member not in self._members):
            raise ValueError(f"На береге нет {member.value}")
        # if member == MEMBER.BOATMAN: raise ValueError(f"Лодочник по умолчанию уплывает с берега на лодке")

        if member: self._members.remove(member)
        if member != MEMBER.BOATMAN: self._members.remove(MEMBER.BOATMAN)

    @property
    def available_take_away(self):
        return [] if MEMBER.BOATMAN not in self._members else list(self._members)

    def __str__(self):
        if not len(self._members): return "пусто"
        return f"На берегу {', '.join(member.value for member in self._members)}"


class River:
    def __init__(self):
        self._shore_start = Shore(MEMBERS.copy())
        self._shore_end = Shore()

    def _from_start_to_end_shore(self, member=None):
        self._shore_start.move_from_shore(member)
        self._shore_end.move_to_shore(member)
        # print(self.__str__())

    def _from_end_to_start_shore(self, member=None):
        self._shore_end.move_from_shore(member)
        self._shore_start.move_to_shore(member)
        # print(self.__str__())

    def move_from_shore(self, member=None):
        if self._shore_start.is_boatman:
            return self._from_start_to_end_shore(member)
        return self._from_end_to_start_shore(member)

    @property
    def can_move(self):
        return self._shore_start.available_take_away if self._shore_start.is_boatman else self._shore_end.available_take_away

    @property
    def members(self):
        return self._shore_start.members, self._shore_end.members

    @property
    def start_shore_empty(self):
        return not bool(self._shore_start.members)

    def __str__(self):
        return f"Начальный берег: {self._shore_start}. Конечный берег: {self._shore_end}."


def get_rivers_after_move(river: River):
    members = river.can_move
    rivers = []
    for member in members:
        new_river = copy.deepcopy(river)
        new_river.move_from_shore(member)
        rivers.append(new_river)
    return rivers


def is_valid(shore):
    if MEMBER.BOATMAN in shore: return True
    if MEMBER.WOLF in shore and MEMBER.GOAT in shore: return False
    if MEMBER.CABBAGE in shore and MEMBER.GOAT in shore: return False
    return True


def bfs():
    river = River()
    visited = set()
    visited.add(river)
    queue = deque(get_rivers_after_move(river))

    while queue:
        current_river = queue.popleft()
        if current_river.start_shore_empty: return current_river

        shore_start, shore_end = current_river.members
        if not is_valid(shore_start) or not is_valid(shore_end):
            # print(f'Решение не верно {current_river}')
            continue
        print(f'Узел вероятен {current_river}')
        queue.extend(get_rivers_after_move(current_river))
    return False

result = bfs()
print(f"Решение найдено {result}" if result else f"Решение не найдено")
