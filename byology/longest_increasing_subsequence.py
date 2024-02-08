from collections import deque

sequence = [10, 33, 34, 21, 22, 25, 50, 41, 60, 80, 111, 3, 0, 3, 80]

queue = deque()
queue.append({'start': [], 'tail': sequence})
solutions = []


def get_first_index(seq, elem):
    return seq.index(elem)


def remove_smaller_from_end(seq, elem):
    first_index = get_first_index(seq, elem)
    return list(filter(lambda x: x > elem, seq[first_index + 1:]))


step = 0
while queue:
    current_item = queue.popleft()
    step += 1
    print(f"step = {step}, current_item = {current_item}")

    if not len(current_item['tail']):
        solutions.append(current_item['start'])
        continue

    sequence = current_item['tail']
    for i in range(len(sequence)):
        local_min = sequence[i]
        queue.append({'start': current_item['start'][:] + [local_min],
                      'tail': remove_smaller_from_end(sequence, local_min)})


# print(solutions)

longest_array = max(solutions, key=len)
print(f"longest length: {len(longest_array)}, longest sequence: {longest_array}")
# longest length: 8, longest sequence: [10, 21, 22, 25, 50, 60, 80, 111]
