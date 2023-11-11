max_indexes = []
indexes_length = []
numbers_n = 1

ethalon_length = float(input("input ethalon length: "))
n = int(input("input n of for: "))

for i in range(n):
    len_ = float(input(f"{i+1} - length: "))
    max_index = int(ethalon_length // len_) + 1
    # print(f"{i+1} - max_index: {max_index}")
    max_indexes.append(max_index)
    indexes_length.append(len_)
    numbers_n = numbers_n * max_index

count = 0
for i in range(numbers_n):
    curr_i = i

    sum_length = 0
    for j in range(n):
        if curr_i > 0:
            sum_length += indexes_length[j] * (curr_i % max_indexes[j])
            curr_i //= max_indexes[j]

    curr_i = i

    if sum_length <= ethalon_length:
        count += 1
        for j in range(n):
            if curr_i > 0:
                print(curr_i % max_indexes[j], end=' ')
                curr_i //= max_indexes[j]
            else:
                print(curr_i, end=' ')
        print(f"- {round(ethalon_length - sum_length, 1)}")

print(f"Total combinations: {count}")
