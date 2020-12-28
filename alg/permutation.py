# coding=utf8


def permutation(arr):
    res = []
    if not arr:
        return None
    visited = [0] * len(arr)
    sorted_arr = sorted(arr)
    path = []
    backtrack(arr, path, visited, res)
    return res


def backtrack(arr, path, visited, res):
    if len(path) == len(arr):
        res.append(path.copy())
        return
    for i in range(len(arr)):
        if visited[i] == 1:
            continue
        if i > 0 and arr[i] == arr[i-1] and visited[i-1] == 0:
            continue
        visited[i] = 1
        path.append(arr[i])
        backtrack(arr, path, visited, res)
        visited[i] = 0
        path.pop()


def permutation2(arr):
    sorted_arr = sorted(arr)
    res = []
    visited = [False] * len(arr)
    tmp_path = []
    trace_back(res, visited, tmp_path, sorted_arr)
    return res


def trace_back(res, visited, tmp_path, arr):
    if len(tmp_path) == len(arr):
        res.append(tmp_path.copy())
        return
    for i in range(len(arr)):
        if i not in visited:
            if i > 0 and arr[i-1] == arr[i] and not visited[i-1]:
                continue
            tmp_path.append(arr[i])
            visited[i] = True
            trace_back(res, visited, tmp_path, arr)
            tmp_path.pop()
            visited[i] = False


if __name__ == "__main__":
    test_arr = [1, 2, 2]
    res = permutation(test_arr)
    print(res)
