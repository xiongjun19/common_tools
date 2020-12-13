# coding=utf8


class HeapSort(object):
    def sort(self, arr):
        res = []
        heap = []
        for elem in arr:
            self._add_heap(heap, elem)
        while len(heap):
            res.append(heap[0])
            tail = heap.pop()
            if len(heap) > 0:
                heap[0] = tail
                self._update_head(heap)
        return res

    def _add_heap(self, dest, x):
        dest.append(x)
        if len(dest) > 1:
            self._update_tail(dest)

    def _update_tail(self, arr):
        """
        用来更新堆的尾部
        """
        idx = len(arr) - 1
        while idx > 0:
            par_id = (idx - 1) // 2
            if arr[par_id] > arr[idx]:
                arr[par_id], arr[idx] = self.swap(arr[par_id], arr[idx])
            else:
                break
            idx = par_id

    def _update_head(self, arr):
        """
        用来更新堆的头部
        """
        idx = 0
        while idx < len(arr) - 1:
            lc_idx = 2 * idx + 1
            rc_idx = 2 * idx + 2
            min_idx = idx
            min_val = arr[idx]
            if lc_idx < len(arr):
                if min_val > arr[lc_idx]:
                    min_idx = lc_idx
                    min_val = arr[lc_idx]
            if rc_idx < len(arr):
                if min_val > arr[rc_idx]:
                    min_idx = rc_idx
            if min_idx == idx:
                break
            arr[idx], arr[min_idx] = self.swap(arr[idx], arr[min_idx])
            idx = min_idx

    def swap(self, x, y):
        tmp = x
        x = y
        y = tmp
        return x, y


if __name__ == "__main__":
    import numpy as np
    t_arr = np.random.randint(100, size=10)
    print("org arr is: ", t_arr)
    t_obj = HeapSort()
    sorted_arr = t_obj.sort(t_arr)
    print("sorted arr is: ")
    print(sorted_arr)
