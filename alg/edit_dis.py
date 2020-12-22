import numpy as np


class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        l1 = len(word1)
        l2 = len(word2)
        dist_arr = np.zeros([l1+1, l2+1])
        for i in range(l1 + 1):
            dist_arr[i][0] = i

        for j in range(l2 + 1):
            dist_arr[0][j] = j

        for i in range(1, l1+1):
            for j in range(l2+1):
                self._calc_dist(word1, word2, dist_arr, i, j)
        return int(dist_arr[l1][l2])

    def _calc_dist(self, word1, word2, dist_arr, i, j):
        if word1[i-1] != word2[j-1]:
            dist_arr[i][j] = min([dist_arr[i - 1][j - 1] + 1, dist_arr[i - 1][j]+1, dist_arr[i][j - 1]+1])
            return
        dist_arr[i][j] = min([dist_arr[i - 1][j]+1, dist_arr[i][j - 1]+1, dist_arr[i - 1][j - 1]])


if __name__ == "__main__":
    obj = Solution()
    w1 = "horse"
    w2 = "ros"
    dis = obj.minDistance(w1, w2)
    print(dis)