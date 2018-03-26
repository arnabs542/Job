/*
    Solutions: http://www.cnblogs.com/grandyang/p/4606334.html
    Youtube: https://www.youtube.com/channel/UCTWuRL33U8xBPqk3LehXjFw/playlists
*/

1 Two Sum
  - Use HashMap to track each visited num
  # HashMap

2 Add Two Numbers
  ? iteration or recursion
  - Use iteration rather than recursion because of the corner case [9], [9,9]. It's hard to deal with this situation with recursion.
  - When using pointer to deal with each node in the linked list. Use a dummy root node to connect the real root node. At the end, return root.next.
  # Linked List

136 Single Number
    - bit manipulation. XOR: x^y

4 Median of Two Sorted Arrays
  - attention to corner cases
  - (https://www.youtube.com/watch?v=do7ibYtv5nk)
  # binary search

461 Hamming Distance
  - int_num % 2 == int_num >> 1
  # bit manipulation

11 Container With Most Water
  - Time O(n), Space O(1)
  # two pointer

3 Longest Substring Without Repeating Characters
  - String: charAt(), length(), isEmpty()
  - Set: contains(), size(), add(), remove()
  - Character -> char: Character.charValue()
  - int[] arr = new int[256]; Arrays.fill(arr,-1);
  - Time O(n), Space O(1)
  - #two pointer

20 Valid Parentheses
  - attention to corner case, e.g. check size before pop, check queue before return
  - only LinkedList has push & pop, List doesnt have
  # Stack

21 Merge Two Sorted Lists
  - deal with linkedlist
  # LinkedList

104 Maximum Depth of Binary Tree

55 Longest Palindromic Substring
  - String.substring(startIndex, endIndex) // "abc".substring(0,2) == "ab"
  - char -> String: Character.toString(char)
  - expand around center Time O(n^2), Space O(1)
  * dynamic programming (dp) Time O(n^2), Space O(n^2)
    boolean[][] dp = new boolean[s.length()][s.length()];
    dp[i][j] = s.charAt(i) == s.charAt(j) && (j-i<=2 || dp[i+1][j-1]);
  # DP

146 LRU(Least Recently Used) Cache
  - HashMap, Double LinkedList
  - head, tail, capacity, hashmap corner cases
  # Double LinkedList, HashMap

206 Reverse Linked List
  # LinkedList

53 Maximum Subarray
  - int[] arr; int length = arr.length // not length()
  # DP

155 Min Stack
  - LinkedList: push(), pop(), peek()
  - attention to size() of linkedlist when push() and pop()
  - 在-128~127的Integer值以int进行比较，而超出-128~127的范围，进行==比较时是进行地址及数值比较。转换成int再进行大小比较
  # Stack LinkedList

169 Majority Element
  - Arrays.sort() // nlog(n)
  # HashMap, Moore voting algorithm

283 Move Zeroes
  - attention to array length when running two pointers
  # Two Pointers

141 Linked List Cycle
  - set fast runner and slow runner, if has cycle, fast == slow
  # Two Pointers

148 Sort List
  - merge sort is preferrable for linked list
  - middle = getMiddle(head);merge(sort(head), sort(middle))
  # Merge Sort, Two Pointers, Merge two lists

121 Best Time to Buy and Sell Stock

70 Climbing Stairs
  - f(result) = f(one step away) + f(two step away)
  - recursion: time O(2^n) iteration: time O(n)
  # Fibonacci, DP

57 Merge k Sorted Lists
  - Merge sort for array, compare with 148(merge sort on list)
  - ListNode[] lists; lists.length // not lists.size()
  # Merge Sort, Divide and Conquer, Merge two lists

42 Trapping Rain Water
  - compare left and right, move towards and keep lMax, rMax
  - main idea is level up
  - Math.max, Math.min
  @ Time: O(n), Space: O(1)
  # Two Pointers

226 Invert Binary Tree
  # Post order traversal of Binary Tree

15 3Sum
  - be careful of IndexOutOfBoundsException for e.g. arr[i+1]
  - Arrays.sort(). Arrays.asList(T... a)
  - ArrayList rather than LinkedList
  @ Time: O(n^2), O(n)
  # Two Pointers, Sort, Edge Case

* 198 House Robber
  - Rob(n) = Max(Rob(n-2)+ Money(n), Rob(n-1))
  @ Time: O(n), Space: O(1)
  # DP

* 10 Regular Expression Matching
  - Run s ={x,a,a,b} p = {x,a,*,b}, dp[s.length()+1][p.length()+1], dp[0][0]=true, i in dp[][], i-1 in p|s
    deal with corner case when dp[0][~] e.g. a * b *
    if p.charAt(j) == s.charAt(i) || p.charAt(j) == ‘.’ : dp[i][j] = dp[i-1][j-1]
    if p.charAt(j-1) != s.charAt(i) : dp[i][j] = dp[i][j-2] //in this case, a* only counts as empty
    if p.charAt(j-1) == s.charAt(i) || p.charAt(j-1) == ‘.’ : dp[i][j] = dp[i][j-2] | dp[i-1][j]
  - (https://www.youtube.com/watch?v=l3hda49XcDE&t=194s)
  # DP

238 Product of Array Except Self
  - Corner case: 0 # 0, 1 # 0, >1 # 0
  @ Time: O(n), Space: O(n)

17 Letter Combinations of a Phone Number
  - Collections.emptyList(), Collections.singletonList()
  - Arrays.asList(T...), String str.toCharArray()
  @ Time: O(n^3), Space: O(n)
  # Recursion, Backtracking

617 Merge Two Binary Trees
  # Tree

22 Generate Parentheses
  - num of left Parentheses always >= right
  # Backtracking

56 Merge Intervals
  - List list.subList()
  - Collections.sort(List list, (a,b)->a-b), sort on list
  - Arrays.sort(), sort on array
  - Object operations are expensive e.g.sort, use primitive as much as possible
  # Sort

338 Counting Bits
  - i&(i - 1)， 这个本来是用来判断一个数是否是2的指数的快捷方法，比如8，二进制位1000, 那么8&(8-1)为0，只要为0就是2的指数
  - 每个i值都是i&(i-1)对应的值加1
  # Bit

200 Number of Islands
  - 2d array, matrix, connected component
  - avoid using collection or object to waste time
  # DFS/BFS, Graph, Connected Component

49 Group Anagrams
  - String s.toCharArray()
  - Arrays.sort(s.toCharArray())
  - new String(s.toCharArray())
  - Collections.singletonList(~) return the abstract list
  - new LinkedList(Collection)
  # Counting Sort, Sort, Array, List

139 Word Break
  - List list.contains()
  - String str.substring()
  - DP for boolean, not only int.
  # DP

152 Maximum Product Subarray
  - keep track max, min, res

48 Rotate Image
  - int[][] matrix, int[] arr = matrix[]  // pass by reference, matrix change, arr change.
  # Array

234 Palindrome Linked List
  - reverse linked list
  # Two Pointers, Reverse LinkedList

33 Search in Rotated Sorted Array
  - either [left, mid] or (mid,right] is sorted, check in sorted side
  - draw pic like (https://www.youtube.com/watch?v=w6nusIojP9c)
  - attention to using >= or >
  # Binary Search

160 Intersection of Two Linked Lists
  - compare object by reference
  # Compare List Node, LinkedList

*72 Edit Distance (template for 2d DP)
  - // insert: dp[i][j] = dp[i][j-1] +1
    // delete: dp[i][j] = dp[i-1][j] +1
    // replace: dp[i][j] = dp[i-1][j-1] +1
    if(word1.charAt(i-1) == word2.charAt(j-1)) {
        dp[i][j] = dp[i-1][j-1];
    } else {
        dp[i][j] = Math.min(Math.min(dp[i-1][j], dp[i][j-1]), dp[j-1][j-1])+1;
    }
  # 2D DP

46 Permutations
  -   transform(List<List<Integer>> list, List<Integer> sublist, int[] nums) {
        if(sublist.size() == nums.length) {
            list.add(new LinkedList<>(sublist)); // new linkedList
            return;
        }
        for(Integer i : nums) {
            if(sublist.contains(i)) {continue;}
            sublist.add(i);
            transform(list, sublist, nums);
            sublist.remove(i); // have to remove }}
  # Backtracking

96 Unique Binary Search Trees
  - f(n) = f(0)*f(n-1) + f(1)*f(n-2) + ... + f(n-1)*f(1)
  # 1D DP

128 Longest Consecutive Sequence
  - keep go down, go up and get difference
  # HashSet

347 Top K Frequent Elements
  - map.put(key, map.getOrDefault(key, 0) + 1);
  - TreeMap, sorted by key, treeMap.pollLastEntry().getValue()
  - List<Integer>[] bucket = new List[nums.length+1];
  # Sort, TreeMap, Butcket Sort

142 Linked List Cycle II
  - slow = a + b, fast = a + b + c + b, 2 * slow = fast
  # Two Pointers

287 Find the Duplicate Number
  - binary search template
      public int findDuplicate(int[] nums) {
        int max = nums.length-1;
        int min = 1;
        while(min<=max) {
            int mid = (min+max)/2, count = 0;
            for(int i = 0; i<nums.length;i++) {
                if(nums[i]<=mid) count++;
            }
            if (count>mid) max = mid - 1; else min = mid + 1;
        }
        return min; // attention
       }

  - HashSet: time O(n), space O(n)
    Sort: time O(nlogn), space O(1)
    Binary Search: time O(nlogn), space O(1)
  # Binary Search, LinkedList Cycle II

406 Queue Reconstruction by Height
  - sort [a,b] first by decending a, then by increasing b. arr = int[][]
    Arrays.sort(arr, (x,y) -> x[0] == y[0]? x[1]-y[1] ? y[0]-x[0])
  - list to array: List.toArray(T[] a)
  - Time: O(n^2), Space: O(n)
  # 2D Array

448 Find All Numbers Disappeared in an Array
  - Use relation between array index ([0, n-1]) and the given value range [1,n]. Set value-1 to index and mark as negative. Then iterate through array and find positive values, which means that index is not visited.
  - Time O(n), Space O(n)
  # Value to Index

279 Perfect Squares
  - for(int i=0;i<=n;i++) {
        for(int j =1;j*j<=i;j++) {
            dp[i] = Math.min(dp[i], dp[i-j*j] + 1);
        }
    }
  # 1D DP

79 Word Search
  - remember to set boolean[][] back if bfs failed
  # BFS, Backtracking









