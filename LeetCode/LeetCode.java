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









