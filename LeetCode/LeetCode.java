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

19 Remove Nth Node From End of List
  - Fast pointer runs nth node first. Be careful of deleting head case.
  # Two Pointers

31 Next Permutation
  - Iterate from right to left in an array to find the first index that arr[index] < arr[index+1], then from right to index find the first index2 that arr[index2]>arr[index], swap these two. Then inverse from index+1 to rightmost. e.g. 4202320 -> 4203220 -> 4203022

62 Unique Paths
  - dp[i][j] = dp[i-1][j] + dp[i][j-1]
  # 2D DP

323 Number of Connected Components in an Undirected Graph
  -     public int find(int[] arr, int i) {
            while(arr[i] != i) {
                arr[i] = arr[arr[i]]; // path compression
                i = arr[i];
            }
            return i;
        }
        public void union(int[] arr, int i, int j) {
            int RootI = find(arr, i);
            int RootJ = find(arr, j);
            if(RootI != RootJ) {
                arr[RootI] = RootJ;
            }
        }
  - union find used for cycle detection, and connected component in undirected graph
  # Union Find

547 Friend Circles
  - LinkedList.removeFirst()
  - For BFS/DFS, use boolean[] visit instead of boolean[][] visit
  - Prefer DFS than BFS cause BFS needs to create and matain Queue.
  - DFS/BFS Time O(n^2), Space O(n)
  - Union Find Time O(n^3), Space O(n)
  - Union find uses path compression and rank
  & Refer Q200
  # DFS, BFS, Union Find

* DFS/BFS vs Union Find
  - if question gives 2D array/matrix, use DFS/BFS. e.g. P323(number of islands)
  - if question gives 2D relation array (e.g. node edge, friends relation), use 1D arry Union Find, DFS/BFS. e.g. P323, P547

218 The Skyline Problem
  - PriorityQueue<Integer> pq, pq.offer(number)
  - Collections.sort(list, (a,b)-> (a[0]==b[0]) ? a[1]-b[1] : a[0]-b[0]);
  - distinguish building start and end by making start as negative, sort first by start and then height in increasing order. Matain priority queue in decending order of height. Conner case would be [[0,2,3],[2,5,3]]
  # Priority Queue, TreeMap, Sweep Line

240 Search a 2D Matrix II
  - think of O(n) for seach first. Then to think O(logn) using binary search
  - For this question, think from 4 coners
  # Array

39 Combination Sum
  - // Use startIndex to avoid duplicate list
    public void find(int[] candidates, List<Integer> list, int target, int startIndex) {
        if(target < 0) return;
        if(target == 0) {
            res.add(new LinkedList<>(list)); // Create new list
            return;
        }
        for(int i=startIndex;i<candidates.length;i++) {
                list.add(candidates[i]);
                // target is not changed
                find(candidates, list, target - candidates[i], i);
                list.remove(list.size()-1);
        }
    }
  - time : O(2^n), space : O(n)
  # Backtracking

64 Minimum Path Sum
  # 2D DP

101 Symmetric Tree
  - In reursion, check treeNode1, treeNode2 null, compare treeNode.val
  - predorder check
  # Inorder Traversal, Binary Tree

124 Binary Tree Maximum Path Sum
  - think of preorder, or postorder
  - Integer.MIN_VALUE
  - postorder, calculate res = Math.max(res, left+right+node.val), return Math.max(left, right) + node.val
  # PostOrder Traversal, Binary Tree

236 Lowest Common Ancestor of a Binary Tree
  - find first parent that includes both nodes.
  - private boolean traverse(TreeNode node, TreeNode p, TreeNode q) {
        if(node == null) return false;
        boolean left = traverse(node.left, p, q);
        boolean right = traverse(node.right, p, q);
        boolean hasPOrQ = node==p || node==q;

        if(left && right) res = node;
        if((left||right) && hasPOrQ) res = node;

        return left || right || hasPOrQ;
    }
  # Postorder Traversal, Binary Tree

76 Minimum Window Substring
  - Use int array to record num of each char. faster pointer runs and decrease num of each char until total becomes 0. Then slow pointer runs and increase num of each char until total becomes 1. Then get i-j+1 compare with the min distance.

    int[] table = new int[128]; // ascii has 128 values
    for(char c:t.toCharArray()) {
        table[c]++;
    }
    int from =0;
    int min = Integer.MAX_VALUE;
    int total = t.length();

    for(int i=0, j=0;i<s.length();i++) {
        if(table[s.charAt(i)]-- > 0) total--;
        while(total ==0) {
            if(i-j+1 < min) {
                min = i-j+1; from = j;}
            if(++table[s.charAt(j++)] > 0) total++;}}

  # Sliding Window, Two Pointers, HashTable

105 Construct Binary Tree from Preorder and Inorder Traversal
  - preorder is easy to find root, then find index of root from inorder.
    public TreeNode helper(int preStart, int inStart, int inEnd, int[] preorder, int[] inorder) {
        if (preStart > preorder.length - 1 || inStart > inEnd) return null;
        TreeNode root = new TreeNode(preorder[preStart]);
        // find inIndex
        root.left = helper(preStart + 1, inStart, inIndex - 1, preorder, inorder);
        root.right = helper(preStart + inIndex - inStart + 1, inIndex + 1, inEnd, preorder, inorder);
  # Tree

32 Longest Valid Parentheses
  -   for(int i=0;i<s.length();i++) {
        if(s.charAt(i) == '(') {
            stack.push(i);
        } else {
          if(stack.empty()) {
              start = i;
          } else {
              stack.pop();
              res = stack.empty() ? Math.max(res, i-start) : Math.max(res, i-stack.peek());}}}

  # Stack

78 Subsets
  - Use startIndex to avoid duplicate
    for(int i = 1;i<=nums.length;i++)
        find(new LinkedList<Integer>(), nums, 0, i);
    }
    private void find(List<Integer> list, int[] nums, int startIndex, int num) {
        if(list.size() == num) {
            res.add(new LinkedList<>(list));
        }
        for(int i=startIndex; i<nums.length;i++) {
            list.add(nums[i]);
            find(list, nums, i+1, num); // use i+1 to avoid dupilcate
            list.remove(list.size()-1);
        }
    }
  # Backtracking

** 301 Remove Invalid Parentheses
  - count ( and ), if ) is more than ( at i. Iterate from j to i to remove ). Then reverse string to remove '('.
  - new StringBuilder(s).reverse().toString()
  - private void construct(List<String> res, String s, int fast, int slow, char[] pair) {
        int count = 0;
        for(int i=fast;i<s.length();i++) {
            if (s.charAt(i) == pair[0]) count++;
            if (s.charAt(i) == pair[1]) count--;
            if(count >= 0 ) continue;
            for(int j=slow;j<=i;j++) {
                if(s.charAt(j) == pair[1] && (j == slow || s.charAt(j-1) != pair[1])) {
                    construct(res, s.substring(0, j) + s.substring(j+1), i, j, pair);
                }
            }
            return;
        }
        String rs = new StringBuilder(s).reverse().toString();
        if(pair[0] == '(') {
            construct(res, rs, 0, 0, new char[]{')', '('});
        } else {
            res.add(rs);
        }
    }
  # Backtracking/DFS, Two Pointers

*297 Serialize and Deserialize Binary Tree
  - Integer.parseInt(str) // parse string to integer
  - Queue<Integer> queue = new LinkedList<>()
    queue.offer(Integer), queue.poll()
  - Deque<> deque = new LinkedList<>(Arrays.asList(data.split(",")))
  - (int)Math.pow(2,n), (int)Math.log(8)
  - convert tree to array, children of arr[i] should be 2*i+1, 2*i+2. i is index rather than tree level.
  # BFS, Preorder Traversal

55 Jump Game
  - Keep track of max (nums[i] + i), compare with i, if i>max, then return false.
  - BFS exceeds time limit, DFS exceeds memory limit
  - time O(n), space o(1)
  # Greedy

215 Kth Largest Element in an Array
  - PriorityQueue is minHeap, poll() removes the least value. Each op takes logk
  - Priority Queue keeps track of k elements. time O(nlogk), space O(n)
  - Quick Select, select pivot, compare and return left index, if left index = k, then return. (http://www.geekviewpoint.com/java/search/quickselect)
  - Quick Select time average O(n), wrost O(n^2)
  # Quick Select, Priority Queue

102 Binary Tree Level Order Traversal
  - time O(n), space O(n)
  # Preorder Traversal

647 Palindromic Substrings
  - interate string, for each iteration, left-- and right++ to find palindrome until s.charAt(left) != s.charAt(right).
  # String

75 Sort Colors
  - 1-pass: track r_index start from 0, track b_index from nums.length-1. Iterate until i<=b
  -   public void sortColors(int[] nums) {
        int r =0; int b = nums.length-1; int i=0;
        while(i<=b) {
            if(nums[i] == 0) {
                swap(nums, i++, r++);
            } else if(nums[i] == 2) {
                swap(nums, i, b--);
            } else {
                i++;}}}
  # Two Pointers, Sort

208 Implement Trie (Prefix Tree)
   -  class Trie {
        Trie[] children  = new Trie[26];
        boolean isWord = false;

        public void insert(String word) {
            Trie trie = this;
            for(int i=0;i<word.length();i++) {
                char c = word.charAt(i);
                if(trie.children[c-'a'] == null) {
                    trie.children[c-'a'] = new Trie();
                }
                trie = trie.children[c-'a'];
            }
            trie.isWord = true;}
    # Trie

























