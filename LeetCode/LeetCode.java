1. Two Sum
  - Use HashMap to track each visited num
  # HashMap

2. Add Two Numbers
  - Use iteration rather than recursion because of the corner case [9], [9,9]. Its hard to deal with this situation with recursion.
  - When using pointer to deal with each node in the linked list. Use a dummy root node to connect the real root node. At the end, return root.next.
  # Linked List

136. Single Number
    - bit manipulation. XOR: x^y

*4. Median of Two Sorted Arrays
  - attention to corner cases
  - (https://www.youtube.com/wa  tch?v=do7ibYtv5nk)
  -     int length1 = nums1.length;
        int length2 = nums2.length;
        int length = length1 + length2;
        if(length1>length2) return findMedianSortedArrays(nums2, nums1);

        int cutL = 0, cutR = length1;
        int cut1 = 0, cut2 = 0;
        while(cut1<=length1) {
            cut1 = cutL + (cutR - cutL)/2;
            cut2 = length/2 - cut1;
            double l1 = cut1 == 0 ? Integer.MIN_VALUE : nums1[cut1-1];
            double l2 = cut2 == 0 ? Integer.MIN_VALUE : nums2[cut2-1];
            double r1 = cut1 == length1 ? Integer.MAX_VALUE : nums1[cut1];
            double r2 = cut2 == length2 ? Integer.MAX_VALUE : nums2[cut2];
            if(l1>r2) {
                cutR--;
            } else if(l2>r1) {
                cutL++;
            } else {
                if(length%2 == 0) {
                    return (Math.max(l1, l2) + Math.min(r1, r2))/2;
                } else {
                    return Math.min(r1, r2);
                }
            }
        }
  # binary search

461. Hamming Distance
  - int_num / 2 == int_num >> 1
  -     int res = x^y;
        while(res>0) {
            num += res & 1;
            res >>= 1;
        }
  # bit manipulation

11. Container With Most Water
  - Time O(n), Space O(1)
  # two pointer

3. Longest Substring Without Repeating Characters
  - String: charAt(), length(), isEmpty()
  - Set: contains(), size(), add(), remove()
  - Character -> char: Character.charValue()
  - int[] arr = new int[256]; Arrays.fill(arr,-1);
  - Time O(n), Space O(1)
  - int[] matains the previous visited index
  # two pointer

20. Valid Parentheses
  - attention to corner case, e.g. check size before pop, check queue before return
  - only LinkedList has push & pop, List doesnt have
  # Stack

21. Merge Two Sorted Lists
  - deal with linkedlist
  # LinkedList

104 Maximum Depth of Binary Tree

*5. Longest Palindromic Substring
  - String.substring(startIndex, endIndex) // "abc".substring(0,2) == "ab"
  - char -> String: Character.toString(char)
  - expand around center Time O(n^2), Space O(1)
  - dynamic programming (dp) Time O(n^2), Space O(n^2)
  -     boolean[][] dp = new boolean[s.length()][s.length()];
        int max = 0;
        String str="";

        for(int j =0;j<s.length();j++) {
            for(int i = 0;i<=j;i++) {
                dp[i][j] = s.charAt(i) == s.charAt(j) && (j-i<=2 || dp[i+1][j-1]);
                if(dp[i][j] && j-i+1>max) {
                    max = j-i;
                    str = s.substring(i, j+1);
                }
            }
        }
  # 2D DP

*146. LRU(Least Recently Used) Cache
  - HashMap, Double LinkedList
  - Create Dummy head and tail node, tail node helps remove node
  - private void addNode(LinkedNode node){
        node.next = head.next;node.pre = head;
        head.next.pre = node;head.next = node;
    }
    private void removeNode(LinkedNode node){node.next.pre = node.pre;node.pre.next = node.next;}
    private void moveToHead(LinkedNode node){removeNode(node); addNode(node);}
    public int get(int key) {
        if (cache.containsKey(key)){
            LinkedNode node = cache.get(key);
            moveToHead(node);
            return node.value;
        }
        return -1;
    }
    public void put(int key, int value) {
        LinkedNode node  = cache.get(key);
        if (node == null){
            node = new LinkedNode(key, value);
            addNode(node);
            cache.put(key,node);
            ++count;
            if (count > _capacity){
                cache.remove(tail.pre.key);
                removeNode(tail.pre);
                count--;
            }
        } else {
            node.value = value;
            moveToHead(node);
        }
    }
  # Double LinkedList, HashMap

*206. Reverse Linked List
  - // Iteration
    public ListNode reverseList(ListNode head) {
      ListNode prev = null;
      ListNode curr = head;
      while (curr != null) {
          ListNode nextTemp = curr.next;
          curr.next = prev;
          prev = curr;
          curr = nextTemp;
      }
      return prev;
    }
    // Recursion
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode p = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return p;
    }
  # LinkedList

53. Maximum Subarray
  - int[] arr; int length = arr.length // not length()
  # Array

155. Min Stack
  - LinkedList: push(), pop(), peek()
  - attention to size() of linkedlist when push() and pop()
  - 在-128~127的Integer值以int进行比较，而超出-128~127的范围，进行==比较时是进行地址及数值比较。转换成int再进行大小比较
  # Stack LinkedList

169. Majority Element
  - Arrays.sort() // nlog(n)
  # HashMap, Moore voting algorithm

283. Move Zeroes
  - attention to array length when running two pointers
  # Two Pointers

141. Linked List Cycle
  - set fast runner and slow runner, if has cycle, fast == slow
  # Two Pointers

*148. Sort List
  - merge sort is preferrable for linked list
  - middle = getMiddle(head);merge(sort(head), sort(middle))
  # Merge Sort, Two Pointers, Merge two lists

121. Best Time to Buy and Sell Stock

70. Climbing Stairs
  - f(result) = f(one step away) + f(two step away)
  - recursion: time O(2^n) iteration: time O(n)
  # Fibonacci, DP

*23. Merge k Sorted Lists
  - Merge sort for array, compare with 148(merge sort on list)
  - ListNode[] lists; lists.length // not lists.size()
  # Merge Sort, Divide and Conquer, Merge two lists

*42. Trapping Rain Water
  - compare left and right, move towards and keep lMax, rMax
  - main idea is level up
  - Math.max, Math.min
  -     int l=0, r=height.length-1, lMax=0, rMax = 0, res=0;
        while(l < r) {
            if(height[l] < height[r]) {
                lMax = Math.max(lMax, height[l]);
                res += lMax - height[l];
                l++;
            } else {
                rMax = Math.max(rMax, height[r]);
                res += rMax - height[r];
                r--;
            }
        }
  @ Time: O(n), Space: O(1)
  # Two Pointers

226. Invert Binary Tree
  # Post order traversal of Binary Tree

*15. 3Sum
  - be careful of IndexOutOfBoundsException for e.g. arr[i+1]
  - Arrays.sort(). Arrays.asList(T... a)
  - ArrayList rather than LinkedList
  -     Arrays.sort(nums);
        for(int i=0;i<nums.length-2;i++) {
            if(i>0 && nums[i] == nums[i-1]) {
                continue;
            }
            int sum = 0-nums[i];
            int l = i+1;
            int r = nums.length-1;
            while(l < r) {
                if(nums[l] + nums[r] == sum) {
                    list.add(Arrays.asList(nums[i], nums[l], nums[r]));
                    while(l<r && nums[l] == nums[l+1]) {l++;}
                    while(l<r && nums[r] == nums[r-1]) {r--;}
                    l++;
                    r--;
                } else if (nums[l] + nums[r] < sum){
                  l++;
                } else {
                    r--;
                }
            }
  @ Time: O(n^2), O(n)
  # Two Pointers, Sort, Edge Case

*198. House Robber
  - Rob(n) = Max(Rob(n-2)+ Money(n), Rob(n-1))
  - public int rob(int[] num) {
        int prevMax = 0;
        int currMax = 0;
        for (int x : num) {
            int temp = currMax;
            currMax = Math.max(prevMax + x, currMax);
            prevMax = temp;
        }
        return currMax;
    }
  - public int rob(int[] nums) {
        if(nums == null || nums.length == 0) {
            return 0;
        }
        if(nums.length == 1) {
            return nums[0];
        }

        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);

        for(int i = 2;i<nums.length;i++) {
            dp[i] = Math.max(dp[i-1], dp[i-2]+nums[i]);
        }

        return dp[nums.length-1];
    }
  @ Time: O(n), Space: O(1)
  # DP

*10. Regular Expression Matching
  - Run s ={x,a,a,b} p = {x,a,*,b}, dp[s.length()+1][p.length()+1]
    when s=p=null, dp[0][0]=true
      initialze dp[0][j] // when p is null
      for dp[s][p] = dp[i][j]
        if s.charAt[i-1] == p.charAt[j-1] || p.charAt[j-1] == '.'
          then dp[i][j] = dp[i-1][j-1]
        else if p.charAt(j-1) == '*'
          dp[i][j] = dp[i][j-2] // p[j-2] has no occurrence
          if p.charAt(j-2) == '.' || p.charAt(j-2) == s.charAt(i-1)
            dp[i][j] = dp[i][j] || dp[i-1][j] // p[j-2] has multiple occurrence
        else
          dp[i][j] = false
  - (https://www.youtube.com/watch?v=l3hda49XcDE&t=194s)
  # 2D DP

*44. Wildcard Matching
  - Similar to 10.
  -     for(int i=1;i<=s.length();i++) {
            for(int j=1;j<=p.length();j++) {
                if(s.charAt(i-1) == p.charAt(j-1) || p.charAt(j-1) == '?') {
                    dp[i][j] = dp[i-1][j-1];
                } else if(p.charAt(j-1) == '*') {
                    dp[i][j] = dp[i-1][j] || dp[i][j-1];
                }
            }
        }
    # 2D DP

*238. Product of Array Except Self
  - Corner case: 0 # 0, 1 # 0, >1 # 0
  - 左到右依次少乘本身，然后右到左乘以right（nums右到左product）
  - public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        res[0] = 1;
        for (int i = 1; i < n; i++) {
            res[i] = res[i - 1] * nums[i - 1];
        }
        int right = 1;
        for (int i = n - 1; i >= 0; i--) {
            res[i] *= right;
            right *= nums[i];
        }
        return res;
    }
  @ Time: O(n), Space: O(n)

17. Letter Combinations of a Phone Number
  - Collections.emptyList(), Collections.singletonList()
  - Arrays.asList(T...), String str.toCharArray()
  @ Time: O(n^3), Space: O(n)
  # Recursion, Backtracking

617. Merge Two Binary Trees
  -  public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if(t1 == null) {
            return t2;
        } else if (t2 == null) {
            return t1;
        }
        TreeNode t = new TreeNode(t1.val + t2.val);
        t.left = mergeTrees(t1.left, t2.left);
        t.right = mergeTrees(t1.right, t2.right);
        return t;
      }
  # Tree

*22. Generate Parentheses
  - num of left Parentheses always >= right
  # Backtracking

*56. Merge Intervals
  - List list.subList()
  - Collections.sort(List list, (a,b)->a-b), sort on list
  - Arrays.sort(), sort on array
  - Object operations are expensive e.g.sort, use primitive as much as possible
  - maintain start index and end index
  -   for (int i = 0; i < n; i++) {
        starts[i] = intervals.get(i).start;
        ends[i] = intervals.get(i).end;
      }
      Arrays.sort(starts);
      Arrays.sort(ends);
      // loop through
      List<Interval> res = new ArrayList<Interval>();
      for (int i = 0, j = 0; i < n; i++) { // j is start of interval.
        if (i == n - 1 || starts[i + 1] > ends[i]) {
          res.add(new Interval(starts[j], ends[i]));
          j = i + 1;
        }
      }
  # Array, Sort

338. Counting Bits
  - i&(i - 1)， 这个本来是用来判断一个数是否是2的指数的快捷方法，比如8，二进制位1000, 那么8&(8-1)为0，只要为0就是2的指数
  - 每个i值都是i&(i-1)对应的值加1
  # Bit Mnaipulation

200. Number of Islands
  - 2d array, matrix, connected component
  - avoid using collection or object to waste time
  # DFS/BFS, Graph, Connected Component

*49. Group Anagrams
  - String s.toCharArray()
  - Arrays.sort(s.toCharArray())
  - new String(s.toCharArray())
  - Collections.singletonList(~) return the abstract list
  - new LinkedList(Collection)
  # Counting Sort, Sort, Array, List

*139 Word Break
  - List list.contains()
  - String str.substring()
  - DP for boolean, not only int.
  -     boolean[] dp = new boolean[s.length()+1];
        dp[0]=true;
        for(int i=1;i<=s.length();i++) {
            for(int j=0;j<i;j++) {
                if(dp[j] && wordDict.contains(s.substring(j,i))) {
                    dp[i] = true;
                    break;}}}
  # 1D DP

** 140. Word Break II
  - // DFS, time out
    public void dfs(String s, int index, String sentence, Set<String> set, List<String> res) {
        if(index == s.length()) {
            res.add(sentence.trim());
        }
        /* Trade off here. Take approach 1
           1. Iterate rest of string, use set contains to check
           2. Iterate through word dictionary
        */
        for(int i=index;i<s.length();i++) {
            if(set.contains(s.substring(index, i+1))) {
                dfs(s, i+1, sentence + s.substring(index, i+1) + " ", set, res);
            }
        }
    }
  - // DP, time out
        for(int i=0;i<s.length();i++) {
            String str = s.substring(0, i+1);
            List<String> list = new ArrayList<>();

            // Take approach 2, iterate through word dictionary
            for(String word : wordDict) {
                if(word.equals(str)) {
                    list.add(word);
                    continue;
                }

                if(word.length() <= str.length() && str.endsWith(word) && map.containsKey(str.substring(0,str.length()-word.length()))){
                    List<String> l = map.get(str.substring(0,str.length()-word.length()));
                    for(String se : l) {
                        list.add(se + " " + word);
                    }

                }
            }

            map.put(str, list);
        }
    - // DFS + Memorization, bottom up. Map keeps String and its combinitions
    List<String> DFS(String s, Set<String> wordDict, HashMap<String, LinkedList<String>>map) {
        // Important memorization
        if (map.containsKey(s))
            return map.get(s);
        // Inlcude all possible solutions for String s
        LinkedList<String>res = new LinkedList<String>();
        if (s.length() == 0) {
            res.add("");
            return res;
        }
        for (String word : wordDict) {
            if (s.startsWith(word)) {
                List<String>sublist = DFS(s.substring(word.length()), wordDict, map);
                for (String sub : sublist)
                    res.add(word + (sub.isEmpty() ? "" : " ") + sub);
            }
        }
        map.put(s, res);
        return res;
    }

152 Maximum Product Subarray
  - keep track max, min, res

48. Rotate Image
  - int[][] matrix, int[] arr = matrix[]  // pass by reference, matrix change, arr change.
  # Array

234 Palindrome Linked List
  - reverse linked list
  -     ListNode pre = null;
        ListNode slow = head;
        ListNode fast = head.next;
        while(fast.next !=null && fast.next.next!=null) {
            fast = fast.next.next;
            ListNode temp = slow.next;
            slow.next = pre;
            pre = slow;
            slow = temp;
        }
        ListNode temp;
        if(fast.next == null) {
            temp = slow.next;
        } else {
            temp = slow.next.next;
        }
        slow.next = pre;
        return compare(temp, slow);
  # Two Pointers, Reverse LinkedList

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

*96 Unique Binary Search Trees
  - f(n) = f(0)*f(n-1) + f(1)*f(n-2) + ... + f(n-1)*f(0)
  -  1         3     3      2      1
      \       /     /      / \      \
       3     2     1      1   3      2
      /     /       \                 \
     2     1         2                 3
     n = 3
     root : 1   left : 0 right : 2   f(0) * f(2);
     root : 2   left : 1 right : 1   f(1) * f(1);
     root : 3   left : 2 right : 0   f(2) * f(0);
  -     dp[0] =1;
        for(int i=1;i<=n;i++) { // 1- n 个节点
            for(int j=0;j<i;j++) {  // 左节点数
                dp[i] += dp[j]*dp[i-j-1];
            }
        }
  # 1D DP

*95. Unique Binary Search Trees II
  - (https://leetcode.com/problems/unique-binary-search-trees-ii/description/)
  - Refer to 96 Unique Binary Search Trees
  - return genTreeList(1, n);
    public List<TreeNode> genTreeList(int start, int end) {
        List<TreeNode> list = new ArrayList<>();
        if (start > end) {
            list.add(null);
        }
        for (int idx = start; idx <= end; idx++) {
            List<TreeNode> leftList = genTreeList(start, idx - 1);
            List<TreeNode> rightList = genTreeList(idx + 1, end);
            for (TreeNode left : leftList) {
                for (TreeNode right : rightList) {
                    TreeNode root = new TreeNode(idx);
                    root.left = left;
                    root.right = right;
                    list.add(root);
                }
            }
        }
        return list;
    }
  # DFS, Tree, PostOrder Traversal

128 Longest Consecutive Sequence
  - keep go down, go up and get difference
  -     Set<Integer> set = new HashSet<>();
        for(int num:nums) set.add(num);
        int res = 0;
        for(int num : nums) {
            if(set.contains(num)) {
                int times = 1;
                int up = num+1;
                while(set.contains(up)) {
                    set.remove(up);
                    times++;
                    up++;
                }
                int down = num-1;
                while(set.contains(down)) {
                    set.remove(down);
                    times++;
                    down--;
                }
                res = Math.max(res, times);
            }
        }
  # HashSet

*347 Top K Frequent Elements
  - (https://leetcode.com/problems/top-k-frequent-elements/discuss/81635/3-Java-Solution-using-Array-MaxHeap-TreeMap)
  - map.put(key, map.getOrDefault(key, 0) + 1);
  - TreeMap, sorted by key, treeMap.pollLastEntry().getValue()
  - List<Integer>[] bucket = new List[nums.length+1];
  # Sort, TreeMap, Bucket Sort

142 Linked List Cycle II
  - slow = a + b, fast = a + b + c + b, 2 * slow = fast -> a = c
  -     while(fast.next != null && fast.next.next != null) {
            slow = slow.next;
            if(fast.next.next == slow) {
                ListNode pointer =head;
                while(pointer != slow) {
                    slow = slow.next;
                    pointer = pointer.next;
                }
                return pointer;
            }
            fast = fast.next.next;
        }
  # Two Pointers

*287 Find the Duplicate Number
  - binary search template
      public int findDuplicate(int[] nums) {
        int max = nums.length-1;
        int min = 1;
        while(min<max) {
            int mid = (min+max)/2, count = 0;
            for(int i = 0; i<nums.length;i++) {
                if(nums[i]<=mid) count++;
            }
            if (min<count) max = mid; else min = mid + 1;
        }
        return min; // attention
       }

  - HashSet: time O(n), space O(n)
    Sort: time O(nlogn), space O(1)
    Binary Search: time O(nlogn), space O(1)
  # Binary Search, LinkedList Cycle II

*406 Queue Reconstruction by Height
  - sort arr[a][b] first by decending a, then by increasing b. arr = int[][]
    Arrays.sort(arr, (x,y) -> x[0] == y[0]? x[1]-y[1] ? y[0]-x[0])
  - list to array: List.toArray(T[] a)
  - Time: O(n^2), Space: O(n)
  # 2D Array

448 Find All Numbers Disappeared in an Array
  - Use relation between array index ([0, n-1]) and the given value range [1,n]. Set value-1 to index and mark as negative. Then iterate through array and find positive values, which means that index is not visited.
  - Time O(n), Space O(n)
  # Value to Index

*279 Perfect Squares
  - for(int i=1;i<=n;i++) {
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

*218 The Skyline Problem
  - PriorityQueue<Integer> pq, pq.offer(number)
  - Collections.sort(list, (a,b)-> (a[0]==b[0]) ? a[1]-b[1] : a[0]-b[0]);
  - distinguish building start and end by making start as negative, sort first by start and then height in increasing order. Matain priority queue in decending order of height. Conner case would be [[0,2,3],[2,5,3]]
  # Priority Queue, TreeMap, Sweep Line

*240 Search a 2D Matrix II
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

*101 Symmetric Tree
  - In reursion, check treeNode1, treeNode2 null, compare treeNode.val
  - predorder check
  - private boolean compare(TreeNode n1, TreeNode n2) {
        if(n1 == null && n2 != null) return false;
        if(n1 != null && n2 == null) return false;
        if(n1 == null && n2 == null) return true;
        if(n1.val != n2.val) return false;

        return compare(n1.left, n2.right) && compare(n1.right, n2.left);

    }
  # Inorder Traversal, Binary Tree

124 Binary Tree Maximum Path Sum
  - think of preorder, or postorder
  - Integer.MIN_VALUE
  - postorder, calculate res = Math.max(res, left+right+node.val), return Math.max(left, right) + node.val
  # PostOrder Traversal, Binary Tree

*236 Lowest Common Ancestor of a Binary Tree
  - find first parent that includes both nodes.
  -  TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if(left != null && right != null) {
            return root;
        }
        return left != null ? left : right;
    }
    # Postorder Traversal, Binary Tree

*76. Minimum Window Substring
  - Use int array to record num of each char. faster pointer runs and decrease num of each char until total becomes 0. Then slow pointer runs and increase num of each char until total becomes 1. Then get i-j+1 compare with the min distance.

        char[] tArr = t.toCharArray();
        char[] sArr = s.toCharArray();
        int[] arr = new int[256];
        for(char c : tArr) {
            arr[c-'A']++;
        }

        int total = tArr.length;
        int sp = 0;
        String res = "";
        for(int i =0;i< sArr.length;i++) {
            if(arr[sArr[i] - 'A']-- > 0) total--;
            if(total == 0) {
                while(++arr[sArr[sp] - 'A'] <= 0) {
                    sp++;
                }
                if(res == "" || i-sp+1 < res.length()) {
                    res = s.substring(sp,i+1);
                }
                sp++;
                total++;
            }
        }

  # Sliding Window, Two Pointers, HashTable

*105 Construct Binary Tree from Preorder and Inorder Traversal
  - preorder is easy to find root, then find index of root from inorder.
    public TreeNode helper(int preStart, int inStart, int inEnd, int[] preorder, int[] inorder) {
        if (preStart > preorder.length - 1 || inStart > inEnd) return null;
        TreeNode root = new TreeNode(preorder[preStart]);
        // find inIndex
        for (int i = inStart; i <= inEnd; i++) {
          if (inorder[i] == root.val) {
              inIndex = i;
            }
        }
        root.left = helper(preStart + 1, inStart, inIndex - 1, preorder, inorder);
        root.right = helper(preStart + inIndex - inStart + 1, inIndex + 1, inEnd, preorder, inorder);
  # Tree

*32 Longest Valid Parentheses
  - push index into stack
  - cant use sum map method in array cause e.g. )(.
  -   for(int i=0;i<s.length();i++) {
        if(s.charAt(i) == '(') {
            stack.push(i);
        } else {
          if(stack.empty()) {
              start = i;
          } else {
              stack.pop();
              res = stack.empty() ? Math.max(res, i-start) : Math.max(res, i-stack.peek());}}}

  # Stack,

78 Subsets
  - Use startIndex to avoid duplicate
  public List<List<Integer>> subsets(int[] nums) {
      List<List<Integer>> list = new ArrayList<>();
      backtrack(list, new ArrayList<>(), nums, 0);
      return list;
  }

  private void backtrack(List<List<Integer>> list , List<Integer> tempList, int [] nums, int start){
      list.add(new ArrayList<>(tempList));
      for(int i = start; i < nums.length; i++){
          tempList.add(nums[i]);
          backtrack(list, tempList, nums, i + 1);
          tempList.remove(tempList.size() - 1);
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
              // ensure no duplicates
                if(s.charAt(j) == pair[1] && (j == slow || s.charAt(j-1) != pair[1])) {
                    construct(res, s.substring(0, j) + s.substring(j+1), i, j, pair);
                }
            }
            return;
        }

        // Check '(' more than ')'
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
  -     // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        traverse(root, sb);
        return sb.toString();
    }

    private void traverse(TreeNode node, StringBuilder sb) {
        if(node == null) {
            sb.append("null,");
            return;
        }
        sb.append(node.val + ",");
        traverse(node.left, sb);
        traverse(node.right, sb);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Deque<String> queue = new LinkedList<>(Arrays.asList(data.split(",")));
        return build(queue);
    }

    private TreeNode build(Deque<String> queue) {
        String val = queue.remove();
        if(val.equals("null")) return null;
        TreeNode node = new TreeNode(Integer.parseInt(val));
        node.left = build(queue);
        node.right = build(queue);
        return node;
    }
  # BFS, Preorder Traversal

55 Jump Game
  - Keep track of max (nums[i] + i), compare with i, if i>max, then return false.
  - BFS exceeds time limit, DFS exceeds memory limit
  - time O(n), space o(1)
  # Greedy

*215 Kth Largest Element in an Array
  - PriorityQueue is minHeap, poll() removes the least value. Each op takes logk
  - Priority Queue keeps track of k elements. time O(nlogk), space O(n)
  - Quick Select, select pivot, compare and return left index, if left index = k, then return. (http://www.geekviewpoint.com/java/search/quickselect)
  -   private int quickSelect(int[] nums, int k, int left, int right) {
          int index = partition(nums, left, right);
          if(index == k) {
              return nums[index];
          } else if(index > k) {
              return quickSelect(nums, k, left, index-1);
          } else {
              return quickSelect(nums, k, index+1, right);
          }
      }
      private int partition(int[] nums, int left, int right) {
          int pivot = nums[right];
          int index = left;
          for(int i = left;i<right;i++) {
              if(nums[i]>=pivot) {
                  int temp = nums[index];
                  nums[index] = nums[i];
                  nums[i] = temp;
                  index++;
              }
          }
          int temp = nums[index];
          nums[index] = nums[right];
          nums[right] = temp;
          return index;
      }
  - Quick Select time average O(n), wrost O(n^2)
  # Quick Select, Priority Queue

102 Binary Tree Level Order Traversal
  - time O(n), space O(n)
  -     int num =1;
        int temp = 0;
        while(num != 0) {
            List<Integer> list = new ArrayList<>();
            for(int i=0;i<num;i++) {
                TreeNode node = queue.poll();
                if(node.left != null) {
                    queue.offer(node.left);
                    temp++;
                }
                if(node.right != null) {
                   queue.offer(node.right);
                    temp++;
                }
                list.add(node.val);
            }
            num = temp;
            temp = 0;
            res.add(list);
        }
  # Preorder Traversal, Queue

*647 Palindromic Substrings
  - interate string, for each iteration, left-- and right++ to find palindrome until s.charAt(left) != s.charAt(right).
  -   for(int i=0;i<s.length();i++) {
          count(s, i, i);
          count(s, i, i+1);
      }
      private void count(String s, int l, int r) {
          while(l>=0 && r<s.length() && s.charAt(l--) == s.charAt(r++)) {
              res++;
          }
      }
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

621 Task Scheduler
  - 1. interval is big enough e.g. n = 1, [A,A,A,B,B,B]
    2. interval is not enough e.g. n = 1, [A,A,A,B,B,B,C,C,C]
  -     public int leastInterval(char[] tasks, int n) {
        int[] arr = new int[26]; int max = 0;
        for(char c : tasks) {
            arr[c-'A']++;
            max = Math.max(max, arr[c-'A']);
        }
        int res = (max-1) * (n+1);
        for(int i:arr) if(i == max) res++;
        // corner case for tasks.length: n = 0, [A,A,A,B,B,B]
        return Math.max(res, tasks.length);}

*84. Largest Rectangle in Histogram
  - maintain increasing order index in stack, for new height, pop up all height that heigher and calculate
  - for(int i =0;i<=heights.length;i++) {
        // deal with corner case: [1,3,4,5]
        int h = (i == heights.length) ? 0 : heights[i];
        while(!stack.isEmpty() && h<heights[stack.peek()]) {
            // corner case: [0,2,1,2], [2,1,2]
            int height = heights[stack.pop()];
            int start = (stack.isEmpty()) ? -1 : stack.peek();
            int area = height * (i-start-1);
            res = Math.max(res, area);
        }
        stack.push(i);
    }
  - time O(n), space O(n)
  # Stack

*315 Count of Smaller Numbers After Self
  - TreeMap.lowerKey() // predecessor, TreeMap.higherKey // successor
  - Arrays.asList(Object only)
  -     private int insert(List<Integer> list, int num) {
        int r = list.size()-1; int l = 0; int mid;
        while(l <= r) {
            mid = (r + l)/2;
            if(list.get(mid)<num) {
                l = mid+1;
            } else {
                r = mid-1;
            }
        }
        // add makes O(n) rather than O(logn) in worst case
        list.add(l, num);
        return l;
    }
  - binary index tree:
    Refer to (https://leetcode.com/problems/count-of-smaller-numbers-after-self/discuss/76611/Short-Java-Binary-Index-Tree-BEAT-97.33-With-Detailed-Explanation)
  - Two ways in Fenwick Tree
    1. Sort array to get rank array, then interate from right most, for each val, get rank, then calculate sum of low rank num and update tree.

    2. find min and max in arr, create size of (max - min) fenwick tree array and size of arr.length diff (arr[i]-min+1) arr. Then iterate from right to left to get sum and update.

        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for(int num : nums) min = Math.min(min, num);

        int[] diffArr = new int[nums.length];
        for(int i=0;i<nums.length;i++) {
            diffArr[i] = nums[i]-min;
            max = Math.max(max, diffArr[i]);
        }

        // fenwickTree has a dummy root at fenwickTree[0]
        int[] fenwickTree = new int[max+1];
        for(int i= nums.length-1;i>=0;i--) {
            // find all ranks < diffArr[i]
            list.add(0, getSum(fenwickTree, diffArr[i]));
            update(fenwickTree, diffArr[i]+1, 1);
        }

    private int getSum(int[] fenwickTree, int index) {
       int sum = 0;
        while(index>0) {
            sum += fenwickTree[index];
            index -= index & (-index);
        }
        return sum;
    }

    private void update(int[] fenwickTree, int index, int diff) {
        while(index < fenwickTree.length) {
            fenwickTree[index] += diff;
            index += index & (-index);
        }
    }

  # Binary Search, Binary Indexed Tree/Fenwick Tree

*34. Search for a Range
   //find first
   private int findFirst(int[] nums, int target) {
        int l = 0; int r = nums.length-1;
        while(l < r-1) {
            int mid = (l+r)/2;
            if(nums[mid] < target) {l = mid;} else {r = mid;}
        }
        if(nums[l] == target) return l;
        if(nums[r] == target) return r;
        return -1;
    }
    //find last
    private int findLast(int[] nums, int target) {
        int l = 0; int r = nums.length-1;
        while(l < r-1) {
            int mid = (l+r)/2;
            if(nums[mid] <= target) {l = mid;} else {r = mid;}
        }
        if(nums[r] == target) return r;
        if(nums[l] == target) return l;
        return -1;
    }
  # Binary Search

*309. Best Time to Buy and Sell Stock with Cooldown
    public int maxProfit(int[] prices) {
        int n = prices.length;
        if(n < 2) return 0;

        int[] buy = new int[n + 1];
        int[] sell = new int[n + 1];
        buy[1] = -prices[0];
        for(int i = 2; i <= n; i++){
            buy[i] = Math.max(buy[i - 1], sell[i - 2] - prices[i - 1]);
            sell[i] = Math.max(sell[i - 1], buy[i - 1] + prices[i - 1]);
        }
        return sell[n];
    }
  # DP

*714. Best Time to Buy and Sell Stock with Transaction Fee
  - 状态转移， Refer to 309. Best Time to Buy and Sell Stock with Cooldown
  - buy[i] = Math.max(buy[i - 1], sell[i - 1] - prices[i]-fee);
    sell[i] = Math.max(sell[i - 1], buy[i - 1] + prices[i]);
    (https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/discuss/108871/2-solutions-2-states-DP-solutions-clear-explanation!)
  -     int days = prices.length;
        int[] buy = new int[days]; // current and last status on i is buy
        int[] sell = new int[days];
        buy[0] = -prices[0]-fee;
        sell[0] = 0;
        for(int i=1;i<days;i++) {
            buy[i] = Math.max(buy[i-1], sell[i-1]-prices[i]-fee);
            sell[i] = Math.max(sell[i-1], buy[i-1]+prices[i]);
        }
        return sell[days-1];
  # DP

122. Best Time to Buy and Sell Stock II
  - for (int i = 1; i < prices.length; i++) {
        if (prices[i] > prices[i - 1]) {
            profit += prices[i] - prices[i - 1];
        }
    }
  # Two Pointers, Greedy

*123. Best Time to Buy and Sell Stock III
  -     int buy1 = Integer.MIN_VALUE;int sell1 = 0;
        int buy2 = Integer.MIN_VALUE;int sell2 = 0;
        for(int price: prices) {
            sell2 = Math.max(sell2, buy2+price);
            buy2 = Math.max(buy2, sell1-price);
            sell1 = Math.max(sell1, buy1+price);
            buy1 = Math.max(buy1, -price);
        }
  # DP

*114. Flatten Binary Tree to Linked List
  - create dummy root for return purpose, cause root is changing. iterate right then left.
  -     TreeNode temp = null;
        public void flatten(TreeNode root) {
            if(root == null) return;
            flatten(root.right);
            flatten(root.left);
            root.right = temp;
            root.left = null;
            temp = root;
        }
  # PostOrder, Tree

*239. Sliding Window Maximum
  - maintain decending order queue, when push new element, remove elements that less than the new element. pop() pops the first element which is also the largest element. Amotized time O(1)
  -     int[] res = new int[n-k+1];
        // Use queue rather than stack
        // Store index into queue rather than value
        Deque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < a.length; i++) {
          // remove numbers out of range k
          while (!q.isEmpty() && q.peek() < i - k + 1) {
            q.poll();
          }
          // remove smaller numbers in k range as they are useless
          while (!q.isEmpty() && a[q.peekLast()] < a[i]) {
            q.pollLast();
          }
          // q contains index... r contains content
          q.offer(i);
          if (i >= k - 1) {
            r[ri++] = a[q.peek()];
          }
  - Priority Queue time O(nlogk), space O(n)
  - Deque time amotized O(n), space O(n)
  - Using queue instead of stack to operate on both side
  # Monotonic Queue / Deque,

322. Coin Change
   -   int[] dp = new int[amount+1];
        for(int i=1;i<=amount;i++) {
            int min = Integer.MAX_VALUE;
            for(int coin : coins) {
                if(i >=coin && dp[i-coin] != -1){
                    min = Math.min(min, dp[i-coin] + 1);
                }
            }
            dp[i] = min == Integer.MAX_VALUE ? -1 : min;}
  - Refer to Word Break and Perfect Squares
  # 1D DP

*94. Binary Tree Inorder Traversal
  - maintain current node and stack
  -     // Iterative
        TreeNode cur = root;
        while(cur!=null || !stack.isEmpty()) {
            while(cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            res.add(cur.val);
            cur = cur.right;
        }
  # Inorder Traversal, Stack

*394. Decode String
  - Stack<> stack = new Stack<>() // Stack<> s = new LinkedList<>() not right
  - Character.isDigit() // check '0' - '9'
  -     //create 2 stacks, one to save number, one to save res.
        while(index < s.length()) {
            // push count
            if(Character.isDigit(s.charAt(index))) {
                int count=0;
                while(Character.isDigit(s.charAt(index))) {
                    count = 10*count + s.charAt(index++) - '0';
                }
                istack.push(count);
            // push res
            } else if(s.charAt(index) == '[') {
                sstack.push(res);
                res = "";
                index++;
            // pop res and count, then append
            } else if(s.charAt(index) == ']') {
                StringBuilder sb = new StringBuilder(sstack.pop());
                int count = istack.pop();
                for(int i=0;i<count;i++) {
                    sb.append(res);
                }
                res = sb.toString();
                index++;
            } else {
                res += s.charAt(index++);
            }
        }
  # Stack

560. Subarray Sum Equals K
  - Map does not have getOrDefault, use HashMap
  - matain sum, hashmap of continuous array sum from index 0 and frequence. Iterate to arr.length, for each sum, find frequence by key = sum - k and add to result:
        map.put(0,1);
        for(int i : nums) {
            sum +=i;
            // sum-k rather than k-sum
            if(map.containsKey(sum-k)) {
                res += map.get(sum-k);
            }
            map.put(sum, map.getOrDefault(sum,0)+1);
        }

  - Time O(n), Space O(n)
  # HashMap, 2Sum

543. Diameter of Binary Tree
  - Similar to 124 Binary Tree Maximum Path Sum
  - private int traverse(TreeNode node) {
        if(node == null) return 0;
        int left = traverse(node.left);
        int right = traverse(node.right);
        res = Math.max(res, left + right);
        return Math.max(left, right) + 1;
    }
  # Tree, Postorder Traversal

*300. Longest Increasing Subsequence
  - Maintain increasing order arr. For new element, find first elem in arr that larger or equal to new elem, then replace it.
  -
            int[] subsequence  = new int[nums.length];
            int numOfSubsequence = 0;
            for(int n : nums) {
                int l = 0; int r = numOfSubsequence;
                while(l < r) {
                    int mid = (l+r)/2;
                    if(subsequence[mid] < n) { // find 1st mid >= n
                      l=mid+1;
                    } else {
                      r= mid;
                    }
                }
                subsequence[l] = n;
                if(l == numOfSubsequence) numOfSubsequence++;
            }

  - 不能用stack，e.g. [1,3,6,7,9,4,10,5,6], 5 but expect 6
  - time O(nlogn), space O(n)
  # Binary Search, patience sort

*337. House Robber III
    public int rob(TreeNode root) {
        if(root == null) return 0;
        int val = 0;
        if(root.left != null)
            val+= rob(root.left.left) + rob(root.left.right);
        if(root.right !=null)
            val+= rob(root.right.left) + rob(root.right.right);
        return Math.max(val + root.val, rob(root.left)+rob(root.right));}
  # Tree, DP

*221. Maximal Square
  - Due to i-1, j-1, new dp[rows+1][cols+1] and i,j start from 1 and i<=rows, matrix[i-1][j-1]

    for(int i=1;i<=rows;i++) {
        for(int j =1;j<=cols;j++) {
            if(matrix[i-1][j-1] == '1') {
                // ensure 正方形而不是长方型，res是最长边长
                dp[i][j] = Math.min(Math.min(dp[i-1][j], dp[i][j-1]), dp[i-1][j-1]) + 1;
                res = Math.max(res, dp[i][j]);}}}
  # 2D DP

207. Course Schedule
  - In directed graph, if there is back edge, then acyclic. Maintain recursion queue/array, if the next visit node is in the recursion queue, its a back edge.
  - Create adjacent list first, then dfs.
  - Topological sort: dfs and if node finish visiting all its adjacent nodes, then push it into stack. Finally pop all nodes from stack.
  -     for(int i=0;i<numCourses;i++) {
            if(isVisited[i]) continue;
            if (!dfs(adj, i, isVisited, isAccessed)) return false;
        }
        private boolean dfs(LinkedList<Integer>[] adj, int i, boolean[] isVisited, boolean[] isAccessed) {
            isVisited[i] = true;
            isAccessed[i] = true;
            for(int j=0;j<adj[i].size();j++) {
                int k = adj[i].get(j);
                if(isAccessed[k]) return false;
                if(!isVisited[k] && !dfs(adj, k, isVisited, isAccessed)) {
                    return false;
                }
            }
            isAccessed[i] = false;
            return true;}

  # DFS, Topological Sort, Graph, Adjacent List, Cycle Detection, Back Edge

**437. Path Sum III
  - path can start not from root.
  -     public int pathSum(TreeNode root, int sum) {
            if(root == null) return 0;
            return dfs(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum); // 吊了，make each child node as root of subtree
        }
        public int dfs(TreeNode node, int sum) {
            int res = 0;
            if (node == null) return res;
            if (sum == node.val) res++; // not return
            res += dfs(node.left, sum - node.val) + dfs(node.right, sum - node.val);
            return res;
        }
  # PreOrder on sum, PostOrder on res, Tree, DFS

98. Validate Binary Search Tree
  - inorder traversal and the new visited node should larger than the previous ndoe. Maintain pre node val.
  - first assign pre = Integer.MIN_VALUE, test case has node.val = Integer.MIN_VALUE. Be careful.
  # Tree, Inorder Traversal, Stack

538. Convert BST to Greater Tree
  - Traverse from right to left
  # Tree

438. Find All Anagrams in a String
  - matain num of characters in String, it num ==0, put into result. Fast pointer and slow pointer = fp - fixedWindowSize
  -     int[] arr = new int[26];
        for(char c : p.toCharArray()) {arr[c-'a']++;}
        int total = p.length();
        for(int i=0;i<s.length();i++) {
            if(arr[s.charAt(i) - 'a']-->0) total--;
            if(i >= p.length() && ++arr[s.charAt(i - p.length()) - 'a']>0) total++;
            if(total == 0) res.add(i-p.length()+1);
        }

  - refer to 76 Minimum Window Substring
  # Two Pointers

572. Subtree of Another Tree
  - Check subtree recursively
    public boolean isSubtree(TreeNode s, TreeNode t) {
        if(s == null && t == null) return true;
        if(s == null || t == null) return false;
        if(compare(s,t)) {
            return true;
        } else {
            return isSubtree(s.left, t) || isSubtree(s.right,t);
        }
    }
    private boolean compare(TreeNode s, TreeNode t) {
        if(s == null && t == null) return true;
        if(s == null || t == null) return false;
        if(s.val != t.val) return false;
        return compare(s.left, t.left) && compare(s.right, t.right);
    }
  # Tree, Subtree

*581. Shortest Unsorted Continuous Subarray
  - Stack<Integer> stack; stack.clear();
  - Matain increasing order in stack, for new elem, find the index and compare with minLeft.
  -     for(int i=0;i<nums.length;i++) {
            while(!stack.isEmpty() && nums[i] < nums[stack.peek()]) {
                minLeft = Math.min(minLeft,stack.pop());
            }
            stack.push(i);
        }
  - Refer to Sliding Window Maximum monotonic queue which has 两头. Stack 只有1头
  - Refer to 769
  # Monotonic, Stack

*494. Target Sum
  -  Given nums, +/- each elem to get target S
      public int findTargetSumWays(int[] nums, int S) {
        int sum = 0;
        for(int num : nums) {
            sum +=num;
        }
        if(sum < S) return 0;
        //dp[array elem][sum range] = 次数
        int[][] dp = new int[nums.length+1][2*sum+1];
        dp[0][sum] = 1; // Initial 0 input, sum = 1
        for(int i=0;i<nums.length;i++) {
            // range is [0, 2sum+1], new elem is nums[i], then 0<=j-nums[i]<=2sum+1, then nums[i]<=j<=2sum+1-nums[i]
            for(int j=nums[i];j<2*sum+1-nums[i];j++) {
                dp[i+1][j+nums[i]] += dp[i][j];
                dp[i+1][j-nums[i]] += dp[i][j];
            }
        }
        return dp[nums.length][S+sum];
  # 2D DP

*273. Integer to English Words
  -     while (num > 0) {
            if (num % 1000 != 0) {
                res = helper(num % 1000) + thousands[i] + " " + res;
            }
            num /= 1000;
            i++;
        }
        return res.trim();

        public String helper(int num) {
            if (num == 0) return "";
            if (num < 20) {
                return less20[num % 20] + " ";
            } else if (num < 100) {
                return tens[num / 10] + " " + helper(num % 10);
            } else {
                return less20[num / 100] + " Hundred " + helper(num % 100);
            }}
  # Recursion

67. Add Binary
  # String

253. Meeting Rooms II
  - Refer to The Skyline Problem
  - Sort by time in increasing order, start marks 1, end marks -1. end executes prior to start if time is the same.
  -     for(Interval interval : intervals) {
            list.add(new int[]{interval.start, 1});
            list.add(new int[]{interval.end,-1});
        }
        Collections.sort(list, (a,b)->a[0] == b[0] ? a[1]-b[1] : a[0]-b[0]);
        int curr = 0;int res = 0;
        for(int[] t:list) {
            curr += t[1];
            res = Math.max(res, curr);
        }
  # Sweep Line

325. Maximum Size Subarray Sum Equals k
  - Matain hashmap of sum and length, for each iteration, check if sum-k exist in hashmap, if yes, compare res and find the larger one. Then put sum to hashMap.
  - Refer to Subarray Sum Equals K
  # HashMap

*91. Decode Ways: decode digit string(e.g. "123") to A-Z
  - each iteration, cur digit!=0, dp[i] = dp[i-1], 9<pre digit + cur digit<27, dp[i] != dp[i-2]
  - 好好区分下dp中index和string中index关系
    public int numDecodings(String s) {
        int[] dp = new int[s.length()+1];
        dp[0] = 1;
        for(int i=1;i<=s.length();i++) {
            if(s.charAt(i-1) != '0') {
                dp[i] = dp[i-1];
            }
            if(i-2>=0 && isDoubleDigits(s, i-2, i)) {
                dp[i] += dp[i-2];
            }
        }

        return dp[s.length()];
    }

    private boolean isDoubleDigits(String s, int start, int end) {
        int n =Integer.parseInt(s.substring(start, end));
        return n>9 && n<27;
    }
        }
  - // 空间复杂度优化
        int c1 = 1;
        int c2 = 1;
        for(int i=1;i<s.length();i++) {
            if (s.charAt(i) == '0') {
                c1 = 0;
            }
            if (s.charAt(i - 1) == '1' || s.charAt(i - 1) == '2' && s.charAt(i) <= '6') {
                c1 = c1 + c2;
                c2 = c1 - c2;
            } else {
                c2 = c1;
            }
        }
        return c1;
  # 1D DP

314. Binary Tree Vertical Order Traversal
  - Use BFS rather than tree traversal due to case e.g.
          1
         2 3
        4 5
           6  should be [[4], [2], [1,5], [3,6]], if tree traversal [...[6,3]]
  - find min and max, left -1, right +1. BFS.
  - create second queue to track index
  - TreeMap map.values()
  -     Queue<TreeNode> queue = new LinkedList<>();
        Queue<Integer> iQueue = new LinkedList<>();
        queue.add(root);
        iQueue.add(0);
        while(!queue.isEmpty()) {
            TreeNode node = queue.poll();
            Integer index = iQueue.poll();

            if(map.containsKey(index)) {
                map.get(index).add(node.val);
            } else {
                List<Integer> list = new LinkedList<>();
                list.add(node.val);
                map.put(index, list);
            }
            if(node.left != null) {
                queue.add(node.left);
                iQueue.add(index-1);
            }
            if(node.right !=null) {
                queue.add(node.right);
                iQueue.add(index+1);
            }
        }
        List<List<Integer>> res = new LinkedList<>();
        map.values().forEach(res::add);
        return res;
  # Tree, BFS

*157. Read N Characters Given Read4
    // read4 read at most 4 into temp, then copy from temp to buf up to n.
    // cases: 1. abcdef n =5  2.abcdef n=10
    public int read(char[] buf, int n) {
        char[] temp = new char[4];
        int index = 0 ;
        while(true) {
            int count = read4(temp);
            count = Math.min(count, n-index);
            for(int i=0;i<count;i++) {
                buf[index++] = temp[i];
            }
            if(index == n || count<4) return index;
        }
    }
    # String, buffer

*158. Read N Characters Given Read4 II - Call multiple times
  -     int count = 0; // # of elements in temp arr
        int pointer = 0; // pointer in temp
        char[] temp = new char[4];

        public int read(char[] buf, int n) {
            int index = 0;
            while(index < n) {
                if(pointer == 0) count = read4(temp);
                if(count == 0) break;
                while(index <n && pointer < count) {
                    buf[index++] = temp[pointer++];
                }
                if(pointer == count) pointer = 0;
            }
            return index;
        }
    # String, Buffer

311. Sparse Matrix Multiplication
  - Refer to (https://leetcode.com/problems/sparse-matrix-multiplication/discuss/76151/54ms-Detailed-Summary-of-Easiest-JAVA-solutions-Beating-99.9)
  -     int rowA = A.length;
        int colA = A[0].length;
        int rowB = B.length;
        int colB = B[0].length;
        int[][] res = new int[rowA][colB];

        for(int i=0;i<rowA;i++) {
            for(int z=0;z<rowB;z++) {
                if(A[i][z] != 0) {
                    for(int j=0;j<colB;j++) {
                        if(B[z][j] !=0) {
                            res[i][j] += A[i][z]*B[z][j];
                        }
                    }
                }

            }
        }
  # Math

*278. First Bad Version
  -     int l = 1;
        int r = n;
        while(l<r-1) {
            int mid = (r-l)/2+l; // dont use (l+r)/2
            if(isBadVersion(mid)) {
                r = mid;
            } else {
                l = mid;
            }
        }
        if(isBadVersion(l)) {return l;} else {return r;}
  # Binary Search

277. Find the Celebrity
  - knows(a,b): true, a is not celebrity; false, b is not celebrity.
  - keep track of celebrity by compare 2 people, finally check if celebrity knows anyone or anyone dont know celebrity
  -     for(int i=1;i<n;i++) {
            celebrity = knows(celebrity, i) ? i : celebrity;
        }
        for(int i=0;i<n;i++) {
            if(celebrity == i) continue;
            if(knows(celebrity, i) || !knows(i,celebrity)) return -1;}
  # Array

257. Binary Tree Paths
  - Integer.toString(int) // int to String

*173. Binary Search Tree Iterator
  - Keep track of current node and stack
  - Refer to 94. Binary Tree Inorder Traversal
  -     public BSTIterator(TreeNode root) {
            cur = root;
            while(cur != null) {
                stack.push(cur);
                cur = cur.left;}}

        public boolean hasNext() {
            return !stack.empty();}

        public int next() {
            if(!hasNext()) return -1;
            cur = stack.pop();
            int res = cur.val;
            cur = cur.right;
            while(cur != null) {
                stack.push(cur);
                cur = cur.left;
            }
            return res;}

  # Stack Implementation of Inorder Traversal

125. Valid Palindrome
  - Ascii: A~Z[\]^_'a~z'
  - Character.isLetterOrDigit()
  - Character.toLowerCase()
  # Two Pointers

680. Valid Palindrome II
  - delete at most one character. Judge if non-empty str can make palindrome.
  - conner case:
  if(s.charAt(l+1) == s.charAt(r) && s.charAt(r-1) == s.charAt(l)) return validate(s,l+2,r-1,true) || validate(s,l+1,r-2,true);
  # Two Pointers

*636. Exclusive Time of Functions
  - matain previous time, for new start or end, calculate curTime-preTime for the id on top of the stack.
  -   for(String logStr : logs){
            String[] log = logStr.split(":");
            int id = Integer.parseInt(log[0]), curTime = Integer.parseInt(log[2]);
            if(log[1].equals("start")){
                if(!stack.isEmpty()) res[stack.peek()] += curTime-prevTime;
                stack.push(id);
                prevTime = curTime;
            } else {
                curTime++;
                res[stack.pop()] += curTime-prevTime;
                prevTime = curTime;
            }
        }
  # Stack

* 689. Maximum Sum of 3 Non-Overlapping Subarrays
  - calculate total sum for each index.
    for (int i = 0; i < n; i++) sum[i+1] = sum[i]+nums[i];
  - 从左到右，posLeft每个记录当前最大的k sum的起始index。从右到左，posRight每个记录当前最大的k sum的起始index。从k到n-2k，循环计算total取最大
    posLeft = new int[n], tracks start index of largest sum of k elements from left to right
    posRight = new int[n], tracks start index of largest sum of k elements from right to left.
  - mid interval is [i, i+k-1], where k<=i<=n-2k, left interval [0,i-1], right interval [i+k, n-1]
  -
          for (int i = k; i <= n-2*k; i++) {
            int l = posLeft[i-1], r = posRight[i+k];
            int tot = (sum[i+k]-sum[i]) + (sum[l+k]-sum[l]) + (sum[r+k]-sum[r])
            if (tot > maxsum) {
                maxsum = tot;
                ans = {l, i, r};
            }
        }
  # Array, DP

252. Meeting Rooms
  - start time arr, end time arr. sort. For each iteration, start time should less than or equals to end time. start time should larger than previous end time.
  # Array, Sort

211. Add and Search Word - Data structure design
  - Create trie of words. Use recursion + dfs for '.' case.

  - public boolean find(WordDictionary node, String word) {
        if(word.length() == 0) {
            if(node.isWord) return true;
            return false;
        }
        if(word.charAt(0) == '.') {
            for(int i=0;i<26;i++) {
                if(node.children[i] !=null && find(node.children[i], word.substring(1))) {
                    return true;}}
        } else {
            if(node.children[word.charAt(0) - 'a'] != null) {
                return find(node.children[word.charAt(0) - 'a'], word.substring(1));}}

        return false;
    }
  # Trie, Recursion, DFS

**282. Expression Add Operators
  - Refer to (https://www.youtube.com/watch?v=v05R1OIIg08)
    public void dfs(String num, int target, int pos, String exp, long pre, long cur){
        if(pos == num.length()) {
            if(cur == target) {
                res.add(exp);
            }
            return;
        }

        for(int i = pos+1;i<=num.length();i++) {
            String temp = num.substring(pos, i);
            // 0x not allowed
            if(temp.charAt(0) == '0' && temp.length()>1) break;
            long n = Long.parseLong(temp);
            if(pos == 0) {
                dfs(num, target, i, temp, n, n);
                continue;
            }
            dfs(num, target, i, exp+'+'+temp, n, cur+n);
            dfs(num, target, i, exp+'-'+temp, -n, cur-n);
            dfs(num, target, i, exp+'*'+temp, pre*n, cur-pre+pre*n);
        }
    }
  # DP

*133. Clone Graph
  - undirected graph which includes self cycle. Maintain Map<Integer, Node>,
  - private void traverse(Node node, Map<Integer, Node> map) {
        Node root = new Node(node.label);
        map.put(root.label, root);
        for(Node n : node.neighbors) {
            // Check if visited
            if(!map.containsKey(n.label)) {
                traverse(n, map);
            }
            root.neighbors.add(map.get(n.label));
        }}
  # DFS, Undirected Graph

161. One Edit Distance
  - conner case "a", "ab"
  - String

*341. Flatten Nested List Iterator
  - For the given nestedList, use dfs to find leaf and add to queue.
  # DFS

597. Friend Requests I: Overall Acceptance Rate
  - count distinct rows from request_accepted divided by count distinct rows from friend_request. Attension friend_request can be 0.
  round(
    ifnull(
    (select count(*) from (select distinct requester_id, accepter_id from request_accepted) as A)
    /
    (select count(*) from (select distinct sender_id, send_to_id from friend_request) as B),
    0)
, 2) as accept_rate;
  # SQL

*43. Multiply Strings
    -   index:   0 1 2
        --------------
           m:    1 2 3
           n:      5 6
                ------
                   1 8
                 1 2
               0 6
                 1 5
               1 0
             0 5
             ----------
             0 6 8 8 8
  - The final result has at most m+n digits. Product digit by digit and set result to pHigh and pLow. Attention to carry. Calculate from right to left.
  -     int[] res = new int[num1.length() + num2.length()];
        for(int i = num1.length() - 1; i >= 0; i--) {
            for(int j = num2.length() - 1; j >= 0; j--) {
                int product = (num1.charAt(i)-'0') * (num2.charAt(j)-'0');
                int pHigh = i+j;
                int pLow = i+j+1;
                int sum = product + res[pLow];
                res[pHigh] += sum/10;
                res[pLow] = sum%10;
            }
        }
        StringBuilder builder = new StringBuilder();
        for(int i=0;i<res.length;i++) {
            if(res[i]==0 && builder.length() == 0) continue;
            builder.append(res[i]);
        }
  # String, Math

*285. Inorder Successor in BST
  - O(n) inorder traversal, O(logn) see follwoing code
  - public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        TreeNode res = null;
        while(root != null) {
            if(root.val > p.val) {
                res = root;
                root = root.left;
            } else {
                root = root.right;
            }
        }
        return res;
    }
  - Normally, to find successor in a BST has 3 cases.
    - it has right subtree. Then successor is the left most node in subtree.
    - it doesnt have right subtree and is the left node of its parent. Then successor is its parent.
    - it doesnt have right subtree and is the right node of its parent. Then successor is the first ancesstor that has it in its left branch.
  # Tree, Successor

*57. Insert Interval
  - find the left and right interval that has overlap with newInterval. Use left.start and right.end to create new interval.
  -   public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
          List<Interval> res = new ArrayList<Interval>();
          Interval tmp = newInterval;
          for (Interval interval : intervals) {
              if (tmp == null || interval.end < tmp.start) {
                  res.add(interval);
              } else if (interval.start > tmp.end) {
                  res.add(tmp);
                  res.add(interval);
                  tmp = null;
              } else {
                  tmp.start = Math.min(tmp.start, interval.start);
                  tmp.end = Math.max(tmp.end, interval.end);
              }
          }
          if (tmp != null) {
              res.add(tmp);
          }
          return res;
      }
  # Range, Sort

*721. Accounts Merge
  - Create edges for each account from all emails from the second place to the first email. Also create edges from first email to all other emails. Then got the adjacent emails for each email. Then use bfs to find connected component.
        // Create adjacent emails for each email.
  -      for(List<String> account : accounts) {
            String firstEmail = account.get(1);
            if(!emailAdjMap.containsKey(firstEmail)) {
                emailNameMap.put(firstEmail, account.get(0));
                emailAdjMap.put(firstEmail, new LinkedList<>());
            }
            for(int i=2;i<account.size();i++) {
                String email = account.get(i);
                // create edge from firstEmail to email for firstEmail
                emailAdjMap.get(firstEmail).add(email);

                if(!emailAdjMap.containsKey(email)) {
                    emailNameMap.put(email, account.get(0));
                    emailAdjMap.put(email, new LinkedList<>());
                }
                // Create edge from email to firstEmail for email
                emailAdjMap.get(email).add(firstEmail);
            }
        }
        // BFS to find connected component
        Set<String> visitedEmails = new HashSet<>();
        for(String email : emailAdjMap.keySet()) {
            if(!visitedEmails.contains(email)) {
                visitedEmails.add(email);
                LinkedList<String> component = new LinkedList<>();
                Queue<String> queue = new LinkedList<>();
                queue.add(email);
                while(!queue.isEmpty()) {
                    String e = queue.poll();
                    component.push(e);
                    for(String adjEmail : emailAdjMap.get(e)) {
                        if(!visitedEmails.contains(adjEmail)) {
                            queue.add(adjEmail);
                            visitedEmails.add(adjEmail);

                        }
                    }
                }
                Collections.sort(component);
                component.addFirst(emailNameMap.get(email));
                res.add(component);
            }
        }
  - Union Find: for each acount, union first email with other emails
    dsu.union(emailToID.get(account.get(1)), emailToID.get(email));
  # BFS/DFS, Union Find, Graph, Connected component

88. Merge Sorted Array
  - Two Pointers

*602. Friend Requests II: Who Has the Most Friends
  # SQL

*670. Maximum Swap
  - if int e.g. 7432 is decreasing, then no swap. Else, find the turning point. e.g 974678 which is 6. Then find max from 6 to the end. Compare max in decreasing order from 4 to 9 and find the largest num that less than max which is 7 and swap. Finnally 984677.
  - new String(charArr) // char arr to String
  # Array

13. Roman to Integer
  - 如果左边的数字小于右边的数字：右-左
        int res = toNum(s.charAt(0));
        for(int i=1;i<s.length();i++) {
            res += toNum(s.charAt(i)) > toNum(s.charAt(i-1)) ? toNum(s.charAt(i)) - 2 * toNum(s.charAt(i-1)) : toNum(s.charAt(i));
        }
        return res;
    }
    private int toNum(char c) {
        switch(c) {
            case 'I' : return 1;case 'V' : return 5;case 'X' : return 10;
            case 'L' : return 50;case 'C' : return 100;case 'D' : return 500;
            case 'M' : return 1000;
        }
        return 0;
    }
  # Roman, String

*12. Integer to Roman
  -     int[] values = {1000,900,500,400,100,90,50,40,10,9,5,4,1};
        String[] strs = {"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};
        StringBuilder sb = new StringBuilder();
        for(int i=0;i<values.length;i++) {
            while(num >= values[i]) {
                num -= values[i];
                sb.append(strs[i]);
            }
        }
  # String, Math

*785. Is Graph Bipartite
  - the adjacent node should have different color.
        // BFS
        // 0(not meet), 1(black), 2(white)
  -     int[] visited = new int[graph.length];
        for (int i = 0; i < graph.length; i++) {
            if (graph[i].length != 0 && visited[i] == 0) {
                visited[i] = 1;
                Queue<Integer> q = new LinkedList<>();
                q.offer(i);
                while(! q.isEmpty()) {
                    int current = q.poll();
                    for (int c: graph[current]) {
                            if (visited[c] == 0) {
                                visited[c] = (visited[current] == 1) ? 2 : 1;
                                q.offer(c);
                            } else {
                                if (visited[c] == visited[current]) return false;
                            }}}}}
  # Graph, BFS

*523. Continuous Subarray Sum
  - attention to mod 0 case
  - if k==0, for continous arr of 0 return true. if k!=0, maintain remainder and map of remainder and index. if remainder sum is exist and its array of at least 2. Then return true;
  -     Map<Integer, Integer> map = new HashMap<>(); // map of remainder and index
        int sum = nums[0] %k;
        map.put(sum, 0);
        for(int i =1;i<nums.length;i++) {
            sum = (sum + nums[i]) %k;
            if(sum == 0) return true;
            if(map.containsKey(sum)) {
                if(i - map.get(sum) > 1) return true;
            } else {
                map.put(sum, i);
            }
        }
  # HashMap

286. Walls and Gates
  - dfs for each 0.
  # BFS/DFS

*477. Total Hamming Distance
  - refer to 461 Hamming Distance
  - Iterate 0-31 bit, collect num of bit 1 and bit 0, then multiply them.
  -     for(int i = 0;i<32;i++) {
            int numOf1 = 0;
            for(int j=0;j<nums.length;j++) {
                if(nums[j] == 0) continue;
                numOf1 += nums[j] & 1;
                nums[j] >>=1;
            }
            res += numOf1 * (nums.length -numOf1);
        }
  # Bit Manipulation

38. Count and Say
  - 数数， 1 -> 11 -> 21 (2个1) -> 1211 ->(1个2，1个1) -> 111221
        while(i < n) {
            int count = 0;
            StringBuilder sb = new StringBuilder();
            char c = res.charAt(0);
            for(int j=0;j<=res.length();j++) {
                if(j!=res.length() && res.charAt(j) == c) {
                    count++;
                } else {
                    sb.append(count);
                    sb.append(c);
                    if(j != res.length()) {
                        count =1;
                        c = res.charAt(j);
                    }
                }
            }
            res = sb.toString();
            i++;
        }
  # String

*398. Random Pick Index
  - int array may have duplicates for a given target. The possibility of output each index of target is equal.
  - Random r = new Random(); r.nextInt(total); // random between [0, total)
  - e.g. {1,5,5,6,5} output index = 1 : 1 * 1/2 * 2/3 = 1/3
  - public int pick(int target) {
        int total = 0, res = -1;
        for(int i=0;i<nums.length;i++) {
            if(nums[i] == target) {
                int randomNum = rand.nextInt(++total); // 机智
                res = randomNum == 0 ? i : res;
    }}}
  # Reservior Sampling

824. Goat Latin
  # String

*825. Friends Of Appropriate Ages
  - B request friend A if B in range (0.5*A+7, A], More than 20000 people. Create sum array of ages. For each age of A, find sum of valid B within that range and multiply.
  -     int[] numOfAges = new int[121];
        int[] sumOfAges = new int[121];
        for(int i:ages) numOfAges[i]++;
        for(int i=1;i<121;i++) sumOfAges[i] = sumOfAges[i-1]+numOfAges[i];
        // B in range (0.5*A+7, A]
        for(int i=15;i<121;i++) {
            if(numOfAges[i] == 0) continue;
            int temp = sumOfAges[i] - sumOfAges[i/2+7];
            res += temp * numOfAges[i] - numOfAges[i]; // exclude self
        }
  # Array, Continuous array, Math

209. Minimum Size Subarray Sum
  - Maintain sum array, fast and slow pointer, if sum[fast] - sum[slow] > s, then increase slow and compare.
  -     sum[0] = nums[0];
        for(int i=1;i<nums.length;i++) sum[i] = sum[i-1]+nums[i];
        int slow=0, fast = 0;
        while(fast<nums.length && sum[fast]<s) fast++;
        if(fast == nums.length && sum[fast-1]<s) return 0;
        res = fast - slow +1;
        for(int i = fast;i<nums.length;i++) {
            while(sum[i] - sum[slow] >=s) {
                res = Math.min(res, i-slow);
                slow++;
            }
        }
  - Time O(n), Space O(n) -> can be reduced to O(1)
  # Two Pointers

*380. Insert Delete GetRandom O(1)
  - Maintain HashMap of val and index, ArrayList of val. For insertion, insert to both. For deletion, swap the val to delete in ArrayList with the last elem, and change the last elem index in HashMap.
  - public boolean remove(int val) {
        if(!map.containsKey(val)) return false;
        int index = map.get(val);
        int v = list.get(list.size()-1);
        list.set(index, v);
        map.put(v, index);
        list.remove(list.size()-1);
        map.remove(val);
        return true;
    }
    public int getRandom() {
        return list.get(random.nextInt(list.size()));
    }
  # Design, HashMap, ArrayList

*269. Alien Dictionary
  - Based on the word list order, figure out character order. Compare each pair to find the charater order and create edge from high order to low order. Then do topological sort.

  -     // create edge by compare each pair of word
        for(int i=1;i<words.length;i++) {
            int minLength = Math.min(words[i-1].length(), words[i].length());
            for(int j=0;j<minLength;j++) {
                if(words[i-1].charAt(j) != words[i].charAt(j)) {
                    if(!map.containsKey(words[i-1].charAt(j))) {
                        map.put(words[i-1].charAt(j), new LinkedList<>());
                    }
                    map.get(words[i-1].charAt(j)).add(words[i].charAt(j));
                    break;
                }
            }
        }

        Stack<Character> stack = new Stack<>();
        Set<Character> visited = new HashSet<>();
        Set<Character> accessed = new HashSet<>();
        for(Character c : set) {
            if(!visited.contains(c)) {
                dfs(visited, map, c, stack, accessed);
            }
        }
        if(hasCycle) return "";

    private void dfs(Set<Character> visited, Map<Character, List<Character>> map, Character c, Stack<Character> stack, Set<Character> accessed) {
        visited.add(c); accessed.add(c);
        if(map.containsKey(c)) {
            for(Character ch : map.get(c)) {
                if(accessed.contains(ch)) {
                    hasCycle = true;
                    return;
                }
                if(!visited.contains(ch)) {
                    dfs(visited, map, ch, stack, accessed);
                }
            }
        }
        accessed.remove(c);
        stack.push(c);
    }
  - Refer to 207 Course Schedule
  # Topological Sort, DFS, Directed Graph, Character Order

*71. Simplify Path
  - "." means stay current dir, ".." means back to last dir. e.g.path = "/a/./b/../../c/", => "/c"
  -     String[] arr = path.split("/+");
        Stack<String> stack = new Stack<>();
        for(String s : arr) {
            if(s.equals("..")) {
                if(!stack.isEmpty()) stack.pop();
            } else if(!s.equals(".") && !s.isEmpty()) {
                stack.push(s);
            }
        }
        if(stack.isEmpty()) return "/";
        String res = "";
        while(!stack.isEmpty()) res = "/" + stack.pop() + res;
  # Stack, Regular Expression

554. Brick Wall
  - Calculate array sum for each row. Use hashmap to track the times of sum. Use height - max times of sum will be the result.
  # HashMap

*90. Subsets II
  - arr has duplicates, Input: [1,2,2] Output:[[2],[1],[1,2,2],[2,2],[1,2],[]]
  - public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> res = new LinkedList<>();
        Arrays.sort(nums);
        dfs(nums, new LinkedList<>(), res, 0);
        return res;
    }
    private void dfs(int[] nums, List<Integer> list, List<List<Integer>> res, int startIndex) {
        res.add(new LinkedList<>(list)); // new list instead of list reference
        for(int i=startIndex;i<nums.length;i++) {
            // e.g. 1,2,2', if startIndex is 1, i=2, val 2 is not called, then 2' should not be called to remove duplicates
            if(i != startIndex && nums[i] == nums[i-1]) continue;
            list.add(nums[i]);
            dfs(nums, list, res, i+1); // use i instead of startIndex
            list.remove(list.size()-1);
        }
    }
  - Refer to 78 Subsets
  # Backtracking

28. Implement strStr()
  - Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.
  -     for(int i=0;i<haystack.length();i++) {
             if(match(haystack, i, needle)) return i;
        }
        private boolean match(String haystack, int i, String needle) {
            // attention to this, needle may more than rest of haystack
            if(needle.length()>haystack.length()-i) return false;
            for(int j=0;j<needle.length();j++) {
                if(haystack.charAt(i+j) != needle.charAt(j)) return false;
            }
            return true;
        }
  # String

674. Longest Continuous Increasing Subsequence
  -     for(int i=1;i<nums.length;i++) {
            cur = nums[i] <= nums[i-1] ? 1 : cur+1;
            res = Math.max(res, cur);
        }
  # Array

*377. Combination Sum IV
  - Integer array with positive numbers and no duplicates, find the number of possible combinations that add up to a positive integer target.
  - // Recursion
    public int combinationSum4(int[] nums, int target) {
        if (target == 0) return 1;
        for (int i = 0, res=0; i < nums.length; i++) {
            if (target >= nums[i])
                res += combinationSum4(nums, target - nums[i]);
        }
        return res;
    }
    // Top Down DP
    private int[] dp;
    public int combinationSum4(int[] nums, int target) {
        dp = new int[target + 1];
        Arrays.fill(dp, -1);
        dp[0] = 1;
        return helper(nums, target);
    }
    private int helper(int[] nums, int target) {
        if (dp[target] != -1) {
            return dp[target];
        }
        int res = 0;
        for (int i = 0; i < nums.length; i++) {
            if (target >= nums[i]) {
                res += helper(nums, target - nums[i]);
            }
        }
        dp[target] = res;
        return res;
    }
    // Bottom Up DP
    comb[0] = 1;
    for (int i = 1; i < comb.length; i++) {
        for (int j = 0; j < nums.length; j++) {
            if (i - nums[j] >= 0) {
                comb[i] += comb[i - nums[j]];
            }
        }
    }
  - (https://leetcode.com/problems/combination-sum-iv/discuss/85036/1ms-Java-DP-Solution-with-Detailed-Explanation)
  - for target issue, create 1d dp array of range [0,target]. Iterate amount and matain dp arr. Refer to 494. Target Sum, 322. Coin Change
  # 1D DP, Backtracking, Target DP

*69. Sqrt(x)
  - Refer to 34 Search for a Range
    // find the left most result, assign mid to r if nums[mid] = target
    while(l < r-1) {
        int mid = (l+r)/2;
        if(nums[mid] < target) {l = mid;} else {r = mid;}
    }
    if(nums[l] == target) return l;
    if(nums[r] == target) return r;
    // find the right most result, assign mid to l if nums[mid] = target
    while(l < r-1) {
        int mid = (l+r)/2;
        if(nums[mid] > target) {r = mid;} else {l = mid;}
    }
    if(nums[r] == target) return r;
    if(nums[l] == target) return l;

  - this question needs to find the left most result
        if(x==0) return 0;
        int left =1, right =x;
        while(left < right-1) {
            int mid = left + (right -left)/2;
            if(x/mid == mid) {
                return mid;
            } else if (x/mid <= mid) {
                right = mid;
            } else {
                left = mid;
            }
        }
        return left;
  # Binary Search

127. Word Ladder
  - Put startWord into queue, for each char, iterate from 'a' to 'z' to check if new created string is exist in wordlist, if yes, add to queue and level map.
  -     HashSet<String> set = new HashSet<>(wordList);
        Map<String, Integer> map = new HashMap<>(); // map(str, level)
        Queue<String> queue = new LinkedList<>(); // bfs queue
        queue.offer(beginWord);
        map.put(beginWord, 1);
        while(!queue.isEmpty()) {
            String word = queue.poll();
            int level = map.get(word);
            for(int i=0;i<word.length();i++) {
                char[] arr = word.toCharArray();
                for(char c = 'a';c<'z';c++) {
                    arr[i] = c;
                    String newWord = new String(arr);
                    if(set.contains(newWord)) {
                        if(newWord.equals(endWord)) return level+1;
                        map.put(newWord, level+1);
                        queue.offer(newWord);
                        set.remove(newWord);
                    }}}}
  - BFS. Create adjacent list for each word, undirected graph. Then bfs to find the shortest path to endWord.
  - Attention, not use DFS to find shortest path. If use DFS, use map(str, distance to startWord) instead of visited set.
  # BFS, Shortest Path, Dijkstra

235. Lowest Common Ancestor of a Binary Search Tree
  - find the first node that have p, and q on left and right
  -     if(root == null) return null;
        if(root == p || root == q) return root;
        TreeNode l = lowestCommonAncestor(root.left, p, q);
        TreeNode r = lowestCommonAncestor(root.right, p, q);
        if(l != null && r != null) return root;
        if(l != null || r != null) return l != null ? l : r;
        return null;
  # BST

*639. Decode Ways II
  - (https://leetcode.com/problems/decode-ways-ii/description/)
  - Refer to 91. Decode Ways. add '*', means 1-9. 细分情况
  # 1 DP

**334. Increasing Triplet Subsequence
  - Refer to 300. Longest Increasing Subsequence, 84 Largest Rectangle in Histogram. 300 and 84 are different, e.g. [8,2,5,3,0] for 300, result is [0,3], for 84, result is [0]. 300 keeps the OLD longest increasing order, 84 tracks the CURRTENT longest increasing order
  -     int[] arr = new int[3];
        int numInArr = 0;
        for(int num : nums) {
            int l=0, r = numInArr;
            while(l < r) {
                int mid = l + (r-l)/2;
                if(arr[mid]<num) {
                    l = mid+1;
                } else {
                    r = mid;
                }
            }
            arr[l]=num;
            if(l == numInArr) numInArr++;
            if(numInArr == 3) return true;
        }
  # Binary Search, patience sort, Longest Increasing subsequence

*117. Populating Next Right Pointers in Each Node II
            1 -> NULL row 1
           /  \
          2 -> 3 -> NULL row 2
         / \    \
        4-> 5 -> 7 -> NULL row 3
  - Popluate next for each node. Iterate through each row. e.g. Iterate row 2 starting by node(2), matain childHead which is 4, and child which for iteration. After row 2 done. Assign childHead 4 to parent and start over.

  -     TreeLinkNode parent = root;
        while(parent != null) {
            TreeLinkNode childHead = null;
            TreeLinkNode child = null;
            while(parent != null) {
                if(parent.left != null) {
                    if(childHead == null) {
                        childHead = parent.left;
                        child = parent.left;
                    } else {
                        child.next = parent.left;
                        child = parent.left;
                    }
                }
                if(parent.right != null) {
                    if(childHead == null) {
                        childHead = parent.right;
                        child = parent.right;
                    } else {
                        child.next = parent.right;
                        child = parent.right;
                    }
                }
                parent = parent.next;
            }
            parent = childHead;
        }
  # Tree, Level Traversal

*50. Pow(x, n)
  - e.g. 2^7 = 2* 2^6 = 2* 4^3 = 2* 4* 4^2 = 2* 4* 16
  - Attention to Integer.MIN_VALUE, e.g. -128 to 127.
      public double myPow(double x, int n)  {
        if(n==0 || x==1) return 1;
        if(n==1) return x;
        // n=0, x=1
        double res = 1;
        if(n<0) return 1/(x*myPow(x, -(n+1))); // deal with Integer.MIN_VALUE
        while(n>1) {
            if(n%2==1) {
                res = x*res;
                n--;
            } else {
                x *= x;
                n /=2;
            }
        }
        return res*x;
    }
  # Binary Search

68. Text Justification
  - 纯String操作，逻辑有点复杂。但无复杂算法数据结构
  # String

*85. Maximal Rectangle
  - matain height arr for each row. For each row iteration, matain index of increasing height in the stack, if height is less than stack.peek(), pop and calculate size.
  -     // Refer example https://github.com/EdwardShi92/Leetcode-Solution-Code/blob/master/MaximalRectangle.java
        int[] heights = new int[matrix[0].length+1];
        for(int i=0;i<matrix.length;i++) {
            Stack<Integer> stack = new Stack<>();
            for(int j=0;j<=matrix[0].length;j++) {
                // calculate heights
                if(j<matrix[0].length) {
                    if(matrix[i][j] == '1') {
                        heights[j] ++;
                    } else {
                        heights[j] = 0;
                    }
                }
                // matain indexes of increasing heights
                while(!stack.isEmpty() && heights[j] < heights[stack.peek()]) {
                    int index = stack.pop();
                    int startIndex = stack.isEmpty()? -1 : stack.peek();
                    // Attention, startIndex using stack.peek() rather than index, endIndex using j rather than index.
                    res = Math.max(res, heights[index] * (j-1-startIndex));
                }
                stack.push(j);
            }
        }
  - Stack: refer to 84 Largest Rectangle in Histogram
  - DP: refer to 221. Maximal Square
  # Stack, DP

261. Graph Valid Tree
  - To find cycle in undirected graph, use union find
  - Use union find to check if two index has same source, if yes, cycle find. Then check the num of connected component, if more than 2, not tree.
  -     int[] arr = new int[n];
        for(int i = 0;i<n;i++) {
            arr[i]=i;
        }
        for(int[] edge : edges) {
            int i = find(arr, edge[0]);
            int j = find(arr, edge[1]);
            if(i == j) return false;
            union(arr, edge[0], edge[1]);
        }
        Set<Integer> set = new HashSet<>();
        for(int i=0;i<n;i++) {
            set.add(find(arr, i));
            if(set.size()>1) return false;
        }
        return true;
    }
    private int find(int[] arr, int index) {
        while(arr[index] != index) {
            arr[index] = arr[arr[index]];
            index = arr[index];
        }
        return index;
    }
    private void union(int[] arr, int i, int j) {
        int a = find(arr, i);
        int b = find(arr, j);
        if(a != b) {
            arr[a] = b;
        }
    }
  # Undirected Graph, Union Find, Connected Component

210. Course Schedule II
  - DFS + accessed[] to find back edge
  # Directed Graph, Topological Sort, Stack

26. Remove Duplicates from Sorted Array
  - slow pointer point to current value, fast pointer iterate
  -     while(fp<nums.length) {
            if(nums[sp] == nums[fp]) {
                fp++;
            } else {
                nums[++sp] = nums[fp++];
            }}
  # Two Pointers

*274. H-Index
  - Bucket each citation value into citations.length+1 buckets. Then iterate from right to left and matain total citations. If total citations >= index, then return index. Clever!
  -     int size = citations.length;
        int[] arr = new int[size + 1];
        for(int i : citations) {
            if(i>size) {
                arr[size]++;
            } else {
                arr[i]++;
            }
        }
        int sum = 0;
        for(int i=size;i>=0;i--) {
            sum +=arr[i];
            if(sum >= i) return i;
        }
  # Bucket Sort

404. Sum of Left Leaves
  # Tree

268. Missing Number
  - Use value range [0, n] and index range [0, n-1]. Map each value to each index and change value to -1 to prove its visited. At last, iterate arr to find the val that is not -1 which is not visited.
  -    for(int num : nums) {
           while(num != -1) {
               if(num == nums.length) {
                   break;
               }
               int temp = nums[num];
               nums[num] = -1;
               num = temp;
           }}
       for(int i=0;i<nums.length;i++) {
           if(nums[i] != -1) return i;
       }
       return nums.length;
  - 高斯公式，算[0,n] sum， iterate and minus from sum. The remainder is the missing number.
  # Array

80. Remove Duplicates from Sorted Array II
  - Refer to 26. Remove Duplicates from Sorted Array
  # Two Pointers

168. Excel Sheet Column Title
  - 10进制变成26进制
  -     while(n>0){
              n--;
            sb.append((char)('A'+n%26));
            n/=26;
        }
        return sb.reverse().toString();
  # Math

535. Encode and Decode TinyURL
  - In industry, shorten url service is by database, one auto increasing long number as primary key.  whenever a long url need to be shorten, append to the database, and return the primary key number. (https://leetcode.com/problems/encode-and-decode-tinyurl/discuss/100276/Easy-solution-in-java-5-line-code.)
  - List<String> list = new LinkedList<>();
    String host = "http://tinyurl.com/";
    // Encodes a URL to a shortened URL.
    public String encode(String longUrl) {
        list.add(longUrl);
        return host + String.valueOf(list.size()-1);
    }

    // Decodes a shortened URL to its original URL.
    public String decode(String shortUrl) {
        int index = Integer.parseInt(shortUrl.replace(host,""));
        return index < list.size() ? list.get(index) : "";
    }
  - Use String.hashCode() and put it into map is not correct. Since 2 different string may have the same hashcode.
  # System Design

*801. Minimum Swaps To Make Sequences Increasing
  - Matain 2 dp arrays, keep[] and swap[]
    Refer to 714. Best Time to Buy and Sell Stock with Transaction Fee
             309. Best Time to Buy and Sell Stock with Cooldown
  - Consider DFS solution for DP question e.g. (http://zxi.mytechroad.com/blog/dynamic-programming/leetcode-801-minimum-swaps-to-make-sequences-increasing/)
  -     int N = A.length;
        int[] swap = new int[N];
        int[] not_swap = new int[N];
        swap[0] = 1;
        for (int i = 1; i < N; ++i) {
            not_swap[i] = swap[i] = N;
            if (A[i - 1] < A[i] && B[i - 1] < B[i]) {
                not_swap[i] = not_swap[i - 1];
                swap[i] = swap[i - 1] + 1;
            }
            if (A[i - 1] < B[i] && B[i - 1] < A[i]) {
                not_swap[i] = Math.min(not_swap[i], swap[i - 1]);
                swap[i] = Math.min(swap[i], not_swap[i - 1] + 1);
            }
        }
        return Math.min(swap[N - 1], not_swap[N - 1]);
  # DP

*525. Contiguous Array
  - For continous array issue, think about iterate and sum up each val and store into hashmap. Map stores <#1-#0 diff, index>, if diff ==0, compare with res, if diff is exist in map, calculate and compare with res.
  -     int[] diffArr = new int[nums.length];
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0;i<nums.length;i++) {
            if (nums[i]==1) count1++; else count0++;
            if(count1-count0 == 0) {
               res = Math.max(res, i+1);
                continue;
            }
            diffArr[i] = count1 - count0;
            if(!map.containsKey(diffArr[i])) {
                map.put(diffArr[i], i);
            } else {
                res = Math.max(res, i-map.get(diffArr[i]));
            }
        }
   - Refer to 523. Continuous Subarray Sum
  # HashMap, Consider Two Pointers and HashMap for continous issue

*750. Number Of Corner Rectangles
  - Fix two rows i, j. iterate col k to find total pairs grid[i][k] == grid[j][k]==1, then calculate combinations. Then continue;
  -     for(int i=0;i<grid.length-1;i++) {
            for(int j=i+1;j<grid.length;j++) {
                int count = 0;
                for(int k=0;k<grid[0].length;k++) {
                    if(grid[i][k] == 1 && grid[j][k] == 1) count++;
                }
                if(count >1) res += count*(count-1)/2;
            }
        }
  - Improvement is compare #row and #cols, start from min.
  # 2D Array

637. Average of Levels in Binary Tree
  # Tree, BFS

**764. Largest Plus Sign
  -  Each val in arr is calculated 4 times that from left to right, from right to left, from up to down and from down to up.
  -     int[][] arr = new int[N][N];
        for(int i=0;i<N;i++) {
            Arrays.fill(arr[i],N);
        }
        for(int[] mine:mines) {
            arr[mine[0]][mine[1]]=0;
        }
        // 神循环，each val is initilzed as N. Each val is calculated to find the minimum from left to right, right to left, up to down and down to up. 前提是N*N，M*N就不行了
        for(int i=0;i<N;i++) {
            for(int j=0, k=N-1,l=0,r=0,u=0,d=0;j<N;j++,k--) {
                arr[i][j] = Math.min(arr[i][j], l=arr[i][j]==0?0:l+1);
                arr[i][k] = Math.min(arr[i][k], r=arr[i][k]==0?0:r+1);
                arr[j][i] = Math.min(arr[j][i], u=arr[j][i]==0?0:u+1);
                arr[k][i] = Math.min(arr[k][i], d=arr[k][i]==0?0:d+1);
            }
        }
        int res = 0;
        for(int i=0;i<N;i++) {
            for(int j=0;j<N;j++) {
                res = Math.max(res, arr[i][j]);
            }
        }
  # 2D Array

784. Letter Case Permutation
  - Character: toUpperCase(), toLowerCase(), isDigit(), isLetter(), isLowerCase()
  # Backtracking

578. Get Highest Answer Rate Question
  # SQL

275. H-Index II
  -     while(l<=r) {
            int mid = l + (r-l)/2;
            if(citations[mid] == size -mid) {
                return size-mid;
            } else if(citations[mid] < size - mid) {
                l = mid +1;
            } else {
                r = mid-1;}}
        return size-l;
  # Binary Search

*745. Prefix and Suffix Search
  - e.g. "apple", instead of create a->p->p->l->e and e->l->p->p->a 2 tries. Create follwing trie structure. Use HashMap to record children rather than fix array size children[27][27], 26 letters + null.
                                           ""
                   (a,e)                      (a,null)            (null,e)
              (p,l) (p,null), (null,l)
          (p,p)
        (l,p)
    (e,a)

  - private void buildTree(TrieNode node, String word, int weight, int l, int r) {
        if(l>=word.length() && r <= -1) return;
        node.maxWeight  = weight;
        int bg = l < word.length() ? word.charAt(l)-'a'+1 : 0;
        int sm = r > -1 ? word.charAt(r)-'a'+1 : 0;
        int num = bg * 27 + sm;  // hashcode
        if(!node.map.containsKey(num)) {
            node.map.put(num, new TrieNode(weight));
        } else {
            node.map.get(num).maxWeight = weight;
        }
        buildTree(node.map.get(num), word, weight, l+1, r-1);
        buildTree(node.map.get(num), word, weight, l+1, -1);
        buildTree(node.map.get(num), word, weight, word.length(), r-1);
    }

    private int find(TrieNode node, String prefix, String suffix, int i, int j){
        if(i>=prefix.length() && j<0) return node.maxWeight;
        int big = i<prefix.length() ? prefix.charAt(i)-'a'+1 : 0;
        int sm = j>= 0 ? suffix.charAt(j)-'a'+1 : 0;
        int num = big*27 + sm;
        if(node.map.containsKey(num)) {
            i = i<prefix.length() ? i+1:i;
            j = j>=0 ? j-1:j;
            return find(node.map.get(num), prefix, suffix, i, j);
        } else {
            return -1;
        }
    }
  # Trie

*36. Valid Sudoku
  - Validate row, column and 3x3 sub-boxes. Matain a set, encode each element and add to set, if it has been added before. Then return false.
      // e.g. board[i][j] = k, encode to "kri", "kcj", "kij" and add to set
  -   Set<String> set = new HashSet<>();
      for(int i=0;i<board.length;i++) {
          for(int j=0;j<board[0].length;j++) {
              char k = board[i][j];
              if(k == '.') continue;
              if(!set.add(k+"r"+i) ||
                 !set.add(k+"c"+j) ||
                 !set.add(Character.toString(k)+i/3+j/3)) {
                  return false;
              }
          }
  - 很聪明的encode节省了时间
  # HashSet, Encode

*37. Sudoku Solver
  - Related to 36. Valid Sudoku. For each empty space, Iterate from 1-9, validate each and call DFS if valid.
  - private boolean dfs(char[][] board) {
        for(int i=0;i<board.length;i++) {
            for(int j=0;j<board[0].length;j++) {
                if(board[i][j] == '.') {
                    for(char k='1';k<='9';k++) {
                        board[i][j]=k;
                        if(validateSudoku(board, i, j) && dfs(board)) {
                            return true;
                        } else {
                            board[i][j]='.';
                        }
                    }
                    return false;
                }}}
        return true;
    }
     private boolean validateSudoku(char[][] board, int row, int col) {
         // Validate row and col
         for(int i=0;i<9;i++) {
             if(i!=row && board[i][col] == board[row][col]) return false;
             if(i!=col && board[row][i] == board[row][col]) return false;
         }
         // Validate 3*3 subbox
         for(int i=row/3*3;i<(row/3+1)*3;i++) {
             for(int j=col/3*3;j<(col/3+1)*3;j++) {
                 if(i == row && j == col) continue;
                 if(board[i][j] == board[row][col]) return false;
             }
         }
         return true;
     }
  # DFS

*41. First Missing Positive
  - Given an unsorted integer array, find the smallest missing positive integer
  - Map each array element to array index.
        for(int i=0;i<nums.length;i++) {
            if(nums[i] == i+1) continue;
            int index = nums[i];
            while(index>0 && index<=nums.length && nums[index-1] != index) {
                int temp = nums[index-1];
                nums[index-1] = index;
                index = temp;
            }
        }
        for(int i=0;i<nums.length;i++) {
            if(nums[i] != i+1) return i+1;
        }
        return nums.length+1;
  # Array, Index

*54. Spiral Matrix
  -     int rowBegin = 0;
        int rowEnd = matrix.length -1;
        int colBegin = 0;
        int colEnd = matrix[0].length -1;
        while(rowBegin<=rowEnd && colBegin<=colEnd) {
            for(int i = colBegin;i<=colEnd;i++) {
                res.add(matrix[rowBegin][i]);
            }
            rowBegin++;
            for(int i=rowBegin;i<=rowEnd;i++) {
                res.add(matrix[i][colEnd]);
            }
            colEnd--;
            if(rowBegin<=rowEnd) { // careful
                for(int i=colEnd;i>=colBegin;i--) {
                    res.add(matrix[rowEnd][i]);
                }
            }
            rowEnd--;
            if(colBegin<=colEnd) {  // careful
                for(int i=rowEnd;i>=rowBegin;i--) {
                    res.add(matrix[i][colBegin]);
                }
            }
            colBegin++;
        }
  # 2D Array

59. Spiral Matrix II
  - Similar to 54. Spiral Matrix
  # 2D Array

*115. Distinct Subsequences
  - Given a string S and a string T, count the number of distinct subsequences of S which equals T.
  -     for(int i=1;i<=s.length();i++) {
            for(int j=1;j<=t.length();j++) {
                if(s.charAt(i-1) == t.charAt(j-1)) {
                    dp[i][j] = dp[i-1][j] + dp[i-1][j-1];
                } else {
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
  # 2D DP, DFS

*134. Gas Station
  - if gas >= cost, gaurantee to have solution
  -     int gasSum = 0, costSum = 0, tank = 0, start = 0;
        for(int i=0; i<gas.length; i++) {
            gasSum += gas[i];
            costSum += cost[i];
            tank += (gas[i] - cost[i]);
            if(tank < 0){
                tank = 0;
                start=i+1;
            }
        }
        if(gasSum < costSum) {
            return -1;
        }
        return start;
  # Greedy

150. Evaluate Reverse Polish Notation
  # Stack

*33. Search in Rotated Sorted Array
  - either [left, mid] or (mid,right] is sorted, check in sorted side
  - draw pic like (https://www.youtube.com/watch?v=w6nusIojP9c)
  - attention to using >= or >
  -       while(l<=r) {
            int mid = l + (r-l)/2;
            if(nums[mid] == target) return mid;
            if(nums[mid] >= nums[l]) {
                if(target>=nums[l] && target<=nums[mid]) {
                    r = mid-1;
                } else {
                    l = mid+1;
                }
            } else {
                if(target>=nums[mid] && target<=nums[r]) {
                    l = mid+1;
                } else {
                    r = mid-1;
                }}}
  # Binary Search

*153. Find Minimum in Rotated Sorted Array
  - [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2].Find the minimum element.
  -     int l = 0, r = nums.length-1;
        while(l < r) {
            int mid = l + (r-l)/2;
            if(mid>0 && nums[mid]<nums[mid-1]) return nums[mid];
            if(nums[mid]>nums[r]) {
                l = mid+1;
            } else {
                r = mid-1;
            }
        }
        return nums[l];
  # Binary Search

*162. Find Peak Element
  - 想象一下爬山，nums[mid] < nums[mid+1] 在上山山峰在右，反之则在下山
  -     // l<r, 保证了存在 mid+1
        // l = mid+1, r = mid 保证了最后的解是l
        int l = 0, r = nums.length-1;
        while(l<r) {
            int mid = l + (r-l)/2;
            if(nums[mid] < nums[mid+1]) {
                l = mid+1;
            } else {
                r = mid;
            }
        }
        return l;
  # Binary Search

165. Compare Version Numbers
  - Use split(regx) to split . e.g. version = "1.0.1", iterate through the larger size.
  -     String[] arr1 = version1.split("\\.");
        String[] arr2 = version2.split("\\.");
        int len = Math.max(arr1.length, arr2.length);
        for(int i=0;i<len;i++) {
            int v1 = 0, v2 =0;
            if(i<arr1.length) {
                v1 = Integer.parseInt(arr1[i]);
            }
            if(i<arr2.length) {
                v2 = Integer.parseInt(arr2[i]);
            }
            if(v1 > v2) {
                return 1;
            } else if(v1<v2){
                return -1;
            }
        }
        return 0;
  # String, Regualr Expression

205. Isomorphic Strings
  -     Map<Character, Character> map = new HashMap<>();
        for(int i=0;i<s.length();i++) {
            char sc = s.charAt(i);
            char tc = t.charAt(i);
            if(map.containsKey(sc)) {
                if(map.get(sc) != tc) return false;
            } else if(map.containsValue(tc)) {
                return false;
            } else {
                map.put(sc, tc);
            }
        }
  # HashMap

222. Count Complete Tree Nodes
  - bit manipulation (https://www.vojtechruzicka.com/bit-manipulation-java-bitwise-bit-shift-operations/)
  - <<, >> 等bit操作优先级小于+ - ，所以要加括号 e.g.(1<<2) == 100 = 2^2 = 4
  - Comapre height of left node and right node. If same height, left node is full subtree, else, right node is full subtree. The precondition is complete tree.

  -   public int countNodes(TreeNode root) {
          if(root == null) return 0;

          int lh = getHeight(root.left);
          int rh = getHeight(root.right);
          if(lh == rh) {
              return (1<<lh) + countNodes(root.right);
          } else {
              return (1<<rh) + countNodes(root.left);
          }
      }
      private int getHeight(TreeNode node) {
          if(node == null) return 0;
          return 1 + getHeight(node.left);
      }
  # Binary Search, Tree, Complete Tree

247. Strobogrammatic Number II
  # String

*248. Strobogrammatic Number III
  - String a.compareTo(b) // - : a<b, 0: a=b, +: a>b
  - Iterate length between [low.length(), high.length()] to build Strobogrammatic number. Then filter by v.compareTo(low)>=0 && v.compareTo(high)<=0
  # String, DFS

249. Group Shifted Strings
  - Encode each string by setting the first char to z and shift other chars. e.g. yab -> zbc. Use encoded string as key in map. Put all strings into map.
  - private String encode(String s) {
        int diff = 'z' - s.charAt(0);
        String str = "z";
        if(diff == 0) {
            return s;
        } else {
            for(int i=1;i<s.length();i++) {
                char c = s.charAt(i);
                if(c + diff >'z') {
                    str +=(char)(c+diff-'z'+'a'-1);
                } else {
                    str += (char)(c+diff);
                }
            }
        }
        return str;
    }
  # String, HashMap

*264. Ugly Number II
  - Ugly numbers are positive numbers whose prime factors only include 2, 3, 5。
  - (https://leetcode.com/problems/ugly-number-ii/discuss/69362/O(n)-Java-solution)
  -     int[] ugly = new int[n];
        ugly[0] = 1;
        int index2 = 0, index3 = 0, index5 = 0;
        int factor2 = 2, factor3 = 3, factor5 = 5;
        for(int i=1;i<n;i++){
            int min = Math.min(Math.min(factor2,factor3),factor5);
            ugly[i] = min;
            if(factor2 == min)
                factor2 = 2*ugly[++index2];
            if(factor3 == min)
                factor3 = 3*ugly[++index3];
            if(factor5 == min)
                factor5 = 5*ugly[++index5];
        }
        return ugly[n-1];
  # Array, 类似DP

289. Game of Life
    - use 01, 10 to represent 0 -> 1, 1 -> 0 status
    # 2D Array

*295. Find Median from Data Stream
  - use minHeap to maintain larger half values, use maxHeap to maintain smller half values.
  - PriorityQueue<Integer> minHeap = new PriorityQueue<>();
    PriorityQueue<Integer> maxHeap = new PriorityQueue<>(1000, Collections.reverseOrder());
    public void addNum(int num) {
        maxHeap.add(num);
        minHeap.add(maxHeap.poll());
        if(maxHeap.size()<minHeap.size()) {
            maxHeap.add(minHeap.poll());
        }
    }
    public double findMedian() {
        if(maxHeap.size() == minHeap.size()) {
            return (maxHeap.peek() + minHeap.peek())/2.0;
        } else {
            return maxHeap.peek();
        }
    }
  # Two PriorityQueue, Heap

298. Binary Tree Longest Consecutive Sequence
  - post order traverse, each time return the current longest consecutive number
  # PostOrder, Tree

*305. Number of Islands II
  - Matain 2D arr of islands and 1D union find arr. For each island placement, check left, right, up, down, if they are islands, check find and then union.
        int[][] arr = new int[m][n];
        int[] ufArr = new int[m*n];
        for(int i=0;i<m*n;i++) ufArr[i] = i;
        for(int[] pos : positions) {
            int islands = res.size() == 0 ? 0 : res.get(res.size()-1);
            int row = pos[0], col = pos[1];
            res.add(islands + check(arr, ufArr, row, col, m, n));
            arr[row][col] = 1;
        }
    private int check(int[][] arr, int[] ufArr, int row, int col, int m, int n){
        int res = 1;
        if(row-1>=0 && arr[row-1][col] == 1) {
            // if not in the same connected componet, union them
            if(find(ufArr,row*n+col) != find(ufArr, (row-1)*n+col)) {
                res--;
                union(ufArr, row*n+col, (row-1)*n+col);
            }
        }
        if(col-1>=0 && arr[row][col-1] == 1) {
            if(find(ufArr,row*n+col) != find(ufArr,row*n+col-1)) {
                res--;
                union(ufArr, row*n+col, row*n+col-1);
            }
        }
        if(row+1<m && arr[row+1][col] == 1) {
            if(find(ufArr,row*n+col) != find(ufArr,(row+1)*n+col)) {
                res--;
                union(ufArr, row*n+col, (row+1)*n+col);
            }
        }
        if(col+1<n && arr[row][col+1] == 1) {
            if(find(ufArr,row*n+col) != find(ufArr,row*n+col+1)) {
                res--;
                union(ufArr, row*n+col, row*n+col+1);
            }
        }
        return res;
    }

  # Union Find

*308. Range Sum Query 2D - Mutable
  - pure 2D binary indexed tree.
  Refer to (https://www.geeksforgeeks.org/two-dimensional-binary-indexed-tree-or-fenwick-tree/)

    public NumMatrix(int[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) return;
        rows = matrix.length;
        cols = matrix[0].length;
        arr = new int[rows][cols];
        tree = new int[rows+1][cols+1];
        for(int i=0;i<rows;i++) {
            for(int j=0;j<cols;j++) {
                update(i, j, matrix[i][j]);
            }
        }
    }
    // row and col in matrix, corresponding to row+1, col+1 in tree
    public void update(int row, int col, int val) {
        if(rows == 0 || cols == 0) return;
        int diff = val - arr[row][col];
        arr[row][col] = val;
        // row+1, col+1 because tree[row+1][col+1]
        for(int r = row+1;r<=rows;r += r&(-r)) {
            for(int c = col+1;c<=cols;c +=c&(-c)) {
                tree[r][c] += diff;
            }
        }
    }
    public int sumRegion(int row1, int col1, int row2, int col2) {
        if(rows == 0 || cols == 0) return 0;
        return sum(row2, col2) + sum(row1-1, col1-1) - sum(row1-1, col2) - sum(row2, col1-1);
    }
    // row and col in matrix, corresponding to row+1, col+1 in tree
    private int sum(int row, int col) {
        int sum = 0;
        for(int r = row+1;r>0;r -=r&(-r)) {
            for(int c = col+1;c>0;c -=c&(-c)) {
                sum += tree[r][c];
            }
        }
        return sum;
    }
  # 2D Fenwik Tree / Binary Indexed Tree

*312. Burst Balloons
  - 从1个balloon arr开始 to nums.length个ballon, 每次index从start=1 to end=nums.length, 从中iterate k, which is the last ballon to burst in [start, end]
  - Refer to (https://www.youtube.com/watch?v=z3hu2Be92UA)
  -     int size = nums.length;
        int[] arr = new int[size+2];
        arr[0] = 1;arr[size+1] = 1;
        for(int i=1;i<=size;i++) arr[i] = nums[i-1];

        int[][] dp = new int[size+2][size+2];

        // burst len start from 1 to size
        for(int len = 1;len<=size;len++) {
            for(int start = 1;start<=size-len+1;start++) {
                int end = start+len-1;
                // k is the last burst ballon in [start,end]
                for(int k = start;k<=end;k++) {
                    dp[start][end] = Math.max(dp[start][end], dp[start][k-1]+arr[start-1]*arr[k]*arr[end+1]+dp[k+1][end]);
                }
            }
        }
        return dp[1][size];
  # 2D DP

340. Longest Substring with At Most K Distinct Characters
  # Two Pointers, HashMap

*346. Moving Average from Data Stream
  - private int [] window;
    private int n, insert=0;
    private long sum=0;

    public MovingAverage(int size) {window = new int[size]；}

    public double next(int val) {
        if (n < window.length)  n++;
        sum -= window[insert];
        sum += val;
        window[insert] = val;
        insert = (insert + 1) % window.length; // 聪明
        return (double)sum / n;
    }
  # Design, Queue, Buffer

*359. Logger Rate Limiter
- Message not print if it has print in the last 10 seconds. Design to scale if there are tremendous messages, so only maintain last 10 second. For a new message, check timestamp%10 slot, if time[timestamp%10] == timestamp, then not clear, else reset time slot and messages. Then check each message bucket. Finally add message.

-   private int[] time;
    private Set[] messageBuckets;

    public Logger() {
        time = new int[10];
        messageBuckets = new Set[10];
        for(int i=0;i<10;i++) {
            messageBuckets[i] = new HashSet<String>();
        }
    }

    public boolean shouldPrintMessage(int timestamp, String message) {
        // Reset time slot and messageBuckets
        int t = timestamp%10;
        if(timestamp != time[t]) {
            time[t] = timestamp;
            messageBuckets[t].clear();
        }
        // Check should print
        for(int i=0;i<10;i++) {
            if(timestamp - time[i]<10 && messageBuckets[i].contains(message)) return false;
        }

        messageBuckets[t].add(message);
        return true;
    }
  # Design, Scale, Buffer, HashSet

*362. Design Hit Counter
  - Design a hit counter which counts the number of hits received in the past 5 minutes.
  - private int[] time = new int[300];
    private int[] hits = new int[300];

    public void hit(int timestamp) {
        int t = timestamp%300;
        if(time[t] != timestamp) {
            time[t] = timestamp;
            hits[t] = 1;
        } else {
            hits[t]++;
        }
    }

    public int getHits(int timestamp) {
        int res = 0;
        for(int i=0;i<300;i++) {
            if(timestamp - time[i]<300) {
                res +=hits[i];
            }
        }
        return res;
    }
  # Design, Scale, Buffer

*361. Bomb Enemy
  - 每一行我们可以在第0列或者当前位置前一列为墙的时候从第当前列开始往右搜索直到撞到墙. 每一列可以在第0行的时候或者在当前行前一行为墙的时候从当前行往下搜索, 直到碰到墙为止. 这样就可以一次计算出一行直到碰到墙之前有几个敌人, 一列在没有碰到墙之前有几个敌人. 直到当某个某位之前位置墙的时候才会重新计算. O(mn + mn + mn)
  -     int rowHit = 0;
        int[] colHit = new int[cols];
        for(int i=0;i<rows;i++) {
            for(int j = 0;j<cols;j++) {
                if(j==0 || grid[i][j-1] == 'W') {
                    rowHit = 0;
                    for(int k = j;k<cols && grid[i][k] != 'W';k++) {
                        if(grid[i][k] == 'E') rowHit++;
                    }
                }

                if(i==0 || grid[i-1][j] == 'W') {
                    colHit[j] = 0;
                    for(int k = i;k<rows && grid[k][j] != 'W';k++) {
                        if(grid[k][j] == 'E') colHit[j]++;
                    }
                }

                if(grid[i][j] == '0') {
                    res = Math.max(res, rowHit + colHit[j]);
                }
            }
        }
  - Refer to 764 神循环
  # Array

*392. Is Subsequence
  - Given a string s and a string t, check if s is subsequence of t by deleting some chars in t. If there are lots of incoming S, say S1, S2, ... , Sk where k >= 1B, and you want to check one by one to see if T has its subsequence. Thus use Binary Search
  - maintain index list for each char in t. Then for each char in s, find if exist in list and index in increasing order.
  -     List<Integer>[] list = new List[256];
        for(int i=0;i<t.length();i++) {
            if(list[t.charAt(i)] == null) {
                list[t.charAt(i)] = new LinkedList<>();
            }
            list[t.charAt(i)].add(i);
        }

        int pre = -1;
        for(int i=0;i<s.length();i++) {
            if(list[s.charAt(i)] == null) return false;
            pre = binarySearch(list[s.charAt(i)], pre);
            if(pre == -1) return false;
        }

    private int binarySearch(List<Integer> list, int index) {
        int start = 0, end = list.size()-1;
        while(start<=end) {
            int mid = start + (end - start)/2;
            if(list.get(mid) <= index)
              start = mid +1;
            else
              end = mid -1;
        }
        return start == list.size() ? -1 : list.get(start);
    }

  - Collections.binarySearch()
  # Binary Search

*399. Evaluate Division
  - Give equations, run queries. Instead of using union find int array, use Map<key, Node>, Node{parent, ratio} to track node parent relationship.
  - convert equation to graph (https://leetcode.com/problems/evaluate-division/description/)
  - Map<String, Node> map = new HashMap<>();
    public double[] calcEquation(String[][] equations, double[] values, String[][] queries) {
        for(int i=0;i<equations.length;i++) {
            union(equations[i][0], equations[i][1], values[i]);
        }

        double[] ans = new double[queries.length];
        for(int i=0;i<queries.length;i++) {
            Node n1 = find(queries[i][0]);
            Node n2 = find(queries[i][1]);
            if(n1 == null || n2 == null || !n1.parent.equals(n2.parent)) {
                ans[i] = -1;
            } else {
                ans[i] = n1.ratio/n2.ratio;
            }
        }

    public void union(String s1, String s2, double ratio) {
        Node n1 = find(s1);
        Node n2 = find(s2);
        if(n1==null && n2==null) {
            map.put(s1, new Node(s2, ratio));
            map.put(s2, new Node(s2, 1));
        } else if(n1 == null) {
            map.put(s1, new Node(s2, ratio));
        } else if(n2 == null) {
            map.put(s2, new Node(s1, 1/ratio));
        } else {
            n1.parent = n2.parent;
            n1.ratio = ratio * n2.ratio  / n1.ratio;
        }
    }

    public Node find(String name) {
        if(!map.containsKey(name)) return null;
        Node node = map.get(name);
        if(!name.equals(node.parent)) {
            Node p = find(node.parent); // path compression
            node.parent = p.parent;
            node.ratio *= p.ratio;
        }
        return node;
    }
  # Union Find, HashMap

*418. Sentence Screen Fitting
  - cols may far bigger than sentence. So for each row, add cols first and then backtrack if cut string until find whitespace. Finally, index/len
  -     for(String str : sentence) {
            if(str.length() > cols) return 0;
        }

        String s = String.join(" ", sentence) + " ";
        int len = s.length();
        int index = 0;
        for(int i=0;i<rows;i++) {
            index +=cols;
            if(s.charAt(index%len) == ' ') {
                index++;
            } else {
                while(s.charAt((index-1)%len) != ' ') {
                    index--;
                }
            }
        }

        return index/len;
  # String

*465. Optimal Account Balancing
  - Calculate balance for each person. The get list of balances and find the minimum num to sum them to 0 which should use DFS.
  - public int minTransfers(int[][] transactions) {
        Map<Integer, Integer> map = new HashMap<>();
        for(int[] t : transactions) {
            map.put(t[0], map.getOrDefault(t[0], 0) - t[2]); // 学习下
            map.put(t[1], map.getOrDefault(t[1], 0) + t[2]);
        }

        return dfs(0, new ArrayList<>(map.values()));
    }

    private int dfs(int index, List<Integer> balances) {
        while(index<balances.size() && balances.get(index) == 0) {
            index++;
        }
        if(index == balances.size()) return 0;

        int transactions = Integer.MAX_VALUE;
        for(int i=index+1;i<balances.size();i++) {
            if(balances.get(index) * balances.get(i) <0) {
                // sum index and i
                balances.set(i, balances.get(index) + balances.get(i));
                // dfs index + 1 rather than i+1
                transactions = Math.min(transactions, 1 + dfs(index+1, balances));
                balances.set(i, balances.get(i) - balances.get(index));
            }
        }

        return transactions;
    }
  # DFS

*490. The Maze
  - Ball movement has direction. Use while loop when ball moving until reach wall. Check stop position if its destination.
  - Add before check
  - Use direction arr instead of up, down, left, right
  - private boolean dfs(int[][] maze, int[] pos, int[] dest, boolean[][] visited) {
        if(pos[0]<0 || pos[0] == maze.length || pos[1]<0 || pos[1] == maze[0].length || visited[pos[0]][pos[1]]) return false;
        if(pos[0] == dest[0] && pos[1] == dest[1]) return true;
        visited[pos[0]][pos[1]] = true; // only mark stop pos as visited
        // down, up, right, left
        int[][] dir = new int[][]{{1, -1, 0, 0},{0, 0, 1, -1}};
        for(int i=0;i<4;i++) {
            int row = pos[0], col = pos[1];
            while(row>=0 && row<maze.length && col>=0 && col<maze[0].length && maze[row][col] !=1) {
                // visited arr ignore non-stop pos
                row += dir[0][i];
                col += dir[1][i];
            }
            row -= dir[0][i];
            col -= dir[1][i];
            if(dfs(maze, new int[]{row, col}, dest, visited)) return true;
        }
        return false;
    }
  # DFS

*505. The Maze II
  - Use bfs. Instead of use boolean[] visited, use int[] visited to track smallest distance at ball stop position.
  -     int[][] visited = new int[rows][cols];
        for(int[] arr : visited) Arrays.fill(arr, Integer.MAX_VALUE);
        int[][] dir = new int[][]{{1,-1,0,0},{0,0,1,-1}};
        Queue<int[]> queue = new LinkedList<>();
        Queue<Integer> distances = new LinkedList<>();
        queue.offer(start);
        distances.offer(0);
        int res = Integer.MAX_VALUE;
        while(queue.size()>0) {
            int[] pos = queue.poll();
            int row = pos[0], col = pos[1];
            int distance = distances.poll();
            if(row == destination[0] && col == destination[1]) {
                res = Math.min(res, distance);
                continue;
            }
            if(distance>=visited[row][col]) {
                continue;
            } else {
                visited[row][col] = distance;
            }

            for(int i=0;i<4;i++) {
                int r = row, c = col, d = distance;
                while(r>=0 && r<rows && c>=0 && c<cols && maze[r][c] != 1) {
                    r += dir[0][i];
                    c += dir[1][i];
                    d++;
                }
                r -= dir[0][i];
                c -= dir[1][i];
                d--;
                queue.offer(new int[]{r, c});
                distances.offer(d);
            }
        }
  # BFS, Shortest Path

*499. The Maze III
  - (https://leetcode.com/problems/the-maze-iii/description/)
  # BFS, Shortest Path

*486. Predict the Winner
  - without memorization, time is O(2^n). With memorization, O(n^2)
  - 每次选左或选右，最大化差值
  -   int[][] memory = new int[nums.length][nums.length];
      for(int[] arr: memory) Arrays.fill(arr, Integer.MIN_VALUE);
      return choose(nums, 0, nums.length-1, memory) >= 0;

      private int choose(int[] nums, int l, int r, int[][] memory) {
          if(l == r) return nums[l];
          if(memory[l][r] != Integer.MIN_VALUE) return memory[l][r];
          memory[l][r] = Math.max(nums[l] - choose(nums, l+1, r, memory), nums[r] - choose(nums, l, r-1, memory)); //状态转移方程
          return memory[l][r];
      }
  # Recursion, Memorization, 2D

496. Next Greater Element I
  - Use stack to keep a decreasing sub-sequence, whenever we see a number x greater than stack.peek() we pop all elements less than x and for all the popped ones, their next greater element is x
  -     Stack<Integer> stack = new Stack<>();
        Map<Integer, Integer> map = new HashMap<>();
        for(int num : nums2) {
            while(!stack.isEmpty() && num>stack.peek()) {
                map.put(stack.pop(), num);
            }
            stack.push(num);
        }
  - interate from right to left, and matains increasing order in stack. Keep track of stack and map.
  -     for(int i = nums2.length -1 ; i>=0;i--) {
            while(!stack.empty()) {
                if(nums2[i]>=stack.peek()) {
                    stack.pop();
                } else {
                    break;
                }
            }

            if(stack.empty()) {
                map.put(nums2[i], -1);
            } else {
                map.put(nums2[i], stack.peek());
            }
            stack.push(nums2[i]);

        }
  # Stack

503. Next Greater Element II
  - it has duplicates. Run second time. Use one more index stack instead of map.
  # Stack

*527. Word Abbreviation
  - build trie, each node matains count (num of words that have this prefix) and isWord. Then iterate trie, count is 1, then create abbreviation, if the abbreviation lenght is same as word length, then use origianl word. This is incorrect for this question e.g. [ "internal","internet", "interval", "intension", "intrusion"], Output: ["internal","i6t","interval","inte4n","intr4n"]
  - Create shortest abbreviation for each word. Then, while we have duplicates, we increase the length of all duplicates.

  # Trie, String

529. Minesweeper
  - dfs to unreveal all adjacent "E", for each "E" get num of Bombs, if more than 1 end, if 0 continue dfs.
  # DFS

*540. Single Element in a Sorted Array
  - Find the single element in a sorted array that include pair elements and one single element.
  -  public static int singleNonDuplicate(int[] nums) {
          int start = 0, end = nums.length - 1;
          while (start < end) { // not start<=end to avoid out of bound e.g. (0,1,1)
              int mid = (start + end) / 2;
              if (mid % 2 == 1) mid--;  // important
              if (nums[mid] != nums[mid + 1]) end = mid; else start = mid + 2;
          }
          return nums[start];
      }
  # Binary Search

*552. Student Attendance Record II
  - /*
    dp[i] = total num of rewardable student records at index i
    1. without A
        1. end with P
        dp[i] = dp[i-1]
        2. end with L
        ...PLLL
        dp[i] = dp[i-1] - dp[i-4]

    2. with A
    assume A is at index i
    [... i-1] A [i+1, ...n]
    dp[i] = dp[i-1] * dp[n-i]
    */
    public int checkRecord(int n) {
        long M = 1000000007;
        long[] dp = new long[n<=3 ? 4 : n+1];
        dp[0]=1; dp[1]=2; dp[2]=4; dp[3]=7;
        for(int i=4;i<=n;i++) {
            dp[i] = ((2* dp[i-1])%M + (M-dp[i-4]))%M;
        }
        long sum = dp[n];
        for(int i=1;i<=n;i++) {
            sum += (dp[i-1] * dp[n-i])%M;
        }

        return (int) (sum%M);
    }
  # 1D DP

*652. Find Duplicate Subtrees
  - postorder traverval and serialize each subtree to string and put into a map<String, Integer>, if Integer == 2, add to res.
  # Tree, Postorder Traversal, Serialize

*659. Split Array into Consecutive Subsequences
  - matain pre, p1, p2, p3 which are the number of consecutive subsequences ending at pre with length of 1, length of 2 and length >= 3. If count<p1+p2 during iteration, false. At end, if p1 !=0 || p2 !=0, false;

  - int pre = Integer.MIN_VALUE, p1 = 0, p2 = 0, p3 = 0;
    int cur = 0, cnt = 0, c1 = 0, c2 = 0, c3 = 0;

    for (int i = 0; i < nums.length; pre = cur, p1 = c1, p2 = c2, p3 = c3) {
        for (cur = nums[i], cnt = 0; i < nums.length && cur == nums[i]; cnt++, i++);

        if (cur != pre + 1) {
            if (p1 != 0 || p2 != 0) return false;
            c1 = cnt; c2 = 0; c3 = 0;

        } else {
            if (cnt < p1 + p2) return false;
            c1 = Math.max(0, cnt - (p1 + p2 + p3));
            c2 = p1;
            c3 = p2 + Math.min(p3, cnt - (p1 + p2));
        }
    }

    return (p1 == 0 && p2 == 0);
  - (https://leetcode.com/problems/split-array-into-consecutive-subsequences/discuss/106495/Java-O(n)-time-and-O(1)-space-solution-greedily-extending-shorter-subsequence)
  # Greedy

684. Redundant Connection
  - cycle detection in undirected graph.
  # Union Find, undirected graph

*685. Redundant Connection II
  - directed graph, 3 cases
    (1) circle, same as 684
    (2) node has 2 parents
    (3) node has 2 parents and circle
  - Find node that has 2 parents and put into candidates. Assume remove one edge. Run union find, if there is circle and candidates are not initialized , case 1 and return the edge. If no circle, return the removed edge. If circle and candidates are initilazed, return another edge.

  -     int[] candidate1 = new int[]{-1, -1}, candidate2 = new int[]{-1,-1};
        int[] arr = new int[edges.length+1];
        // If node has two parents, pop to candidates and remove one edge
        for(int[] edge : edges) {
            if(arr[edge[1]] != 0 ) {
                candidate1 = new int[]{edge[0], edge[1]};
                candidate2 = new int[]{arr[edge[1]], edge[1]};
                edge[1] = 0;
                break;
            } else {
                arr[edge[1]] = edge[0];
            }
        }
        for(int i=1;i<=edges.length;i++) arr[i] = i;
        for(int[] edge : edges) {
            if(edge[1] == 0) continue;
            int parent = find(arr, edge[0]);
            int child = find(arr, edge[1]);
            if(parent == child) {
                if(candidate2[0] != -1) {
                    return candidate2;
                }
                return edge;
            } else {
                arr[parent] = child;
            }
        }
        return candidate1;
  # Union Find, directed graph

*727. Minimum Window Subsequence
  - Given strings S and T, find the minimum (contiguous) substring W of S, so that T is a subsequence of W.
  - Track the start index. e.g. S = abcde, T = be
              a b c d e b
            0 1 2 3 4 5 6
          b 0 0 2 2 2 2 6
          e 0 0 0 0 0 2 2
  -     int lenT = T.length(), lenS = S.length();
        int[][] dp = new int[lenT + 1][lenS + 1];
        for (int j = 0; j <= lenS; j++) {
            dp[0][j] = j + 1; // Store start index
        }
        for (int i = 1; i <= lenT; i++) {
            for (int j = 1; j <= lenS; j++) {
                if (T.charAt(i - 1) == S.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = dp[i][j - 1];
                }
            }
        }

        int start = 0, len = lenS + 1;
        for (int j = 1; j <= lenS; j++) {
            if (dp[lenT][j] != 0) {
                if (j - dp[lenT][j] + 1 < len) {
                    start = dp[lenT][j] - 1;
                    len = j - dp[lenT][j] + 1;
                }
            }
        }
        return len == lenS + 1 ? "" : S.substring(start, start + len);
  # 2D DP

*731. My Calendar II
  - Insert (start, end), if there is triple event intersection, return false, else return true. Construct binary search tree. Corner case start == end
  - private Node insert(Node node, int start, int end) {
        if(start==end) return node;
        if(node == null) return new Node(start, end);
        if(start >= node.end) {
            node.r = insert(node.r, start, end);
        } else if(end<=node.start){
            node.l = insert(node.l, start, end);
        } else {
            node.overlap = true;
            int a = Math.min(node.start, start);
            int b = Math.max(node.start, start);
            int c = Math.min(node.end, end);
            int d = Math.max(node.end, end);
            node.l = insert(node.l,a,b);
            node.r = insert(node.r,c,d);
            node.start = b;
            node.end = c;
        }
        return node;
    }

    private boolean insertable(Node node, int start, int end) {
        if(start==end) return true;
        if(node == null) return true;
        if(start>=node.end) {
            return insertable(node.r, start, end);
        } else if(end <= node.start) {
            return insertable(node.l, start, end);
        } else {
            if(node.overlap) {
                return false;
            } else {
                return insertable(node.l, start, node.start) && insertable(node.r, node.end, end);
            }
        }
    }

    private class Node {
        int start, end;
        boolean overlap;
        Node l, r;
        public Node(int start, int end) {
            this.start = start;
            this.end = end;
        }
    }

  # Binary Search Tree

753. Cracking the Safe
  - Total number of differences should be Math.pow(k,n). Matain set for string. Run dfs until num matches total.
  -     int total = (int)Math.pow(k,n);
        for(int i=0;i<n;i++) res +="0";
        Set<String> set = new HashSet<>();
        set.add(res);
        dfs(total, 1, set, n, k);

        private boolean dfs(int total, int cur, Set<String> set, int n, int k) {
            if(cur == total) return true;

            String substring = res.substring(res.length()-n+1,res.length());
            for(int i=0;i<k;i++) {
                String s = substring + i;
                if(!set.contains(s)) {
                    cur++;
                    set.add(s);
                    res +=i;
                    if (dfs(total, cur, set, n, k)) return true;
                    cur--;
                    set.remove(s);
                    res = res.substring(0, res.length()-1);
                }
            }
            return false;
        }
  # DFS

765. Couples Holding Hands
  - For given n people, divide into n/2 groups. e.g. people 0/2 is group 0, people 1/2 is group 0. Iterate pair by pair, if same group e.g. 0 == 0, means couple, else use find to check if these two people have been union before, if have been union before, it means no more swap, e.g. 0312 -> only 1 swap to become 0132. Else res++ and union group.
  -     int couples = row.length/2;
        int[] group = new int[couples];
        for(int i=0;i<group.length;i++) group[i] = i;

        for(int i=0;i<group.length;i++) {
            int g1 = row[2*i]/2;
            int g2 = row[2*i+1]/2;
            if(g1 == g2) {
                continue;
            } else if(find(group, g1) != find(group, g2)) {
                res++;
                union(group, g1, g2);
            }
        }

  # Union Find

769. Max Chunks To Make Sorted
  - Arr is a permutation of [0, 1, ..., arr.length - 1], we split the array into some number of "chunks" (partitions), and individually sort each chunk.  After concatenating them, the result equals the sorted array.
  - Matain the max array. Iterate, if max[i] = i, then gaurantee 左边的数字都小于这条线右边的数字
  - /*
    original: 0, 2, 1, 4, 3, 5, 7, 6
    max:      0, 2, 2, 4, 4, 5, 7, 7
    sorted:   0, 1, 2, 3, 4, 5, 6, 7
    index:    0, 1, 2, 3, 4, 5, 6, 7
    */
        int[] max = new int[arr.length];
        max[0] = arr[0];
        for (int i = 1; i < arr.length; i++) {
            max[i] = Math.max(max[i - 1], arr[i]);
        }
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            if (max[i] == i) {
                count++;
            }
        }
  # Array

773. Sliding Puzzle
  - encode to string instead of keep using int[][]
  - Queue<String> queue = new LinkedList<>();
    queue.offer(sb.toString());
    visited.add(sb.toString());
    String end = "123450";
    int count = 0;
    while (!queue.isEmpty()) {
        int size = queue.size();
        for (int index = 0; index < size; index++) {
            String cur = queue.poll();
            if (cur.equals(end))
                return count;
            int i = cur.indexOf('0');
            int[] dir = {1, -1, 3, -3};
            for (int d : dir) {
                char[] ch = cur.toCharArray();
                int j = i + d;
                if (j >= ch.length || j < 0 || (i == 2 && j == 3) || (i == 3 && j == 2))
                    continue;
                char temp = ch[i];
                ch[i] = ch[j];
                ch[j] = temp;
                String newBoard = new String(ch);
                if (visited.add(newBoard))
                    queue.offer(newBoard);
            }
        }
        count++; // 聪明，每次每种情况走一步，算最短路径
    }
  # BFS

*787. Cheapest Flights Within K Stops
  - 保证queue里头的每一批都走一步，直到k stops. int[] visisted matains minimum total price for each position.
  -     while(stops++<=K+1) {
            int size = queue.size();
            for(int i=0;i<size;i++) { // 每一批都走一步
                int[] pos = queue.poll(); // pos[0] position, pos[1] totalPrice
                if(pos[0] == dst) {
                    res = Math.min(res, pos[1]);
                    continue;
                }
                if(pos[1] >= visited[pos[0]]) continue;
                visited[pos[0]] = pos[1];
                if(map.containsKey(pos[0])) {
                    List<int[]> adjNodes = map.get(pos[0]);
                    for(int[] node : adjNodes) {
                        queue.offer(new int[]{node[0], node[1] + pos[1]});
                    }
                }
            }
        }
  - Similar to 773 Sliding Puzzle
  # BFS

734. Sentence Similarity
  # HashMap， Set

737. Sentence Similarity II
  -     private void union(Map<String, String> map, String s1, String s2) {
            String f1 = find(map, s1);
            String f2 = find(map, s2);
            if(f1 == null && f2 == null) {
                map.put(s1, s2);
                map.put(s2, s2);
            } else if(f1 == null) {
                map.put(s1, f2);
            } else if(f2 == null) {
                map.put(s2, f1);
            } else {
                map.put(f1, f2);
            }
        }
        private String find(Map<String, String> map, String s1) {
            if(!map.containsKey(s1)) return null;
            String s = s1;
            while(!s.equals(map.get(s))) {
                s = map.get(s);
            }
            map.put(s1, s);
            return s;
        }
  # String, Union Find

739. Daily Temperatures
  - Put index into stack instead of value because its more easy to calculte index diff.
  -     for(int i=len-1;i>=0;i--) {
            while(stack.size() != 0 && temperatures[stack.peek()]<=temperatures[i]) {
                stack.pop();
            }
            res[i] = stack.size() == 0 ? 0 : stack.peek()-i;
            stack.push(i);
        }
  # Stack

802. Find Eventual Safe States
  - If has cycle, mark all nodes in accessed as has cycle, if not, mark all nodes in visited as no cycle.
  # DFS, Directed Graph

805. Split Array With Same Average
  - 取1，2...A.length/2 个，每次用dfs/backtracking遍历所有情况。超时了。很多的duplicates， 可以用memorization进行优化
      for(int i = A.length/2;i>=1;i--) {
          if(dfs(A, 0, i, 0, 0, sum)) {
              return true;
          }
      }
    private boolean dfs(int[] arr, int start, int num, int count, double sum, double total) {
        if(count == num) {
            return sum/num == average && (total-sum)/(arr.length-num)==average;
        }
        if(num-count > arr.length-start) return false;

        for(int i=start;i<arr.length;i++) {
            if(dfs(arr, i+1, num, count+1, sum+arr[i], total)){
                return true;
            }
        }
        return false;
    }
  # DFS(TLE), Backtracking

815. Bus Routes
  - map each stop to routes. Then use start and end stop to find start routes set and end routes set. Create route to route adjacent list. Iterate start routes, use bfs to find the first end route and compare with the res.
  - Lots of code at (https://leetcode.com/problems/bus-routes/description/)
  # BFS

------------------------------------Follwing are not added to Category ------

686. Repeated String Match
  - need 2 more copies e.g. a: "abc" "abc" "abc", b: "c abc a"
  -   public int repeatedStringMatch(String a, String b) {
        String as = a;
        for (int rep = 1; rep <= b.length() / a.length() + 2; rep++, as += a)
            if (as.indexOf(b) != -1) return rep;
        return -1;
    }
  # String

889. Construct Binary Tree from Preorder and Postorder Traversal
  - (https://zxi.mytechroad.com/blog/tree/leetcode-889-construct-binary-tree-from-preorder-and-postorder-traversal/)

*568. Maximum Vacation Days
  -   private int dfs(int city, int curWeek, int numOfWeeks) {
        if(curWeek == numOfWeeks) {
            return 0;
        }

        if(map[city][curWeek] != 0) {
            return map[city][curWeek];
        }

        int maxPlayDays = Integer.MIN_VALUE;
        for(int i=0;i<flights.length;i++) {
            if(flights[city][i] == 1 || i== city) {
                maxPlayDays = Math.max(maxPlayDays, days[i][curWeek] + dfs(i, curWeek+1, numOfWeeks));
            }
        }
        map[city][curWeek] = maxPlayDays;
        return maxPlayDays;
    }
    # DFS + memorization

270. Closest Binary Search Tree Value
  -     while(root != null) {
            if(Math.abs(root.val-target)<Math.abs(res-target)) {
                res = root.val;
            }
            if(root.val > target) {
                root = root.left;
            } else {
                root = root.right;
            }
        }
    # Tree

1007. Minimum Domino Rotations For Equal Row
  - 4 个loop
  - https://www.youtube.com/watch?v=yjfGwiYPdfc

953. Verifying an Alien Dictionary
  # String

1249. Minimum Remove to Make Valid Parentheses
  # String

973. K Closest Points to Origin
  - PriorityQueue<int[]> queue = new PriorityQueue<>(K, (a,b) -> a[0]*a[0]+a[1]*a[1] - b[1]*b[1] - b[0]*b[0]);
  # PriorityQueue, Quick Select

426. Convert Binary Search Tree to Sorted Doubly Linked List
  - public void helper(Node node) {
      if (node != null) {
        helper(node.left);
        if (last != null) {
          last.right = node;
          node.left = last;
        }
        else {
          first = node;
        }
        last = node;
        helper(node.right);
      }}
  # Inorder Traversal

415. Add Strings
  # String

199. Binary Tree Right Side View
  # Level Order Tree Traversal

986. Interval List Intersections
  # Interval

349. Intersection of Two Arrays
  # Set, Two Pointers

987. Vertical Order Traversal of a Binary Tree
  - 在（x, y）坐标相同下，val小的优先。所以不能用level traversal， 反例 [0,2,1,3,null,null,null,4,5,null,7,6,null,10,8,11,9]， 7和6有相同的坐标，根据level traver，7先，但6数值小应该排前头。所以traversal给所有点标上坐标，先根据x分类，然后先比较y然后val
  - Collections.sort(l, (a,b) -> a.y == b.y ? a.val - b.val : b.y - a.y);
  # Tree Traversal, Compare

304. Range Sum Query 2D - Immutable
  # DP

65. Valid Number
         // flags
        int signCount = 0;
        boolean hasE = false;
        boolean hasNum = false;
        boolean hasPoint = false;

        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);
            // invalid character
            if (!isValid(c)) return false;
            // digit is always fine
            if (c >= '0' && c <= '9') hasNum = true;
            // e or E
            if (c == 'e' || c == 'E') {
                // e cannot appear twice and digits must be in front of it
                if (hasE || !hasNum) return false;
                // e cannot be the last one
                if (i == n - 1) return false;
                hasE = true;
            }
            // decimal place
            if (c == '.') {
                // . cannot appear twice and it cannot appear after e
                if (hasPoint || hasE) return false;
                // if . is the last one, digits must be in front of it, e.g. "7."
                if (i == n - 1 && !hasNum) return false;

                hasPoint = true;
            }
            // signs
            if (c == '+' || c == '-') {
                // no more than 2 signs
                if (signCount == 2) return false;
                // sign cannot be the last one
                if (i == n - 1) return false;
                // sign can appear in the middle only when e appears in front
                if (i > 0 && !hasE) return false;

                signCount++;}}
  # String

767. Reorganize String
  - Build string based on its count in decreasing order
  -     // Greedy: fetch char of max count as next char in the result.
        // Use PriorityQueue to store pairs of (char, count) and sort by count DESC.
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> b[1] - a[1]);
        for (char c : map.keySet()) {
            pq.add(new int[] {c, map.get(c)});
        }
        // Build the result. Get highest count char and gaurantee same char is not adjacent
        StringBuilder sb = new StringBuilder();
        while (!pq.isEmpty()) {
            int[] first = pq.poll();
            if (sb.length() == 0 || first[0] != sb.charAt(sb.length() - 1)) {
                sb.append((char) first[0]);
                if (--first[1] > 0) {
                    pq.add(first);
                }
            } else {
                int[] second = pq.poll();
                sb.append((char) second[0]);
                if (--second[1] > 0) {
                    pq.add(second);
                }
                pq.add(first);
            }
        }
  # String, Greedy, PriorityQueue

863. All Nodes Distance K in Binary Tree
  - Build non-directional graph and run bfs
  # Graph, BFS

1027. Longest Arithmetic Sequence
  - dp[diff][idx]表示等差为diff，以系数idx结尾的最长子序列长度。
  - https://www.youtube.com/watch?v=mSplC7Q-Rm8
  - public int longestArithSeqLength(int[] A) {
        int res = 2, n = A.length;
        HashMap<Integer, Integer>[] dp = new HashMap[n];
        for (int j = 0; j < A.length; j++) {
            dp[j] = new HashMap<>();
            for (int i = 0; i < j; i++) {
                int d = A[j] - A[i];
                dp[j].put(d, dp[i].getOrDefault(d, 1) + 1);
                res = Math.max(res, dp[j].get(d));
            }
        }
        return res;
    }
  # DP

1197. Minimum Knight Moves
  - 仅看第一象限就好，x>-1, y>-1.走BFS
  -    queue.add(new int[]{0,0});
        int[][] steps = new int[][]{{2, 1}, {1, 2}, {-1, 2}, {-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {2, -1}};
        int s = 0;
        x = Math.abs(x);
        y = Math.abs(y);
        set.add("0#0");
        while(++s>0) {
            int size = queue.size();
            while(size-- > 0) {
                int[] pos = queue.poll();
                for(int[] step : steps) {
                    int newX = pos[0]+step[0];
                    int newY = pos[1]+step[1];
                    if(newX<-1 || newY<-1) continue;
                    if(newX == x && newY == y) {
                        return s;
                    }
                    if(!set.contains(newX+"#"+newY)) {
                        set.add(newX+"#"+newY);
                        queue.add(new int[]{newX,newY});
                    }}}}
  # BFS

*1004. Max Consecutive Ones III
  - 注意快慢指针的位置！！！ fp - sp 得到距离
  -     int fp = 0;
        int sp = 0;
        int res = Integer.MIN_VALUE;
        // write as fp++ will out of bound
        while(fp< A.length) {
            if(A[fp] == 0) {
                // always minus 1 incase of K=0
                if(K-- == 0) {
                  res = Math.max(res, fp - sp);
                  // Bypass one 0
                  while(A[sp++] != 0) {};
                  K++;
                }
            }
            fp++;
        }
        // Coner case that [0,0,0,1] K=4
        res = Math.max(res, fp - sp);
  # Two Pointers, Sliding Window

* 29. Divide Two Integers
  - 神操作 1<<1,左移1位乘以2， 1>>1,右移1位除以2
  -     // 1<<31 = Integer.MIN_VALUE, (1<<31)-1 = Integer.MAX_VALUE
        if (A == 1 << 31 && B == -1) return (1 << 31) - 1;
        int a = Math.abs(A), b = Math.abs(B), res = 0, x = 0;
        while (a - b >= 0) {
            // A = 50, B = 4
            // x = 0, 50 - 4*2 >0
            // x= 1, 50 - 4*2^(1+1)>0
            // x = 2, 50 - 4*2^(2+1)>0
            // x = 3, 50 - 4*2^(3+1)<0, thus res += 2^3
            for (x = 0; a - (b << x << 1) >= 0; x++);
            res += 1 << x;
            a -= b << x;
        }
        return (A > 0) == (B > 0) ? res : -res;
  # Bit Manipulation

958. Check Completeness of a Binary Tree
  - level order traversal, if start getting null, all the rest should be null for complete binary tree, if not, then wrong
  -     while(!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if(node == null) {
                if(!startToHaveNull) {
                    startToHaveNull = true;
                }
            } else {
                if(startToHaveNull) {
                    return false;
                }
                queue.add(node.left);
                queue.add(node.right);
            }
        }
  # Level Order Traversal,

143. Reorder List
  - 快慢指针，找到中点。后半部分指针转向。然后头指针和尾指针轮流指
  - 细心，举单双数例子跑
  -     ListNode sp = head;
        ListNode fp = head;
        //找中点
        while(fp.next != null && fp.next.next!=null) {
            fp = fp.next.next;
            sp = sp.next;
        }

        fp = sp.next;
        sp.next = null;
        //后半部分转向
        while(fp != null) {
            ListNode temp = fp.next;
            fp.next = sp;
            sp = fp;
            fp = temp;
        }

        ListNode h = head;
        // 轮流指
        while(h != sp && h != sp.next) {
            ListNode temp = h.next;
            h.next = sp;
            h = temp;
            temp = sp.next;
            sp.next = h;
            sp = temp;
        }

        if(h == sp) {
            h.next = null;
        } else {
            h.next = sp;
            sp.next = null;
        }
  # Two Pointers

350. Intersection of Two Arrays II
  # HashMap, Two Pointers

246. Strobogrammatic Number
  # Two Pointers

708. Insert into a Sorted Circular Linked List
  - duplicates exist in the list
        Node n = new Node(insertVal);
        if(head == null) {
            n.next = n;
            return n;
        } else if(head.next == head) {
            head.next = n;
            n.next = head;
            return head;
        }

        max = head;
        min = head;
        Node pre = head;
        Node cur = head.next;
        do {
            if(cur.val>=max.val) { // due to duplicates, >= not >
                max = cur;
            }
            if(cur.val <= min.val) { // duplicates, <= not <
                min = cur;
            }
            if(n.val<=cur.val && n.val>=pre.val) { // dup, 得有=
                pre.next = n;
                n.next = cur;
                return head;
            }

            pre = pre.next;
            cur = cur.next;
        } while(pre != head); // 保证绕一圈

        max.next = n;
        n.next = min;
  # Two Pointers

480. Sliding Window Median
  - 坑是minQ里新加val，maxQ加minQ里最小的。如果减掉的val在minQ里，minQ加maxQ最大的
  - Refer to 295. Find Median from Data Stream
  -     initializeQueue(nums,k);
        double[] res = new double[nums.length - k + 1];
        res[0] = getMedian(k);
        for(int i = k,j=1;i<nums.length;i++,j++) {
            res[j] = processAndGetMedian(nums, i-k , i, k);
        }

    // maxQ keeps 1 more if k is odd
    private void initializeQueue(int[] nums, int k) {
        for(int i = 0;i<k;i++) {
            minQ.add(nums[i]);
        }
        for(int i=0;i<k/2;i++) {
            maxQ.add(minQ.poll());
        }
    }

    //先加入minQ,然后maxQ，如果减的在minQ里，再从MaxQ里加一个
    private double processAndGetMedian(int[] nums, int remove, int add, int k) {
        minQ.add(nums[add]);
        maxQ.add(minQ.poll());
        if(minQ.remove(nums[remove])) {
            minQ.add(maxQ.poll());
        } else {
            maxQ.remove(nums[remove]);
        }

        return getMedian(k);
    }

    private double getMedian(int k) {
        if(k%2 ==1) {
            return minQ.peek();
        } else {
          // 小心 2<<31-1 coner case
            return ((double)minQ.peek() + (double)maxQ.peek()) / 2;
        }
    }
  # Sliding Window, PriorityQueue

**1216. Valid Palindrome III
  - 因为delete string会出现奇变偶或偶变奇的情况，所以中间数不确定，不能从中间到两边。e.g. abbababa。 所以用l,r两边往中间走。
  - DFS + Memorization

    Integer[][] cache = new Integer[s.length()][s.length()];
    return aux(s, 0, s.length()-1, cache) <= k;

    private int aux(String s, int left, int right, Integer[][] cache) {
        // l == r || right < left
        if (right - left < 1) return 0;
        if (cache[left][right] != null) return cache[left][right];

        int step = 0;
        if (s.charAt(left) == s.charAt(right)) {
            step = aux(s, left+1, right-1, cache);
        } else {
            step = 1 + Math.min(aux(s, left+1, right, cache), aux(s, left, right-1, cache));
        }
        cache[left][right] = step;
        return step;

  - 2D DP
        int[][] dp = new int[s.length()][s.length()];
        for(int i =0; i<s.length();i++) {
            for(int j=i;j>=0;j--) {
                // 奇数i==j或者偶数j>i两种情况
                if(i-j<1) {
                    dp[i][j] = 0;
                } else if(s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i-1][j+1];
                } else {
                    // 因为 dp[i][j+1]。所以j从右往左走
                    dp[i][j] = 1 + Math.min(dp[i-1][j], dp[i][j+1]);
                }
            }
        }
        return dp[s.length()-1][0] <= k;
     e.g. abcdea
      0 0 0 0 0 0
      1 0 0 0 0 0
      2 1 0 0 0 0
      3 2 1 0 0 0
      4 3 2 1 0 0
      3 4 3 2 1 0

  # 2D DP, DFS + Memorization，Two Pointers

528. Random Pick with Weight
  - Calculate prefix sum of each elem in array and total weight. Use random to pick from total weight. Then binary search prefix sum and get index
  - public Solution(int[] w) {
        random = new Random();
        weight = new int[w.length];
        for(int i=0;i<w.length;i++) {
            totalWeight += w[i];
            weight[i] = totalWeight;
        }
    }
    public int pickIndex() {
        // [1,totalWight]
        int target  = random.nextInt(totalWeight)+1;
        int l = 0;
        int r = weight.length;
        while(l < r) {
            int mid = (l+r)/2;
            if(target > weight[mid]) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }

        return l;
    }
  - Refer to 398. Random Pick Index
  # Random, Binary Search

938. Range Sum of BST
  # Binary Search Tree

921. Minimum Add to Make Parentheses Valid
     -  int res = 0;
        // ‘(’比 ‘)'多的个数
        int lMinusR=0;
        for(char c : S.toCharArray()) {
            if(c == '(') {
                lMinusR++;
            } else {
                if(lMinusR == 0) {
                    res++;
                } else {
                    lMinusR--;
                }
            }
        }

        res += lMinusR;
  # Parentheses

* 1060. Missing Element in Sorted Array
  - 用不了 l<r, l = mid+ 1,所以用 l<r-1, l=mid, r=mid。
  - public int missingElement(int[] nums, int k) {
        int l = 0;
        int r = nums.length-1;
        while(l < r-1) {
            int mid = (l+r)/2;
            if(nums[0] + k + mid > nums[mid]) {
                l = mid;
            } else {
                r = mid;
            }
        }
        int res = nums[0] + k + l;
        if(res < nums[r]) {
            return res;
        } else {
            return res + 1;
        }
    }
  # Binary Search

138. Copy List with Random Pointer
  - Matain Map<oldNode, newNode>, then connect through oldNode link
    Map<Node, Node> map = new HashMap<Node, Node>();

    // loop 1. copy all the nodes
    Node node = head;
    while (node != null) {
      map.put(node, new Node(node.val));
      node = node.next;
    }
    // loop 2. assign next and random pointers
    node = head;
    while (node != null) {
      map.get(node).next = map.get(node.next);
      map.get(node).random = map.get(node.random);
      node = node.next;
    }
  # HashMap

*126. Word Ladder II
  - Diff is that this quesiton asks for shortest *PAHTS*, not single shortest path.
  - Build undirected graph, then run bfs to find shortest path. Note that can not use visited set due to corner case:
    beginWord:"red", endWord: "tax", wordList: ["ted","tex","red","tax","tad","den","rex","pee"]
    output with visited: [["red","ted","tex","tax"],["red","ted","tad","tax"]]
    expected: [["red","ted","tad","tax"],["red","ted","tex","tax"],["red","rex","tex","tax"]]
        -> ted
    red         -> tex -> tax
        -> rex
    If maintain visited. the rex route is missing. Thus only maintain accessed list for each path.
  # Grpah, BFS

843. Guess the Word
  - Come up with match method to find num of matching chars of two words. Then pick least 0 matching word to guess cause word mostly get 0 matches so that it can eliminate the most words. Or scoring, for each char in 6-letter, find most char, calculate words and find highest score.
  - https://leetcode.com/problems/guess-the-word/discuss/133862/Random-Guess-and-Minimax-Guess-with-Comparison
    public void findSecretWord(String[] wordlist, Master master) {
        for (int i = 0, x = 0; i < 10 && x < 6; ++i) {
            HashMap<String, Integer> count = new HashMap<>();
            // cal match for each word
            for (String w1 : wordlist)
                for (String w2 : wordlist)
                    //
                    if (match(w1, w2) == 0)
                        count.put(w1, count.getOrDefault(w1 , 0) + 1);
            String guess = "";
            int min0 = 100;
            // find word with lowest num of 0 match
            for (String w : wordlist)
                if (count.getOrDefault(w, 0) < min0) {
                    guess = w;
                    min0 = count.getOrDefault(w, 0);
                }
            x = master.guess(guess);
            List<String> wordlist2 = new ArrayList<String>();
            for (String w : wordlist)
                if (match(guess, w) == x)
                    wordlist2.add(w);
            wordlist = wordlist2.toArray(new String[0]);
        }
    }
  # interactive problem, Minimax

1153. String Transforms Into Another String
  - 1. one char cant transform to two char 2. if cycle e.g. a->b->c->a, it should have 1 more ununsed char
  -   public boolean canConvert(String str1, String str2) {
        // If same, return true
        if (str1.equals(str2)) return true;
        Map<Character, Character> map = new HashMap<>();
        for(int i=0;i<str1.length();i++) {
            // 1 char can't transform to 2 char
            if(map.containsKey(str1.charAt(i)) && map.get(str1.charAt(i)) != str2.charAt(i)) {
                return false;
            }
            map.put(str1.charAt(i), str2.charAt(i));
        }

        // if cycle e.g. a ->b ->c ->a, it has to have 1 extra ununsed char
        return new HashSet<>(map.values()).size()<26;
    }
  # Graph

1088. Confusing Number II
  - Calculate num of all combinations of 0, 1, 6, 8, 9 less than N. Then build Strobogrammatic numbers less than N. Total减去 Strobogrammatic numbers就是答案
  - 代码太多太复杂，自己看：https://leetcode.com/problems/confusing-number-ii/
  # Math, Backtracking














