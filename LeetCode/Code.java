/* Heap Sort */
// complete binary tree structure
public void removeMax(int[] arr, int index) {
    int size = arr.length;
    int left = 2 * index + 1;
    int right = left + 1;
    if( left >= size) {
        return;
    }

    if (left == size - 1) {
        exchange (arr, index, left);
        return;
    }

    if(arr[left] < arr[right]) {
        exchange(index, right);
        removeMax(arr, right);
    } else {
        exchange(index, left);
        removeMax(arr, left);
    }
}

public void bottomUp(int[] arr, int index) {
    if(index - 1 < 0) {
        return;
    }

    int parent = (index - 1)/2;
    if (arr[parent] < arr[index]) {
        int temp = arr[index];
        arr[index] = arr[parent];
        arr[parent] = temp;
        bottomUp(arr, parent);
    } else {
        return;
    }
}
/* Insertion Sort */
public void insertionSort(int[] arr) {
    if(arr == null) {
        return;
    }

    for(int i=0; i < arr.length;i++) {
        int j=i-1;
        int value = arr[i]
        while(j>=0) {
            if(arr[j]>value) {
                arr[j] = arr[j+1];
                j--;
            } else {
                break;
            }
        }
        arr[j+1] = value;
    }
}

/* Selection Sort */
public void selectionSort(int[] arr) {
    if(arr == null) {
        return;
    }

    for (int i = 0; i<arr.length-1;i++) {
        int minIndex = i;
        for(int j = i+1;j<arr.length;j++) {
            if(arr[minIndex] > arr[j] ) {
                minIndex = j;
            }
        }
        int temp = arr[i];
        arr[i] = arr[minIndex];
        arr[minIndex] = temp;
    }
}

/* Quick Sort */
public void partition(int[] arr, int l, int r) {
    int pivot = arr[r];
    while(l != r) {
        if(arr[l] <= pivot) {
            l++;
        } else {
            int temp = arr[l];
            arr[l] = arr[r];
            arr[r] = temp;
            r--;
        }
    }
    return l;
}

/* Merge Sort LinkedList */
public Node sortList(Node head) {
    if(head == null || head.next == null){
        return head;
    }
    Node middle = getMiddle(head);
    Node next = middle.next;
    middle.next = null;
    return merge(sortList(head), sortList(next));
}

public Node getMiddle(Node node) {
    Node slow = node;
    Node fast  = node;
    while(fast.next != null && fast.next.next !=null) {
        slow = node.next;
        fast = fast.next.next;
    }
    return slow;
}

public merge(Node n1, Node n2) {
    Node dummy = new Node(0);
    Node node = dummy;
    while(n1 != null && n2 != null) {
        if(n1.value < n2.value) {
            node.next = n1;
            n1 = n1.next;
        } else {
            node.next = n2;
            n2 = n2.next;
        }
        node = node.next;
    }
    node.next = n1==null?n2:n1;
    return dummy.next;
}

/* Couting Sort*/
public void coutingSort(String s) {
    int[] arr = new int[26];
    for(char c : s.toCharArray) {
        arr[c - 'a']++;
    }
    String s = "";
    for(int i=0;i<26;i++) {
        if(arr[i] !=0) {
            while(arr[i] != 0) {
                s+=(char)(i+'a');
                arr[i]--;
            }
        }
    }
}

/* Union Find */
public int find(int[] arr, int index) {
    while(arr[index] != index) {
        arr[index] = arr[arr[index]];
        index = arr[index];
    }
    return index;
}

public int union(int[] arr, int[] rank, int i, int j) {
    int rootI = find(arr, i);
    int rootJ = find(arr, j);
    if (rootI == rootJ) return;

    if(rank[rootI] < rank[rootJ]) arr[rootI] = rootJ;
    else if(rank[rootI]) > rank[rootJ]) arr[rootJ] = rootI;
    else {
        arr[rootI] = rootJ;
        rank[rootJ] ++;
    }
}

public int find(int[] arr, int index) {
    if(arr[index] == index) {
        return index;
    }
    arr[index] = find(arr, arr[index])
    return arr[index];
}

pubic int union(int[] arr, int i, int j) {
    int rootI = find(arr, i);
    int rootJ = find(arr, j);

    if(rootI != rootJ) {
      arr[rootI] = root];
    }
}

// Segment Tree
public int construct(int[] arr, int[] tree, int l, int r, int index) {
    if (l == r) {
        tree[index] = arr[l];
        return tree[index];
    }
    int mid = mid(l, r);
    tree[index] = construct(arr, tree, l, m, 2*index+1) + construct(arr, tree, m+1, r, 2*index+2);
    return tree[index];
}

public int sum(int[] tree, int tl, int tr, int l, int r, int index) {
    if(tr < l || r < tl) {
        return 0;
    }
    if (l <= tl && r >= tr) {
        return tree[index];
    }
    int mid = mid(tl, tr);
    return sum(tree, tl, m , l , r, 2*index+1) + sum(tree, m+1, tr, l, r, 2*index+2);
}

// fibonacci time: O(2^n)
public int fibonacci(int n) {
    if(n<3) {
        return 1;
    }
    return fibonacci(n-1) + fibonacci(n-2);
}

// fibonacci time: O(n)
public int fibonacci(int n) {
    if(n < 3) return 1;
    int f1 = 1;
    int f2 = 1;
    int f3 = 0;

    for(int i =2;i<n;i++) {
        f3 = f1 + f2;
        f1 = f2;
        f2 = f3;
    }
    return f3;
}

// BFS: Serialize and desearialize binary tree
    public String serialize(TreeNode root) {
        if(root == null) return "";
        StringBuilder sb = new StringBuilder();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node == null) {
                sb.append("null,");
            } else {
                sb.append(node.val + ",");
                queue.offer(node.left);
                queue.offer(node.right);
            }
        }
        return sb.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if(data == "") return null;
        String[] arr = data.split(",");
        TreeNode root = new TreeNode(Integer.parseInt(arr[0]));
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        for(int i=1;i<arr.length;i++) {
            TreeNode node = queue.poll();
            if(!arr[i].equals("null")) {
                node.left = new TreeNode(Integer.parseInt(arr[i]));
                queue.offer(node.left);
            }
            if(!arr[++i].equals("null")) {
                node.right = new TreeNode(Integer.parseInt(arr[i]));
                queue.offer(node.right);
            }
        }
        return root;
    }

// PreOrder: Serialize and desearialize binary tree
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        sBuild(root, sb);
        return sb.toString();
    }

    private void sBuild(TreeNode node, StringBuilder sb) {
        if(node == null) {
            sb.append("null,");
        } else {
            sb.append(node.val + ",");
            sBuild(node.left, sb);
            sBuild(node.right, sb);
        }
    }

    public TreeNode deserialize(String data) {
        Deque<String> nodes = new LinkedList<>();                         nodes.addAll(Arrays.asList(data.split(",")));
        return dBuild(nodes);
    }

    private TreeNode dBuild(Deque<String> nodes) {
        String n = nodes.remove();
        if(n.equals("null")) return null;
        TreeNode node  = new TreeNode(Integer.parseInt(n));
        node.left = dBuild(nodes);
        node.right= dBuild(nodes);
        return node;
    }






























