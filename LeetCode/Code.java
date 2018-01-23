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





































