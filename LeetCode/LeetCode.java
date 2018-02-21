/* Array, HashTable */

1 Two Sum
  - Use HashMap to track each visited num
  # HashMap

2 Add Two Numbers
  ? iteration or recursion
  - Use iteration rather than recursion because of the corner case [9], [9,9]. It's hard to deal with this situation with recursion.
  - When using pointer to deal with each node in the linked list. Use a dummy root node to connect the real root node. At the end, return root.next.
  # Linked List

126 Single Number
    - bit manipulation. XOR

4 Median of Two Sorted Arrays
  - attention to corner cases
  - (https://www.youtube.com/watch?v=do7ibYtv5nk)
  # binary search

461 Hamming Distance
  - int_num % 2 == int_num >> 1
  # bit manipulation
