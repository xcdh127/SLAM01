/* This file is part of the SceneLib2 Project.
 * http://hanmekim.blogspot.com/2012/10/scenelib2-monoslam-open-source-library.html
 * https://github.com/hanmekim/SceneLib2
 *
 * Copyright (c) 2012 Hanme Kim (hanme.kim@gmail.com)
 *
 * SceneLib2 is an open-source C++ library for SLAM originally designed and
 * implemented by Andrew Davison and colleagues at the University of Oxford.
 *
 * I reimplemented his version with the following objectives;
 *  1. Understand his MonoSLAM algorithm in code level.
 *  2. Replace older libraries (i.e. VW34, GLOW, VNL, Pthread) with newer ones
 *     (Pangolin, Eigen3, Boost).
 *  3. Support USB camera instead of IEEE1394.
 *  4. Make it more portable and convenient by using CMake and git repository.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef KALMAN_H
#define KALMAN_H

#include "monoslam.h"

namespace SceneLib2 {

// Implements an Extended Kalman Filter.
class Kalman {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Kalman();
  ~Kalman();

  void KalmanFilterPredict(MonoSLAM *monoslam, Eigen::Vector3d &u);
  void KalmanFilterUpdate(MonoSLAM *monoslam);
};

} // namespace SceneLib2

#endif // KALMAN_H
//111. 二叉树的最小深度
/*给定一个二叉树，找出其最小深度。
最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
说明：叶子节点是指没有子节点的节点。
示例 1：
输入：root = [3,9,20,null,null,15,7]
输出：2
示例 2：
输入：root = [2,null,3,null,4,null,5,null,6]
输出：5
提示：
    树中节点数的范围在 [0, 105] 内
    -1000 <= Node.val <= 1000
*/
class Solution {
    public int minDepth(TreeNode root) {

    return recur(root);
    }
    
    public int recur(TreeNode root){
    
    if(root==null){
    return 0;
    }
    int left=recur(root.left);
    int right=recur(root.right);
    
    if(left==0 && right!=0){
    return 1+right;
    }
    else if(left!=0 && right==0){
    return 1+left;
    }
    return 1+Math.min(left,right);
    }
    
}
  
//222. 完全二叉树的节点个数
/*给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。
完全二叉树 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层，则该层包含 1~ 2h 个节点。
示例 1：
输入：root = [1,2,3,4,5,6]
输出：6
示例 2：
输入：root = []
输出：0
示例 3：
输入：root = [1]
输出：1
提示：
    树中节点的数目范围是[0, 5 * 104]
    0 <= Node.val <= 5 * 104
    题目数据保证输入的树是 完全二叉树
进阶：遍历树来统计节点是一种时间复杂度为 O(n) 的简单解决方案。你可以设计一个更快的算法吗？
*/
//没有利用完全二叉树的性质解题
class Solution {
    public int countNodes(TreeNode root) {
int total=0;
Stack<TreeNode> stack=new Stack<TreeNode>();
  TreeNode cur=root;
while(!stack.isEmpty() || cur!=null){
  while(cur!=null){
  stack.push(cur);
  cur=cur.left;
  }
  cur=stack.pop();
  total++;
  cur=cur.right;
  }
  return total;

    }
}
  //利用完全二叉树的性质解题
class Solution {
    public int countNodes(TreeNode root) {
  if(root==null){
  return 0;
  }
  //定义两个指针分别指向当前节点的左右孩子
  //当节点只有一个根节点时，此时两个孩子节点就是空，不会累加层数，
  TreeNode left=root.left;
  TreeNode right=root.right;
  //记录当前层数
  int leftHight=0;
  int rightHight=0;
  while(left!=null){
  left=left.left;
  leftHight++;
  }
  while(right!=null){
  right=right.right;
  rightHight++;
  }
  //当左子树和右子树的深度相同时，直接套入满树公式计算这颗子树的节点总数
  if(leftHight==rightHight){
  return (2<<leftHight)-1;
  }
  //如果当前节点的左右子树不是满树时，分别计算子树的节点树相加，再加上根节点，向上返回
  return countNodes(root.left)  + countNodes(root.right) +1;
  }
}
  //107. 二叉树的层序遍历 II
  /*给定一个二叉树，返回其节点值自底向上的层序遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
例如：
给定二叉树 [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
返回其自底向上的层序遍历为：
[
  [15,7],
  [9,20],
  [3]
]
*/
  class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
List<List<Integer>> res=new ArrayList<List<Integer>>();
  if(root==null){
  return res;
  }
  Queue<TreeNode> queue1=new LinkedList<TreeNode>();
  
  Queue<TreeNode> queue2=new LinkedList<TreeNode>();
  
  queue1.offer(root);
  List<Integer> list=new ArrayList<Integer>();
  
  while(!queue1.isEmpty()){
  TreeNode temp=queue1.poll();
  if(temp!=null){
  list.add(temp.val);
  
  if(temp.left!=null){
  queue2.offer(temp.left);
  }
  if(temp.right!=null){
  queue2.offer(temp.right);
  }
  }
  if(queue1.isEmpty()){
  res.add(new ArrayList<Integer>(list));
  queue1=queue2;
  queue2=new LinkedList<TreeNode>();
  list=new ArrayList<Integer>();
  }
  }
  Collections.reverse(res);
  return res;
  
    }
}
  
  
  //199. 二叉树的右视图
  /*给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
示例 1:
输入: [1,2,3,null,5,null,4]
输出: [1,3,4]
示例 2:
输入: [1,null,3]
输出: [1,3]
示例 3:
输入: []
输出: []
提示:
    二叉树的节点个数的范围是 [0,100]
    -100 <= Node.val <= 100 

*/
  class Solution {
    public List<Integer> rightSideView(TreeNode root) {
  Queue<TreeNode> queue1=new LinkedList<TreeNode>();
  
  Queue<TreeNode> queue2=new LinkedList<TreeNode>();
  
  List<Integer> list=new ArrayList<Integer>();
  
  queue1.offer(root);
  
  while(!queue1.isEmpty()){
  TreeNode temp=queue1.poll();
  if(temp!=null){
  if(temp.left!=null){
  queue2.offer(temp.left);
  }
  
  if(temp.right!=null){
  queue2.offer(temp.right);
  }
  
  if(queue1.isEmpty()){
  
  list.add(temp.val);
  queue1=queue2;
  queue2=new LinkedList<TreeNode>();
  }
  
  
  }
  
  }
  
  return list;
  

    }
}
  
  //637. 二叉树的层平均值
  /*给定一个非空二叉树, 返回一个由每层节点平均值组成的数组。
示例 1：
输入：
    3
   / \
  9  20
    /  \
   15   7
输出：[3, 14.5, 11]
解释：
第 0 层的平均值是 3 ,  第1层是 14.5 , 第2层是 11 。因此返回 [3, 14.5, 11] 。
提示：

    节点值的范围在32位有符号整数范围内。
*/
  class Solution {
    public List<Double> averageOfLevels(TreeNode root) {
  
  Queue<TreeNode> queue1=new LinkedList<TreeNode>();
  Queue<TreeNode> queue2=new LinkedList<TreeNode>();
  
  List<Double> res=new ArrayList<Double>();
  double sum=0;
  int count=0;
  
  queue1.offer(root);
  
  while(!queue1.isEmpty()){
  
  TreeNode temp=queue1.poll();
  
  if(temp!=null){
  sum+=temp.val;
  count++;
  if(temp.left!=null){
  queue2.offer(temp.left);
  }
  
  if(temp.right!=null){
  queue2.offer(temp.right);
  }
  }
  
  if(queue1.isEmpty()){
  
  res.add(sum/count);
  sum=0;
  count=0;
  queue1=queue2;
  queue2=new LinkedList<TreeNode>();
  }
  }
  return res;
  

    }
}
  
  //429. N 叉树的层序遍历
  /*给定一个 N 叉树，返回其节点值的层序遍历。（即从左到右，逐层遍历）。
树的序列化输入是用层序遍历，每组子节点都由 null 值分隔（参见示例）。
示例 1：
输入：root = [1,null,3,2,4,null,5,6]
输出：[[1],[3,2,4],[5,6]]
示例 2：
输入：root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
输出：[[1],[2,3,4,5],[6,7,8,9,10],[11,12,13],[14]]
提示：
    树的高度不会超过 1000
    树的节点总数在 [0, 10^4] 之间
*/
  /*
// Definition for a Node.
class Node {
    public int val;
    public List<Node> children;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, List<Node> _children) {
        val = _val;
        children = _children;
    }
};
*/
class Solution {
    public List<List<Integer>> levelOrder(Node root) {
    
  List<List<Integer>> res=new ArrayList<List<Integer>>();
  if(root==null){
  return res;
  }
  List<Integer> list=new ArrayList<Integer>();
  Queue<Node> queue1=new LinkedList<Node>();
  Queue<Node> queue2=new LinkedList<Node>();
  
  queue1.offer(root);
  while(!queue1.isEmpty()){
  
  Node temp=queue1.poll();
  list.add(temp.val);
  if(temp!=null){
  for(Node child : temp.children){
  queue2.offer(child);
  }
  
  if(queue1.isEmpty()){
  
  res.add(new ArrayList<Integer>(list));
  queue1=queue2;
  queue2=new LinkedList<Node>();
  list=new ArrayList<Integer>();
  }
  
  }
  
  }
  return res;
  
    }
}
 
 
 //515. 在每个树行中找最大值
 /*给定一棵二叉树的根节点 root ，请找出该二叉树中每一层的最大值。
示例1：
输入: root = [1,3,2,5,3,null,9]
输出: [1,3,9]
解释:
          1
         / \
        3   2
       / \   \  
      5   3   9 

示例2：
输入: root = [1,2,3]
输出: [1,3]
解释:
          1
         / \
        2   3

示例3：
输入: root = [1]
输出: [1]
示例4：
输入: root = [1,null,2]
输出: [1,2]
解释:      
           1 
            \
             2     

示例5：
输入: root = []
输出: []
提示：
    二叉树的节点个数的范围是 [0,104]
    -231 <= Node.val <= 231 - 1
*/
  class Solution {
    public List<Integer> largestValues(TreeNode root) {
 
 int maxValue=Integer.MIN_VALUE;
 Queue<TreeNode> queue1=new LinkedList<TreeNode>();
 Queue<TreeNode> queue2=new LinkedList<TreeNode>();
 
 List<Integer> res=new ArrayList<Integer>();
 if(root==null){
 return res;
 }
 
 queue1.offer(root);
 while(!queue1.isEmpty()){
 
 TreeNode temp=queue1.poll();
 
 if(temp!=null){
  maxValue= Math.max(maxValue,temp.val);
 
 if(temp.left!=null){
 queue2.offer(temp.left);
 }
 
 if(temp.right!=null){
 queue2.offer(temp.right);
 }
 }

 if(queue1.isEmpty()){
 res.add(maxValue);
 queue1=queue2;
 queue2=new LinkedList<TreeNode>();
 maxValue=Integer.MIN_VALUE;
 }
 
 }
 
 return res;
 

    }
}
 
 //116. 填充每个节点的下一个右侧节点指针
 /*给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
初始状态下，所有 next 指针都被设置为 NULL。
进阶：
    你只能使用常量级额外空间。
    使用递归解题也符合要求，本题中递归程序占用的栈空间不算做额外的空间复杂度。
示例：
输入：root = [1,2,3,4,5,6,7]
输出：[1,#,2,3,#,4,5,6,7,#]
解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。序列化的输出按层序遍历排列，同一层节点由 next 指针连接，'#' 标志着每一层的结束。
提示：
    树中节点的数量少于 4096
    -1000 <= node.val <= 1000
 */
 class Solution {
    public Node connect(Node root) {
  Queue<Node> queue1=new LinkedList<Node>();
  Queue<Node> queue2=new LinkedList<Node>();
  Node prev=null;
  queue1.offer(root);
  while(!queue1.isEmpty()){
 
 Node temp=queue1.poll();
 
 if(temp!=null){
 if(prev!=null){
 prev.next=temp;
 }
 prev=temp;
 
 if(temp.left!=null){
 queue2.offer(temp.left);
 }
 if(temp.right!=null){
 queue2.offer(temp.right);
 }
 
 }
 
 if(queue1.isEmpty()){
 prev=null;
 queue1=queue2;
 queue2=new LinkedList<Node>();
 }
 
 
 }
 
 return root;
 
 
        
    }
}
 
 //117. 填充每个节点的下一个右侧节点指针 II
 /*给定一个二叉树
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
初始状态下，所有 next 指针都被设置为 NULL。
进阶：

    你只能使用常量级额外空间。
    使用递归解题也符合要求，本题中递归程序占用的栈空间不算做额外的空间复杂度。
示例：
输入：root = [1,2,3,4,5,null,7]
输出：[1,#,2,3,#,4,5,7,#]
解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。序列化输出按层序遍历顺序（由 next 指针连接），'#' 表示每层的末尾。
提示：
    树中的节点数小于 6000
    -100 <= node.val <= 100
*/
 class Solution {
    public Node connect(Node root) {
  Queue<Node> queue1=new LinkedList<Node>();
  Queue<Node> queue2=new LinkedList<Node>();
  Node prev=null;
  queue1.offer(root);
  while(!queue1.isEmpty()){
 
 Node temp=queue1.poll();
 
 if(temp!=null){
 if(prev!=null){
 prev.next=temp;
 }
 prev=temp;
 
 if(temp.left!=null){
 queue2.offer(temp.left);
 }
 if(temp.right!=null){
 queue2.offer(temp.right);
 }
 
 }
 
 if(queue1.isEmpty()){
 prev=null;
 queue1=queue2;
 queue2=new LinkedList<Node>();
 }
 
 
 }
 
 return root;
 
 
        
    }
}
 //226. 翻转二叉树
 /*翻转一棵二叉树。

示例：
输入：
     4
   /   \
  2     7
 / \   / \
1   3 6   9
输出：

     4
   /   \
  7     2
 / \   / \
9   6 3   1
备注:
这个问题是受到 Max Howell 的 原问题 启发的 ：

谷歌：我们90％的工程师使用您编写的软件(Homebrew)，但是您却无法在面试时在白板上写出翻转二叉树这道题，这太糟糕了。
*/
 class Solution {
    public TreeNode invertTree(TreeNode root) {
 recur(root);
 return root;
    }
 
 public void recur(TreeNode root){
 
 if(root==null){
 return;
 }
 TreeNode left=root.left;
 root.left=root.right;
 root.right=left;
 recur(root.left);
 recur(root.right);
 }
 
}
 //589. N 叉树的前序遍历
 /*给定一个 N 叉树，返回其节点值的 前序遍历 。
N 叉树 在输入中按层序遍历进行序列化表示，每组子节点由空值 null 分隔（请参见示例）。
进阶：
递归法很简单，你可以使用迭代法完成此题吗?
示例 1：
输入：root = [1,null,3,2,4,null,5,6]
输出：[1,3,5,6,2,4]
示例 2：
输入：root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
输出：[1,2,3,6,7,11,14,4,8,12,5,9,13,10]
提示：
N 叉树的高度小于或等于 1000
节点总数在范围 [0, 10^4] 内
*/
 /*
// Definition for a Node.
class Node {
    public int val;
    public List<Node> children;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, List<Node> _children) {
        val = _val;
        children = _children;
    }
};
*/
 //递归解法
 class Solution {
    List<Integer> res=new ArrayList<Integer>();
    public List<Integer> preorder(Node root) {
        recur(root);
        return res;
    }
 
 public void recur(Node root){
 if(root==null){
 return ;
 }
 
 res.add(root.val);
 for(Node child : root.children){
 recur(child);
 }
 }
}
 //迭代解法
 class Solution {
    List<Integer> res=new ArrayList<Integer>();
 
    public List<Integer> preorder(Node root) {
 
 if(root==null){
 return res;
 }
        Deque<Node> stack=new LinkedList<Node>();
        stack.offer(root);
 while(!stack.isEmpty()){
 Node temp=stack.pollLast();
 //前
 res.add(temp.val);
 //将节点的孩子们逆个序,最后添加进去的节点先遍历到
 Collections.reverse(temp.children);
 for(Node child : temp.children){
 stack.addLast(child);
 }
 }
 return res;
    }
}
 方法：迭代
由于递归实现 N 叉树的前序遍历较为简单，因此我们只讲解如何使用迭代的方法得到 N 叉树的前序遍历。

我们使用栈来帮助我们得到前序遍历，需要保证栈顶的节点就是我们当前遍历到的节点。

我们首先把根节点入栈，因为根节点是前序遍历中的第一个节点。随后每次我们从栈顶取出一个节点 u，它是我们当前遍历到的节点，并把 u 的所有子节点逆序推入栈中。例如 u 的子节点从左到右为 v1, v2, v3，那么推入栈的顺序应当为 v3, v2, v1，这样就保证了下一个遍历到的节点（即 u 的第一个子节点 v1）出现在栈顶的位置。
class Solution {
    public List<Integer> preorder(Node root) {
        LinkedList<Integer> output = new LinkedList<>();
        if (root == null) {
            return output;
        }

        LinkedList<Node> stack = new LinkedList<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.pollLast();
            output.add(node.val);
            Collections.reverse(node.children);
            for (Node item : node.children) {
                stack.add(item);
            }
        }
        return output;
    }
}
 
 //590. N 叉树的后序遍历
 /*给定一个 N 叉树，返回其节点值的 后序遍历 。
N 叉树 在输入中按层序遍历进行序列化表示，每组子节点由空值 null 分隔（请参见示例）。
进阶：
递归法很简单，你可以使用迭代法完成此题吗?
示例 1：
输入：root = [1,null,3,2,4,null,5,6]
输出：[5,6,3,2,4,1]
示例 2：
输入：root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
输出：[2,6,14,11,7,3,12,8,4,13,9,10,5,1]
提示：
N 叉树的高度小于或等于 1000
节点总数在范围 [0, 10^4] 内
*/
 class Solution {
    List<Integer> res=new ArrayList<Integer>();
    public List<Integer> postorder(Node root) {
        recur(root);
        return res;
    }
 public void recur(Node root){
 
 if(root==null){
 return;
 }
 for(Node child : root.children){
 recur(child);
 }
 res.add(root.val);
 }
}
 
class Solution {
    public List<Integer> postorder(Node root) {
 List<Integer> res=new ArrayList<Integer>();
 if(root==null){
 return res;
 }
 Stack<Node> stack=new Stack<Node>();
 
 stack.push(root);
 
 while(!stack.isEmpty()){
 
 Node temp=stack.pop();
 //根
 res.add(temp.val);
 //右最后进入栈，最先出栈->(根->右->左)
 for(Node child :temp.children){
 stack.push(child);
 }
 }
 //反转结果,左->右->根
 Collections.reverse(res);
 return res;
    }
}
                                          

/* This file is part of the SceneLib2 Project.
 * http://hanmekim.blogspot.com/2012/10/scenelib2-monoslam-open-source-library.html
 * https://github.com/hanmekim/SceneLib2
 *
 * Copyright (c) 2012 Hanme Kim (hanme.kim@gmail.com)
 *
 * SceneLib2 is an open-source C++ library for SLAM originally designed and
 * implemented by Andrew Davison and colleagues at the University of Oxford.
 *
 * I reimplemented his version with the following objectives;
 *  1. Understand his MonoSLAM algorithm in code level.
 *  2. Replace older libraries (i.e. VW34, GLOW, VNL, Pthread) with newer ones
 *     (Pangolin, Eigen3, Boost).
 *  3. Support USB camera instead of IEEE1394.
 *  4. Make it more portable and convenient by using CMake and git repository.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */



