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
 
 //剑指 Offer II 079. 所有子集
 /*给定一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
示例 1：
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
示例 2：
输入：nums = [0]
输出：[[],[0]]
提示：
1 <= nums.length <= 10
-10 <= nums[i] <= 10
nums 中的所有元素 互不相同
*/
 class Solution {
    public List<List<Integer>> subsets(int[] nums) {
List<List<Integer>> res=new ArrayList<List<Integer>>();
LinkedList<Integer> subset=new LinkedList<Integer>(); 
 
 recur(nums,res,subset,0);
 return res;
    }
 
 public void recur(int[] nums,List<List<Integer>> res,LinkedList<Integer> subset,int index){
 
 if(index==nums.length){
 res.add(new LinkedList<Integer>(subset));
 }
 else if(index<nums.length){
 recur(nums,res,subset,index+1);
 
 subset.add(nums[index]);
recur(nums,res,subset,index+1);
subset.removeLast();                             
 }
 
 }
 
}
                             
                             
//剑指 Offer II 080. 含有 k 个元素的组合
/*给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
示例 1:
输入: n = 4, k = 2
输出:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
示例 2:
输入: n = 1, k = 1
输出: [[1]]
提示:
1 <= n <= 20
1 <= k <= n
*/                             
class Solution {
    public List<List<Integer>> combine(int n, int k) {
List<List<Integer>> res=new ArrayList<List<Integer>>();
 LinkedList<Integer> subset=new LinkedList<Integer>();
 recur(n,k,res,subset,1);
 return res;
    }
public void recur(int n,int k,List<List<Integer>> res,LinkedList<Integer> subset,int index){
 
 if(index==n+1){
 
 if(subset.size()==k){
 
 res.add(new LinkedList<Integer>(subset));
 }
 }
 else if(index<n+1){
 recur(n,k,res,subset,index+1);
subset.add(index);                     
recur(n,k,res,subset,index+1);                     
subset.removeLast(); 
 }
 } 
}                             
//剑指 Offer II 081. 允许重复选择元素的组合
/*给定一个无重复元素的正整数数组 candidates 和一个正整数 target ，找出 candidates 中所有可以使数字和为目标数 target 的唯一组合。
candidates 中的数字可以无限制重复被选取。如果至少一个所选数字数量不同，则两种组合是唯一的。 
对于给定的输入，保证和为 target 的唯一组合数少于 150 个。
示例 1：
输入: candidates = [2,3,6,7], target = 7
输出: [[7],[2,2,3]]
示例 2：
输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]
示例 3：
输入: candidates = [2], target = 1
输出: []
示例 4：
输入: candidates = [1], target = 1
输出: [[1]]
示例 5：
输入: candidates = [1], target = 2
输出: [[1,1]]
提示：
1 <= candidates.length <= 30
1 <= candidates[i] <= 200
candidate 中的每个元素都是独一无二的。
1 <= target <= 500
*/                     
class Solution {
int sum=0;                 
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
List<List<Integer>> res=new ArrayList<List<Integer>>();
 LinkedList<Integer> subset=new LinkedList<Integer>();
 recur(candidates,target,res,subset,0);
 return res;
    }
 
 public void recur(int[] candidates,int target,List<List<Integer>> res,LinkedList<Integer> subset,int index){
 
 if(index==candidates.length){
 return ;
 }
 if(sum==target){
 res.add(new LinkedList<Integer>(subset));
 return;
 }
 if(sum<target && index<candidates.length){
 recur(candidates,target,res,subset,index+1);
 
 subset.add(candidates[index]);
 sum+=candidates[index];
 recur(candidates,target,res,subset,index);
 subset.removeLast();
 sum-=candidates[index];
 }
 }
}
 //794. 有效的井字游戏
 /*用字符串数组作为井字游戏的游戏板 board。当且仅当在井字游戏过程中，玩家有可能将字符放置成游戏板所显示的状态时，才返回 true。
该游戏板是一个 3 x 3 数组，由字符 " "，"X" 和 "O" 组成。字符 " " 代表一个空位。
以下是井字游戏的规则：
玩家轮流将字符放入空位（" "）中。
第一个玩家总是放字符 “X”，且第二个玩家总是放字符 “O”。
“X” 和 “O” 只允许放置在空位中，不允许对已放有字符的位置进行填充。
当有 3 个相同（且非空）的字符填充任何行、列或对角线时，游戏结束。
当所有位置非空时，也算为游戏结束。
如果游戏结束，玩家不允许再放置字符。
示例 1:
输入: board = ["O  ", "   ", "   "]
输出: false
解释: 第一个玩家总是放置“X”。
示例 2:
输入: board = ["XOX", " X ", "   "]
输出: false
解释: 玩家应该是轮流放置的。
示例 3:
输入: board = ["XXX", "   ", "OOO"]
输出: false
示例 4:
输入: board = ["XOX", "O O", "XOX"]
  ["XOX",
   "O O", 
   "XOX"]
输出: true
说明:
游戏板 board 是长度为 3 的字符串数组，其中每个字符串 board[i] 的长度为 3。
 board[i][j] 是集合 {" ", "X", "O"} 中的一个字符。
*/
 class Solution {
    public boolean validTicTacToe(String[] board) {
int n=board.length;
 int countOfX=0;
 int countOfO=0;
 int[] row=new int[n];
 int[] col=new int[n];
 int duiJiao1=0;
 int duiJiao2=0;
 for(int i=0;i<n;i++){
for(int j=0;j<n;j++){
 char character=board[i].charAt(j);
 //处理' '
 if(character==' '){
 row[i]+=0;
 col[j]+=0;
 if(i==j){
 duiJiao1+=0;
 }
 if(i==n-j-1){
 duiJiao2+=0;
 }
 }
 //处理'O'
 else if(character=='O'){
 countOfO++;
 row[i]+=1;
 col[j]+=1;
 if(i==j){
 duiJiao1+=1;
 }
 if(i==n-j-1){
 duiJiao2+=1;
 }
 }
 //处理'X'
 else if(character=='X'){
 countOfX++;
 row[i]-=1;
 col[j]-=1;
 if(i==j){
 duiJiao1-=1;
 }
 if(i==n-j-1){
 duiJiao2-=1;
 }
 }
 }                       
}
 if(countOfO>countOfX || countOfX-countOfO>1){
 return false;
 }
 
 if(!isValid(row,col,duiJiao1,duiJiao2)){
 return false;
 }
 return true;
 
    }
 
 public boolean isValid(int[] row,int[] col,int duiJiao1,int duiJiao2){
 
 int sum1=0;
 for(int num : row){
 sum1+=num;
 }
 
 int sum2=0;
 for(int num : col){
 sum2+=num;
 }
 int count=0;
 int[] arr=new int[]{sum1,sum2,duiJiao1,duiJiao2};
 for(int num : arr){
 if(num==3 || num==-3){
 count++;
 }
 }
 return count<=1;
 }
 
}
                 
                 
class Solution {
    public boolean validTicTacToe(String[] board) {
char[][] ch=new char[3][3];
int countOfO=0;
int countOfX=0;                 
for(int i=0;i<board.length;i++){
 for(int j=0;j<board.length;j++){
 
if(board[i].charAt(j)=='X'){
 countOfX++;                                 
 }         
if(board[i].charAt(j)=='O'){
 countOfO++;                                 
 }

ch[i][j]=board[i].charAt(j);                                  
                                  
                                
 }
                   
 } 
                                                                    
boolean a=check(ch,'X');
boolean b=check(ch,'O');                                  
if(countOfX<countOfO || countOfX-countOfO>1){
 return false;
 }                                  
if(a && countOfX<=countOfO){
 return false;                           
 }                                  
 
if(b && countOfO!=countOfX){
  return false;                              
 }                            

if(a && b){
   return false;                                     
}    
        return true;          
    }
                                  
 public boolean check(char[][] ch,char c){
for(int i=0;i<3;i++){

 if(ch[i][0]==c && ch[i][1]==c && ch[i][2]==c){
return true;                       
  }                      
if(ch[0][i]==c && ch[1][i]==c && ch[2][i]==c){
 return true;
 }
 
 }   
 
 boolean a=true;
 boolean b=true;
 for(int i=0;i<3;i++){
 for(int j=0;j<3;j++){
 
 if(i==j){
 a=a&ch[i][j]==c;
 }
 if(i+j==2){
 b=b&ch[i][j]==c;
 }
 
 }                      
 }
 return a || b;
                                  
  }                                
                            
}                 
              
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 给定的棋盘大小固定，对于无效情况进行分情况讨论即可：

由于 X 先手，O 后手，两者轮流下子。因此 O 的数量不会超过 X，且两者数量差不会超过 1，否则为无效局面；
若局面是 X 获胜，导致该局面的最后一个子必然是 X此时必然有 X 数量大于 O（X 为先手），否则为无效局面；
若局面是 O 获胜，导致该局面的最后一个子必然是 O，此时必然有 X 数量等于 O（X 为先手），否则为无效局面；
局面中不可能出现两者同时赢（其中一方赢后，游戏结束）。
class Solution {
    public boolean validTicTacToe(String[] board) {
        //用于将字符串放进字符数组，便于统计字符出现的位置
        char[][] cs = new char[3][3];
        int x = 0, o = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                char c = board[i].charAt(j);
                if (c == 'X') x++;
                else if (c == 'O') o++;
                cs[i][j] = c;
            }
        }
       //判断X字符是不是胜出         判断O字符是不是胜出                             
        boolean a = check(cs, 'X'), b = check(cs, 'O');
       //由于X总是先手，O后手，交替出牌，所以O的数量小于等于X的数量，但是由于交替出牌，所以X的数量不会大于O超过1个子               
        if (o > x || x - o > 1) return false;
      //X获得胜利，并且X的子数小于等于O,不可能X想获得胜利，必须比O的数量多1
        if (a && x <= o) return false;        
      //O获得胜利，并且O的子数不等于X,不可能，O想要获得胜利，必须和X的数量相同                       
        if (b && o != x) return false;
      //a和b同时获得胜利，不可能,当其中一个人获得胜利后比赛就结束了                
        if (a && b) return false;
        return true;
    }
    boolean check(char[][] cs, char c) {
        for (int i = 0; i < 3; i++) {
            if (cs[i][0] == c && cs[i][1] == c && cs[i][2] == c) return true;
            if (cs[0][i] == c && cs[1][i] == c && cs[2][i] == c) return true;
        }
//确定对角线上的字符情况，结果将给出给定符号'O'或者'X'是不是填满了整个对角线
 //首先假定字符'O'和字符'X'都能够填满整个对角线(只要主对角线上有一个字符不相同就会返回false)
        boolean a = true, b = true;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
 //记录主对角线上的字符情况
                if (i == j) a &= cs[i][j] == c;
 //记录副对角线上的字符情况(只要副对角线上有一个字符不相同就会返回false)
                if (i + j == 2) b &= cs[i][j] == c;
            }
        }
 //只要主对线或者副对角线上有3个相同的字符就返回true
        return a || b;
    }
}

//113. 路径总和 II
 /*给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
叶子节点 是指没有子节点的节点。
示例 1：
输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：[[5,4,11,2],[5,8,4,5]]
示例 2：
输入：root = [1,2,3], targetSum = 5
输出：[]
示例 3：
输入：root = [1,2], targetSum = 0
输出：[]
提示：
树中节点总数在范围 [0, 5000] 内
-1000 <= Node.val <= 1000
-1000 <= targetSum <= 1000
*/
 class Solution {
 int sum=0;
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
List<List<Integer>> res=new ArrayList<List<Integer>>();
 LinkedList<Integer> path=new LinkedList<Integer>();
 recur(res,path,root,targetSum);
 return res;
    }
 
 public void recur(List<List<Integer>> res,LinkedList<Integer> path,TreeNode root,int targetSum){
 if(root==null){
 return ;
 }
 sum+=root.val;
 path.add(root.val);
 if(root.left==null && root.right==null && sum==targetSum){
 res.add(new LinkedList<Integer>(path)); 
 }
 recur(res,path,root.left,targetSum);
 recur(res,path,root.right,targetSum);
 path.removeLast();
 sum-=root.val;
 }
}
 //112. 路径总和
 /*给你二叉树的根节点 root 和一个表示目标和的整数 targetSum 。判断该树中是否存在 
 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。如果存在，返回 true ；否则，返回 false 。
叶子节点 是指没有子节点的节点。
示例 1：
输入：root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
输出：true
解释：等于目标和的根节点到叶节点路径如上图所示。
示例 2：
输入：root = [1,2,3], targetSum = 5
输出：false
解释：树中存在两条根节点到叶子节点的路径：
(1 --> 2): 和为 3
(1 --> 3): 和为 4
不存在 sum = 5 的根节点到叶子节点的路径。
示例 3：
输入：root = [], targetSum = 0
输出：false
解释：由于树是空的，所以不存在根节点到叶子节点的路径。
提示：
树中节点的数目在范围 [0, 5000] 内
-1000 <= Node.val <= 1000
-1000 <= targetSum <= 1000
*/
 class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {

 return recur(root,targetSum);
    }
 public boolean recur(TreeNode root,int targetSum){
 if(root==null){
 return false;
 }
 if(root.left==null && root.right==null){
 return root.val==targetSum;
 }
 
 return recur(root.left,targetSum-root.val) || recur(root.right,targetSum-root.val);
 
 }
}
 //106. 从中序与后序遍历序列构造二叉树
 /*根据一棵树的中序遍历与后序遍历构造二叉树。
注意:
你可以假设树中没有重复的元素。
例如，给出
中序遍历 inorder = [9(l),3(i),15,20,7(r)]
后序遍历 postorder = [9(root-(r-i)-1),15,7,20(root-1),3(root)]
返回如下的二叉树
    3
   / \
  9  20
    /  \
   15   7
*/
 //中序: 左 根 右
 //后序: 左 右 根
 //中序遍历 inorder = [9(l),3(i),15,20,7(r)]
 //后序遍历 postorder = [9(root-(r-i)-1),15,7,20(root-1),3(root)]
 class Solution {
 Map<Integer,Integer> map;
 int[] postorder;
    public TreeNode buildTree(int[] inorder, int[] postorder) {
map=new HashMap<Integer,Integer>();
 this.postorder=postorder;
 for(int i=0;i<inorder.length;i++){
 map.put(inorder[i],i);                                   
 }
 return recur(postorder,inorder.length-1,0,inorder.length-1);
 
 
    }
 
 public TreeNode recur(int[] postorder,int root,int l,int r){
 if(l>r){
 return null;
 }
 int i=map.get(postorder[root]);
 TreeNode node=new TreeNode(postorder[root]);
 node.left=recur(postorder,root-(r-i)-1,l,i-1);
 node.right=recur(postorder,root-1,i+1,r);
 return node;
}
 
 
}
 //105. 从前序与中序遍历序列构造二叉树
 /*给定一棵树的前序遍历 preorder 与中序遍历  inorder。请构造二叉树并返回其根节点。
示例 1:
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
示例 2:
Input: preorder = [-1], inorder = [-1]
Output: [-1]
提示:
1 <= preorder.length <= 3000
inorder.length == preorder.length
-3000 <= preorder[i], inorder[i] <= 3000
preorder 和 inorder 均无重复元素
inorder 均出现在 preorder
preorder 保证为二叉树的前序遍历序列
inorder 保证为二叉树的中序遍历序列
*/
 //前序: 根 左 右 preorder = [3(root),9(root+1),20(root+i-l+1),15,7], 
 //后序: 左 根 右 inorder = [9(l),3(i),15,20,7(r)]
 class Solution {
 Map<Integer,Integer> map;
 int[] preorder;
    public TreeNode buildTree(int[] preorder, int[] inorder) {
 map=new HashMap<Integer,Integer>();
 this.preorder=preorder;
 for(int i=0;i<inorder.length;i++){
  map.put(inorder[i],i);                                   
 }
  return recur(preorder,0,0,inorder.length-1);                                  
                                 
    }
                                    
 public TreeNode recur(int[] preorder,int root,int l,int r){
 if(l>r){
 return null;
 }                                   
 
 int i=map.get(preorder[root]);
 TreeNode node=new TreeNode(preorder[root]);
 node.left=recur(preorder,root+1,l,i-1);
 node.right=recur(preorder,root+i-l+1,i+1,r);
 return node;
 }                                   
                                    
                                    
}
 //654. 最大二叉树
 /*给定一个不含重复元素的整数数组 nums 。一个以此数组直接递归构建的 最大二叉树 定义如下：
二叉树的根是数组 nums 中的最大元素。
左子树是通过数组中 最大值左边部分 递归构造出的最大二叉树。
右子树是通过数组中 最大值右边部分 递归构造出的最大二叉树。
返回有给定数组 nums 构建的 最大二叉树 。
示例 1：
输入：nums = [3,2,1,6,0,5]
输出：[6,3,5,null,2,0,null,null,1]
解释：递归调用如下所示：
- [3,2,1,6,0,5] 中的最大值是 6 ，左边部分是 [3,2,1] ，右边部分是 [0,5] 。
    - [3,2,1] 中的最大值是 3 ，左边部分是 [] ，右边部分是 [2,1] 。
        - 空数组，无子节点。
        - [2,1] 中的最大值是 2 ，左边部分是 [] ，右边部分是 [1] 。
            - 空数组，无子节点。
            - 只有一个元素，所以子节点是一个值为 1 的节点。
    - [0,5] 中的最大值是 5 ，左边部分是 [0] ，右边部分是 [] 。
        - 只有一个元素，所以子节点是一个值为 0 的节点。
        - 空数组，无子节点。
示例 2：
输入：nums = [3,2,1]
输出：[3,null,2,null,1]
提示：
1 <= nums.length <= 1000
0 <= nums[i] <= 1000
nums 中的所有整数 互不相同
*/
 class Solution {
    public TreeNode constructMaximumBinaryTree(int[] nums) {
int n=nums.length;
 return recur(nums,0,n-1);
    }
public TreeNode recur(int[] nums,int l,int r){
 if(l>r){
 return null;
 }
 //当子区间内只有一个节点时,将当前数字作为根节点返回
 if(l==r){
 return new TreeNode(nums[l]);
 }
 int maxValue=0;
 int maxIndex=0;
 //寻找最大值，得到最大值在数组中的下标
 for(int i=l;i<=r;i++){
 if(nums[i]>maxValue){
 maxIndex=i;
 maxValue=nums[i];
 }
 }
 TreeNode node=new TreeNode(nums[maxIndex]);
 node.left=recur(nums,l,maxIndex-1);
 node.right=recur(nums,maxIndex+1,r);
 return node;
 } 
 
}
 
 class Solution {
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return dc(nums, 0, nums.length - 1);
    }
    private TreeNode dc(int[] nums, int start, int end) {
        //判断区间内没有数字，返回null
        if (start > end) {
            return null;
        }
        //区间内只有一个数组，返回这个数字作为节点
        if (start == end) {
            return new TreeNode(nums[start]);
        }
        //寻找区间内最大值下标，并传入之后的两个递归
        int max = findMaxIndex(nums, start, end);
        //将下标对应的最大值构建为节点
        TreeNode node = new TreeNode(nums[max]);
        //体现分治思想的两个递归
        node.left = dc(nums, start, max - 1);
        node.right = dc(nums, max + 1, end);
        //返回上面构建的最大值节点
        return node;
    }
    private int findMaxIndex(int[] nums, int start, int end) {
        //假定最大值
        int maxindex = start;
        for (int i = start + 1; i <= end; i++) {
            //有更大的数值，更新maxindex的值
            if (nums[i] > nums[maxindex]) {
                maxindex = i;
            }
        }
        return maxindex;
    }
}
//617. 合并二叉树
 /*给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。
你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。
示例 1:
输入: 
	Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
输出: 
合并后的树:
	     3
	    / \
	   4   5
	  / \   \ 
	 5   4   7
注意: 合并必须从两个树的根节点开始。
*/
 class Solution {
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {

 return recur(root1,root2);
    }
 
 public TreeNode recur(TreeNode root1,TreeNode root2){
 
 if(root1==null && root2==null){
 return null;
 }
 else if(root1!=null && root2==null){
 return root1;
 }
  else if(root1==null && root2!=null){
 return root2;
 }
 TreeNode node=new TreeNode(root1.val+root2.val);
 node.left=recur(root1.left,root2.left);
 node.right=recur(root1.right,root2.right);
 return node;
 }
 
}
 //700. 二叉搜索树中的搜索
 /*给定二叉搜索树（BST）的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 NULL。
例如，
给定二叉搜索树:
        4
       / \
      2   7
     / \
    1   3
和值: 2
你应该返回如下子树:
      2     
     / \   
    1   3
在上述示例中，如果要找的值是 5，但因为没有节点值为 5，我们应该返回 NULL。
*/
 class Solution {
 TreeNode res;
    public TreeNode searchBST(TreeNode root, int val) {
recur(root,val);
 return res;
    }
 
 public void recur(TreeNode root,int val){
 
 if(root==null){
 return ;
 }
 
 if(root.val<val){
 recur(root.right,val);                  
 }
 if(root.val>val){
 recur(root.left,val);
 }                  
if(root.val==val){
 res=root;
 } 
 }
}
 class Solution {
    public TreeNode searchBST(TreeNode root, int val) {
return recur(root,val);
    }
 
 public TreeNode recur(TreeNode root,int val){
 
 if(root==null || root.val==val){
 return root;
 }
 if(root.val<val){
 return recur(root.right,val);                  
 }

 return recur(root.left,val);                  
           
 }
}
                  
98. 验证二叉搜索树
给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
有效 二叉搜索树定义如下：
节点的左子树只包含 小于 当前节点的数。
节点的右子树只包含 大于 当前节点的数。
所有左子树和右子树自身必须也是二叉搜索树。
示例 1：
输入：root = [2,1,3]
输出：true
示例 2：
输入：root = [5,1,4,null,null,3,6]
输出：false
解释：根节点的值是 5 ，但是右子节点的值是 4 。
提示：
树中节点数目范围在[1, 104] 内
-231 <= Node.val <= 231 - 1
 class Solution {
TreeNode prev;                      
    public boolean isValidBST(TreeNode root) {

TreeNode cur=root;
Stack<TreeNode> stack=new Stack<TreeNode>();
 while(!stack.isEmpty() || cur!=null){
 while(cur!=null){
 
 stack.push(cur);
 cur=cur.left;
 }
 cur=stack.pop();
 if(prev!=null){
 if(prev.val>=cur.val){
 return false;
 }
 }
 prev=cur;
 cur=cur.right;
 }
  return true;
    }                                   
}
//669. 修剪二叉搜索树
/*给你二叉搜索树的根节点 root ，同时给定最小边界low 和最大边界 high。通过修剪二叉搜索树，
	使得所有节点的值在[low, high]中。修剪树不应该改变保留在树中的元素的相对结构（即，如果没有被移除，原有的父代子代关系都应当保留）。 可以证明，存在唯一的答案。
所以结果应当返回修剪好的二叉搜索树的新的根节点。注意，根节点可能会根据给定的边界发生改变。
示例 1：
输入：root = [1,0,2], low = 1, high = 2
输出：[1,null,2]
示例 2：
输入：root = [3,0,4,null,2,null,null,1], low = 1, high = 3
输出：[3,2,null,1]
示例 3：
输入：root = [1], low = 1, high = 2
输出：[1]
示例 4：
输入：root = [1,null,2], low = 1, high = 3
输出：[1,null,2]
示例 5：
输入：root = [1,null,2], low = 2, high = 4
输出：[2]
提示：
树中节点数在范围 [1, 104] 内
0 <= Node.val <= 104
树中每个节点的值都是唯一的
题目数据保证输入是一棵有效的二叉搜索树
0 <= low <= high <= 104*/
class Solution {
    public TreeNode trimBST(TreeNode root, int low, int high) {
return recur(root,low,high);		       
    }
public TreeNode recur(TreeNode root,int low,int high){
if(root==null){
return null;		  		       
}
//根		       
//由于根节点的值小于low的值，所以将搜索当前节点的右子树，返回符合条件的头结点
if(root.val<low){
TreeNode right=recur(root.right,low,high);
return right;	
}
//由于根节点的值大于high的值，所以将搜索当前节点的左子树，返回符合条件的头结点	
if(root.val>high){
TreeNode left=recur(root.left,low,high);
return left;	
}	
//左
root.left=recur(root.left,low,high);	
//右
root.right=recur(root.right,low,high);	
//将根节点返回	
return root;	
}	
}
	
//108. 将有序数组转换为二叉搜索树
/*给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。
高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。
示例 1：
输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也将被视为正确答案：
示例 2：
输入：nums = [1,3]
输出：[3,1]
解释：[1,3] 和 [3,1] 都是高度平衡二叉搜索树。
提示：
1 <= nums.length <= 104
-104 <= nums[i] <= 104
nums 按 严格递增 顺序排列	*/
class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
        int n=nums.length;
	return recur(nums,0,n-1);
    }
public TreeNode recur(int[] nums,int l,int r){
if(l>r){
return null;	 
}
int m=(l+r)/2;	
TreeNode node=new TreeNode(nums[m]);
node.left=recur(nums,l,m-1);
node.right=recur(nums,m+1,r);
return node;	
}	
}	
//538. 把二叉搜索树转换为累加树
/*给出二叉 搜索 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。
提醒一下，二叉搜索树满足下列约束条件：
节点的左子树仅包含键 小于 节点键的节点。
节点的右子树仅包含键 大于 节点键的节点。
左右子树也必须是二叉搜索树。
注意：本题和 1038: https://leetcode-cn.com/problems/binary-search-tree-to-greater-sum-tree/ 相同
示例 1：
输入：[4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
输出：[30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]
示例 2：
输入：root = [0,null,1]
输出：[1,null,1]
示例 3：
输入：root = [1,0,2]
输出：[3,3,2]
示例 4：
输入：root = [3,2,4,1]
输出：[7,9,4,10]
提示：
树中的节点数介于 0 和 104 之间。
每个节点的值介于 -104 和 104 之间。
树中的所有值 互不相同 。
给定的树为二叉搜索树。*/
//逆中序遍历	
class Solution {
int sum;
    public TreeNode convertBST(TreeNode root) {
	recur(root);
	return root;
    }
	
public void recur(TreeNode root){
if(root==null){
return ;	
}
	
//右
recur(root.right);
//根
sum+=root.val;
root.val=sum;
//左
recur(root.left);
}	
}	
//77. 组合
/*给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
你可以按 任何顺序 返回答案。
示例 1：
输入：n = 4, k = 2
输出：
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
示例 2：
输入：n = 1, k = 1
输出：[[1]]
提示：
1 <= n <= 20
1 <= k <= n
*/	
class Solution {	
    public List<List<Integer>> combine(int n, int k) {

List<List<Integer>> res=new ArrayList<List<Integer>>();
LinkedList<Integer> subset=new LinkedList<Integer>();
	recur(res,subset,n,k,1);
	return res;
    }
public void recur(List<List<Integer>> res,LinkedList<Integer> subset,int n,int k,int index){
if(index==n+1){
if(subset.size()==k){
res.add(new LinkedList<Integer>(subset));	
}	
return; 	
}	
if(index<n+1){
recur(res,subset,n,k,index+1);	       
subset.add(index);
recur(res,subset,n,k,index+1);	
subset.removeLast();	
}
}	
}
//216. 组合总和 III
/*找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。
说明：
所有数字都是正整数。
解集不能包含重复的组合。 
示例 1:
输入: k = 3, n = 7
输出: [[1,2,4]]
示例 2:
输入: k = 3, n = 9
输出: [[1,2,6], [1,3,5], [2,3,4]]
*/	
class Solution {
int sum;
int num=9;	       
    public List<List<Integer>> combinationSum3(int k, int n) {
List<List<Integer>> res=new ArrayList<List<Integer>>();
LinkedList<Integer> subset=new LinkedList<Integer>();	
	recur(res,subset,k,n,1);
	return res;
    }
public void recur(List<List<Integer>> res,LinkedList<Integer> subset,int k,int n,int index){
if(sum==n && subset.size()==k){
res.add(new LinkedList<Integer>(subset));	
return ;	
}
if(sum<n && subset.size()<k && index<=num){
recur(res,subset,k,n,index+1);	
sum+=index;					   
subset.add(index);
recur(res,subset,k,n,index+1);	
subset.removeLast();	
sum-=index;					   
}	
	
}	
	
}
//17. 电话号码的字母组合		
/*给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
示例 1：
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
示例 2：
输入：digits = ""
输出：[]
示例 3：
输入：digits = "2"
输出：["a","b","c"]
提示：
0 <= digits.length <= 4
digits[i] 是范围 ['2', '9'] 的一个数字。
*/	
class Solution {
    public List<String> letterCombinations(String digits) {

Map<Character,String> map=new HashMap<Character,String>();
List<String> res=new ArrayList<String>();
if(digits.length()==0){
return res;
}	
StringBuilder sb=new StringBuilder();	
	map.put('2',"abc");	
	map.put('3',"def");	
	map.put('4',"ghi");	
	map.put('5',"jkl");	
	map.put('6',"mno");	
	map.put('7',"pqrs");	
	map.put('8',"tuv");	
	map.put('9',"wxyz");	
	recur(res,map,digits,0,sb);
	return res;
    }
	
public void recur(List<String> res,Map<Character,String> map,String digits,int index,StringBuilder sb){
	
if(index==digits.length()){
res.add(sb.toString());
return;	
}
char ch=digits.charAt(index);
String str=map.get(ch);
for(int i=0;i<str.length();i++){
sb.append(str.charAt(i));				 
recur(res,map,digits,index+1,sb);				 
sb.deleteCharAt(sb.length()-1);				 
}	
}	
}
				 
//39. 组合总和
/*给定一个无重复元素的正整数数组 candidates 和一个正整数 target ，找出 candidates 中所有可以使数字和为目标数 target 的唯一组合。
candidates 中的数字可以无限制重复被选取。如果至少一个所选数字数量不同，则两种组合是唯一的。 
对于给定的输入，保证和为 target 的唯一组合数少于 150 个。
示例 1：
输入: candidates = [2,3,6,7], target = 7
输出: [[7],[2,2,3]]
示例 2：
输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]
示例 3：
输入: candidates = [2], target = 1
输出: []
示例 4：
输入: candidates = [1], target = 1
输出: [[1]]
示例 5：
输入: candidates = [1], target = 2
输出: [[1,1]]
提示：
1 <= candidates.length <= 30
1 <= candidates[i] <= 200
candidate 中的每个元素都是独一无二的。
1 <= target <= 500
*/	
class Solution {
int sum;		 
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
List<List<Integer>> res=new ArrayList<List<Integer>>();
LinkedList<Integer> subset=new LinkedList<Integer>();
	recur(res,subset,candidates,target,0);
	return res;
    }
	
public void recur(List<List<Integer>> res,LinkedList<Integer> subset,int[] candidates, int target,int index){

if(sum==target){
res.add(new LinkedList<Integer>(subset));
return;	
}	
if(index==candidates.length){
return;	
}
if(sum<target){
recur(res,subset,candidates,target,index+1);
sum+=candidates[index];		
subset.add(candidates[index]);
recur(res,subset,candidates,target,index);	
subset.removeLast();
sum-=candidates[index];		
}		
}		
}
//40. 组合总和 II
/*给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
candidates 中的每个数字在每个组合中只能使用一次。
注意：解集不能包含重复的组合。 
示例 1:
输入: candidates = [10,1,2,7,6,1,5], target = 8,
输出:
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
示例 2:
输入: candidates = [2,5,2,1,2], target = 5,
输出:
[
[1,2,2],
[5]
]
提示:
1 <= candidates.length <= 100
1 <= candidates[i] <= 50
1 <= target <= 30
*/	
class Solution {
int sum;		
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
Arrays.sort(candidates);
	List<List<Integer>> res=new ArrayList<List<Integer>>();
	LinkedList<Integer> subset=new LinkedList<Integer>();
	recur(res,subset,candidates,target,0);
	return res;
    }
	
public void recur(List<List<Integer>> res,LinkedList<Integer> subset,int[] candidates, int target,int index){
	
if(sum==target){
res.add(new LinkedList<Integer>(subset));	
return;	
}	
if(index==candidates.length){
return;	
}	
if(sum<target && index<candidates.length){
recur(res,subset,candidates,target,getNext(candidates,index));
subset.add(candidates[index]);
sum+=candidates[index];	
recur(res,subset,candidates,target,index+1);	
subset.removeLast();	
sum-=candidates[index];	
}	
}	
		
public int getNext(int[] candidates,int index){
int next=index;
while(next<candidates.length && candidates[index]==candidates[next]){
next++;			     
}
return next;			     
}		
}	
			     
//131. 分割回文串
/*给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。
回文串 是正着读和反着读都一样的字符串。
示例 1：
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
示例 2：
输入：s = "a"
输出：[["a"]]
提示：
1 <= s.length <= 16
s 仅由小写英文字母组成
*/	
class Solution {
    public List<List<String>> partition(String s) {
List<List<String>> res=new ArrayList<List<String>>();
LinkedList<String> subset=new LinkedList<String>();	
	recur(res,s,0,subset);
	return res;
    }
	
public void recur(List<List<String>> res,String s,int index,LinkedList<String> subset){
if(index==s.length()){
res.add(new LinkedList<String>(subset));	
return;	
}
	
if(index<s.length()){
for(int i=index;i<s.length();i++){
	if(isHuiWen(s.substring(index,i+1))){
	subset.add(s.substring(index,i+1));
	recur(res,s,i+1,subset);
	subset.removeLast();
}
}		      
}	
}	
	
public boolean isHuiWen(String s){
int start=0;
int end=s.length()-1;	
while(start<end){
if(s.charAt(start)!=s.charAt(end)){
return false;	
}	
start++;
end--;		  
}
return true;		  
}	
}
//93. 复原 IP 地址
/*有效 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。
例如："0.1.2.201" 和 "192.168.1.1" 是 有效 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效 IP 地址。
给定一个只包含数字的字符串 s ，用以表示一个 IP 地址，返回所有可能的有效 IP 地址，这些地址可以通过在 s 中插入 '.' 来形成。
你不能重新排序或删除 s 中的任何数字。你可以按 任何 顺序返回答案。
示例 1：
输入：s = "25525511135"
输出：["255.255.11.135","255.255.111.35"]
示例 2：
输入：s = "0000"
输出：["0.0.0.0"]
示例 3：
输入：s = "1111"
输出：["1.1.1.1"]
示例 4：
输入：s = "010010"
输出：["0.10.0.10","0.100.1.0"]
示例 5：
输入：s = "101023"
输出：["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
提示：
0 <= s.length <= 20
s 仅由数字组成
*/	
class Solution {
    public List<String> restoreIpAddresses(String s) {
List<String> res=new ArrayList<String>();
recur(res,"","",s,0,0);	
return res;	
    }
	
public void recur(List<String> res,String curIp,String part,String s,int partId,int index){
if(partId==3 && isValid(part) && index==s.length()){
res.add(curIp+part);
return;	
}
if(index<s.length()){
char ch=s.charAt(index);	
if(partId<=3){
if(isValid(part+ch+"")){
recur(res,curIp,part+ch+"",s,partId,index+1);	      
}	      
}
if(partId<3 && part.length()>0){
recur(res,curIp+part+".",ch+"",s,partId+1,index+1);	
}	      		      
}	      
}
	
public boolean isValid(String s){
if(s.charAt(0)=='0' && s.length()>1 || Integer.valueOf(s)>255){
return false;	
}
return true;	
}	
}	
//78. 子集
/*给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
示例 1：
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
示例 2：
输入：nums = [0]
输出：[[],[0]]
提示：
1 <= nums.length <= 10
-10 <= nums[i] <= 10
nums 中的所有元素 互不相同
*/
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
List<List<Integer>> res=new ArrayList<List<Integer>>();
LinkedList<Integer> subset=new LinkedList<Integer>();
recur(res,subset,nums,0);
return res;	
    }
	
public void recur(List<List<Integer>> res,LinkedList<Integer> subset,int[] nums,int index){
if(index==nums.length){
res.add(new LinkedList<Integer>(subset));	
return;
}
recur(res,subset,nums,index+1);
subset.add(nums[index]);
recur(res,subset,nums,index+1);	
subset.removeLast();	
}	
	
}	
//491. 递增子序列
/*给你一个整数数组 nums ，找出并返回所有该数组中不同的递增子序列，递增子序列中 至少有两个元素 。你可以按 任意顺序 返回答案。
数组中可能含有重复元素，如出现两个整数相等，也可以视作递增序列的一种特殊情况。
示例 1：
输入：nums = [4,6,7,7]
输出：[[4,6],[4,6,7],[4,6,7,7],[4,7],[4,7,7],[6,7],[6,7,7],[7,7]]
示例 2：
输入：nums = [4,4,3,2,1]
输出：[[4,4]]
提示：
1 <= nums.length <= 15
-100 <= nums[i] <= 100
*/	
class Solution {
    public List<List<Integer>> findSubsequences(int[] nums) {
List<List<Integer>> res=new ArrayList<List<Integer>>();
LinkedList<Integer> subset=new LinkedList<Integer>();
	
	recur(res,subset,nums,0);
	return res;
    }
			
public void recur(List<List<Integer>> res,LinkedList<Integer> subset,int[] nums,int index){
//当subset集合数量大于等于2时,放进结果集中
if(subset.size()>=2){
res.add(new LinkedList<Integer>(subset));	
}


Set<Integer> set=new HashSet<Integer>();	
for(int i=index;i<nums.length;i++){
//处理树的层				    
//当subset不为空，并且最后放进去的数字比当前数字大时，或者当前数字在这一层已经搜索过，跳过当前循环				    
if(!subset.isEmpty() && subset.getLast()>nums[i] || set.contains(nums[i])){
	continue;
}
set.add(nums[i]);
subset.add(nums[i]);、
//处理树的枝	
recur(res,subset,nums,i+1);	
subset.removeLast();				    
}	


}	
	
public boolean isTrue(LinkedList<Integer> subset){
int size=subset.size();
int[] arr=new int[size];
int index=0;	
for(int num : subset){
arr[index++]=num;	
}	
for(int i=0;i<arr.length-1;i++){
if(arr[i]>arr[i+1]){
return false;	
}				 
}
return true;	
}
	
public int getNext(int[] nums,int index){
int next=index;
while(next<nums.length && nums[next]==nums[index]){
next++;		       
}
return next;		       
}	
	
}
	
//46. 全排列
/*给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
示例 1：
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
示例 2：
输入：nums = [0,1]
输出：[[0,1],[1,0]]
示例 3：
输入：nums = [1]
输出：[[1]]
提示：
1 <= nums.length <= 6
-10 <= nums[i] <= 10
nums 中的所有整数 互不相同
*/	
class Solution {
    public List<List<Integer>> permute(int[] nums) {
List<List<Integer>> res=new ArrayList<List<Integer>>();
recur(res,nums,0);	
return res;	
    }
	
public void recur(List<List<Integer>> res,int[] nums,int index){
if(index==nums.length){
List<Integer> list=new ArrayList<Integer>();
for(int num :nums){
list.add(num);	
}
res.add(list);	
return;	
}
for(int i=index;i<nums.length;i++){
swap(nums,i,index);
recur(res,nums,index+1);
swap(nums,i,index);				    
}	
		
}
public void swap(int[] nums,int a,int b){
int temp=nums[a];
nums[a]=nums[b];
nums[b]=temp;				    
}				    
				    
}
				    
				    
//	47. 全排列 II	
/*给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
示例 1：
输入：nums = [1,1,2]
输出：
[[1,1,2],
 [1,2,1],
 [2,1,1]]
示例 2：
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
提示：
1 <= nums.length <= 8
-10 <= nums[i] <= 10
*/		
class Solution {
    public List<List<Integer>> permuteUnique(int[] nums) {
List<List<Integer>> res=new ArrayList<List<Integer>>();
	recur(res,nums,0);
	return res;
    }
public void recur(List<List<Integer>> res,int[] nums,int index){
if(index==nums.length){
List<Integer> list=new ArrayList<Integer>();	
for(int num :nums){
list.add(num);	
}
res.add(new ArrayList<Integer>(list));
return;	
}	

Set<Integer> set=new HashSet<Integer>();
	
for(int i=index;i<nums.length;i++){
	if(set.contains(nums[i])){
	continue;			    
	}
        set.add(nums[i]);				    
	swap(nums,i,index);
      recur(res,nums,index+1);
       swap(nums,i,index);				    
}		
}	
public void swap(int[] nums,int a,int b){
int temp=nums[a];
nums[a]=nums[b];
nums[b]=temp;				    
}	
	
}
				    //定义两个函数f(i),g(i)
//S是一个长度为i+1的字符串,下标从0~i,我们假设字符串的前i个字符是已经按照递增顺序翻转好的字符串;
//f(i)表示字符串按照递增的顺序翻转好之后,第i+1位的  字符是0  所需要的最少的翻转次数;
//因为我们的字符串是按照递增的顺序翻转好的,所以第i位的字符是0,那么第i-1位的字符一定是0,否则就不符合递增排序的规则了;
//在决定反转第i位字符的时候,我们面临两种选择,
//(1)一种选择是字符串第i位字符是0时,此时我们就不需要翻转字符串的这个位置的数字,因此f(i)=f(i-1);
//(2)另一种选择是字符串第i位字符是1时,此时我们需要将字符串这个位置的数字翻转成0,因此f(i)=f(i-1)+1;

//g(i)表示字符串按照递增的顺序翻转好之后,第i+1位的  字符是1  所需要的最少翻转次数,此时我们翻转字符串时,面临两种选择
//因为我们的字符串是按照递增的顺序翻转好的,所以当最后第i为字符是1时,我们无论i-1位置上是0,还是1,都是满足题目递增的要求的;
//(1)一种选择是字符串的第i位的字符是0,我们需要将最后字符串一位字符翻转成1,g(i)=min(f(i-1,0),f(i-1,1))+1;
//(2)另一种选择是字符串的第i位字符是1,此时我们不需要将最后的一位字符翻转,g(i)=min(f(i-1,0),f(i-1,0));


//设置初始值:
//f(0)表示,字符串的前i位按照递增顺序排序后,我们第0位置的字符是0的最少翻转次数;
//(1)当字符串的第一位的字符是0时,我们不需要翻转,f(0)=0
//(2)当字符串的第一位的字符是1时,我们就要反转这个字符为1,f(0)=1;

//g(0)表示,字符串的前i位按照递增的顺序排序后,我们第0位置的字符是1的最小翻转次数;
//(1)当字符串的第一位的字符是0时,我们需要将这个字符翻转成1,g(0)=1;
//(2)当字符串的第一位字符是1时,我们不需要将这个字符翻转,g(0)=0;
				    char ch=s.charAt(i);
        	if(ch=='0'){
        		dp[0][i]=dp[0][i-1];
        		dp[1][i]=Math.min(dp[0][i-1],dp[1][i-1])+1;
        	}
        	else{
        		dp[0][i]=dp[0][i-1]+1;
        		dp[1][i]=Math.min(dp[0][i-1],dp[1][i-1]);
        	}

//剑指 Offer II 092. 翻转字符
/*如果一个由 '0' 和 '1' 组成的字符串，是以一些 '0'（可能没有 '0'）后面跟着一些 '1'（也可能没有 '1'）
的形式组成的，那么该字符串是 单调递增 的。
我们给出一个由字符 '0' 和 '1' 组成的字符串 s，我们可以将任何 '0' 翻转为 '1' 或者将 '1' 翻转为 '0'。
返回使 s 单调递增 的最小翻转次数。
示例 1：
输入：s = "00110"
输出：1
解释：我们翻转最后一位得到 00111.
示例 2：
输入：s = "010110"
输出：2
解释：我们翻转得到 011111，或者是 000111。
示例 3：
输入：s = "00011000"
输出：2
解释：我们翻转得到 00000000。
提示：
1 <= s.length <= 20000
s 中只包含字符 '0' 和 '1'
*/
//定义f(i,0)表示根据反转规则将第i位翻转为0的最小翻转次数
//定义f(i,1)表示根据反转规则将第i位翻转为1的最小翻转次数

//因为我们默认字符串已经按照递增的顺序排序，所以根据翻转规则此位应该是'0'时,他的前一位一定是'0'
//f(i,0)有两种选择,(1)第i位字符是'0'时,f(i,0)=f(i-1,0)它的前一位只能是'0'并且不需要进行翻转
//(2)第i位字符是'1'时,f(i,0)=f(i-1,0)+1,它的前一位是'0',并且此时需要再翻转一次,此时需要在f(i-1,0)的基础上再翻转一个字符

//当我们想把某一位翻转为1时，这个字符前面的一个字符可以是'0',也可以是'1'
//f(i,1)有两种选择,当第i位字符是'0'时,min(f(i-1,0),f(i-1,1))+1,因为当前字符是'0'需要再翻转一次
//当第i位字符是'1'时,min(f(i-1,0),f(i-1,1)),因为当前字符是'1'所以不需要再翻转一次

//确定初始值,f(0,0)当第一位为0时f(0,0)=0,当第一位为1时f(0,0)=1
//f(0,1)当第一位为1时f(0,0)=0,当第一位为0时f(0,0)=1
class Solution {
    public int minFlipsMonoIncr(String s) {

			int n=s.length();
			int[][] dp=new int[n][2];
			if(s.charAt(0)=='0'){
			
				dp[0][0]=0;
				dp[0][1]=1;
			}
			else{
			
				dp[0][0]=1;
				dp[0][1]=0;
			}
			
			for(int i=1;i<n;i++){
                               //当前字符是'0'时，我们将当前字符翻转成'0'或者'1'的最小翻转次数
				if(s.charAt(i)=='0'){
				       //根据翻转规则，此位应该反转为'0'
					dp[i][0]=dp[i-1][0];
	                               //根据翻转规则，此位应该反转为'1'
					dp[i][1]=Math.min(dp[i-1][0],dp[i-1][1])+1;
				
				}
	                        //当前字符是'1'时，我们将当前字符翻转成'0'或者'1'的最小翻转次数
				else{
				        //根据翻转规则，此位应该翻转为'0'
					dp[i][0]=dp[i-1][0]+1;
	                                //根据翻转规则，此位应该翻转为'1'
				        dp[i][1]=Math.min(dp[i-1][0],dp[i-1][1]);
				}
				
			}
	                 //最小翻转次数是一下两个结果中较小的一个
	      return Math.min(dp[n-1][0],dp[n-1][1]);
    }
}   
	
//剑指 Offer II 093. 最长斐波那契数列
/*如果序列 X_1, X_2, ..., X_n 满足下列条件，就说它是 斐波那契式 的：
n >= 3
对于所有 i + 2 <= n，都有 X_i + X_{i+1} = X_{i+2}
给定一个严格递增的正整数数组形成序列 arr ，找到 arr 中最长的斐波那契式的子序列的长度。如果一个不存在，返回  0 。
（回想一下，子序列是从原序列  arr 中派生出来的，它从 arr 中删掉任意数量的元素（也可以不删），而不改变其余元素的顺序。例如， [3, 5, 8] 是 [3, 4, 5, 6, 7, 8] 的一个子序列）
示例 1：
输入: arr = [1,2,3,4,5,6,7,8]
输出: 5
解释: 最长的斐波那契式子序列为 [1,2,3,5,8] 。
示例 2：
输入: arr = [1,3,7,11,12,14,18]
输出: 3
解释: 最长的斐波那契式子序列有 [1,11,12]、[3,11,14] 以及 [7,11,18] 。
提示：
3 <= arr.length <= 1000
1 <= arr[i] < arr[i + 1] <= 10^9*/
//定义f(i,j)表示以j为倒数第2个字符，i为倒数第一个字符的斐波那契子序列的长度
//我们需要在0~j内寻找数字k,使得	
class Solution {
    public int lenLongestFibSubseq(int[] arr) {

	
	int length=2;
	int n=arr.length;
	int[][] dp=new int[n][n];
	int maxLen=Integer.MIN_VALUE;
	Map<Integer,Integer> map=new HashMap<>();
	for(int i=0;i<n;i++){
	map.put(arr[i],i);					  
	}
	for(int i=1;i<n;i++){
	for(int j=0;j<n-1;j++){
	int k=map.getOrDefault(arr[i]-arr[j],-1);
	
	if(k>=0 && k<j){
	dp[i][j]=Math.max(dp[j][k]+1,dp[i][j]);		 
	}
	else{
	dp[i][j]=2;		
	}		 
	maxLen=Math.max(maxLen,dp[i][j]);			
	}
	
	}
	
	return maxLen>2?maxLen:0;
	
	
    }
}
	//我们定义f(i,j)表示,以下标i为最后一个数字,以下标j为倒数第二个数字形成的斐波那契数列的长度;
//A[i]表示下标为i的数字,A[j]表示下标为j的数字(0<=j<i),
//如果能在(0,j)之间找到一个下标位置k(0<=k<j),使得A[k]+A[j]=A[i]那么就构成了一个斐波那栔数列;
//此时斐波那契数列的长度是在以A[j]为倒数第一个字符,A[k]为倒数第二个字符的斐波那契字符的长度的基础上+1;
//f(i,j)=f(j,k)+1;(关键就是寻找这个k值,使得A[k]+A[j]=A[i])
//由于状态转移方程存在两个变量,所以我们定义一个二维数组来缓存数据;
//定义dp数组,为一个二维数组int[arr.length][arr.length];
	class Solution {
    public int lenLongestFibSubseq(int[] arr) {

    	int length=2;
    	Map<Integer,Integer> map=new HashMap<Integer,Integer>();
        for(int i=0;i<arr.length;i++){
        	map.put(arr[i],i);

        }

        int[][] dp=new int[arr.length][arr.length];
        for(int i=1;i<arr.length;i++){
        	for(int j=0;j<i;j++){

        		int k=map.getOrDefault(arr[i]-arr[j],-1);

        		dp[i][j]=k>=0 && k<j?dp[j][k]+1:2;

        		length=Math.max(length,dp[i][j]);
        	}

        }

        return length>2?length:0;
    }
}
//115. 不同的子序列	
/*给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。
字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）
题目数据保证答案符合 32 位带符号整数范围。
示例 1：
输入：s = "rabbbit", t = "rabbit"
输出：3
解释：
如下图所示, 有 3 种可以从 s 中得到 "rabbit" 的方案。
rabbbit
rabbbit
rabbbit
示例 2：
输入：s = "babgbag", t = "bag"
输出：5
解释：
如下图所示, 有 5 种可以从 s 中得到 "bag" 的方案。 
babgbag
babgbag
babgbag
babgbag
babgbag
提示：
0 <= s.length, t.length <= 1000
s 和 t 由英文字母组成
*/
//f(i,j)表示字符串s[0,i]中t[0,j]出现的次数
//当字符串s[i-1]和字符串t[j-1]的字符相同时,我们有两个选择，(1)不用s[i-1]去匹配t[j-1],相当于删除字符串s第i-1个字符
//f(i,j)=f(i-1,j)
//(2)用s[i-1]去匹配t[j-1],相当于两个字符串后面都分别添加一个字符，个数不变f(i,j)=f(i-1,j-1)
//所以当s[i-1]=t[j-1]时,f(i,j)=f(i-1,j)+f(i-1,j-1)	
//当字符串s[i-1]和字符串t[j-1]的字符不相同时,只能选择不用字符s[i-1]去匹配字符t[j-1],f(i,j)=f(i-1,j)
//确定初始值，f(-1,j)表示非空字符串在空字符串中出现的次数，f(-1,j)=0
//f(i,-1)表示空字符串在非空字符串中出现的次数，f(i,-1)=1;	
class Solution {
    public int numDistinct(String s, String t) {

	if(s.length()<t.length()){
	return 0;
	}
	int m=s.length();
	int n=t.length();			   
	int[][] dp=new int[m+1][n+1];
	for(int i=0;i<=m;i++){
	dp[i][0]=1;
	}
	for(int i=0;i<m;i++){
			      
	for(int j=0;j<n;j++){
	if(s.charAt(i)==t.charAt(j)){
	dp[i+1][j+1]=dp[i][j]+dp[i][j+1];
	}
	
	else if(s.charAt(i)!=t.charAt(j)){
	
        dp[i+1][j+1]=dp[i][j+1];
	}
	
	}		      
        }
	
	return dp[m][n];
    }
}
//583. 两个字符串的删除操作
/*给定两个单词 word1 和 word2，找到使得 word1 和 word2 相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。
示例：
输入: "sea", "eat"
输出: 2
解释: 第一步将"sea"变为"ea"，第二步将"eat"变为"ea"
提示：
给定单词的长度不超过500。
给定单词中的字符只含有小写字母。*/
//定义f(i,j)表示字符串word1[0,i]删除一个字符得到word2[0,j]的最小步数
//当字符串word1[i-1]==字符串word2[j-1]时，此时不需要对字符串做删除,f(i,j)=f(i-1,j-1)
//当字符串word1[i-1]!=字符串word2[j-1]时,有两种选择(1)删除word1[i-1],f(i,j)=f(i-1,j)+1
//(2)删除word2[j-1],f(i,j)=f(i,j-1)+1
//所以f(i,j)=min(f(i-1),f(i,j-1))+1;
	
//确定初始值，f(-1,j)空字符串word2通过删除字符得到字符word1,f(-1,j)=word2.length();
//f(i,-1)空字符串word1通过删除字符得到字符word2,f(i,-1)=word1.length();	
class Solution {
    public int minDistance(String word1, String word2) {

	int m=word1.length();
	int n=word2.length();
	
	int[][] dp=new int[m+1][n+1];
	
	for(int i=0;i<=m;i++){
	dp[i][0]=i;		      
	}
			      
	for(int j=0;j<=n;j++){
	dp[0][j]=j;		      
	}	
	
	for(int i=0;i<m;i++){
	for(int j=0;j<n;j++){
	if(word1.charAt(i)==word2.charAt(j)){
	dp[i+1][j+1]=dp[i][j];
	}
	else{
	dp[i+1][j+1]=Math.min(dp[i+1][j],dp[i][j+1])+1;
	}
	
	}		      
			      
	}
	return dp[m][n];
    }
}
	
//72. 编辑距离
/*两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。
你可以对一个单词进行如下三种操作：
插入一个字符
删除一个字符
替换一个字符
示例 1：
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
示例 2：
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
提示：
0 <= word1.length, word2.length <= 500
word1 和 word2 由小写英文字母组成	*/
//我们将对字符串的操作压缩为3种，
//(1)对字符串word1插入一个字符得到word2
//(2)对字符串word2插入一个字符得到word1
//(3)对字符串word1替换一个字符得到word2
//定义f(i,j)表示字符串word1[0,i]与字符串word2[0,j]相等的最小操作次数	
//当word1[i-1]==word2[j-1]时，f(i,j)=f(i-1,j-1)
//当word1[i-1]!=word2[j-1]时，f(i,j)=f(i-1,j-1)+1
//f(i-1,j)对字符串word2添加末尾添加一个和字符串word1末尾相同的字符f(i,j)=f(i-1,j)+1	
//f(i,j-1)对字符串word1添加末尾添加一个和字符串word2末尾相同的字符f(i,j)=f(i,j-1)+1

//确定初始值:f(-1,j)表示将空字符串word1通过操作变成word2 f(-1,j)=word2.length
//f(i,-1)表示将空字符串word2通过操作变成word1 f(i,-1)=word1.length	
class Solution {
    public int minDistance(String word1, String word2) {

	int m=word1.length();
	int n=word2.length();
	int[][] dp=new int[m+1][n+1];
	
	for(int i=0;i<=word1.length();i++){
	dp[i][0]=i;				   
	}
	for(int j=0;j<=word2.length();j++){
	dp[0][j]=j;				   
	}				   
	
	for(int i=0;i<m;i++){
	for(int j=0;j<n;j++){
	
	if(word1.charAt(i)==word2.charAt(j)){
	dp[i+1][j+1]=Math.min(Math.min(dp[i][j+1],dp[i+1][j]),dp[i][j]-1)+1;
	}
	else{
	
	dp[i+1][j+1]=Math.min(Math.min(dp[i][j+1],dp[i+1][j]),dp[i][j])+1;
	}
	}		      
	}
	
	return dp[m][n];
    }
}	
	
//647. 回文子串
/*给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。
回文字符串 是正着读和倒过来读一样的字符串。
子字符串 是字符串中的由连续字符组成的一个序列。
具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。
示例 1：
输入：s = "abc"
输出：3
解释：三个回文子串: "a", "b", "c"
示例 2：
输入：s = "aaa"
输出：6
解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
提示：
1 <= s.length <= 1000
s 由小写英文字母组成*/
//定义f(i,j)表示字符串[i,j]是不是回文字符串，当字符串s[i]!=s[j]时，一定不是回文字符串
//当字符串s[i]==s[j]时，可以分成以下3种情况:
//(1)i==j时,s[i,j]一定是回文字符串f(i,j)=true
//(2)j-i=1时,s[i,j]一定是回文字符串 f(i,j)=true
//(3)j-i>1时，s[i,j]是不是回文字符串取决于 f(i+1,j-1)是true还是false
//所以递推公式可以表达为
//s[i]==s[j]时	
//j-i<=1时:dp[i][j]=true
//j-i>1时: dp[i][j]=dp[i+1][j-1]		
//由于递推公式中有i+1是先于i的所以我们的遍历顺序是有所变化
//我们要从后向前遍历i,又由定义知道j是大于等于i的所以j从i开始遍历
//确定初始值，初始值设置为false	
class Solution {
    public int countSubstrings(String s) {

	int n=s.length();
	boolean[][] dp=new boolean[n][n];
	int count=0;
	//倒序遍历i，为了使用正确的dp[i+1][j-1]推出dp[i][j]
	for(int i=n;i>=0;i--){
	//定义时，j是大于等于i的，所以j从i开始遍历
	for(int j=i;j<n;j++){
	//只考虑两个字符相同的时候		      
	if(s.charAt(i)==s.charAt(j)){	      
	//当j和i相差不超过1时,直接判断为true		      
	if(j-i<=1){
	dp[i][j]=true;
	count++;
	}
	//当j和i相差超过1时,此时就要判断dp[i+1][j-1]是不是为true,如果为true，那么此时dp[i][j]就为true
	if(j-i>1){
	if(dp[i+1][j-1]){
	count++;
	dp[i][j]=true;
	}
	}		      
	}		      	      
        }
	}
	return count;	
    }
}	
	
	
//回文串的另外一种判别方式,将一个字符向两边扩展形成回文
//双指针法	
class Solution {
    public int countSubstrings(String s) {
        int n=s.length();
	int count=0;
	for(int i=0;i<n;i++){
	//记录一个字符形成的回文字符串的个数		      
	count+=	count(s,i,i);
	//记录两个相同的字符形成的回文字符串的个数		      
        count+= count(s,i,i+1);			      
	}
	return count;
	
    }
//统计一个字符和两个相同的字符能够形成的回文字符串的个数	
public int count(String s,int start,int end){
int count=0;
while(start>=0 && end<s.length() && s.charAt(start)==s.charAt(end)){
count++;								     
start--;
end++;								     
}	
return count;	
}		
}	
				 
//516. 最长回文子序列
/*给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。
子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。
示例 1：
输入：s = "bbbab"
输出：4
解释：一个可能的最长回文子序列为 "bbbb" 。
示例 2：
输入：s = "cbbd"
输出：2
解释：一个可能的最长回文子序列为 "bb" 。
提示：
1 <= s.length <= 1000
s 仅由小写英文字母组成*/
//定义f(i,j)表示字符串s[i,j]最长回文子序列的长度
//当s[i]==s[j]时,f(i,j)=f(i+1,j-1)+2(回文字符串s[i+1,j-1]基础上加上这两个字符后形成的，所以长度+2)
//当s[i]!=s[j]时,将s[i]添加到字符串中，或者将s[j]添加到字符串中,取两个字符串最长的回文子序列的长度  (i [i+1 ... j-1) j]从图中清晰地看出回文字符串的长度是两者中较大的一个
//f(i,j)=max(f(i+1,j),f(i,j-1))	
		    
//确定初始值:f(i,j)=1,(其中i==j)当i==j时,此时构成长度为1的字符串,一个字符的回文字串长度为1
//确定遍历顺序,由于递推公式中,f(i,j)=f(i+1,j-1)+2,可以知道遍历顺序是i是倒序遍历的，j要从i开始遍历   
//最后结果是,f(0,n-1)表示字符串s[0~n-1]中的最长回文子序列		    
class Solution {
    public int longestPalindromeSubseq(String s) {

	int n=s.length();
		    
        int[][] dp=new int[n][n];
	//当只有一个字符时，回文串的长度是1	    
	for(int i=0;i<n;i++){
	dp[i][i]=1;
	}
	//倒序遍历i,由于f(i,j)由f(i+1,j-1)推出
	for(int i=n-1;i>=0;i--){
	for(int j=i+1;j<n;j++){
        //如果字符s[i]==s[j],将这两个字符都添加到字符串中,此时字符串的长度在s[i+1,j-1]的基础上+2			
	if(s.charAt(i)==s.charAt(j)){
	dp[i][j]=dp[i+1][j-1]+2;		      
        }
	//如果字符s[i]!=s[j],那就不能同时将这两个字符添加进去,将他们一个一个添加进去,此时求出两者最长的子序列长度作为f(i,j)				
	else{
	dp[i][j]=Math.max(dp[i+1][j],dp[i][j-1]);		      
	}		      
	}
	}	    
	return dp[0][n-1];	    
		    
    }
}	
				
//5. 最长回文子串
/*给你一个字符串 s，找到 s 中最长的回文子串。
示例 1：
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
示例 2：
输入：s = "cbbd"
输出："bb"
示例 3：
输入：s = "a"
输出："a"
示例 4：
输入：s = "ac"
输出："a"
提示：
1 <= s.length <= 1000
s 仅由数字和英文字母（大写和/或小写）组成*/
class Solution {
    public String longestPalindrome(String s) {

	int n=s.length();
	int maxLen=0;
	int maxStart=0;
	int maxEnd=0;
	String res="";	    
	for(int i=0;i<n;i++){
	//由一个字符形成的回文子字符串的长度
	int len1=recur(s,i,i);
	//由两个字符形成的回文子字符串的长度
	int len2=recur(s,i,i+1);
	//得到形成的子字符串的较长者
        int len=Math.max(len1,len2);	
	//如果新形成的子字符串的长度大于之前形成的字符串的长度,那么就将当前字符串替换到之前的字符串
	if(len>maxLen){
	maxLen=len;
	//获取生成的最大的回文串的开始位置
	maxStart=i-(maxLen-1)/2;         
	//获取生成的最大的回文串的结束位置
	maxEnd=i+maxLen/2;
	//截取当前最大回文字符串
	res=s.substring(maxStart,maxEnd+1);
	}
	}	     
	return res;	    
    }
		    
		    
   public int recur(String s,int start,int end){	    
   while(start>=0 && end<s.length() && s.charAt(start)==s.charAt(end)){
   start--;
   end++;				    				    
  }		    
   return end-start-1;		    
  }		    
}
//704. 二分查找
/*给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的 target，如果目标值存在返回下标，否则返回 -1。
示例 1:
输入: nums = [-1,0,3,5,9,12], target = 9
输出: 4
解释: 9 出现在 nums 中并且下标为 4
示例 2:
输入: nums = [-1,0,3,5,9,12], target = 2
输出: -1
解释: 2 不存在 nums 中因此返回 -1
提示：
你可以假设 nums 中的所有元素是不重复的。
n 将在 [1, 10000]之间。
nums 的每个元素都将在 [-9999, 9999]之间。*/
class Solution {
    public int search(int[] nums, int target) {
				    
int n=nums.length;
int i=0;
int j=n-1;				    
while(i<=j){
int m=(i+j)/2;
if(nums[m]>target){
j=m-1;	
}	
else if(nums[m]<target){
i=m+1;	
}	
else{
return m;		
}	
}				    
				    
return -1;				    

    }
}	
			 
//35. 搜索插入位置
/*给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
请必须使用时间复杂度为 O(log n) 的算法。
示例 1:
输入: nums = [1,3,5,6], target = 5
输出: 2
示例 2:
输入: nums = [1,3,5,6], target = 2
输出: 1
示例 3:
输入: nums = [1,3,5,6], target = 7
输出: 4
示例 4:
输入: nums = [1,3,5,6], target = 0
输出: 0
示例 5:
输入: nums = [1], target = 0
输出: 0
提示:
1 <= nums.length <= 104
-104 <= nums[i] <= 104
nums 为无重复元素的升序排列数组
-104 <= target <= 104*/
class Solution {
public int searchInsert(int[] nums, int target) {

	int n=nums.length;
  int i=0;                   
  int j=n-1;                   
  while(i<=j){
  
  int m=(i+j)/2;
  if(nums[m]<target){
  i=m+1;                                      
  }
  else if(nums[m]>target){
  j=m-1;
  }
   else{
  return m;
  }
  }                   
                     
	return i;	     
		     
		     
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



