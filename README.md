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



