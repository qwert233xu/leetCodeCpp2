#pragma once
#include <vector>
#include "TreeNode.h"
class Solution
{
public:
	Solution();
	~Solution();
	// 爬楼梯
	int climbStairs(int n);
	// 使用最小花费爬楼梯
	int minCostClimbingStairs(std::vector<int>& cost);
	// 不同路径
	int uniquePaths(int m, int n);
	// 不同路径II
	int uniquePathsWithObstacles(std::vector<std::vector<int>>& obstacleGrid);
	// 整数拆分
	int integerBreak(int n);
	// 分割等和子集
	bool canPartition(std::vector<int>& nums);
	// 完全平方数
	int numSquares(int n);
	// 零钱兑换     
	int coinChange(std::vector<int>& coins, int amount);
	// 单词拆分
	bool wordBreak(std::string s, std::vector<std::string>& wordDict);
	// 打家劫舍
	int rob(std::vector<int>& nums);
	// 打家劫舍II
	int rob2(std::vector<int>& nums);
	// 打家劫舍III
	int rob3(TreeNode* root);
	// 买卖股票的最佳时机
	int maxProfit(std::vector<int>& prices);
	// 买卖股票的最佳时机II
	int maxProfit2(std::vector<int>& prices);
	// 买卖股票的最佳时机III
	int maxProfit3(std::vector<int>& prices);
	// 买卖股票的最佳时机IV
	int maxProfit4(int k, std::vector<int>& prices);
	// 最佳买卖股票时机含冷冻期
	int maxProfit5(std::vector<int>& prices);
	// 买卖股票的最佳时机含手续费
	int maxProfit6(std::vector<int>& prices, int fee);
	// 最长上升子序列
	int lengthOfLIS(std::vector<int>& nums);
	// 最长连续递增序列
	int findLengthOfLCIS(std::vector<int>& nums);
	// 最长重复子数组
	int findLength(std::vector<int>& nums1, std::vector<int>& nums2);
	// 最长公共子序列
	int longestCommonSubsequence(std::string text1, std::string text2);
private:

};