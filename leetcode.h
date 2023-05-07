#pragma once
#include <vector>
#include "TreeNode.h"
class Solution
{
public:
	Solution();
	~Solution();
	// ��¥��
	int climbStairs(int n);
	// ʹ����С������¥��
	int minCostClimbingStairs(std::vector<int>& cost);
	// ��ͬ·��
	int uniquePaths(int m, int n);
	// ��ͬ·��II
	int uniquePathsWithObstacles(std::vector<std::vector<int>>& obstacleGrid);
	// �������
	int integerBreak(int n);
	// �ָ�Ⱥ��Ӽ�
	bool canPartition(std::vector<int>& nums);
	// ��ȫƽ����
	int numSquares(int n);
	// ��Ǯ�һ�     
	int coinChange(std::vector<int>& coins, int amount);
	// ���ʲ��
	bool wordBreak(std::string s, std::vector<std::string>& wordDict);
	// ��ҽ���
	int rob(std::vector<int>& nums);
	// ��ҽ���II
	int rob2(std::vector<int>& nums);
	// ��ҽ���III
	int rob3(TreeNode* root);
	// ������Ʊ�����ʱ��
	int maxProfit(std::vector<int>& prices);
	// ������Ʊ�����ʱ��II
	int maxProfit2(std::vector<int>& prices);
	// ������Ʊ�����ʱ��III
	int maxProfit3(std::vector<int>& prices);
	// ������Ʊ�����ʱ��IV
	int maxProfit4(int k, std::vector<int>& prices);
	// ���������Ʊʱ�����䶳��
	int maxProfit5(std::vector<int>& prices);
	// ������Ʊ�����ʱ����������
	int maxProfit6(std::vector<int>& prices, int fee);
	// �����������
	int lengthOfLIS(std::vector<int>& nums);
	// �������������
	int findLengthOfLCIS(std::vector<int>& nums);
	// ��ظ�������
	int findLength(std::vector<int>& nums1, std::vector<int>& nums2);
	// �����������
	int longestCommonSubsequence(std::string text1, std::string text2);
private:

};