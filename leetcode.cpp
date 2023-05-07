#include <iostream>
#include <vector>
#include "leetcode.h"
#include <algorithm>
#include <unordered_set>

Solution::Solution()
{
}

Solution::~Solution()
{
}

int Solution::climbStairs(int n)
{	
	if (n == 0) return 0;
	if (n == 1) return 1;
	int* dp = new int[n + 1];
	dp[0] = 0;
	dp[1] = 1;
	dp[2] = 2;
	for (int i = 3; i <= n; i++)
	{
		dp[i] = dp[i - 1] + dp[i - 2];
	}


	return dp[n];
}

int Solution::minCostClimbingStairs(std::vector<int>& cost)
{	
	//dp[i]表示到达第i台阶花费的最小体力
	std::vector<int> dp(cost.size() + 1, 0);
	//初始化
	dp[0] = 0; // 默认第一步都是不花费体力的
	dp[1] = 0;



	for (int i = 2; i <= cost.size(); i++)
	{
		dp[i] = std::min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
	}


	return dp[cost.size()];
}

int Solution::uniquePaths(int m, int n)
{	
	// 定义dp数组
	std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));

	// 初始化
	for (int i = 0; i < n; i++)
	{
		dp[0][i] = 1;
	}

	for (int i = 0; i < m; i++)
	{
		dp[i][0] = 1;
	}

	for (int i = 1; i < m; i++)
	{
		for (int j = 1; j < n; j++)
		{
			dp[i][j] = dp[i-1][j] + dp[i][j-1];
		}
	}


	return dp[m-1][n-1];

}

int Solution::uniquePathsWithObstacles(std::vector<std::vector<int>>& obstacleGrid)
{
	int m = obstacleGrid.size();
	int n = obstacleGrid[0].size();

	
	// 创建dp
	std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));

	// 初始化
	int i = 0;
	for (; i < n; i++)
	{
		if (obstacleGrid[0][i] == 0)
		{
			dp[0][i] = 1;
		}
		else
		{
			break;
		}
	}
	for (; i < n; i++)
	{
		dp[0][i] = 0;
	}
	if (m == 1) {
		return dp[0][n - 1];
	}

	
	int j = 0;
	for (; j < m; j++)
	{
		if (obstacleGrid[j][0] == 0)
		{
			dp[j][0] = 1;
		}
		else
		{
			break;
		}
	}
	for (; j < m; j++)
	{
		dp[j][0] = 0;
	}

	if (n == 1) {
		return dp[m - 1][0];
	}

	for (int i = 1; i < m; i++)
	{
		for (int j = 1; j < n; j++)
		{
			if (obstacleGrid[i][j] != 1)
			{
				dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
			}
		}
	}

	
	return dp[m-1][n-1];
}

int Solution::integerBreak(int n)
{	
	// 1、数组定义：拆分数字i，得到最大乘积dp[i]
	// 2、将 i 拆分为 j 、i - j
	// 3、那么乘积就是 j *（i - j）
	std::vector<int> dp(n + 1, 0);
	dp[2] = 1;

	//也可以这么理解，j * (i - j) 是单纯的把整数拆分为两个数相乘，
	//而j * dp[i - j]是拆分成两个以及两个以上的个数相乘。

	//j的结束条件是 j < i - 1 ，其实 j < i 也是可以的，
	//不过可以节省一步，例如让j = i - 1，的话，
	//其实在 j = 1的时候，这一步就已经拆出来了，重复计算，所以 j < i - 1
	for (int i = 3; i <= n; i++) // i 表示长度
	{	
		for (int j = 1; j < i - 1; j++) // j 表示截成几段
		{
			dp[i] = std::max(dp[i], std::max(j * dp[i - j], j * (i - j)));
		}
	}


	//因为拆分一个数n 使之乘积最大，那么一定是拆分成m个近似相同的子数相乘才是最大的。
	//那么 j 遍历，只需要遍历到 n / 2 就可以，后面就没有必要遍历了，一定不是最大值。

	//for (int i = 3; i <= n; i++) // i 表示长度
	//{
	//	for (int j = 1; j <= i/2; j++) // j 表示截成几段
	//	{
	//		dp[i] = std::max(dp[i], std::max(j * dp[i - j], j * (i - j)));
	//	}
	//}


	return dp[n];
}

bool Solution::canPartition(std::vector<int>& nums)
{	
	int sum = 0;

	// dp[i]中的i表示背包内总和
	// 题目中说：每个数组中的元素不会超过 100，数组的大小不会超过 200
	// 总和不会大于20000，背包最大只需要其中一半，所以10001大小就可以了
	std::vector<int> dp(10001, 0);
	for (int i = 0; i < nums.size(); i++) {
		sum += nums[i];
	}
	// 也可以使用库函数一步求和
	// int sum = accumulate(nums.begin(), nums.end(), 0);
	if (sum % 2 == 1) return false;
	int target = sum / 2;

	// 开始 01背包
	for (int i = 0; i < nums.size(); i++) {
		for (int j = target; j >= nums[i]; j--) { // 每一个元素一定是不可重复放入，所以从大到小遍历
			dp[j] = std::max(dp[j], dp[j - nums[i]] + nums[i]);
		}
	}
	// 集合中的元素正好可以凑成总和target
	if (dp[target] == target) return true;
	return false;
}


int Solution::numSquares(int n)
{
	// 思路：转换成完全背包问题
	// 完全平方数就是物品，且可以选取多次，所以是完全背包问题
	// 凑成正整数 n 就是容量，求完全平方数的最小个数
	std::vector<int> dp(n + 1, INT32_MAX);
	dp[0] = 0;

	for (int i = 0; i <= n; i++) // 遍历容量
	{	
		for (int j = 1; j * j <= i; j++) // 遍历物品
		{
			dp[i] = std::min(dp[i], dp[i - j*j] + 1);
		}
	}
	
	
	return dp[n] == INT32_MAX? -1: dp[n];
}

int Solution::coinChange(std::vector<int>& coins, int amount)
{	

	//这道题和上一道题都是完全背包问题
	//完全背包问题：即物品可以无限次选取，但也存在背包的概念
	//这道题是求可以凑成总金额最少的硬币个数  物品为 coins，容量为 amount
	std::vector<int> dp(amount + 1, INT32_MAX);// 首先初始化为最大值
	//初始化
	dp[0] = 0; // 这里dp[0] == 0 无实际含义。只是便于递推


	for (int i = 1; i <= amount; i++) // 遍历容量
	{
		for (int j = 0; j < coins.size(); j++)  // 遍历物品
		{
			if (coins[j] <= i && dp[i - coins[j]] != INT32_MAX)
			{
				dp[i] = std::min(dp[i], dp[i - coins[j]] + 1);
			}
		}
	}

	// 凑不出来，则dp[amount] == INT32_MAX
	return dp[amount] == INT32_MAX? -1: dp[amount];
}

bool Solution::wordBreak(std::string s, std::vector<std::string>& wordDict)
{	
	std::unordered_set<std::string> wordSet(wordDict.begin(), wordDict.end());

	// dp[i] 表示以 i 结尾的子串 可以由字符串列表wordDict拼接而成
	// 背包容量是 s.length  物品是 wordDict，且可以重复选取，则为完全背包问题
	// 这里是排列问题！！！先遍历容量，再遍历物品
	std::vector<bool> dp(s.size() + 1, false);
	dp[0] = true;
	for (int i = 1; i <= s.size(); i++) // 遍历容量
	{
		for (int j = 0; j < i; j++) // 遍历物品
		{
			std::string word = s.substr(j, i - j); // substr(起始位置，截取的个数）
			std::cout << word << "     ";
			if (wordSet.find(word) != wordSet.end() && dp[j])
			{
				dp[i] = true;
			}

			if (dp[i]) {
				break; // 剪枝
			}
		}
	}


	return dp[s.size()];
}

int Solution::rob(std::vector<int>& nums)
{   
	if (nums.size() == 1)
	{
		return nums[0];
	}
	if (nums.size() == 2)
	{
		return std::max(nums[0], nums[1]);
	}

	// 思路：就是分为两种情况 偷 1 与  不偷 0
	std::vector<int> dp(nums.size(), 0);  // 表示 到第 i 家 所能偷盗的总价钱
	// 初始化
	dp[0] = nums[0];
	dp[1] = std::max(nums[0], nums[1]);

	for (int i = 2; i < nums.size(); i++)
	{
		dp[i] = std::max(dp[i - 2] + nums[i], dp[i - 1]);
	}



	return dp[nums.size() - 1];
}


int rob2_range(std::vector<int>& nums, int start, int end) {
	if (start == end)
	{
		return nums[start];
	}

	std::vector<int> dp(nums.size());
	dp[start] = nums[start];
	dp[start + 1] = std::max(nums[start], nums[start + 1]);

	for (int i = start + 2; i <= end; i++)
	{
		dp[i] = std::max(dp[i - 2] + nums[i], dp[i - 1]);
	}
	return dp[end];
}

int Solution::rob2(std::vector<int>& nums)
{	
	if (nums.size() == 0)
	{
		return 0;
	}
	if (nums.size() == 1)
	{
		return nums[0];
	}

	// 环形
    // 偷第一家
	int res1 = rob2_range(nums, 0, nums.size() - 2);
    //偷最后一家
	int res2 = rob2_range(nums, 1, nums.size() - 1);

	return std::max(res1, res2);
}

std::vector<int> robTree(TreeNode* cur) {
	if (cur == NULL) return std::vector<int>{0, 0};
	std::vector<int> left = robTree(cur->left);
	std::vector<int> right = robTree(cur->right);
	// 偷cur，那么就不能偷左右节点。
	int val1 = cur->val + left[0] + right[0];
	// 不偷cur，那么可以偷也可以不偷左右节点，则取较大的情况
	int val2 = std::max(left[0], left[1]) + std::max(right[0], right[1]);
	return { val2, val1 };
}


int Solution::rob3(TreeNode* root)
{
	//本题一定是要后序遍历，因为通过递归函数的返回值来做下一步计算。
	std::vector<int> result = robTree(root);
	return std::max(result[0], result[1]); // 0 表示 不偷  1 表示 偷
}

int Solution::maxProfit(std::vector<int>& prices)
{	
	if (prices.size() == 0)
	{
		return 0;
	}

	// 注意：本题中只能买卖一次
	std::vector<std::vector<int>> dp(prices.size(), std::vector<int>(2, 0));

	dp[0][0] = 0;
	dp[0][1] = -prices[0];

	// 持股 1  不持股 0
	for (int i = 1; i < prices.size(); i++)
	{
		dp[i][0] = std::max(dp[i-1][0], dp[i-1][1] + prices[i]);
		dp[i][1] = std::max(dp[i-1][1], -prices[i]); // 只能买卖一次
	}

	return dp[prices.size() - 1][0];
}

int Solution::maxProfit2(std::vector<int>& prices)
{	
	//在每一天，你可以决定是否购买和 / 或出售股票
	if (prices.size() == 0)
	{
		return 0;
	}

	std::vector<std::vector<int>> dp(prices.size() + 1, std::vector<int>(2, 0));

	// 0 不持有   1  持有
	dp[0][0] = 0;
	dp[0][1] = -prices[0];


	for (int i = 1; i < prices.size(); i++)
	{
		dp[i][0] = std::max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
		dp[i][1] = std::max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
	}

	return dp[prices.size() - 1][0];
}

int Solution::maxProfit3(std::vector<int>& prices)
{	

	// 至多两笔交易   这意味着可以买卖一次，可以买卖两次，也可以不买卖。
	// 存在多个状态
	// dp[i][0] 不买卖
	// dp[i][1] 表示第一次买入
	// dp[i][2] 表示第一次卖出
	// dp[i][3] 表示第二次买入
	// dp[i][4] 表示第二次卖出
	std::vector<std::vector<int>> dp(prices.size(), std::vector<int>(5, 0));

	// 初始化
	dp[0][0] = 0;
	dp[0][1] = -prices[0];
	dp[0][2] = 0;
	dp[0][3] = -prices[0];
	dp[0][4] = 0;

	for (int i = 1; i < prices.size(); i++)
	{
		dp[i][0] = dp[i - 1][0];
		dp[i][1] = std::max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
		dp[i][2] = std::max(dp[i - 1][2], dp[i - 1][1] + prices[i]);
		dp[i][3] = std::max(dp[i - 1][3], dp[i - 1][2] - prices[i]);
		dp[i][4] = std::max(dp[i - 1][4], dp[i - 1][3] + prices[i]);
	}

	return dp[prices.size()-1][4];
}

int Solution::maxProfit4(int k, std::vector<int>& prices)
{	
	// 至多k笔交易
	// 存在多个状态
	// dp[i][0] 不买卖
	// dp[i][1] 表示第一次买入
	// dp[i][2] 表示第一次卖出
	// dp[i][3] 表示第二次买入
	// dp[i][4] 表示第二次卖出
	// ...
	std::vector<std::vector<int>> dp(prices.size(), std::vector<int>(2 * k + 1, 0));

	// 初始化
	for (int i = 0; i < 2 * k + 1; i++)
	{
		if (i % 2 == 1)
		{
			dp[0][i] = -prices[0];
		}
	}


	for (int i = 1; i < prices.size(); i++)
	{	
		for (int j = 1; j < 2 * k + 1; j++)
		{	
			if (j % 2 == 1)
			{
				dp[i][j] = std::max(dp[i - 1][j], dp[i - 1][j - 1] - prices[i]);
			}
			else 
			{
				dp[i][j] = std::max(dp[i - 1][j], dp[i - 1][j - 1] + prices[i]);
			}
		}
	}

	return dp[prices.size() - 1][2 * k];
}

int Solution::maxProfit5(std::vector<int>& prices)
{
	// 多次买卖一支股票，卖出股票后你无法在第二天买入股票
	std::vector<std::vector<int>> dp(prices.size(), std::vector<int>(4, 0));

	// 初始化  // 持有状态 0  不持有状态 1  今天卖出 2  今天为冷冻期 3
	dp[0][0] = -prices[0];
	dp[0][1] = 0;
	dp[0][2] = 0;
	dp[0][3] = 0;

	for (int i = 1; i < prices.size(); i++)
	{

		dp[i][0] = std::max(dp[i - 1][0], std::max(dp[i - 1][1] - prices[i], dp[i - 1][3] - prices[i]));
		dp[i][1] = std::max(dp[i - 1][1], dp[i - 1][3]);
		dp[i][2] = dp[i - 1][0] + prices[i];
		dp[i][3] = dp[i - 1][2];

	}

	return std::max(dp[prices.size() - 1][1], std::max(dp[prices.size() - 1][2], dp[prices.size() - 1][3]));
}

int Solution::maxProfit6(std::vector<int>& prices, int fee)
{
	// 持有 1 不持有 0
	std::vector<std::vector<int>> dp(prices.size(), std::vector<int>(2, 0));

	// 初始化
	dp[0][0] = 0;
	dp[0][1] = -prices[0];

	for (int i = 1; i < prices.size(); i++)
	{
		dp[i][0] = std::max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee); // 只有今天卖出的时候才会减去手续费
		dp[i][1] = std::max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
	}

	return dp[prices.size() - 1][0];
}

int Solution::lengthOfLIS(std::vector<int>& nums)
{	
	if (nums.size() <= 1) return nums.size();
	// 子序列：删除（或不删除）数组中的元素而不改变其余元素的顺序都属于子序列
	// dp表示以 i 结尾 最长递增子序列长度为 dp[i]
	std::vector<int>dp(nums.size(), 1); 
	// 初始化
	int result = 0;
	for (int i = 1; i < nums.size(); i++)
	{
		for (int j = 0; j < i; j++)
		{	
			if (nums[i] > nums[j])
			{
				dp[i] = std::max(dp[i], dp[j] + 1);
			}
		}
		if (dp[i] > result) result = dp[i]; // 取长的子序列
	}

	return result;
}

int Solution::findLengthOfLCIS(std::vector<int>& nums)
{
	if (nums.size() == 1) return 1;
	//要求：连续性
	std::vector<int> dp(nums.size(), 1);
	int res = 0;
	for (int i = 1; i < nums.size(); i++)
	{
		if (nums[i] > nums[i - 1])
		{
			dp[i] = dp[i - 1] + 1;
		}
		if (res < dp[i])
		{
			res = dp[i];
		}
	}


	return res;
}

int Solution::findLength(std::vector<int>& nums1, std::vector<int>& nums2)
{
	//要求：连续性
	std::vector<std::vector<int>> dp(nums1.size() + 1, std::vector<int>(nums2.size() + 1, 0));
	int res = 0;
	for (int i = 1; i <= nums1.size(); i++)
	{
		for (int j = 1; j <= nums2.size(); j++)
		{
			if (nums1[i - 1] == nums2[j - 1])
			{
				dp[i][j] = dp[i - 1][j - 1] + 1;
			}
			if (res < dp[i][j])
			{
				res = dp[i][j];
			}
		}
	}

	return res;

}

int Solution::longestCommonSubsequence(std::string text1, std::string text2)
{
	// 不用连续
	std::vector<std::vector<int>> dp(text1.size() + 1, std::vector<int>(text2.size() + 1, 0));
	int res = 0;
	for (int i = 1; i <= text1.size(); i++)
	{
		for (int j = 1; j <= text2.size(); j++)
		{
			if (text1[i - 1] == text2[j - 1])
			{
				dp[i][j] = dp[i - 1][j - 1] + 1;
			}
			else
			{
				dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
			}


			if (res < dp[i][j])
			{
				res = dp[i][j];
			}
		}
	}

	return res;

}
