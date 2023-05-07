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
	//dp[i]��ʾ�����į�׻��ѵ���С����
	std::vector<int> dp(cost.size() + 1, 0);
	//��ʼ��
	dp[0] = 0; // Ĭ�ϵ�һ�����ǲ�����������
	dp[1] = 0;



	for (int i = 2; i <= cost.size(); i++)
	{
		dp[i] = std::min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
	}


	return dp[cost.size()];
}

int Solution::uniquePaths(int m, int n)
{	
	// ����dp����
	std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));

	// ��ʼ��
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

	
	// ����dp
	std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));

	// ��ʼ��
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
	// 1�����鶨�壺�������i���õ����˻�dp[i]
	// 2���� i ���Ϊ j ��i - j
	// 3����ô�˻����� j *��i - j��
	std::vector<int> dp(n + 1, 0);
	dp[2] = 1;

	//Ҳ������ô��⣬j * (i - j) �ǵ����İ��������Ϊ��������ˣ�
	//��j * dp[i - j]�ǲ�ֳ������Լ��������ϵĸ�����ˡ�

	//j�Ľ��������� j < i - 1 ����ʵ j < i Ҳ�ǿ��Եģ�
	//�������Խ�ʡһ����������j = i - 1���Ļ���
	//��ʵ�� j = 1��ʱ����һ�����Ѿ�������ˣ��ظ����㣬���� j < i - 1
	for (int i = 3; i <= n; i++) // i ��ʾ����
	{	
		for (int j = 1; j < i - 1; j++) // j ��ʾ�سɼ���
		{
			dp[i] = std::max(dp[i], std::max(j * dp[i - j], j * (i - j)));
		}
	}


	//��Ϊ���һ����n ʹ֮�˻������ôһ���ǲ�ֳ�m��������ͬ��������˲������ġ�
	//��ô j ������ֻ��Ҫ������ n / 2 �Ϳ��ԣ������û�б�Ҫ�����ˣ�һ���������ֵ��

	//for (int i = 3; i <= n; i++) // i ��ʾ����
	//{
	//	for (int j = 1; j <= i/2; j++) // j ��ʾ�سɼ���
	//	{
	//		dp[i] = std::max(dp[i], std::max(j * dp[i - j], j * (i - j)));
	//	}
	//}


	return dp[n];
}

bool Solution::canPartition(std::vector<int>& nums)
{	
	int sum = 0;

	// dp[i]�е�i��ʾ�������ܺ�
	// ��Ŀ��˵��ÿ�������е�Ԫ�ز��ᳬ�� 100������Ĵ�С���ᳬ�� 200
	// �ܺͲ������20000���������ֻ��Ҫ����һ�룬����10001��С�Ϳ�����
	std::vector<int> dp(10001, 0);
	for (int i = 0; i < nums.size(); i++) {
		sum += nums[i];
	}
	// Ҳ����ʹ�ÿ⺯��һ�����
	// int sum = accumulate(nums.begin(), nums.end(), 0);
	if (sum % 2 == 1) return false;
	int target = sum / 2;

	// ��ʼ 01����
	for (int i = 0; i < nums.size(); i++) {
		for (int j = target; j >= nums[i]; j--) { // ÿһ��Ԫ��һ���ǲ����ظ����룬���ԴӴ�С����
			dp[j] = std::max(dp[j], dp[j - nums[i]] + nums[i]);
		}
	}
	// �����е�Ԫ�����ÿ��Դճ��ܺ�target
	if (dp[target] == target) return true;
	return false;
}


int Solution::numSquares(int n)
{
	// ˼·��ת������ȫ��������
	// ��ȫƽ����������Ʒ���ҿ���ѡȡ��Σ���������ȫ��������
	// �ճ������� n ��������������ȫƽ��������С����
	std::vector<int> dp(n + 1, INT32_MAX);
	dp[0] = 0;

	for (int i = 0; i <= n; i++) // ��������
	{	
		for (int j = 1; j * j <= i; j++) // ������Ʒ
		{
			dp[i] = std::min(dp[i], dp[i - j*j] + 1);
		}
	}
	
	
	return dp[n] == INT32_MAX? -1: dp[n];
}

int Solution::coinChange(std::vector<int>& coins, int amount)
{	

	//��������һ���ⶼ����ȫ��������
	//��ȫ�������⣺����Ʒ�������޴�ѡȡ����Ҳ���ڱ����ĸ���
	//�����������Դճ��ܽ�����ٵ�Ӳ�Ҹ���  ��ƷΪ coins������Ϊ amount
	std::vector<int> dp(amount + 1, INT32_MAX);// ���ȳ�ʼ��Ϊ���ֵ
	//��ʼ��
	dp[0] = 0; // ����dp[0] == 0 ��ʵ�ʺ��塣ֻ�Ǳ��ڵ���


	for (int i = 1; i <= amount; i++) // ��������
	{
		for (int j = 0; j < coins.size(); j++)  // ������Ʒ
		{
			if (coins[j] <= i && dp[i - coins[j]] != INT32_MAX)
			{
				dp[i] = std::min(dp[i], dp[i - coins[j]] + 1);
			}
		}
	}

	// �ղ���������dp[amount] == INT32_MAX
	return dp[amount] == INT32_MAX? -1: dp[amount];
}

bool Solution::wordBreak(std::string s, std::vector<std::string>& wordDict)
{	
	std::unordered_set<std::string> wordSet(wordDict.begin(), wordDict.end());

	// dp[i] ��ʾ�� i ��β���Ӵ� �������ַ����б�wordDictƴ�Ӷ���
	// ���������� s.length  ��Ʒ�� wordDict���ҿ����ظ�ѡȡ����Ϊ��ȫ��������
	// �������������⣡�����ȱ����������ٱ�����Ʒ
	std::vector<bool> dp(s.size() + 1, false);
	dp[0] = true;
	for (int i = 1; i <= s.size(); i++) // ��������
	{
		for (int j = 0; j < i; j++) // ������Ʒ
		{
			std::string word = s.substr(j, i - j); // substr(��ʼλ�ã���ȡ�ĸ�����
			std::cout << word << "     ";
			if (wordSet.find(word) != wordSet.end() && dp[j])
			{
				dp[i] = true;
			}

			if (dp[i]) {
				break; // ��֦
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

	// ˼·�����Ƿ�Ϊ������� ͵ 1 ��  ��͵ 0
	std::vector<int> dp(nums.size(), 0);  // ��ʾ ���� i �� ����͵�����ܼ�Ǯ
	// ��ʼ��
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

	// ����
    // ͵��һ��
	int res1 = rob2_range(nums, 0, nums.size() - 2);
    //͵���һ��
	int res2 = rob2_range(nums, 1, nums.size() - 1);

	return std::max(res1, res2);
}

std::vector<int> robTree(TreeNode* cur) {
	if (cur == NULL) return std::vector<int>{0, 0};
	std::vector<int> left = robTree(cur->left);
	std::vector<int> right = robTree(cur->right);
	// ͵cur����ô�Ͳ���͵���ҽڵ㡣
	int val1 = cur->val + left[0] + right[0];
	// ��͵cur����ô����͵Ҳ���Բ�͵���ҽڵ㣬��ȡ�ϴ�����
	int val2 = std::max(left[0], left[1]) + std::max(right[0], right[1]);
	return { val2, val1 };
}


int Solution::rob3(TreeNode* root)
{
	//����һ����Ҫ�����������Ϊͨ���ݹ麯���ķ���ֵ������һ�����㡣
	std::vector<int> result = robTree(root);
	return std::max(result[0], result[1]); // 0 ��ʾ ��͵  1 ��ʾ ͵
}

int Solution::maxProfit(std::vector<int>& prices)
{	
	if (prices.size() == 0)
	{
		return 0;
	}

	// ע�⣺������ֻ������һ��
	std::vector<std::vector<int>> dp(prices.size(), std::vector<int>(2, 0));

	dp[0][0] = 0;
	dp[0][1] = -prices[0];

	// �ֹ� 1  ���ֹ� 0
	for (int i = 1; i < prices.size(); i++)
	{
		dp[i][0] = std::max(dp[i-1][0], dp[i-1][1] + prices[i]);
		dp[i][1] = std::max(dp[i-1][1], -prices[i]); // ֻ������һ��
	}

	return dp[prices.size() - 1][0];
}

int Solution::maxProfit2(std::vector<int>& prices)
{	
	//��ÿһ�죬����Ծ����Ƿ���� / ����۹�Ʊ
	if (prices.size() == 0)
	{
		return 0;
	}

	std::vector<std::vector<int>> dp(prices.size() + 1, std::vector<int>(2, 0));

	// 0 ������   1  ����
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

	// �������ʽ���   ����ζ�ſ�������һ�Σ������������Σ�Ҳ���Բ�������
	// ���ڶ��״̬
	// dp[i][0] ������
	// dp[i][1] ��ʾ��һ������
	// dp[i][2] ��ʾ��һ������
	// dp[i][3] ��ʾ�ڶ�������
	// dp[i][4] ��ʾ�ڶ�������
	std::vector<std::vector<int>> dp(prices.size(), std::vector<int>(5, 0));

	// ��ʼ��
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
	// ����k�ʽ���
	// ���ڶ��״̬
	// dp[i][0] ������
	// dp[i][1] ��ʾ��һ������
	// dp[i][2] ��ʾ��һ������
	// dp[i][3] ��ʾ�ڶ�������
	// dp[i][4] ��ʾ�ڶ�������
	// ...
	std::vector<std::vector<int>> dp(prices.size(), std::vector<int>(2 * k + 1, 0));

	// ��ʼ��
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
	// �������һ֧��Ʊ��������Ʊ�����޷��ڵڶ��������Ʊ
	std::vector<std::vector<int>> dp(prices.size(), std::vector<int>(4, 0));

	// ��ʼ��  // ����״̬ 0  ������״̬ 1  �������� 2  ����Ϊ�䶳�� 3
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
	// ���� 1 ������ 0
	std::vector<std::vector<int>> dp(prices.size(), std::vector<int>(2, 0));

	// ��ʼ��
	dp[0][0] = 0;
	dp[0][1] = -prices[0];

	for (int i = 1; i < prices.size(); i++)
	{
		dp[i][0] = std::max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee); // ֻ�н���������ʱ��Ż��ȥ������
		dp[i][1] = std::max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
	}

	return dp[prices.size() - 1][0];
}

int Solution::lengthOfLIS(std::vector<int>& nums)
{	
	if (nums.size() <= 1) return nums.size();
	// �����У�ɾ������ɾ���������е�Ԫ�ض����ı�����Ԫ�ص�˳������������
	// dp��ʾ�� i ��β ����������г���Ϊ dp[i]
	std::vector<int>dp(nums.size(), 1); 
	// ��ʼ��
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
		if (dp[i] > result) result = dp[i]; // ȡ����������
	}

	return result;
}

int Solution::findLengthOfLCIS(std::vector<int>& nums)
{
	if (nums.size() == 1) return 1;
	//Ҫ��������
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
	//Ҫ��������
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
	// ��������
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
