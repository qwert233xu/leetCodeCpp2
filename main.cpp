#include <iostream>
#include "leetcode.h"

int main() {
	Solution s1;

	/*auto res = s1.climbStairs(3);
	std::cout << res << std::endl;*/

	/*auto res = s1.uniquePaths(3, 7);
	std::cout << res << std::endl;*/

	/*std::vector<std::vector<int>> obstacleGrid({ {0,0} });
	auto res = s1.uniquePathsWithObstacles(obstacleGrid);
	std::cout << res << std::endl;*/

	/*auto res = s1.integerBreak(10);
	std::cout << res << std::endl;*/

	std::vector<std::string> des({ "leet", "code" });
	auto res = s1.wordBreak("leetcode", des);
	std::cout << "============" << std::endl;
	std::cout << res << std::endl;


	return 0;
}