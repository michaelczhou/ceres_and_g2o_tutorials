#include<iostream>
#include<ceres/ceres.h>

using namespace std;
using namespace ceres;

//第一部分：构建代价函数，重载（）符号，仿函数的小技巧
struct CostFunctor {
   template <typename T>
   bool operator()(const T* const x, T* residual) const {
     residual[0] = T(7.0) - x[0];
     return true;
   }
};

//主函数
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // 寻优参数x的初始值，为5
  double initial_x = 5.0;
  double x = initial_x;

// 第二部分：构建寻优问题.构建残差方程
Problem problem;
  CostFunction* cost_function =
      new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor); //使用自动求导，将之前的代价函数结构体传入，第一个1是输出维度，即残差的维度，第二个1是输入维度，即待寻优参数x的维度。
  problem.AddResidualBlock(cost_function, NULL, &x); //向问题中添加误差项，本问题比较简单，添加一个就行。&x表示x是待寻优参数
//编写一个重载()运算，而且必须使用模板类型，所有的输入参数和输出参数都要使用T类型。
//第一个模板参数为残差对象，也就是最开始写的那个那个带有重载()运算符的结构体，第二个模板参数为残差个数，第三个模板参数为未知数个数，最后参数是结构体对象。
  /*Ceres库中其实还有更多的求导方法可供选择（虽然自动求导的确是最省心的，而且一般情况下也是最快的。。。）。这里就简要介绍一下其他的求导方法：
数值求导法（一般比自动求导法收敛更慢，且更容易出现数值错误）
  CostFunction* cost_function =
      new NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL, 1, 1>(
      new NumericDiffCostFunctor);
   problem.AddResidualBlock(cost_function, NULL, &x);
 */

  //第三部分： 配置并运行求解器
  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR; //配置增量方程的解法
  options.minimizer_progress_to_stdout = true;//输出到cout
  Solver::Summary summary;//优化信息
  Solve(options, &problem, &summary);//求解!!!

  std::cout << summary.BriefReport() << "\n";//输出优化的简要信息
//最终结果
  std::cout << "x : " << initial_x
            << " -> " << x << "\n";
  return 0;
}
//通过实验发现除了多线程以及linear_solver_type，别的对优化性能和结果影响不是很大：
