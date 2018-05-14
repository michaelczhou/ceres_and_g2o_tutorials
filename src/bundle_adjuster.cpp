#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bal_problem.h"
#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "snavely_reprojection_error.h"


namespace ceres {
namespace examples {

void SetLinearSolver(Solver::Options* options) {
  // 设置LinearSolver，可选项为："sparse_schur, dense_schur, iterative_schur,   
  // sparse_normal_cholesky, ""dense_qr, dense_normal_cholesky and cgnr."
  CHECK(StringToLinearSolverType(FLAGS_linear_solver,
                                 &options->linear_solver_type));
  // 设置PreconditionerType，可选项为："identity, jacobi, 
  // schur_jacobi, cluster_jacobi, ""cluster_tridiagonal."
  CHECK(StringToPreconditionerType(FLAGS_preconditioner,
                                   &options->preconditioner_type));
  // 设置VisibilityClusteringType，可选项为："single_linkage, 
  // canonical_views"
  CHECK(StringToVisibilityClusteringType(FLAGS_visibility_clustering,
                                         &options->visibility_clustering_type));
  // 设置SparseLinearAlgebraLibraryType，可选项为："suite_sparse,
  // cx_sparse"
  CHECK(StringToSparseLinearAlgebraLibraryType(
            FLAGS_sparse_linear_algebra_library,
            &options->sparse_linear_algebra_library_type));
  // 设置DenseLinearAlgebraLibraryType，可选项为："eigen,
  // lapack."
  CHECK(StringToDenseLinearAlgebraLibraryType(
            FLAGS_dense_linear_algebra_library,
            &options->dense_linear_algebra_library_type));
  // 线程数
  options->num_linear_solver_threads = FLAGS_num_threads;
  // 是否使用舍尔补
  options->use_explicit_schur_complement = FLAGS_explicit_schur_complement;
}

void SetOrdering(BALProblem* bal_problem, Solver::Options* options) {
  // 3D点数量
  const int num_points = bal_problem->num_points();
  // 点的维度（3维）
  const int point_block_size = bal_problem->point_block_size();
  // 点数据起始地址
  double* points = bal_problem->mutable_points();

  // 相机姿态数
  const int num_cameras = bal_problem->num_cameras();
  // 相机参数数（9/10）
  const int camera_block_size = bal_problem->camera_block_size();
  // 相机参数起始地址
  double* cameras = bal_problem->mutable_cameras();
  // true（使用内迭代对非线性进行）
  // false（细化每个成功的信任区域步骤）
  if (options->use_inner_iterations) {
    // 可选参数类型automatic, cameras, points，(points,cameras),(cameras,points)
    if (FLAGS_blocks_for_inner_iterations == "cameras") {
      LOG(INFO) << "Camera blocks for inner iterations";
      options->inner_iteration_ordering.reset(new ParameterBlockOrdering);
      for (int i = 0; i < num_cameras; ++i) {
        // 添加元素到一个组
        options->inner_iteration_ordering->AddElementToGroup(cameras + camera_block_size * i, 0);
      }
    } else if (FLAGS_blocks_for_inner_iterations == "points") {
      LOG(INFO) << "Point blocks for inner iterations";
      options->inner_iteration_ordering.reset(new ParameterBlockOrdering);
      for (int i = 0; i < num_points; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(points + point_block_size * i, 0);
      }
    } else if (FLAGS_blocks_for_inner_iterations == "cameras,points") {
      LOG(INFO) << "Camera followed by point blocks for inner iterations";
      options->inner_iteration_ordering.reset(new ParameterBlockOrdering);
      for (int i = 0; i < num_cameras; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(cameras + camera_block_size * i, 0);
      }
      for (int i = 0; i < num_points; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(points + point_block_size * i, 1);
      }
    } else if (FLAGS_blocks_for_inner_iterations == "points,cameras") {
      LOG(INFO) << "Point followed by camera blocks for inner iterations";
      options->inner_iteration_ordering.reset(new ParameterBlockOrdering);
      for (int i = 0; i < num_cameras; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(cameras + camera_block_size * i, 1);
      }
      for (int i = 0; i < num_points; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(points + point_block_size * i, 0);
      }
    } else if (FLAGS_blocks_for_inner_iterations == "automatic") {
      LOG(INFO) << "Choosing automatic blocks for inner iterations";
    } else {
      LOG(FATAL) << "Unknown block type for inner iterations: "
                 << FLAGS_blocks_for_inner_iterations;
    }
  }

  // BA问题有一个稀疏的结构，使它们能够适应更专业、更高效的解决方案策略。斯帕塞舒尔、登斯舒尔和迭代器的解决者利用了这种特殊的结构。
  //
  // 这可以通过指定Options::orderingtype=ceres::SCHUR，
  // 在这种情况下，Ceres将自动确定正确的参数块排序，
  // 或者手动指定一个合适的排序向量，定义Options::num_eliminate_blocks。
  if (FLAGS_ordering == "automatic") {
    return;
  }

  ceres::ParameterBlockOrdering* ordering =
      new ceres::ParameterBlockOrdering;

  // The points come before the cameras.
  for (int i = 0; i < num_points; ++i) {
    ordering->AddElementToGroup(points + point_block_size * i, 0);
  }

  for (int i = 0; i < num_cameras; ++i) {
    // When using axis-angle, there is a single parameter block for
    // the entire camera.
    ordering->AddElementToGroup(cameras + camera_block_size * i, 1);
  }

  options->linear_solver_ordering.reset(ordering);
}

void SetMinimizerOptions(Solver::Options* options) {
  // 最大迭代次数
  options->max_num_iterations = FLAGS_num_iterations;
  // 迭代过程是否输出
  options->minimizer_progress_to_stdout = true;
  // 线程数
  options->num_threads = FLAGS_num_threads;
  // 每次迭代的精度
  options->eta = FLAGS_eta;
  // 求解时间
  options->max_solver_time_in_seconds = FLAGS_max_solver_time;
  // 信任区间算法/nonmonotic
  options->use_nonmonotonic_steps = FLAGS_nonmonotonic_steps;
  if (FLAGS_line_search) {
    options->minimizer_type = ceres::LINE_SEARCH;
  }   
  CHECK(StringToTrustRegionStrategyType(FLAGS_trust_region_strategy,
                                        &options->trust_region_strategy_type));
  // 可选项：raditional_dogleg,subspace_dogleg
  CHECK(StringToDoglegType(FLAGS_dogleg, &options->dogleg_type));
  options->use_inner_iterations = FLAGS_inner_iterations;
}

void SetSolverOptionsFromFlags(BALProblem* bal_problem,
                               Solver::Options* options) {
  SetMinimizerOptions(options);
  SetLinearSolver(options);
  SetOrdering(bal_problem, options);
}

void BuildProblem(BALProblem* bal_problem, Problem* problem) {
  const int point_block_size = bal_problem->point_block_size();
  const int camera_block_size = bal_problem->camera_block_size();
  double* points = bal_problem->mutable_points();
  double* cameras = bal_problem->mutable_cameras();

  // Observations是特征的坐标 u v
  const double* observations = bal_problem->observations();
  for (int i = 0; i < bal_problem->num_observations(); ++i) {
       CostFunction* cost_function;
    // Each Residual block takes a point and a camera as input and
    // outputs a 2 dimensional residual.
    // 加入约束 特征位置
    cost_function =
        (FLAGS_use_quaternions)
        ? SnavelyReprojectionErrorWithQuaternions::Create(
            observations[2 * i + 0],
            observations[2 * i + 1])
        : SnavelyReprojectionError::Create(
            observations[2 * i + 0],
            observations[2 * i + 1]);

    // If enabled use Huber's loss function.
    LossFunction* loss_function = FLAGS_robustify ? new HuberLoss(1.0) : NULL;

    // 没一个特征对应这一个相机姿态和一个三维点
    // 加入点 和 相机
    double* camera =
        cameras + camera_block_size * bal_problem->camera_index()[i];
    double* point = points + point_block_size * bal_problem->point_index()[i];
    problem->AddResidualBlock(cost_function, loss_function, camera, point);
  }

  if (FLAGS_use_quaternions && FLAGS_use_local_parameterization) {
    LocalParameterization* camera_parameterization =
        new ProductParameterization(
            new QuaternionParameterization(),
            new IdentityParameterization(6));
    for (int i = 0; i < bal_problem->num_cameras(); ++i) {
      // ？？？？
      problem->SetParameterization(cameras + camera_block_size * i,
                                   camera_parameterization);
    }
  }
}


void SolveProblem(const char* filename) {
  // 实例化BALProblem 导入文件信息
  BALProblem bal_problem(filename, FLAGS_use_quaternions);

  if (!FLAGS_initial_ply.empty()) {
    bal_problem.WriteToPLYFile(FLAGS_initial_ply);
  }

  Problem problem;

  srand(FLAGS_random_seed);
  bal_problem.Normalize();
  // 添加噪声
  bal_problem.Perturb(FLAGS_rotation_sigma,
                      FLAGS_translation_sigma,
                      FLAGS_point_sigma);
  // 添加约束
  BuildProblem(&bal_problem, &problem);
  Solver::Options options;
  // 设置优化选项
  SetSolverOptionsFromFlags(&bal_problem, &options);
  options.gradient_tolerance = 1e-16;
  options.function_tolerance = 1e-16;
  Solver::Summary summary;
  //求解
  Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  if (!FLAGS_final_ply.empty()) {
    bal_problem.WriteToPLYFile(FLAGS_final_ply);
  }
}

}  // namespace examples
} 

// main
int main(int argc, char** argv) {
  CERES_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  if (FLAGS_input.empty()) {
    LOG(ERROR) << "Usage: bundle_adjuster --input=bal_problem";
    return 1;
  }

  CHECK(FLAGS_use_quaternions || !FLAGS_use_local_parameterization)
      << "--use_local_parameterization can only be used with "
      << "--use_quaternions.";
  ceres::examples::SolveProblem(FLAGS_input.c_str());
  return 0;
}
