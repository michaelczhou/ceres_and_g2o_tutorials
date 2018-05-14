#ifndef CERES_EXAMPLES_BAL_PROBLEM_H_
#define CERES_EXAMPLES_BAL_PROBLEM_H_
#include <string>

namespace ceres {
namespace examples {

class BALProblem {
 public:
  explicit BALProblem(const std::string& filename, bool use_quaternions);
  ~BALProblem();

  void WriteToFile(const std::string& filename) const;
  void WriteToPLYFile(const std::string& filename) const;


  // 将重建的“中心”移到原点，中心是通过计算点的marginal median来确定的。
  // 然后对重构进行缩放，以便从原点处得到的点的 median absolute deviation 是100.0
  //
  // 问题的重投影误差保持不变
  void Normalize();

  // 用特定标准差的随机正态分布值扰乱相机姿态和几何结构
  void Perturb(const double rotation_sigma,
               const double translation_sigma,
               const double point_sigma);

    //相机参数的个数
  int camera_block_size()      const { return use_quaternions_ ? 10 : 9; }
  // 点参数的个数
  int point_block_size()       const { return 3;                         }
  int num_cameras()            const { return num_cameras_;              }
  int num_points()             const { return num_points_;               }
  int num_observations()       const { return num_observations_;         }
  int num_parameters()         const { return num_parameters_;           }
  const int* point_index()     const { return point_index_;              }
  const int* camera_index()    const { return camera_index_;             }
  const double* observations() const { return observations_;             }
  const double* parameters()   const { return parameters_;               }
  const double* cameras()      const { return parameters_;               }
  double* mutable_cameras()          { return parameters_;               }
  double* mutable_points() {
    return parameters_  + camera_block_size() * num_cameras_;
  }

 private:
  void CameraToAngleAxisAndCenter(const double* camera,
                                  double* angle_axis,
                                  double* center) const;

  void AngleAxisAndCenterToCamera(const double* angle_axis,
                                  const double* center,
                                  double* camera) const;
  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;
  bool use_quaternions_;

  int* point_index_;
  int* camera_index_;
  double* observations_;
  // The parameter vector is laid out as follows
  // [camera_1, ..., camera_n, point_1, ..., point_m]
  double* parameters_;
};

}  // namespace examples
}  // namespace ceres

#endif  // CERES_EXAMPLES_BAL_PROBLEM_H_
