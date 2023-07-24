#ifndef _TRACK_HPP_
#define _TRACK_HPP_

#include <base.hpp>

namespace hw
{
using namespace Eigen;

//* Optimize the pose of the frame by minimizing the photometric error of feature patches.
class Tracker
{
    typedef std::vector<std::pair<int, int> > PatternType;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 2> MatX2;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Tracker(
        int         top_level,      //!< start level numbers of alignment       
        int         bottom_level,   //!< end level numbers of alignment
        size_t      num_iter_max,   //!< max iteration from father class
        PatternType pattern         //!< patch pattern of alignment
    );

    size_t run(Frame::Ptr &frame_ref, Frame::Ptr &frame_cur);

private:

    int                 level_cur_;             //!< current level for optimizing
    int                 num_level_max_;         //!< top level
    int                 num_level_min_;         //!< bottom level
    size_t              num_ftr_active_;        //!< number of feature used to aligne image
    cv::Mat             patch_ref_;             //!< the reference patch rows=featurenum cols=patchsize 
    MatX2               patch_dI_;              //!< gradient of patch
    PatternType         pattern_;               //!< pattern of patch, should put feature(center) to (0,0)
    std::vector<bool>   visible_ftr_;           //!< the visibility of feature projected from reference frame  
    Frame::Ptr          frame_ref_;             //!< frame of reference 
    Frame::Ptr          frame_cur_;             //!< frame of current

    size_t              num_iter_max_;          //!< max numbers of iteration
    size_t              num_iter_;              //!< current number of iteration
    double              epsilon_;               //!< convergence condition
    bool                stop_;                  //!< stop flag
    size_t              num_obser_;             //!< numbers of measurement]

    Matrix<double, Dynamic, 6>  jacobian_;      //!< Jacobians of state 
    Matrix<double, Dynamic, 1>  residual_;      //!< residuals between prediction and observation
    Matrix<double, 6, 6>        hessian_;       //!< Hessians Matrix
    Matrix<double, 6, 1>        jres_;          //!< Jacobians * residuals
    Matrix<double, 6, 1>        delta_x_;       //!< increment
    MatrixXd                    cov_;           //!< covariance of observation
    double                      chi2_;          //!< Sum of squares of random variables obeying Gaussian distribution
    bool                        converged_;     //!< whether converge to enough small

    bool runOptimize(Sophus::SE3d& state);
    void optimizeGaussNetow(Sophus::SE3d& state);
    void preCompute();
    void computeResidual(const Sophus::SE3d& state);
    void update(const Sophus::SE3d& old_state, Sophus::SE3d& new_state);
    bool solve();

    double getChi2()
    {
        return residual_.transpose()*residual_;
    };

    // Reset
    void reset()
    {
        chi2_=0;
        num_obser_=0;
        num_iter_ = 0;
        stop_=false;
        converged_=false;
    };


};

}




#endif // _TRACK_HPP_