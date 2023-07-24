#include "track.hpp"
#include <eigen3/Eigen/StdVector>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Cholesky>
#include <iostream>

hw::Tracker::Tracker(
    int         top_level,   
    int         bottom_level,
    size_t      num_iter_max,     
    PatternType pattern ) :
    num_level_max_(top_level),
    num_level_min_(bottom_level),
    pattern_(pattern),
    num_iter_max_(num_iter_max)
{
    num_iter_ = 0;
    epsilon_ = 0.00000001;
    stop_ = false;
    num_obser_ = 0;
};

size_t hw::Tracker::run(Frame::Ptr &frame_ref, Frame::Ptr &frame_cur)
{
    if(frame_ref->features.empty())
    {
        std::cout<<"[error]: There is nothing in features!"<<std::endl;
    }

    patch_ref_ = cv::Mat::zeros(frame_ref->features.size(), pattern_.size(), CV_32F);
    patch_dI_.resize(frame_ref->features.size()*pattern_.size(), Eigen::NoChange);
    visible_ftr_.resize(frame_ref->features.size(), false);
    jacobian_.resize(frame_ref->features.size()*pattern_.size(), Eigen::NoChange);
    residual_.resize(frame_ref->features.size()*pattern_.size(), Eigen::NoChange);
    
    num_ftr_active_ = 0;
    
    // Get the optimizing state
    Sophus::SE3d T_r_w = frame_ref->getPose();
    Sophus::SE3d T_c_w = frame_cur->getPose();
    Sophus::SE3d T_cur_ref = T_c_w * T_r_w.inverse();

    // size_t all_obser;
    // Frames to aligne 
    frame_cur_ = frame_cur;
    frame_ref_ = frame_ref;
    
    for(level_cur_=num_level_max_; level_cur_ >= num_level_min_; level_cur_--)
    {
        preCompute();  // get reference patch
 
        runOptimize(T_cur_ref); // first compute Jacobian and residual, then solve

        // all_obser += num_ftr_active_; // all pixels we used
    }
    
    // update result
    T_c_w = T_cur_ref*T_r_w;
    frame_cur->setPose(T_c_w);

    return num_ftr_active_;

}


void hw::Tracker::preCompute()
{
    const cv::Mat img_ref_pyr = frame_ref_->getImage(level_cur_);
    patch_dI_.setZero();
    size_t num_ftr = frame_ref_->features.size();
    const double scale = 1.f/(1<<level_cur_); // current level
    int stride = img_ref_pyr.cols; // step

    auto iter_vis_ftr = visible_ftr_.begin();

    for(size_t i_ftr=0; i_ftr<num_ftr; ++i_ftr, ++iter_vis_ftr)
    {
        //! feature on level_cur
        const int ftr_x = frame_ref_->features[i_ftr].pt.x; // feature coordiante
        const int ftr_y = frame_ref_->features[i_ftr].pt.y;
        const float ftr_pyr_x = ftr_x*scale; // current level feature coordiante
        const float ftr_pyr_y = ftr_y*scale;

        bool is_in_frame = true;
        //* patch pointer on head
        float* data_patch_ref = reinterpret_cast<float*>(patch_ref_.data)+i_ftr*pattern_.size();
        //* img_ref_pyr pointer on feature
        // float* data_img_ref = reinterpret_cast<float*> (img_ref_pyr.data)+ftr_pyr_y_i*stride+ftr_pyr_x_i;
        int pattern_count = 0;

        for(auto iter_pattern : pattern_)
        {
            int x_pattern = iter_pattern.first;     // offset from feature 
            int y_pattern = iter_pattern.second;    
            // Is in the image of current level
            if( x_pattern + ftr_pyr_x < 1 || y_pattern + ftr_pyr_y < 1 || 
                x_pattern + ftr_pyr_x > img_ref_pyr.cols - 2 || 
                y_pattern + ftr_pyr_y > img_ref_pyr.rows - 2 )
            {
                is_in_frame = false;
                break;
            }
            
            // reference patch
            // const float* data_img_pattern = data_img_ref + y_pattern*stride + x_pattern;
            const float x_img_pattern = ftr_pyr_x + x_pattern;
            const float y_img_pattern = ftr_pyr_y + y_pattern;
            data_patch_ref[pattern_count] = utils::interpolate_uint8((img_ref_pyr.data), x_img_pattern, y_img_pattern, stride);
            
            ++pattern_count;
        }
        
        if(is_in_frame)
        {
            *iter_vis_ftr = true; 
        }
    }
}

//! NOTE myself: need to make sure residual match Jacobians
void hw::Tracker::computeResidual(const Sophus::SE3d& state)
{
    residual_.setZero();
    jacobian_.setZero();
    patch_dI_.setZero();
    Sophus::SE3d T_cur_ref = state;
    // cv::Mat img_cur_pyr = frame_cur_->getImage(level_cur_);
    const cv::Mat img_cur_pyr = frame_cur_->getImage(level_cur_);
    size_t num_ftr = frame_ref_->features.size();
    int num_pattern = pattern_.size();
    int stride = img_cur_pyr.cols;
    const double scale = 1.f/(1<<level_cur_);
    auto iter_vis_ftr = visible_ftr_.begin();

    for(size_t i_ftr=0; i_ftr<num_ftr; ++i_ftr, ++iter_vis_ftr)
    {
        if(!*iter_vis_ftr)
            continue;
        
        // get point in reference
        const int ftr_ref_x = frame_ref_->features[i_ftr].pt.x;
        const int ftr_ref_y = frame_ref_->features[i_ftr].pt.y;
        const double ftr_depth_ref = frame_ref_->getDepth(ftr_ref_x, ftr_ref_y);
        // project to current 
        Eigen::Vector3d point_ref(Cam::pixel2unitPlane(ftr_ref_x, ftr_ref_y)*ftr_depth_ref);
        Eigen::Vector3d point_cur(T_cur_ref*point_ref);
        Eigen::Vector2d ftr_cur = Cam::project(point_cur)*scale;

        const float ftr_cur_x = ftr_cur.x();
        const float ftr_cur_y = ftr_cur.y();
        // const int ftr_cur_x_i = floor(ftr_cur_x);
        // const int ftr_cur_y_i = floor(ftr_cur_y);

        bool is_in_frame = true;
        Eigen::VectorXd residual_pattern = Eigen::VectorXd::Zero(num_pattern);
        size_t pattern_count = 0; 
        float hw = 1;

        float* data_patch_ref = reinterpret_cast<float*>(patch_ref_.data) + i_ftr*num_pattern;

        for(auto iter_pattern : pattern_)
        {
            int x_pattern = iter_pattern.first;     // offset from feature 
            int y_pattern = iter_pattern.second;    
            // Is in the image of current level
            if( x_pattern + ftr_cur_x < 1 || y_pattern + ftr_cur_y < 1 || 
                x_pattern + ftr_cur_x > img_cur_pyr.cols - 2 || 
                y_pattern + ftr_cur_y > img_cur_pyr.rows - 2 )
            {
                is_in_frame = false;
                break;
            }

            const float x_img_pattern = ftr_cur_x + x_pattern;
            const float y_img_pattern = ftr_cur_y + y_pattern;
            float pattern_value = utils::interpolate_uint8((img_cur_pyr.data), x_img_pattern, y_img_pattern, stride);

            // derive patch
            float dx = 0.5*(utils::interpolate_uint8((img_cur_pyr.data), x_img_pattern+1, y_img_pattern, stride) - 
                            utils::interpolate_uint8((img_cur_pyr.data), x_img_pattern-1, y_img_pattern, stride));
            float dy = 0.5*(utils::interpolate_uint8((img_cur_pyr.data), x_img_pattern, y_img_pattern+1, stride) - 
                            utils::interpolate_uint8((img_cur_pyr.data), x_img_pattern, y_img_pattern-1, stride));
            patch_dI_.row(i_ftr*pattern_.size()+pattern_count) = Eigen::Vector2d(dx, dy);

            
            // TODO calculate residual res = ref - cur may need to add huber

            // residual_pattern[pattern_count] = ***;
            
            ++pattern_count;
        }

        if(is_in_frame)
        {
            residual_.segment(i_ftr*num_pattern, num_pattern) = residual_pattern;
            num_ftr_active_++;
        }
        else
        {
            *iter_vis_ftr = false;
            continue;
        }

        // TODO calculate Jacobian
        {
            Eigen::Matrix<double, 2, 6> jacob_xyz2uv;
            
            // jacobian_.block(num_pattern*i_ftr, 0, num_pattern, 6) = ;
        }
    }

}

bool hw::Tracker::runOptimize(Sophus::SE3d& state)
{
    optimizeGaussNetow(state);

    if(converged_ == true)
        return true;
    else
        return false;
}

void hw::Tracker::optimizeGaussNetow(Sophus::SE3d& state)
{
    reset();
    computeResidual(state);

    while(num_iter_ < num_iter_max_ && !stop_)
    {        
        Sophus::SE3d state_new(state);
        // new chi2
        double chi2_new = 0;
        // old chi2
        chi2_ = getChi2();

        // TODO calculate H and b
        // hessian_ = ;
        // jres_ = ;

        if( !solve())
        {
            stop_ = true;
        }
        else
        {
            update(state, state_new);
            // NOTICE: this will change residual
            computeResidual(state_new); 
            chi2_new = getChi2();
        }
        
        num_obser_ = jacobian_.rows();

        if(chi2_ >= chi2_new && !stop_)
        {
            state = state_new;
            chi2_ = chi2_new;
        }
        else
        {
            stop_ = true;
            converged_ = false;
        }
        
        // converged condition 

        if( utils::maxFabs(delta_x_) < epsilon_ && !std::isnan(delta_x_[0]))
        {
            converged_=true;
            stop_ = true;
        }
        ++num_iter_;

        std::cout<<"[message]"<<std::setprecision(4)
            <<"\t NO."<<num_iter_
            <<"\t Obser "<<num_obser_
            <<"\t Chi2 "<<chi2_
            <<"\t delta_x "<<delta_x_.transpose()
            <<"\t Stop "<<std::boolalpha<<stop_
            <<"\t Converged "<<std::boolalpha<<converged_
            <<std::endl;
    }

}


bool hw::Tracker::solve()
{
    delta_x_ = hessian_.ldlt().solve(jres_);
    if((double)std::isnan(delta_x_[0]))
        return false;
    return true;
}

void hw::Tracker::update(const Sophus::SE3d& old_state, Sophus::SE3d& new_state)
{
    // TODO update new state
    // new_state = ;
}