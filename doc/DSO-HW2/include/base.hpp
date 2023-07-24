#ifndef _BASE_HPP_
#define _BASE_HPP_

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <eigen3/Eigen/Core>

#include <vector>
#include <memory>
#include <string>

namespace hw
{

typedef std::vector<cv::Mat> ImgPyr;

class Cam
{

public:
    // only one
    static Cam &getCamera()
    {
        static Cam instance;
        return instance;
    }
    static double &fx() { return getCamera().fx_; }
    static double &fy() { return getCamera().fy_; }
    static double &cx() { return getCamera().cx_; }
    static double &cy() { return getCamera().cy_; }
    static int &height() { return getCamera().height_; }
    static int &width() { return getCamera().width_; }
    static Eigen::Matrix3d &K() { return getCamera().K_; }

    static Eigen::Vector3d pixel2unitPlane(const double &u, const double &v)
    {
        Eigen::Vector3d xyz;
        xyz[0] = (u - cx()) / fx();
        xyz[1] = (v - cy()) / fy();
        xyz[2] = 1.0;
        return xyz;
    }

    static Eigen::Vector2d project(const Eigen::Vector3d &p)
    {
        return (K() * (p / p[2])).head(2);
    }

private:

    Cam() : fx_(315.5),
            fy_(315.5),
            cx_(376.0),
            cy_(240.0),
            height_(480),
            width_(752)
    {
        K_ << fx_, 0, cx_,
            0, fy_, cy_,
            0, 0, 1;
    }

    Cam(Cam const &);
    void operator=(Cam const &);

    double fx_, fy_, cx_, cy_;
    int height_, width_;
    Eigen::Matrix3d K_;
};

class Frame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Frame> Ptr;

    size_t num_max_level_;              //!< Max numbers of levels of pyramid
    std::vector<cv::KeyPoint> features; //!< the corner of image
    static int frame_count_;            //!< Count the number of frames

    void createImgPyr(const cv::Mat &img);
    void setPose(const Sophus::SE3d &pose);
    inline Sophus::SE3d getPose() const { return T_c_w_; }
    inline const cv::Mat &getImage(size_t level) const
    {
        assert(level <= num_max_level_ && level >= 0);
        return img_pyr_[level];
    }
    inline float getDepth(int u, int v) const { return img_depth_.at<float>(v, u); }

    inline static Ptr create(const cv::Mat &img, const cv::Mat &depth, const size_t max_pyr)
    {
        return Ptr(new Frame(img, depth, max_pyr));
    }

protected:
    Frame(const cv::Mat &img, const cv::Mat &depth, const size_t max_pyr);
    Frame(const Frame &) = delete;
    Frame &operator=(const Frame &) = delete;

    Sophus::SE3d T_c_w_; //!< Pose Transform from world to camera
    cv::Mat img_depth_;  //!< image of depth
    ImgPyr img_pyr_;     //!< Image pyramid
};

class DatasetReader
{
public:
    DatasetReader(const std::string &path) : dataset_file(path)
    {
        size_t found = path.find_last_of("/\\");
        if (found + 1 != path.size())
            dataset_file += dataset_file.substr(found, 1);

        dataset_file += "trajectory.txt";
        std::ifstream data_stream;
        data_stream.open(dataset_file.c_str());
        assert(data_stream.is_open());

        while (!data_stream.eof())
        {
            std::string s;
            std::getline(data_stream, s);
            if (!s.empty())
            {
                std::stringstream ss;
                ss << s;
                double times, tx, ty, tz, qx, qy, qz, qw;
                Eigen::Quaterniond q;
                std::string file_name;
                ss >> times;
                timestamps_.push_back(times);
                ss >> file_name;
                image_names_.push_back(file_name);
                ss >> tx;
                ss >> ty;
                ss >> tz;
                t_.push_back(Eigen::Vector3d(tx, ty, tz));
                ss >> qx;
                ss >> qy;
                ss >> qz;
                ss >> qw;
                q = Eigen::Quaterniond(qw, qx, qy, qz);
                q.normalize();
                q_.push_back(q);
            }
        }

        data_stream.close();

        N_ = timestamps_.size();
        if (N_ == 0)
            std::cout << "Nothing in datasets " << dataset_file << std::endl;
        else
            std::cout << "Find items in datasets: num = " << N_ << std::endl;
    }

    bool readByIndex(size_t index, double &timestamp, std::string &filename, Eigen::Vector3d &t, Eigen::Quaterniond &q)
    {
        if (index >= N_)
        {
            std::cerr << " Index(" << index << ") is out of scape, max should be (0~" << N_ - 1 << ")";
            return false;
        }
        timestamp = timestamps_[index];
        filename = image_names_[index];
        t = t_[index];
        q = q_[index];
        return true;
    }

    int size() { return N_; }

public:
    size_t N_;
    std::string dataset_file;
    std::vector<double> timestamps_;
    std::vector<std::string> image_names_; // include rgb and depth
    std::vector<Eigen::Vector3d> t_;
    std::vector<Eigen::Quaterniond> q_;
};

namespace utils
{
typedef unsigned char uint8_t;

inline double maxFabs(const Eigen::VectorXd &v)
{
    double max = -1.;
    // turn Eigen::vectorXd to std::vector
    std::vector<double> value(v.data(), v.data() + v.rows() * v.cols());
    // find max
    std::for_each(value.begin(), value.end(), [&](double &val) {
        max = fabs(val) > max ? fabs(val) : max;
    });
    return max;
}

// from rpg_vikit vision.cpp
inline bool is_aligned16(const void *ptr)
{
    return ((reinterpret_cast<size_t>(ptr)) & 0xF) == 0;
}

inline void halfSampleSSE2(const unsigned char *in, unsigned char *out, int w, int h)
{
    const unsigned long long mask[2] = {0x00FF00FF00FF00FFull, 0x00FF00FF00FF00FFull};
    const unsigned char *nextRow = in + w;
    __m128i m = _mm_loadu_si128((const __m128i *)mask);
    int sw = w >> 4;
    int sh = h >> 1;
    for (int i = 0; i < sh; i++)
    {
        for (int j = 0; j < sw; j++)
        {
            __m128i here = _mm_load_si128((const __m128i *)in);
            __m128i next = _mm_load_si128((const __m128i *)nextRow);
            here = _mm_avg_epu8(here, next);
            next = _mm_and_si128(_mm_srli_si128(here, 1), m);
            here = _mm_and_si128(here, m);
            here = _mm_avg_epu16(here, next);
            _mm_storel_epi64((__m128i *)out, _mm_packus_epi16(here, here));
            in += 16;
            nextRow += 16;
            out += 8;
        }
        in += w;
        nextRow += w;
    }
}

inline void halfSample(const cv::Mat &in, cv::Mat &out)
{
    assert(in.rows / 2 == out.rows && in.cols / 2 == out.cols);
    assert(in.type() == CV_8U && out.type() == CV_8U);

    if (is_aligned16(in.data) && is_aligned16(out.data) && ((in.cols % 16) == 0))
    {
        halfSampleSSE2(in.data, out.data, in.cols, in.rows);
        return;
    }
    // four pixel merge to one
    const int stride = in.step.p[0];
    uint8_t *top = (uint8_t *)in.data;
    uint8_t *bottom = top + stride;
    uint8_t *end = top + stride * in.rows;
    const int out_width = out.cols;
    uint8_t *p = (uint8_t *)out.data;
    while (bottom < end)
    {
        for (int j = 0; j < out_width; j++)
        {
            *p = static_cast<uint8_t>((uint16_t(top[0]) + top[1] + bottom[0] + bottom[1]) / 4);
            p++;
            top += 2;
            bottom += 2;
        }
        top += stride;
        bottom += stride;
    }
}

/******************
 * 坐标系  -------> x
 *         |
 *         |
 *         y
 ******************/
//! the data should be the origin, or it may appear memory leak
inline float interpolate_uint8(const uint8_t *data, const float x, const float y, const int stride)
{
    const int x_i = floor(x);
    const int y_i = floor(y);
    const float subpix_x = x - x_i;
    const float subpix_y = y - y_i;
    const float w_tl = (1.f - subpix_x) * (1.f - subpix_y);
    const float w_tr = (subpix_x) * (1.f - subpix_y);
    const float w_bl = (1.f - subpix_x) * (subpix_y);
    const float w_br = subpix_x * subpix_y;

    const uint8_t *data_cur = data + y_i * stride + x_i;

    return w_tl * data_cur[0] + w_tr * data_cur[1] + w_bl * data_cur[stride] + w_br * data_cur[stride + 1];
}

} // end namespace utils
} // end namespace hw

#endif //_BASE_HPP_