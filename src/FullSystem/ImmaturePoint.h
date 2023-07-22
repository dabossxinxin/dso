/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once
 
#include "util/NumType.h"
#include "FullSystem/HessianBlocks.h"

namespace dso
{
	struct ImmaturePointTemporaryResidual
	{
	public:
		ResState state_state;
		double state_energy;
		ResState state_NewState;
		double state_NewEnergy;
		FrameHessian* target;
	};

	enum ImmaturePointStatus {
		IPS_GOOD = 0,				// 表示最近一次在极线上进行特征点匹配成功
		IPS_OOB,					// 表示跟踪结束或其投影在当前帧中已越界
		IPS_OUTLIER,				// 表示未在极线上搜索到点跟踪失败
		IPS_SKIPPED,				// 表示跟踪成功但是最大深度与最小深度投影点距离很近不用再优化
		IPS_BADCONDITION,			// 表示一些异常情况导致匹配跟踪失败
		IPS_UNINITIALIZED			// 表示还未进行过深度更新
	};

	class ImmaturePoint
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		// static values
		float color[MAX_RES_PER_POINT];		// 特征点周围八领域的像素值
		float weights[MAX_RES_PER_POINT];	// 特征点周围八领域的梯度值

		Mat22f gradH;
		Vec2f gradH_ev;
		Mat22f gradH_eig;
		float energyTH;
		float u, v;
		FrameHessian* host;					// 特整点的主导帧
		int idxInImmaturePoints;			// 当前特征点序号

		float quality;						// 当前特征点质量
		float my_type;

		float idepth_min;					// 深度滤波器中估计的最小逆深度
		float idepth_max;					// 深度滤波器中估计的最大逆深度
		ImmaturePoint(int u_, int v_, FrameHessian* host_, float type, CalibHessian* HCalib);
		~ImmaturePoint();

		ImmaturePointStatus traceOn(FrameHessian* frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f &hostToFrame_affine, CalibHessian* HCalib, bool debugPrint = false);

		ImmaturePointStatus lastTraceStatus;// 上一次跟踪深度滤波器的状态
		Vec2f lastTraceUV;					// 上一次跟踪深度滤波器最佳匹配点位置
		float lastTracePixelInterval;		// 上一次跟踪最佳匹配位置视差

		float idepth_GT;

		// 计算当前未成熟点投影到关键帧上的残差以及雅可比
		double linearizeResidual(
			CalibHessian *  HCalib, const float outlierTHSlack,
			ImmaturePointTemporaryResidual* tmpRes,
			float &Hdd, float &bd,
			float idepth);

		float getdPixdd(
			CalibHessian *  HCalib,
			ImmaturePointTemporaryResidual* tmpRes,
			float idepth);

		float calcResidual(
			CalibHessian *  HCalib, const float outlierTHSlack,
			ImmaturePointTemporaryResidual* tmpRes,
			float idepth);

	private:
	};
}